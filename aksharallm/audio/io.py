"""Getting a waveform off the disk, and changing its sample rate — both from scratch.

This is the least glamorous file in the audio phase and the one most likely to ruin a run
silently, so it is also the strictest. Two jobs:

**1. Read a WAV.** The stdlib's `wave` module hands over a bag of bytes and the header that
describes them; interpreting those bytes as numbers is ours. A WAV frame is *interleaved* —
`L R L R L R` for stereo — and the samples are little-endian signed integers (except 8-bit,
which is unsigned with a bias of 128, a genuine wart of the format). We normalise to
**float32 in [-1, 1]** by dividing by the type's maximum, because every later stage (the
STFT, the codec's reconstruction loss, the decoder's `tanh`) assumes that range.

**2. Resample.** LJSpeech is 22,050 Hz, LibriSpeech is 16,000, and the codec must be shown
one number or the frame rate of its tokens means two different things in the same dataset.
Resampling is not "take every other sample": that folds every frequency above the new
Nyquist rate back down into the audible band as a lower tone that was never there
(*aliasing*), and it cannot be undone afterwards. The correct operation is to reconstruct
the continuous signal implied by the samples and read it at new times, which is a
convolution with a `sinc`:

```mermaid
flowchart LR
    A["samples at 22,050 Hz"] --> B["windowed sinc<br/>low-pass at the LOWER Nyquist"]
    B --> C["read at 16,000 Hz<br/>positions"]
    C --> D["samples at 16,000 Hz"]
```

The low-pass has to happen *before* the new samples are taken, and its cutoff is the lower
of the two Nyquist rates — that is the whole of anti-aliasing.

**The assertion rule (trap 5 of the phase, written down before it bit us):** sample rate,
channel count and amplitude are **reported and checked, never silently repaired**. A dataset
that is half 22 kHz and half resampled is a bug you hear only at the end of a day of
training, and every convenience function that quietly fixes its input is how one is built.
`read_wav` tells you what it found; `load_audio` converts only what you asked it to convert
and records what it did in the `Clip` it returns.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

#: Sample rate everything downstream speaks. 16 kHz keeps frequencies to 8 kHz, which holds
#: all of speech (the highest fricatives sit around 6-8 kHz) and is half the data of 22 kHz.
#: Music would want 44.1; we are building a speech codec.
TARGET_SR = 16_000

#: How many sinc side-lobes the resampling kernel keeps. 16 is inaudibly good and cheap;
#: below ~8 the stop-band leaks and you get a faint whistle on tonal material.
SINC_ZEROS = 16

#: Resampling builds an (output x kernel) matrix. Chunked so a 10-minute clip does not ask
#: for a gigabyte to change its sample rate.
_CHUNK = 65_536


@dataclass(frozen=True)
class Clip:
    """One mono waveform plus the provenance of how it got that way.

    The fields after `samples` exist so a dataset builder can *assert* rather than hope:
    `orig_sr` and `orig_channels` are what was on disk, not what we turned it into.
    """

    samples: np.ndarray  # float32, shape (n,), nominally in [-1, 1]
    sample_rate: int
    path: str = ""
    orig_sr: int = 0
    orig_channels: int = 0
    peak: float = 0.0  # |x|.max() BEFORE any normalisation, so clipping is visible

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    def describe(self) -> str:
        bits = [f"{self.seconds:.2f}s", f"{self.sample_rate} Hz", f"peak {self.peak:.3f}"]
        if self.orig_sr and self.orig_sr != self.sample_rate:
            bits.append(f"resampled from {self.orig_sr}")
        if self.orig_channels > 1:
            bits.append(f"downmixed from {self.orig_channels}ch")
        return " · ".join(bits)


# ---------------------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------------------


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into float32 in [-1, 1], shape `(n_samples, n_channels)`.

    Channels are kept. Deciding what to do with them is the caller's problem, on purpose —
    see the assertion rule in the module docstring.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    # `wave` guarantees PCM but not a width numpy has a dtype for. 24-bit exists in the wild
    # and needs its three bytes widening by hand; refusing loudly beats a garbled read.
    if width == 1:
        # 8-bit WAV is UNSIGNED, biased by 128. Everything wider is signed. Yes, really.
        data = (raw_to_array(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = raw_to_array(raw, np.int16).astype(np.float32) / 32768.0
    elif width == 3:
        b = raw_to_array(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        packed = np.where(packed >= 1 << 23, packed - (1 << 24), packed)  # sign-extend
        data = packed.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = raw_to_array(raw, np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path.name}: {width * 8}-bit WAV is not supported")

    return data.reshape(-1, n_channels), sr


def raw_to_array(raw: bytes, dtype) -> np.ndarray:
    """Little-endian bytes to numbers. Split out because it is the one line that would be
    wrong on a big-endian machine, and naming it makes that visible."""
    return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 in [-1, 1] as 16-bit PCM.

    Values outside the range are **clipped, and that is a real loss** — a decoder whose
    output regularly saturates is a decoder with a problem, so `Clip.peak` keeps the number
    around rather than letting the clip hide it.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    ints = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(ints.tobytes())


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average the channels. Mean, not sum: summing doubles the amplitude of a stereo file
    and every clip in the dataset then sits 6 dB louder than every mono one."""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1) if x.ndim == 1 else x.mean(axis=1)


def load_audio(
    path: str | Path,
    sample_rate: int | None = TARGET_SR,
    *,
    normalize: bool = False,
) -> Clip:
    """Read a WAV and put it in the shape the rest of the phase expects.

    `sample_rate=None` means "whatever is on disk" — used by the inspector so you can see
    the original. `normalize` rescales the peak to 0.95; it is **off by default**, because
    per-clip normalisation destroys the relative loudness of a corpus and a codec trained on
    it learns nothing about quiet speech.
    """
    data, sr = read_wav(path)
    channels = data.shape[1]
    mono = to_mono(data)
    peak = float(np.abs(mono).max()) if mono.size else 0.0

    out = mono
    if sample_rate is not None and sample_rate != sr:
        out = resample(mono, sr, sample_rate)
    if normalize and peak > 0:
        out = out * (0.95 / peak)

    return Clip(
        samples=out.astype(np.float32, copy=False),
        sample_rate=sample_rate if sample_rate is not None else sr,
        path=str(path),
        orig_sr=sr,
        orig_channels=channels,
        peak=peak,
    )


# ---------------------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------------------


def kaiser(u: np.ndarray, half: float, beta: float = 8.6) -> np.ndarray:
    """A Kaiser window evaluated at continuous offsets `u`, zero outside `|u| <= half`.

    Continuous rather than "an array of length N" because each resampling phase samples the
    window a fraction of a step off the integer grid. Reusing one integer-indexed window for
    every phase leaves a step at the kernel's edge — small, but a discontinuity convolved
    with the signal is the least forgiving thing to leave in a filter.

    Kaiser rather than Hann because `beta` trades stop-band attenuation against transition
    width on a dial, and for a resampler we want the stop band very deep: energy that leaks
    through it is aliasing, which is audible and permanent. `beta = 8.6` is about −90 dB.
    """
    r = np.clip(np.asarray(u, dtype=np.float64) / half, -1.0, 1.0)
    w = np.i0(beta * np.sqrt(1.0 - r * r)) / np.i0(beta)
    return np.where(np.abs(u) <= half, w, 0.0)


def resample(x: np.ndarray, sr_in: int, sr_out: int, *, zeros: int = SINC_ZEROS) -> np.ndarray:
    """Change the sample rate of a mono signal by windowed-sinc interpolation.

    The maths, in four lines:

    * output sample `j` sits at input position `p = j · sr_in / sr_out`;
    * `x` at a non-integer position is `Σ_k x[k] · sinc(p − k)` — the exact reconstruction
      of a band-limited signal, which is the sampling theorem read forwards;
    * `sinc` is infinite, so it is truncated to `zeros` lobes and tapered by a Kaiser window
      (truncating without a window rings, and the ringing is audible);
    * when *downsampling*, the sinc is stretched by `sr_out / sr_in` so its cutoff drops to
      the **new** Nyquist rate. That single factor is the anti-aliasing filter; without it
      an 8 kHz tone in a 22 kHz file becomes a 6 kHz tone in a 16 kHz one.

    Only `L = sr_out / gcd` distinct fractional offsets ever occur, so the kernel is built
    once per *phase* and reused — `j` and `j + L` land at the same place between samples.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x.copy()
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError(f"sample rates must be positive, got {sr_in} -> {sr_out}")

    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g  # p = j * down / up

    # zeta < 1 when downsampling: it both stretches the sinc (lower cutoff) and is the
    # gain correction that keeps the amplitude the same. Upsampling needs neither.
    zeta = min(1.0, up / down)
    half = int(np.ceil(zeros / zeta))
    width = 2 * half + 1

    # One kernel per fractional phase. Output j sits at p = j·down/up, whose fractional part
    # is `((j·down) mod up) / up` — so the *residue* is the table index, and entry r is the
    # phase r/up. (Indexing a table built in j-order by that residue is a real bug and an
    # invisible one: it stays a valid resampler and merely time-warps by up to one sample,
    # which is inaudible on a low tone and destroys a high one.)
    phases = np.arange(up, dtype=np.float64) / up
    offs = np.arange(width, dtype=np.float64) - half
    u = offs[None, :] - phases[:, None]  # (up, width) distance in input samples
    kernels = (zeta * np.sinc(zeta * u) * kaiser(u, half)).astype(np.float32)

    n_out = int(np.ceil(x.size * up / down))
    # Zero-pad so the first and last outputs can see a full kernel without index games.
    padded = np.concatenate([np.zeros(half, np.float32), x, np.zeros(half + width, np.float32)])

    out = np.empty(n_out, dtype=np.float32)
    for start in range(0, n_out, _CHUNK):
        j = np.arange(start, min(start + _CHUNK, n_out))
        base = (j * down) // up  # integer part of the input position
        idx = base[:, None] + np.arange(width)[None, :]  # already offset by the left pad
        out[start : start + len(j)] = np.einsum(
            "ij,ij->i", padded[idx], kernels[(j * down) % up]
        )
    return out
