"""Turning a waveform into a picture, and the picture back into sound — from scratch.

A waveform at 16 kHz is 16,000 numbers a second and almost none of them mean anything on
their own. What means something is *which frequencies are present, and when* — so the first
thing anyone does with audio is cut it into short overlapping frames, take the Fourier
transform of each, and stack the results into an image with time across and frequency up.
That image is the **spectrogram**, and everything in this phase is built on it:

```mermaid
flowchart LR
    W["waveform<br/>16,000 samples/s"] --> F["frame<br/>1024 wide, hop 256"]
    F --> H["multiply by a<br/>Hann window"]
    H --> R["rfft<br/>513 complex bins"]
    R --> M["magnitude"]
    M --> B["mel filterbank<br/>513 -> 80"]
    B --> L["log"]
    L --> S["log-mel<br/>80 x frames"]
```

Three of those steps deserve a sentence before you read the code.

**Why a window.** Cutting a frame out of a signal multiplies it by a rectangle, and a
rectangle has a Fourier transform full of side-lobes, so a single pure tone smears across
every bin — *spectral leakage*. A Hann window tapers the frame to zero at both ends, which
trades a slightly wider main lobe for side-lobes ~30 dB lower. It is also what makes the
transform invertible: with a hop of `n_fft/4`, the overlapping Hann windows sum to a
constant (the COLA condition), so overlap-add reconstructs the original **exactly**.
`test_audio.py` asserts that to 1e-6, because if it is not true then every reconstruction
loss in the codec is measuring the window rather than the model.

**Why mel.** 513 linear frequency bins spend most of their resolution above 4 kHz, where
human hearing barely distinguishes anything. The mel scale is roughly linear below 1 kHz and
logarithmic above it — it is a fit to how far apart two tones must be to sound different.
Collapsing 513 bins onto 80 triangular mel filters throws away detail we cannot hear and
keeps the detail we can, which is why 80 mel bands is the standard input to a speech model.

**Why log.** Loudness is perceived logarithmically, and the raw magnitudes span six orders
of magnitude. A network trained on linear magnitudes spends all of its capacity on the
loudest frames. The floor (`log_eps`) matters: it sets how far down silence is allowed to
go, and too small a floor means the quietest frames dominate the loss with pure noise.

**Griffin-Lim, and why it is here.** A magnitude spectrogram has thrown away phase, so it
cannot be inverted directly. Griffin-Lim guesses: start from random (or zero) phase, invert,
re-transform, keep the *original* magnitudes with the *new* phase, repeat. It converges to
something that sounds like the original — muffled, slightly metallic, unmistakably the same
words. The reason to build it is not quality; it is that **you can listen to what the model
sees**. A log-mel with a broken filterbank is a plausible-looking picture and an obviously
wrong sound.

Everything here is torch, on purpose: the codec's reconstruction loss is a multi-scale STFT
and has to be differentiable and run on the GPU. `io.py` is the numpy half.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MelConfig:
    """The spectrogram everything downstream agrees on.

    These five numbers are a contract: change `hop` and every codec checkpoint's token rate
    changes, change `n_mels` and every saved feature file is a different shape. They live in
    one frozen object so a checkpoint can record which one it was trained with.
    """

    sample_rate: int = 16_000
    n_fft: int = 1024  # 64 ms — long enough to resolve pitch, short enough to be local
    hop: int = 256  # 16 ms, n_fft/4: the COLA hop for a Hann window
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None  # None -> Nyquist
    log_eps: float = 1e-5  # the floor, i.e. how far below full scale silence lands

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop

    @property
    def top_hz(self) -> float:
        return self.fmax if self.fmax is not None else self.sample_rate / 2


# ---------------------------------------------------------------------------------------
# windows and framing
# ---------------------------------------------------------------------------------------


def hann(n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """A periodic Hann window: `0.5 · (1 − cos(2πk/n))`.

    **Periodic, not symmetric** — the denominator is `n`, not `n − 1`. The symmetric version
    is for filter design; for an STFT the periodic one is what satisfies COLA, and using the
    wrong one leaves a slow ripple in the reconstruction that looks like a bug in the model.
    """
    k = torch.arange(n, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * k / n)


def frame(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Cut `(..., n)` into overlapping frames `(..., frames, n_fft)` with no copy.

    `as_strided` through `unfold`: the frames overlap by `n_fft − hop` samples and share
    that memory rather than duplicating it, which for hop = n_fft/4 is the difference
    between one copy of the audio and four.
    """
    return x.unfold(-1, n_fft, hop)


def stft(x: torch.Tensor, cfg: MelConfig, *, center: bool = True) -> torch.Tensor:
    """Short-time Fourier transform. `(..., n)` -> complex `(..., n_freqs, frames)`.

    `center=True` pads by `n_fft/2` at both ends in **reflect** mode, so frame *t* is
    centred on sample `t · hop` and the first frame describes the beginning of the signal
    rather than starting half a window in. Reflect rather than zeros because a hard edge to
    silence is a broadband click, and the first frame would be a picture of the click.
    """
    lead, n = x.shape[:-1], x.shape[-1]
    win = hann(cfg.n_fft, device=x.device, dtype=x.dtype)
    flat = x.reshape(-1, n)
    if center:
        pad = cfg.n_fft // 2
        # `reflect` needs a channel dim; add one and take it back off.
        flat = torch.nn.functional.pad(flat.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    frames = frame(flat, cfg.n_fft, cfg.hop) * win
    spec = torch.fft.rfft(frames, n=cfg.n_fft, dim=-1).transpose(-1, -2)
    return spec.reshape(*lead, cfg.n_freqs, spec.shape[-1])


def istft(spec: torch.Tensor, cfg: MelConfig, *, center: bool = True, length: int | None = None):
    """Invert an STFT by weighted overlap-add. Complex `(..., n_freqs, frames)` -> `(..., n)`.

    Each frame is transformed back, multiplied by the window a *second* time and added into
    the output. Two multiplications, not one: the synthesis window suppresses the
    discontinuities that a modified spectrogram (Griffin-Lim's, or a codec's) leaves at the
    frame edges. That means the overlap sums to `Σ w²` rather than `Σ w`, so the result is
    divided by exactly that — computed here rather than assumed, which is what makes this
    exact for any hop instead of only for `n_fft/4`.
    """
    win = hann(cfg.n_fft, device=spec.device, dtype=torch.float32)
    lead = spec.shape[:-2]
    spec = spec.reshape(-1, cfg.n_freqs, spec.shape[-1])
    frames = torch.fft.irfft(spec.transpose(-1, -2), n=cfg.n_fft, dim=-1) * win  # (B, T, n_fft)

    n_frames = frames.shape[1]
    out_len = cfg.n_fft + cfg.hop * (n_frames - 1)
    # fold() is overlap-add: it is the transpose of unfold, which is exactly what we want.
    out = torch.nn.functional.fold(
        frames.transpose(1, 2), (1, out_len), (1, cfg.n_fft), stride=(1, cfg.hop)
    ).reshape(-1, out_len)
    norm = torch.nn.functional.fold(
        (win * win).reshape(1, cfg.n_fft, 1).expand(1, cfg.n_fft, n_frames),
        (1, out_len), (1, cfg.n_fft), stride=(1, cfg.hop),
    ).reshape(out_len)
    out = out / norm.clamp_min(1e-8)

    if center:
        # Strip the analysis padding from the FRONT only. Stripping the same amount off the
        # back throws away real samples: `unfold` already dropped the trailing partial frame,
        # so the tail is short by up to one hop and cutting another n_fft/2 loses audio that
        # was reconstructed correctly. `length` is what trims the end, and the caller knows it.
        out = out[:, cfg.n_fft // 2 :]
    if length is not None:
        out = (
            out[:, :length]
            if out.shape[-1] >= length
            else torch.nn.functional.pad(out, (0, length - out.shape[-1]))
        )
    return out.reshape(*lead, out.shape[-1])


# ---------------------------------------------------------------------------------------
# the mel scale
# ---------------------------------------------------------------------------------------


def hz_to_mel(f):
    """The HTK formula: `2595 · log10(1 + f/700)`.

    There are two conventions in the wild (this one and Slaney's piecewise linear-then-log)
    and they disagree by a few percent. Neither is more correct; what matters is that the
    *same* one is used to build the filterbank and to read it. Written as one function for
    exactly that reason.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: MelConfig, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """The `(n_mels, n_freqs)` matrix that collapses an FFT onto mel bands.

    `n_mels + 2` points equally spaced *on the mel scale* become the left edge, peak and
    right edge of `n_mels` overlapping triangles, each sharing its peak with the next one's
    edge. A band therefore rises linearly from the previous peak to its own and falls to the
    next — the classic overlapping-triangle picture.

    **Each filter is normalised to unit area** (Slaney normalisation): without it the wide
    high-frequency triangles pass many more bins than the narrow low ones, so a flat-spectrum
    input comes out as a rising ramp and the network spends its first epochs unlearning the
    filterbank. A test asserts a white-noise spectrum stays roughly flat through it.
    """
    n_freqs = cfg.n_freqs
    fft_hz = np.linspace(0.0, cfg.sample_rate / 2.0, n_freqs)
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.top_hz), cfg.n_mels + 2))

    fb = np.zeros((cfg.n_mels, n_freqs), dtype=np.float64)
    for m in range(cfg.n_mels):
        left, peak, right = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rise = (fft_hz - left) / max(peak - left, 1e-9)
        fall = (right - fft_hz) / max(right - peak, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(rise, fall))
        # Unit area in Hz, so a filter's output does not scale with its width.
        width = max(right - left, 1e-9)
        fb[m] *= 2.0 / width

    return torch.as_tensor(fb, device=device, dtype=dtype)


# ---------------------------------------------------------------------------------------
# the feature
# ---------------------------------------------------------------------------------------


def magnitude(x: torch.Tensor, cfg: MelConfig) -> torch.Tensor:
    """`|STFT|` — the picture without its phase. `(..., n_freqs, frames)`."""
    return stft(x, cfg).abs()


def log_mel(x: torch.Tensor, cfg: MelConfig, *, fb: torch.Tensor | None = None) -> torch.Tensor:
    """Waveform `(..., n)` -> log-mel `(..., n_mels, frames)`.

    The feature. Pass `fb` to reuse a filterbank across a batch — building it is a Python
    loop over 80 triangles and doing that once per training step is measurable.
    """
    if fb is None:
        fb = mel_filterbank(cfg, device=x.device, dtype=torch.float32)
    mag = magnitude(x, cfg)
    mel = torch.matmul(fb.to(mag.dtype), mag)
    return torch.log(mel.clamp_min(cfg.log_eps))


def mel_to_magnitude(mel_log: torch.Tensor, cfg: MelConfig) -> torch.Tensor:
    """Undo the filterbank, approximately, so a log-mel can be listened to.

    80 numbers cannot be turned back into 513: the filterbank is not invertible and this is
    a pseudo-inverse, i.e. the least-squares best guess. What comes back is recognisably the
    same speech with the fine harmonic structure smeared — which is the honest demonstration
    that the mel front end is *lossy* and that the codec's job is harder than it looks.
    """
    fb = mel_filterbank(cfg, device=mel_log.device, dtype=torch.float32)
    mel = torch.exp(mel_log)
    inv = torch.linalg.pinv(fb)  # (n_freqs, n_mels)
    return torch.matmul(inv, mel).clamp_min(0.0)


def griffin_lim(
    mag: torch.Tensor,
    cfg: MelConfig,
    *,
    n_iter: int = 60,
    momentum: float = 0.99,
    length: int | None = None,
    seed: int | None = 0,
) -> torch.Tensor:
    """Recover a waveform from magnitudes alone, by iterated projection.

    Each round: invert with the current phase estimate, transform forward again, and throw
    away the magnitude that comes back — keeping only its *phase*, paired with the
    magnitudes we were given. Two projections, onto "signals with this magnitude" and onto
    "spectrograms that are the STFT of some real signal"; the intersection is what we want.

    `momentum` is the fast Griffin-Lim trick: step a little past the new estimate along the
    direction of travel. 0.99 is the published value and roughly halves the iterations. At
    `momentum = 0` this is the original 1984 algorithm, and the test suite checks that
    convergence is monotone-ish either way.

    `seed=None` starts from zero phase, which is deterministic but gives every frame the
    same phase at every frequency — an audible buzz on the first iterations. Random phase
    converges faster; the seed keeps it reproducible.
    """
    mag = mag.to(torch.float32)
    # Without a target length the loop is not stable: `istft` returns a few samples more
    # than the frames strictly cover, `stft` then reports one or two extra frames, and the
    # next iteration cannot subtract the previous estimate at all. Pin it to the length that
    # reproduces exactly this many frames — `stft` makes `1 + n//hop` of them.
    if length is None:
        length = cfg.hop * (mag.shape[-1] - 1)
    if seed is None:
        angle = torch.zeros_like(mag)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        angle = (torch.rand(mag.shape, generator=g) * 2 * math.pi).to(mag.device)
    spec = torch.polar(mag, angle)

    # The published update is `c = t_n + momentum·(t_n − t_{n−1})` renormalised, which after
    # the magnitude projection is the same direction as `t_n − (momentum/(1+momentum))·t_{n−1}`.
    # That denominator is not decoration: feeding the raw `momentum` in overshoots, and 0.99
    # then converges *worse* than no momentum at all — measured, not assumed.
    beta = momentum / (1.0 + momentum)

    prev = torch.zeros_like(spec)
    for _ in range(n_iter):
        wave = istft(spec, cfg, length=length)
        rebuilt = stft(wave, cfg)
        # The momentum term looks at where the estimate came from, not just where it is.
        step = rebuilt - beta * prev
        prev = rebuilt
        spec = torch.polar(mag, torch.angle(step))

    return istft(spec, cfg, length=length)


def spectral_convergence(target: torch.Tensor, pred: torch.Tensor) -> float:
    """`‖|T| − |P|‖ / ‖|T|‖` — how far two magnitude spectrograms are apart, scale-free.

    The number Griffin-Lim is minimising, and the one to watch when deciding how many
    iterations are enough. Around 0.3 is "muffled but clearly the same words"; under 0.1 is
    hard to tell from the original on speech.

    **The arguments are ordered, and it matters.** The denominator is the *target*'s norm, so
    passing the reconstruction first divides by the wrong thing — and because a half-trained
    decoder is usually quieter than its input, the mistake makes the number *worse* than the
    truth, which is the direction nobody double-checks. The parameters are named rather than
    `a, b` for exactly that reason.
    """
    return float(torch.linalg.norm(target - pred) / torch.linalg.norm(target).clamp_min(1e-9))
