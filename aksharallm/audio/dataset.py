"""Getting a speech corpus onto disk in a shape the codec can read at full speed.

This is `data/prepare.py` for sound, and it is deliberately the same design: walk a corpus
once, convert everything to the one format the trainer understands, and write a **single
flat file** plus a manifest. The trainer then `np.memmap`s it and takes random windows —
zero copies, no dataloader workers, the OS page cache doing the buffering. The only
difference from the text pipeline is the dtype: `int16` samples rather than `uint16` tokens.

```mermaid
flowchart LR
    W["a folder of<br/>WAV files"] --> R["read, downmix,<br/>resample to 16 kHz"]
    R --> C["concatenate into<br/>audio.bin (int16)"]
    C --> M["manifest.json<br/>clip offsets + provenance"]
    M --> D["AudioDataset<br/>random windows"]
```

**Why int16 on disk and float32 in memory.** 24 hours of 16 kHz mono is 2.7 GB as int16 and
5.5 GB as float32, and the conversion is one divide on a window of 32,000 samples. Storing
floats would double the disk for a precision the source WAV never had in the first place.

**The one rule that is not obvious: a window must not straddle two clips.** The bin is a
plain concatenation, so a naive random offset can land 10 ms before the end of one utterance
and take the rest of its window from the start of another. That join is a discontinuity that
occurs nowhere in real speech, and a codec trained on it spends capacity learning to
reconstruct an artefact of our file format. `AudioDataset` samples a clip first and an offset
within it second, which costs one extra lookup and removes the problem entirely.

**Assertions, not repairs** (trap 5): every clip's original sample rate and channel count are
recorded in the manifest, and `pack` **refuses** a corpus whose files disagree about sample
rate unless it was told what to resample to. A corpus half at 22 kHz and half at 16 is a bug
you hear only after a day of training.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .io import TARGET_SR, load_audio, write_wav

#: LJSpeech: 24 hours of one reader, public domain, 2.6 GB compressed. The right first
#: target because one speaker removes speaker identity as a variable — the codec has enough
#: to learn without also learning what different people sound like.
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

#: Names for the five synthetic vowel targets, in the order `synth_corpus` defines them.
#: These become the *transcript*, which is what makes TTS and ASR measurable on a corpus
#: nobody had to download.
VOWEL_NAMES = ("aa", "ee", "eh", "oh", "oo")


@dataclass
class Manifest:
    """What is in an `audio.bin`, and where each clip starts.

    `offsets` has `n_clips + 1` entries — the last is the total length — so clip *i* is
    `samples[offsets[i]:offsets[i+1]]` with no special case for the end.
    """

    sample_rate: int
    offsets: list[int]
    names: list[str]
    sources: list[dict]  # per clip: original sr, channels, peak
    seconds: float
    built: str
    #: Where the WAVs came from. Recorded because a packed corpus usually lives somewhere
    #: else entirely (`data/audio/lj` from `data/audio/ljspeech/LJSpeech-1.1/wavs`), and the
    #: transcripts stay with the originals — so without this, `load_transcripts` has nowhere
    #: to look. Defaulted so manifests written before this existed still load.
    source_dir: str = ""

    @property
    def n_clips(self) -> int:
        return len(self.names)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))


def pack(
    wavs: list[Path],
    out_dir: str | Path,
    *,
    sample_rate: int | None = TARGET_SR,
    min_seconds: float = 0.5,
    progress=print,
) -> Manifest:
    """Convert a list of WAV files into `<out_dir>/audio.bin` + `manifest.json`.

    `sample_rate=None` means "keep whatever is on disk", which is only legal if every file
    agrees — otherwise the bin would be a mixture of two time bases with nothing recording
    which is which, and every frame rate downstream would be a lie.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "audio.bin"

    offsets, names, sources = [0], [], []
    total = 0
    seen_sr: set[int] = set()

    with open(bin_path, "wb") as f:
        for i, wav in enumerate(wavs):
            clip = load_audio(wav, sample_rate=sample_rate)
            seen_sr.add(clip.orig_sr)
            if sample_rate is None and len(seen_sr) > 1:
                raise ValueError(
                    f"{wav.name}: corpus mixes sample rates {sorted(seen_sr)}. Pass an "
                    "explicit sample_rate so every clip is resampled to one of them."
                )
            if clip.seconds < min_seconds:
                continue  # a 200 ms file is a labelling error more often than it is speech
            # int16 is the storage format; the conversion back is one divide in the loader.
            f.write(np.clip(clip.samples * 32767.0, -32768, 32767).astype("<i2").tobytes())
            total += len(clip.samples)
            offsets.append(total)
            names.append(wav.name)
            sources.append(
                {"sr": clip.orig_sr, "channels": clip.orig_channels, "peak": round(clip.peak, 4)}
            )
            if progress and (i + 1) % 200 == 0:
                progress(f"  {i + 1}/{len(wavs)} clips, {total / (sample_rate or 16000) / 3600:.2f}h")

    sr = sample_rate if sample_rate is not None else seen_sr.pop()
    man = Manifest(
        sample_rate=sr,
        offsets=offsets,
        names=names,
        sources=sources,
        seconds=total / sr,
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
        # The common parent of the inputs, so transcripts can be found later.
        source_dir=str(Path(os.path.commonpath([str(w) for w in wavs])).resolve())
        if wavs else "",
    )
    man.save(out_dir / "manifest.json")
    if progress:
        progress(
            f"{len(names)} clips, {man.seconds / 3600:.2f} h, "
            f"{bin_path.stat().st_size / 1e9:.2f} GB -> {bin_path}"
        )
    return man


class AudioDataset:
    """Random fixed-length windows of a packed corpus, as float32 in [-1, 1].

    Mirrors `data/loader.py`'s `TokenDataset` — a memmap, a generator, and a `batch()` that
    hands back a tensor on the right device. The one structural difference is the
    clip-boundary rule in the module docstring.
    """

    def __init__(
        self,
        path: str | Path,
        window: int,
        device: str = "cpu",
        *,
        seed: int | None = None,
        split: str = "train",
        val_clips: int = 16,
    ):
        path = Path(path)
        d = path if path.is_dir() else path.parent
        self.manifest = Manifest.load(d / "manifest.json")
        self.samples = np.memmap(d / "audio.bin", dtype="<i2", mode="r")
        self.window = window
        self.device = device
        self.rng = np.random.default_rng(seed)

        # Held out by CLIP, never by offset: two windows from the same utterance are close
        # to duplicates of each other, so splitting inside a clip leaks the validation set
        # into training and makes the val curve look better than the model is.
        n = self.manifest.n_clips
        val_clips = min(val_clips, max(1, n // 10))
        clips = range(n - val_clips, n) if split == "val" else range(n - val_clips)
        off = self.manifest.offsets
        # Only clips long enough to hold a whole window are usable.
        self.clips = [i for i in clips if off[i + 1] - off[i] >= window]
        if not self.clips:
            raise ValueError(
                f"no {split} clip is at least {window} samples "
                f"({window / self.manifest.sample_rate:.2f}s) long"
            )
        # Sampling clips uniformly would over-represent short utterances; weighting by
        # length makes every SECOND of the corpus equally likely, which is what "a random
        # window of the corpus" is supposed to mean.
        lens = np.array([off[i + 1] - off[i] - window + 1 for i in self.clips], dtype=np.float64)
        self.weights = lens / lens.sum()

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def seconds(self) -> float:
        off = self.manifest.offsets
        return sum(off[i + 1] - off[i] for i in self.clips) / self.manifest.sample_rate

    def batch(self, batch_size: int) -> torch.Tensor:
        """`(batch_size, window)` float32 on `self.device`."""
        off = self.manifest.offsets
        picks = self.rng.choice(len(self.clips), size=batch_size, p=self.weights)
        out = np.empty((batch_size, self.window), dtype=np.float32)
        for row, p in enumerate(picks):
            c = self.clips[p]
            start = off[c] + self.rng.integers(0, off[c + 1] - off[c] - self.window + 1)
            out[row] = self.samples[start : start + self.window].astype(np.float32) / 32768.0
        return torch.from_numpy(out).to(self.device)

    def clip(self, i: int) -> torch.Tensor:
        """One whole clip, for listening rather than training."""
        off = self.manifest.offsets
        raw = self.samples[off[i] : off[i + 1]].astype(np.float32) / 32768.0
        return torch.from_numpy(raw)


# ---------------------------------------------------------------------------------------
# from waveforms to codes
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def encode_corpus(codec, corpus: str | Path, out_dir: str | Path, *, batch_clips: int = 8,
                  progress=print) -> Manifest:
    """Run a trained codec over a packed corpus and write `codes.bin` + `manifest.json`.

    The audio LM's equivalent of `data/prepare.py`'s tokenization, and done once for the same
    reason: the codec's encoder is a stack of strided convolutions over raw samples, and
    paying for it on every training step of the language model would make the *tokenizer* the
    expensive part of the run.

    Codes are `int16`. A codebook of 1,024 fits with room to spare, and a 24-hour corpus at
    50 frames a second × 8 codebooks is 34 M integers = 69 MB, against 2.7 GB of audio. That
    ratio is the compression, made visible on disk.
    """
    corpus, out_dir = Path(corpus), Path(out_dir)
    src = Manifest.load(corpus / "manifest.json")
    samples = np.memmap(corpus / "audio.bin", dtype="<i2", mode="r")
    device = next(codec.parameters()).device
    n_books, hop = codec.cfg.n_codebooks, codec.cfg.hop

    if src.sample_rate != codec.cfg.sample_rate:
        raise ValueError(
            f"corpus is {src.sample_rate} Hz and the codec is {codec.cfg.sample_rate} Hz. "
            "Re-pack the corpus; resampling here would leave the two disagreeing about "
            "what a frame is."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    offsets, names = [0], []
    total = 0
    was = codec.training
    codec.eval()
    with open(out_dir / "codes.bin", "wb") as f:
        for i in range(src.n_clips):
            a, b = src.offsets[i], src.offsets[i + 1]
            wave = torch.from_numpy(samples[a:b].astype(np.float32) / 32768.0)
            codes = codec.encode(wave.unsqueeze(0).to(device))[0].cpu().numpy()
            # (N, frames) written codebook-major per clip, so a window is one contiguous
            # slice per codebook rather than a strided gather.
            f.write(codes.astype("<i2").tobytes())
            total += codes.shape[1]
            offsets.append(total)
            names.append(src.names[i])
            if progress and (i + 1) % 200 == 0:
                progress(f"  {i + 1}/{src.n_clips} clips, {total} frames")
    codec.train(was)

    man = Manifest(
        sample_rate=src.sample_rate, offsets=offsets, names=names,
        sources=[{"n_codebooks": n_books, "hop": hop, "codebook_size": codec.cfg.codebook_size}],
        seconds=total * hop / src.sample_rate, built=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    man.save(out_dir / "manifest.json")
    if progress:
        progress(f"{len(names)} clips, {total} frames ({man.seconds / 3600:.2f} h), "
                 f"{n_books} codebooks -> {out_dir}/codes.bin")
    return man


class CodeDataset:
    """Random windows of codec codes. `(B, n_codebooks, window)` int64.

    Same clip-boundary rule as `AudioDataset` and for the same reason: a window spanning the
    join of two utterances is a transition that occurs nowhere, and here it is worse than in
    the codec — a language model would learn it as a *pattern*, and generate it.
    """

    def __init__(self, path: str | Path, window: int, device: str = "cpu", *,
                 n_codebooks: int, seed: int | None = None, split: str = "train",
                 val_clips: int = 16):
        d = Path(path)
        self.manifest = Manifest.load(d / "manifest.json")
        self.n_codebooks = n_codebooks
        flat = np.memmap(d / "codes.bin", dtype="<i2", mode="r")
        total_frames = self.manifest.offsets[-1]
        if flat.size != total_frames * n_codebooks:
            raise ValueError(
                f"{d}/codes.bin holds {flat.size} integers, but the manifest says "
                f"{total_frames} frames x {n_codebooks} codebooks = "
                f"{total_frames * n_codebooks}. Re-encode with the right codec."
            )
        self.codes = flat  # clip-major, then codebook-major within a clip
        self.window, self.device = window, device
        self.rng = np.random.default_rng(seed)

        n = self.manifest.n_clips
        val_clips = min(val_clips, max(1, n // 10))
        clips = range(n - val_clips, n) if split == "val" else range(n - val_clips)
        off = self.manifest.offsets
        self.clips = [i for i in clips if off[i + 1] - off[i] >= window]
        if not self.clips:
            raise ValueError(f"no {split} clip has {window} frames")
        lens = np.array([off[i + 1] - off[i] - window + 1 for i in self.clips], dtype=np.float64)
        self.weights = lens / lens.sum()

    def _clip_view(self, i: int) -> np.ndarray:
        """The `(n_codebooks, frames)` block for clip `i`, as a view into the memmap."""
        off = self.manifest.offsets
        a, b = off[i], off[i + 1]
        return self.codes[a * self.n_codebooks : b * self.n_codebooks].reshape(
            self.n_codebooks, b - a
        )

    def __len__(self) -> int:
        return len(self.clips)

    def batch(self, batch_size: int) -> torch.Tensor:
        picks = self.rng.choice(len(self.clips), size=batch_size, p=self.weights)
        out = np.empty((batch_size, self.n_codebooks, self.window), dtype=np.int64)
        for row, p in enumerate(picks):
            view = self._clip_view(self.clips[p])
            start = self.rng.integers(0, view.shape[1] - self.window + 1)
            out[row] = view[:, start : start + self.window]
        return torch.from_numpy(out).to(self.device)

    def clip(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self._clip_view(i).astype(np.int64))


# ---------------------------------------------------------------------------------------
# a corpus that needs no download
# ---------------------------------------------------------------------------------------


def synth_corpus(
    out_dir: str | Path,
    *,
    n_clips: int = 200,
    seconds: float = 2.0,
    sample_rate: int = TARGET_SR,
    seed: int = 0,
    progress=print,
) -> Manifest:
    """Write a corpus of synthetic vowel babble — no download, no licence, ~2 minutes of audio.

    This is `prepare_sft synthetic`'s counterpart, and it exists for the same reason: every
    command in the phase should be runnable before anyone has spent 2.6 GB of bandwidth, and
    a smoke test on real speech is slow enough that it stops being run.

    It is **source-filter synthesis**, which is the classical model of how a voice works and
    is worth understanding for its own sake:

    * a **source** — a pulse train at the fundamental `f0`, i.e. the vocal folds opening
      once per period. Its spectrum is a harmonic stack: energy at `f0, 2·f0, 3·f0, …`;
    * a **filter** — three two-pole resonators at the **formant** frequencies, i.e. the
      resonances of the throat and mouth. They boost the harmonics nearest them.

    Which vowel you hear is set entirely by where the first two formants sit ("ee" is a low
    F1 and a very high F2; "ah" is the two close together). Gliding between vowel targets
    gives something with the *statistics* of speech — harmonics, formants, transitions,
    silences — without being speech. A codec that reconstructs this well has genuinely
    learned something; it is just an easier something than LJSpeech.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    # Five vowel targets, as (F1, F2, F3) in Hz. Textbook values for a male speaker.
    vowels = np.array(
        [[730, 1090, 2440], [270, 2290, 3010], [530, 1840, 2480],
         [570, 840, 2410], [300, 870, 2240]], dtype=np.float64
    )
    n = int(seconds * sample_rate)
    paths = []
    # **The transcripts are free, and that is why this corpus is worth having.** We chose the
    # vowel sequence, so we know it exactly — which makes TTS and ASR trainable and, more to
    # the point, *measurable* (a real word error rate) before anyone has downloaded 2.6 GB.
    transcripts: dict[str, str] = {}

    for c in range(n_clips):
        f0 = rng.uniform(90, 190)
        # A glottal pulse train with jitter, so the harmonics are not perfectly stationary.
        phase = np.cumsum(np.full(n, f0 / sample_rate) * rng.normal(1.0, 0.01, n))
        source = (np.diff(np.floor(phase), prepend=0) > 0).astype(np.float64)
        source += 0.005 * rng.standard_normal(n)  # breath

        # Glide between a few vowel targets over the clip.
        n_targets = rng.integers(3, 6)
        which = rng.integers(0, len(vowels), n_targets)
        targets = vowels[which]
        knots = np.linspace(0, n, n_targets)
        track = np.stack([np.interp(np.arange(n), knots, targets[:, k]) for k in range(3)])

        x = source
        for k, bw in enumerate((80.0, 90.0, 120.0)):
            x = _resonate(x, track[k], bw, sample_rate)

        # An amplitude envelope with pauses, so the corpus contains silence to reconstruct.
        env = np.clip(np.interp(np.arange(n), np.linspace(0, n, 9),
                                rng.uniform(0.0, 1.0, 9)), 0, 1) ** 1.5
        x = x * env
        x = (x / max(np.abs(x).max(), 1e-9) * rng.uniform(0.3, 0.8)).astype(np.float32)

        p = out_dir / "wavs" / f"synth-{c:04d}.wav"
        write_wav(p, x, sample_rate)
        paths.append(p)
        transcripts[p.name] = " ".join(VOWEL_NAMES[k] for k in which)
        if progress and (c + 1) % 50 == 0:
            progress(f"  {c + 1}/{n_clips} clips")

    (out_dir / "transcripts.json").write_text(json.dumps(transcripts, indent=1))
    return pack(paths, out_dir, sample_rate=sample_rate, min_seconds=0.1, progress=progress)


def _resonate(x: np.ndarray, freq: np.ndarray, bandwidth: float, sr: int) -> np.ndarray:
    """A time-varying two-pole resonator: `y[n] = x[n] + 2r·cos(θ)·y[n−1] − r²·y[n−2]`.

    `r = exp(−π·BW/sr)` puts the poles just inside the unit circle — the closer to it, the
    narrower and louder the resonance. This is a Python loop over samples because the
    coefficients change every sample; it is the slowest thing in this file by an order of
    magnitude and it only ever runs on the synthetic corpus.
    """
    theta = 2.0 * math.pi * freq / sr
    r = math.exp(-math.pi * bandwidth / sr)
    a1 = 2.0 * r * np.cos(theta)
    a2 = -(r**2)
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for i in range(len(x)):
        y0 = x[i] + a1[i] * y1 + a2 * y2
        y[i] = y0
        y2, y1 = y1, y0
    return y * (1.0 - r)  # keep the gain roughly independent of bandwidth


# ---------------------------------------------------------------------------------------
# the real corpora
# ---------------------------------------------------------------------------------------


def fetch_ljspeech(dest: str | Path = "data/audio/ljspeech", progress=print) -> Path:
    """Download and extract LJSpeech-1.1. **2.6 GB over the network, ~6.5 GB on disk.**

    Kept as a plain `urlretrieve` + `tarfile` rather than going through `datasets`: the Hub's
    copy needs an audio backend to decode, and the whole point of this package is that the
    only thing between a WAV file and a tensor is code in this repo.
    """
    import tarfile
    import urllib.request

    dest = Path(dest)
    wavs = dest / "LJSpeech-1.1" / "wavs"
    if wavs.is_dir():
        progress(f"already extracted: {wavs}")
        return wavs

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "LJSpeech-1.1.tar.bz2"
    if not archive.exists():
        progress(f"downloading {LJSPEECH_URL} (2.6 GB) -> {archive}")
        urllib.request.urlretrieve(LJSPEECH_URL, archive)  # noqa: S310 (fixed https URL)
    progress(f"extracting {archive} (~3.6 GB)")
    with tarfile.open(archive, "r:bz2") as t:
        t.extractall(dest, filter="data")
    progress(f"extracted to {wavs}. The archive can be deleted: {archive}")
    return wavs


def find_wavs(root: str | Path) -> list[Path]:
    """Every `.wav` under a directory, in a stable order.

    Sorted, because `pack` writes clips in the order it is given and the validation split is
    the *last* N clips — an unsorted walk would hold out a different set on every machine
    and no two val numbers would be comparable.
    """
    return sorted(Path(root).rglob("*.wav"))
