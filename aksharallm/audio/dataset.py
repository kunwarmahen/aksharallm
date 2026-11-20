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

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
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


class MixedAudioDataset:
    """Windows from several packed corpora, mixed by weight within every batch.

    The many-voices codec reads LJSpeech (one reader, 24 h) beside LibriSpeech (~1,170
    readers, 460 h). Sampled by duration, LJ would be 5% of what the codec sees and the voice
    it already knows would drift; a weight per corpus keeps it at the share the config says.

    Each batch has `floor(batch * weight)` rows from every corpus, and the leftover rows go
    to corpora drawn by their fractional parts. So a weight is the expected share of the
    batch exactly, and every corpus with weight >= 1/batch appears in every batch. Inside a
    corpus, windows are drawn the way `AudioDataset` draws them: every second equally likely.
    """

    def __init__(self, sources: list[dict], window: int, device: str = "cpu", *,
                 seed: int | None = None, split: str = "train", val_clips: int = 16):
        if not sources:
            raise ValueError("data.sources is empty")
        w = np.array([float(s.get("weight", 1.0)) for s in sources], dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("every source weight must be positive")
        self.weights = w / w.sum()
        base = 0 if seed is None else seed
        self.parts = [AudioDataset(s["corpus"], window, device, seed=base + 1000 * i,
                                   split=split, val_clips=val_clips)
                      for i, s in enumerate(sources)]
        self.names = [Path(s["corpus"]).name for s in sources]
        rates = {p.manifest.sample_rate for p in self.parts}
        if len(rates) != 1:
            # Asserted, never repaired: one corpus at another rate is a codec that learns two
            # meanings of a frame.
            raise ValueError(f"sources disagree on sample rate: {sorted(rates)}")
        self.manifest = self.parts[0].manifest
        self.window = window
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(len(p) for p in self.parts)

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.parts)

    def counts(self, batch_size: int) -> np.ndarray:
        exact = self.weights * batch_size
        n = np.floor(exact).astype(int)
        left = batch_size - int(n.sum())
        if left:
            frac = exact - n
            n[self.rng.choice(len(n), size=left, replace=False, p=frac / frac.sum())] += 1
        return n

    def batch(self, batch_size: int) -> torch.Tensor:
        rows = [p.batch(int(k)) for p, k in zip(self.parts, self.counts(batch_size)) if k]
        return torch.cat(rows)

    def clip(self, i: int) -> torch.Tensor:
        return self.parts[0].clip(i)


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


class UtteranceDataset:
    """Whole utterances with their transcripts, for TTS and ASR. `batch` -> `(clips, texts)`.

    Not windows: a sentence's audio has to be all of that sentence, or the text and the
    sound disagree about what was said. So clips longer than `max_frames`, or whose text is
    longer than `max_text` ids, are dropped rather than cut, and the count is reported.
    The validation clips are the last `val_clips` of the manifest, the same rule
    `CodeDataset` uses, so the two kinds of run hold out the same audio.
    """

    def __init__(self, codes_dir: str | Path, transcripts: dict[str, str], tokenizer, *,
                 n_codebooks: int, max_frames: int, max_text: int, split: str = "train",
                 val_clips: int = 16, seed: int | None = None, speakers: bool = False):
        self.base = CodeDataset(codes_dir, 1, "cpu", n_codebooks=n_codebooks, seed=seed,
                                split=split, val_clips=val_clips)
        self.speaker_emb = None
        if speakers:
            self._load_speakers(Path(codes_dir), val_clips)
        names, off = self.base.manifest.names, self.base.manifest.offsets
        self.items, self.dropped = [], {"no_transcript": 0, "too_long_audio": 0,
                                        "too_long_text": 0}
        for i in self.base.clips:
            text = transcripts.get(Path(names[i]).name)
            if text is None:
                self.dropped["no_transcript"] += 1
                continue
            ids = tokenizer.encode(normalise_text(text))
            if off[i + 1] - off[i] > max_frames:
                self.dropped["too_long_audio"] += 1
            elif len(ids) > max_text:
                self.dropped["too_long_text"] += 1
            else:
                self.items.append((i, ids))
        if not self.items:
            raise ValueError(f"no {split} utterance fits {max_frames} frames / {max_text} ids")
        self.rng = np.random.default_rng(seed)
        self._transcripts, self._tok = transcripts, tokenizer

    def _load_speakers(self, d: Path, val_clips: int) -> None:
        """`speakers.json` + `speaker_emb.npy`, from `encode_many`. A row's reference is a
        DIFFERENT clip by the same speaker, drawn from the training clips, so the model learns
        the voice from a reference and not this clip's own words or recording."""
        names = self.base.manifest.names
        n = len(names)
        emb_path = d / "speaker_emb.npy"
        if not emb_path.is_file():
            raise FileNotFoundError(f"{emb_path} missing: encode with `encode-many --speaker`")
        self.speaker_emb = np.load(emb_path)
        if len(self.speaker_emb) != n:
            raise ValueError(f"{emb_path} has {len(self.speaker_emb)} rows for {n} clips")
        who = json.loads((d / "speakers.json").read_text())
        self.speaker = [who[Path(x).name] for x in names]
        n_train = n - min(val_clips, max(1, n // 10))     # CodeDataset's split rule
        self.pool: dict[str, list[int]] = {}
        for i in range(n_train):
            self.pool.setdefault(self.speaker[i], []).append(i)

    def reference_index(self, j: int, rng: np.random.Generator | None = None) -> int:
        """The manifest index of item `j`'s reference: another training clip by its speaker."""
        i = self.items[j][0]
        pool = self.pool.get(self.speaker[i], [i])
        rng = rng or self.rng
        r = int(pool[rng.integers(len(pool))])
        if r == i and len(pool) > 1:
            r = int(pool[(pool.index(i) + 1) % len(pool)])
        return r

    def reference(self, j: int, rng: np.random.Generator | None = None) -> torch.Tensor:
        """A speaker embedding for item `j`: another training clip by the same speaker."""
        return torch.from_numpy(self.speaker_emb[self.reference_index(j, rng)].astype(np.float32))

    def prompt_index(self, j: int, max_frames: int) -> int | None:
        """The SHORTEST training clip by item `j`'s speaker, other than `j`, if it is at most
        `max_frames` long: the voice prompt for cloning (`speak(prompt_codes=...)`). Shortest,
        because the prompt and the new sentence share one context; whole, because a cropped
        clip's transcript would no longer say what its codes say."""
        i = self.items[j][0]
        off = self.base.manifest.offsets
        frames = lambda k: off[k + 1] - off[k]  # noqa: E731  (a codes manifest counts frames)
        cands = [k for k in self.pool.get(self.speaker[i], []) if k != i]
        if not cands:
            return None
        k = min(cands, key=frames)
        return k if frames(k) <= max_frames else None

    def prompt(self, j: int, max_frames: int) -> tuple[list[int], torch.Tensor] | None:
        """`(transcript ids, codes (N, P))` of item `j`'s voice prompt, or None (see
        `prompt_index`)."""
        k = self.prompt_index(j, max_frames)
        if k is None:
            return None
        text = self._transcripts.get(Path(self.base.manifest.names[k]).name)
        if text is None:
            return None
        return self._tok.encode(normalise_text(text)), self.base.clip(k)

    def __len__(self) -> int:
        return len(self.items)

    def get(self, j: int) -> tuple[torch.Tensor, list[int]]:
        i, ids = self.items[j]
        return self.base.clip(i), ids

    def frames_per_char(self) -> float:
        """Median frames per character (BOS/EOS excluded) over these utterances."""
        off = self.base.manifest.offsets
        return float(np.median([(off[i + 1] - off[i]) / max(1, len(ids) - 2)
                                for i, ids in self.items]))

    def batch(self, batch_size: int) -> tuple[list[torch.Tensor], list[list[int]]]:
        picks = self.rng.choice(len(self.items), size=batch_size)
        self.last_picks = picks
        pairs = [self.get(int(j)) for j in picks]
        return [c for c, _ in pairs], [t for _, t in pairs]

    def speaker_batch(self) -> torch.Tensor | None:
        """`(B, D)` references for the rows of the last `batch`; None for a one-voice set."""
        if self.speaker_emb is None:
            return None
        return torch.stack([self.reference(int(j)) for j in self.last_picks])


def normalise_text(text: str) -> str:
    """Lower case, whitespace collapsed. Case carries no sound, and a smaller alphabet is a
    smaller table to learn; punctuation stays, because a comma is a pause."""
    return " ".join(text.lower().split())


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


def speaker_of(name: str, corpus_speaker: str) -> str:
    """LibriSpeech names are `<speaker>-<chapter>-<utt>`; anything else is one speaker,
    `corpus_speaker` (`encode_many` uses `LJ` for LJSpeech)."""
    stem = Path(name).name.rsplit(".", 1)[0].split("-")
    return stem[0] if len(stem) == 3 and all(p.isdigit() for p in stem) else corpus_speaker


def encode_many(codec, corpora: list[str | Path], out_dir: str | Path, *, speaker_model=None,
                seed: int = 0, max_batch: int = 32, max_batch_s: float = 240.0,
                one_speaker: str | None = None, progress=print) -> Manifest:
    """Encode several packed corpora into ONE codes directory, for a multi-speaker audio LM.

    Writes, beside `codes.bin` + `manifest.json` (the layout `CodeDataset` reads):

    * `transcripts.json` -- every corpus's transcripts merged, keyed by clip file name (the
      names are unique across LJSpeech and LibriSpeech, and a clash is refused);
    * `speakers.json` -- `{clip name: speaker}`, from `speaker_of` (LJSpeech is `LJ`), or
      `one_speaker` for every clip;
    * `speaker_emb.npy` -- `(n_clips, D)` float16, one `speaker-libri` embedding per clip,
      in manifest order, when a speaker model is given. Computed here because the audio is
      already being read; a second pass over 488 hours would cost as much again.

    **The clips are written in a shuffled order** (fixed `seed`). Every audio run holds out
    the *last* `val_clips` of the manifest, and the corpora are sorted by speaker, so in
    corpus order the validation set would be one or two LibriSpeech readers. Shuffled, it is
    a sample of the whole mix, LJ included.
    """
    from .text import load_transcripts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(codec.parameters()).device
    n_books, hop = codec.cfg.n_codebooks, codec.cfg.hop

    clips, transcripts, who = [], {}, {}
    for c in corpora:
        c = Path(c)
        man = Manifest.load(c / "manifest.json")
        if man.sample_rate != codec.cfg.sample_rate:
            raise ValueError(f"{c} is {man.sample_rate} Hz and the codec {codec.cfg.sample_rate} Hz")
        tr = load_transcripts(c)
        clash = set(tr) & set(transcripts)
        if clash:
            raise ValueError(f"{c}: {len(clash)} clip names already used by another corpus")
        transcripts.update(tr)
        mm = np.memmap(c / "audio.bin", dtype="<i2", mode="r")
        clips += [(mm, man.offsets[i], man.offsets[i + 1], man.names[i]) for i in range(man.n_clips)]
        one = "LJ" if c.name == "lj" else c.name
        # `one_speaker`: every clip is one voice (e.g. the author's test and training takes,
        # two corpora that would otherwise read as two speakers)
        who.update({Path(x).name: one_speaker or speaker_of(x, one) for x in man.names})
    order = np.random.default_rng(seed).permutation(len(clips))

    # Encoded in BATCHES of similar length (one clip at a time left the card 60% idle: 56 min
    # for 488 h), then written in the shuffled order. Zero-padding a clip changes only its
    # last two frames' codes (the codec's receptive field reaching past the end; measured on
    # codec-many, the same two frames whatever the padding), which is trailing silence.
    # The speaker model takes true lengths and masks, so its embeddings are exact.
    lengths = np.array([b - a for _, a, b, _ in clips])
    by_len = np.argsort(lengths, kind="stable")
    codes_of, emb_of = {}, {}
    was = codec.training
    codec.eval()
    t0, done = time.time(), 0
    with torch.no_grad():
        i = 0
        while i < len(by_len):
            # as many clips as fit in max_batch_s seconds of padded audio
            longest = lengths[by_len[min(i + max_batch - 1, len(by_len) - 1)]]
            n = max(1, min(max_batch, int(max_batch_s * codec.cfg.sample_rate // max(1, longest))))
            idx = by_len[i : i + n]
            longest = int(lengths[idx].max())
            wave = torch.zeros(len(idx), longest)
            for r, j in enumerate(idx):
                mm, a, b, _ = clips[j]
                wave[r, : b - a] = torch.from_numpy(mm[a:b].astype(np.float32) / 32768.0)
            wave = wave.to(device)
            codes = codec.encode(wave).cpu().numpy().astype("<i2")
            for r, j in enumerate(idx):
                frames = -(-int(lengths[j]) // hop)           # what one-at-a-time would give
                codes_of[int(j)] = codes[r, :, :frames]
            if speaker_model is not None:
                e = speaker_model(wave, torch.as_tensor(lengths[idx], device=device))
                for r, j in enumerate(idx):
                    emb_of[int(j)] = e[r].float().cpu().numpy().astype(np.float16)
            i += n
            if progress and i // 4000 > done:
                done = i // 4000
                rate = i / (time.time() - t0)
                progress(f"  {i}/{len(by_len)} clips, eta {(len(by_len) - i) / rate / 60:.0f} min")
    codec.train(was)

    offsets, names, embs, total = [0], [], [], 0
    with open(out_dir / "codes.bin", "wb") as f:
        for j in order:
            c = codes_of.pop(int(j))
            f.write(c.tobytes())
            total += c.shape[1]
            offsets.append(total)
            names.append(clips[j][3])
            if speaker_model is not None:
                embs.append(emb_of.pop(int(j)))

    man = Manifest(
        sample_rate=codec.cfg.sample_rate, offsets=offsets, names=names,
        sources=[{"n_codebooks": n_books, "hop": hop, "codebook_size": codec.cfg.codebook_size,
                  "corpora": [str(c) for c in corpora], "seed": seed}],
        seconds=total * hop / codec.cfg.sample_rate, built=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out_dir / "transcripts.json").write_text(json.dumps({n: transcripts[n] for n in names if n in transcripts}))
    (out_dir / "speakers.json").write_text(json.dumps({n: who[Path(n).name] for n in names}))
    if embs:
        np.save(out_dir / "speaker_emb.npy", np.stack(embs))
    man.save(out_dir / "manifest.json")   # last: a manifest means the directory is complete
    if progress:
        progress(f"{len(names)} clips, {man.seconds / 3600:.2f} h, "
                 f"{len(set(who.values()))} speakers -> {out_dir}")
    return man
