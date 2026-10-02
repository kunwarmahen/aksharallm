"""Speech with transcripts: getting LibriSpeech onto disk, and handing the trainer whole utterances.

The codec trains on random one-second windows and never needs to know what was said. A
recogniser needs the opposite: **whole utterances, each with its transcript**, because a
window cut out of the middle of a sentence has no transcript of its own. So this file reuses
the codec's storage — one flat `audio.bin` of int16 plus a `manifest.json` (`audio/dataset.py`)
— and adds what a recogniser needs on top:

```mermaid
flowchart LR
    F["LibriSpeech .flac<br/>+ *.trans.txt"] --> D["decode<br/>(ffmpeg, decoder only)"]
    D --> P["audio.bin + manifest.json<br/>+ transcripts.json"]
    P --> U["Utterances<br/>normalised text, speaker, length"]
    U --> K["length buckets<br/>-> padded batches"]
```

**FLAC is decoded by `ffmpeg`, and only decoded.** Writing a FLAC decoder would be a week spent
on a file format, which is plumbing by this repo's own rule. But `ffmpeg` will also quietly
resample or downmix whatever it is given, and that is *not* plumbing — it is a change to the
data. So the sample rate and channel count are read from the file's own STREAMINFO header by
`flac_info` (forty lines, below), **asserted**, and `ffmpeg` is told nothing it could use to
convert: the decoded sample count must equal the header's or the file is refused. Trap 5 of the
audio phase, applied to a new door.

**Length buckets.** LibriSpeech utterances run from 1 to 35 seconds. A batch is padded to its
longest member, so a random batch of 32 is mostly padding. Sorting by length and cutting the
sorted list into batches under a *padded-seconds* budget makes every batch nearly rectangular;
the batches themselves are then drawn at random, so the model never sees the corpus in length
order (which would be a curriculum nobody chose).

**Every dropped utterance is counted by reason.** Too long for memory, too short to be speech,
a transcript that normalises to nothing, or more letters than CTC has frames for (trap 2).
`Utterances.dropped` is printed in the trainer's header, because "trained on 28,539
utterances" and "trained on 26,000 of them" are different claims.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tarfile
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..audio.dataset import Manifest
from ..audio.text import load_transcripts
from . import vocab

#: OpenSLR resource 12. `train-clean-100` is the first real run; `dev-clean` is validation and
#: `test-clean` is the number everyone publishes.
LIBRISPEECH = "https://www.openslr.org/resources/12/{split}.tar.gz"
LIBRISPEECH_SIZES_GB = {
    "dev-clean": 0.34, "test-clean": 0.35, "dev-other": 0.31, "test-other": 0.33,
    "train-clean-100": 6.3, "train-clean-360": 23.0, "train-other-500": 30.0,
}


# ---------------------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------------------


def fetch_librispeech(split: str, dest: str | Path = "data/asr/librispeech", progress=print) -> Path:
    """Download and extract one LibriSpeech split. Returns `<dest>/LibriSpeech/<split>`.

    The download **resumes** — a 6.3 GB file over a home connection will be interrupted at
    least once, and `urlretrieve` starts again from zero. It writes `<archive>.part` with an
    HTTP Range request and renames it only once complete, so a half-downloaded archive is
    never mistaken for a whole one.
    """
    if split not in LIBRISPEECH_SIZES_GB:
        raise ValueError(f"unknown split {split!r}; one of {sorted(LIBRISPEECH_SIZES_GB)}")
    dest = Path(dest)
    out = dest / "LibriSpeech" / split
    if out.is_dir() and any(out.iterdir()):
        progress(f"already extracted: {out}")
        return out
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{split}.tar.gz"
    download(LIBRISPEECH.format(split=split), archive, progress=progress,
             note=f"~{LIBRISPEECH_SIZES_GB[split]} GB")
    progress(f"extracting {archive}")
    with tarfile.open(archive, "r:gz") as t:
        t.extractall(dest, filter="data")
    progress(f"extracted to {out}. The archive can be deleted: {archive}")
    return out


def download(url: str, out: Path, *, progress=print, note: str = "") -> Path:
    """Fetch `url` to `out`, **resuming** an interrupted download.

    Writes `<out>.part` with an HTTP Range request and renames it only once complete, so a
    half-downloaded file is never mistaken for a whole one. A 6 GB file over a home connection
    will be interrupted at least once, and `urlretrieve` starts again from zero.
    """
    out = Path(out)
    if out.exists():
        progress(f"already downloaded: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    progress(f"downloading {url}" + (f" ({note})" if note else "")
             + (f", resuming at {have / 1e9:.2f} GB" if have else ""))
    with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as f:  # noqa: S310
        if have and r.status != 206:
            # The server ignored the Range header and is sending the whole file again.
            f.seek(0)
            f.truncate()
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        done, last = have, time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                progress(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
                last = time.time()
    part.rename(out)
    return out


#: OpenSLR resource 11: the text the word LM is counted from (asr/ngram.py). Public-domain
#: books with the dev and test books excluded by its authors -- checked, not trusted, by
#: `asr lm build --overlap`.
LM_TEXT_URL = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"


def fetch_lm_text(dest: str | Path = "data/asr/lm", progress=print) -> Path:
    """Download the LibriSpeech LM corpus (1.5 GB, 803M words). Resumes if interrupted."""
    return download(LM_TEXT_URL, Path(dest) / "librispeech-lm-norm.txt.gz", progress=progress,
                    note="1.5 GB")


# ---------------------------------------------------------------------------------------
# FLAC: read the header ourselves, let ffmpeg decode
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlacInfo:
    sample_rate: int
    channels: int
    bits: int
    samples: int  # per channel; 0 means "unknown" in the spec


def flac_info(path: str | Path) -> FlacInfo:
    """Parse a FLAC file's STREAMINFO block. The spec puts it first, always.

    Layout: the 4-byte marker `fLaC`, a 4-byte metadata block header (type 0 = STREAMINFO,
    length 34), then 34 bytes of which bytes 10–17 pack, big-endian: sample rate (20 bits),
    channels − 1 (3), bits per sample − 1 (5), total samples (36).
    """
    with open(path, "rb") as f:
        head = f.read(42)
    if head[:4] != b"fLaC":
        raise ValueError(f"{path}: not a FLAC file")
    if head[4] & 0x7F != 0:
        raise ValueError(f"{path}: first metadata block is not STREAMINFO")
    (packed,) = struct.unpack(">Q", head[8 + 10 : 8 + 18])
    return FlacInfo(
        sample_rate=packed >> 44,
        channels=((packed >> 41) & 0x7) + 1,
        bits=((packed >> 36) & 0x1F) + 1,
        samples=packed & ((1 << 36) - 1),
    )


def decode_flac(path: str | Path, *, sample_rate: int = 16_000) -> np.ndarray:
    """int16 mono samples, **refusing** anything that would need converting."""
    info = flac_info(path)
    if info.sample_rate != sample_rate or info.channels != 1 or info.bits != 16:
        raise ValueError(
            f"{path}: {info.sample_rate} Hz / {info.channels} ch / {info.bits}-bit, expected "
            f"{sample_rate} Hz mono 16-bit. Convert it deliberately rather than here."
        )
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    x = np.frombuffer(raw, dtype="<i2")
    if info.samples and len(x) != info.samples:
        raise ValueError(f"{path}: decoded {len(x)} samples, header says {info.samples}")
    return x


def librispeech_transcripts(split_dir: str | Path) -> dict[str, str]:
    """`{ "103-1240-0000.flac": "CHAPTER ONE ..." }` from every `*.trans.txt` under a split."""
    out = {}
    for t in sorted(Path(split_dir).rglob("*.trans.txt")):
        for line in t.read_text().splitlines():
            uid, _, text = line.partition(" ")
            if uid:
                out[f"{uid}.flac"] = text.strip()
    return out


def pack_librispeech(split_dir: str | Path, out_dir: str | Path, *, workers: int = 8,
                     progress=print) -> Manifest:
    """One split -> `audio.bin` + `manifest.json` + `transcripts.json`, in the codec's format.

    Sorted file order, for the same reason as `find_wavs`: a different order on another
    machine would be a different corpus with the same name. Decoding runs on a thread pool —
    `ffmpeg` is a subprocess, so the threads spend their time waiting on it, not on the GIL —
    and `map` returns in submission order, so the bin is written in that sorted order.
    """
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flacs = sorted(split_dir.rglob("*.flac"))
    if not flacs:
        raise FileNotFoundError(f"no .flac files under {split_dir}")
    texts = librispeech_transcripts(split_dir)
    missing = [p.name for p in flacs if p.name not in texts]
    if missing:
        raise ValueError(f"{len(missing)} files have no transcript, e.g. {missing[:3]}")

    offsets, names, sources, total = [0], [], [], 0
    t0 = time.time()
    with open(out_dir / "audio.bin", "wb") as f, ThreadPoolExecutor(workers) as pool:
        for i, (p, x) in enumerate(zip(flacs, pool.map(decode_flac, flacs), strict=True)):
            f.write(x.tobytes())
            total += len(x)
            offsets.append(total)
            names.append(p.name)
            peak = float(np.abs(x).max()) / 32768.0 if len(x) else 0.0
            sources.append({"sr": 16_000, "channels": 1, "peak": round(peak, 4)})
            if progress and (i + 1) % 2000 == 0:
                progress(f"  {i + 1}/{len(flacs)} files, {total / 16_000 / 3600:.1f} h, "
                         f"{(i + 1) / (time.time() - t0):.0f} files/s")
    man = Manifest(sample_rate=16_000, offsets=offsets, names=names, sources=sources,
                   seconds=total / 16_000, built=time.strftime("%Y-%m-%d %H:%M:%S"),
                   source_dir=str(split_dir.resolve()))
    man.save(out_dir / "manifest.json")
    (out_dir / "transcripts.json").write_text(json.dumps({n: texts[n] for n in names}, indent=0))
    if progress:
        progress(f"{len(names)} utterances, {man.seconds / 3600:.2f} h -> {out_dir}")
    return man


# ---------------------------------------------------------------------------------------
# utterances and batches
# ---------------------------------------------------------------------------------------


def speaker_of(name: str) -> str:
    """LibriSpeech names are `<speaker>-<chapter>-<utt>`; anything else is its corpus's own
    single speaker. Per-speaker WER is the accent check (day-two problem 2), so this matters."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return parts[0]
    return stem.split("-")[0].rstrip("0123456789") or "?"


@dataclass
class Utt:
    clip: int
    start: int
    n: int
    text: str
    ids: list[int]
    speaker: str
    name: str


class Utterances:
    """A packed corpus with transcripts, filtered, as whole utterances.

    `feasible(n_samples, ids)` is supplied by the caller because it depends on the model's
    frame rate — the data does not know how many frames the encoder will give it.
    """

    def __init__(self, corpus: str | Path, *, split: str = "all", val_clips: int = 0,
                 min_seconds: float = 0.5, max_seconds: float = 20.0,
                 feasible: Callable[[int, list[int]], bool] | None = None,
                 limit: int | None = None):
        d = Path(corpus)
        self.corpus = d
        self.manifest = Manifest.load(d / "manifest.json")
        self.samples = np.memmap(d / "audio.bin", dtype="<i2", mode="r")
        self.sr = self.manifest.sample_rate
        texts = load_transcripts(d)
        n = self.manifest.n_clips
        if split == "all":
            clips = range(n)
        elif split == "val":
            clips = range(n - val_clips, n)
        elif split == "train":
            clips = range(n - val_clips)
        else:
            raise ValueError(f"split must be all/train/val, not {split!r}")

        self.dropped: Counter = Counter()
        self.utts: list[Utt] = []
        off = self.manifest.offsets
        for c in clips:
            name = self.manifest.names[c]
            raw = texts.get(name)
            if raw is None:
                self.dropped["no transcript"] += 1
                continue
            text = vocab.normalise(raw)
            ns = off[c + 1] - off[c]
            if not text:
                self.dropped["empty transcript"] += 1
            elif ns < min_seconds * self.sr:
                self.dropped[f"shorter than {min_seconds:g}s"] += 1
            elif ns > max_seconds * self.sr:
                self.dropped[f"longer than {max_seconds:g}s"] += 1
            else:
                ids = vocab.encode(text)
                if feasible is not None and not feasible(ns, ids):
                    self.dropped["more letters than frames"] += 1
                    continue
                self.utts.append(Utt(c, off[c], ns, text, ids, speaker_of(name), name))
            if limit and len(self.utts) >= limit:
                break
        if not self.utts:
            raise ValueError(f"no usable utterances in {d} ({dict(self.dropped)})")

    def __len__(self) -> int:
        return len(self.utts)

    @property
    def seconds(self) -> float:
        return sum(u.n for u in self.utts) / self.sr

    def wave(self, u: Utt) -> np.ndarray:
        return self.samples[u.start : u.start + u.n].astype(np.float32) / 32768.0

    def collate(self, idx: list[int], device: str = "cpu") -> dict:
        """Padded waves, their lengths, padded targets, their lengths — what `ctc_loss` takes."""
        us = [self.utts[i] for i in idx]
        N = max(u.n for u in us)
        L = max(len(u.ids) for u in us)
        wave = np.zeros((len(us), N), dtype=np.float32)
        tgt = np.zeros((len(us), L), dtype=np.int64)
        for r, u in enumerate(us):
            wave[r, : u.n] = self.wave(u)
            tgt[r, : len(u.ids)] = u.ids
        return {
            "wave": torch.from_numpy(wave).to(device),
            "n_samples": torch.tensor([u.n for u in us], device=device),
            "targets": torch.from_numpy(tgt).to(device),
            "target_lengths": torch.tensor([len(u.ids) for u in us], device=device),
            "utts": us,
        }

    def buckets(self, max_batch_seconds: float, max_batch: int = 256) -> list[list[int]]:
        """Indices sorted by length, cut into batches whose *padded* size fits the budget."""
        order = sorted(range(len(self.utts)), key=lambda i: self.utts[i].n)
        out, cur, longest = [], [], 0
        budget = max_batch_seconds * self.sr
        for i in order:
            n = self.utts[i].n
            if cur and (max(longest, n) * (len(cur) + 1) > budget or len(cur) >= max_batch):
                out.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, n)
        if cur:
            out.append(cur)
        return out


class BatchSampler:
    """Random batches from fixed length buckets, reproducible and resumable.

    Batches are drawn **with replacement**, like `AudioDataset` and `TokenDataset`: a step is a
    random batch, not a position in an epoch. That makes resume trivial — the generator's
    *state* goes into `ckpt_last.pt` (gotcha 16: saving the seed alone re-shows the batches the
    run just trained on).
    """

    def __init__(self, utts: Utterances, max_batch_seconds: float, *, seed: int = 0):
        self.utts = utts
        self.batches = utts.buckets(max_batch_seconds)
        # Weighted by audio in the batch, so a second of a long utterance is as likely to be
        # seen as a second of a short one.
        w = np.array([sum(utts.utts[i].n for i in b) for b in self.batches], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def next(self, device: str = "cpu") -> dict:
        b = self.batches[int(self.rng.choice(len(self.batches), p=self.p))]
        return self.utts.collate(b, device)

    def state(self) -> dict:
        return self.rng.bit_generator.state

    def load_state(self, s: dict) -> None:
        self.rng.bit_generator.state = s


def corpus_hours(path: str | Path) -> float:
    return Manifest.load(Path(path) / "manifest.json").seconds / 3600


__all__ = ["fetch_librispeech", "flac_info", "decode_flac", "pack_librispeech", "Utterances",
           "BatchSampler", "Utt", "speaker_of", "corpus_hours"]
