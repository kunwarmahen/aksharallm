"""Does a codec trained on one voice reconstruct other people's?

`codec-lj` learned from 24 hours of a single reader. A multi-speaker TTS built on its codes
can only sound like other people if the codec can *carry* other people, so this is the
question to answer before deciding whether the codec or the audio LM is the thing to retrain.

It measures two corpora the same way, side by side:

    home    held-out clips of the voice the codec trained on (its own val split)
    other   clips of voices it never heard (LibriSpeech test-clean, a few per speaker)

For every clip: MCD at each rung of the bitrate ladder, and one independent recogniser's
transcript of the original and of the reconstruction. The reading is **paired, per clip**:
for each corpus, how many clips the codec made *worse* to the recogniser, better, or left
alone, with a sign test. Comparing two corpora's totals would mix up "the codec hurts these
voices" with "these sentences are harder", and the recogniser's error on the original of the
same clip is exactly the control that separates them.

Two things this cannot see, said here so they are not forgotten: MCD measures the spectral
envelope, not whether the voice still sounds like *that person* (a codec that turns every
speaker into the same voice could still score well on both columns); and the judge
(`asr-libri460`) trained on LibriSpeech, so it is on home ground for the "other" corpus —
which is why each row is read against its own original, never against the other corpus.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .dataset import Manifest, speaker_of  # noqa: F401  (speaker_of: re-exported)
from .features import MelConfig
from .measure import mcd
from .speech import plain_words
from .text import character_error_rate, load_transcripts, word_error_rate


def pick_clips(corpus: str | Path, *, per_speaker: int | None = None, last: int | None = None,
               min_seconds: float = 2.0, max_seconds: float = 15.0,
               sample_rate: int = 16_000) -> list[dict]:
    """Clips to measure, as `{name, speaker, text, wave}`, deterministically.

    `last=N` takes the corpus's final N clips — the held-out split `AudioDataset` uses, so
    for the codec's own corpus these are clips it never trained on. `per_speaker=K` takes
    the first K usable clips of every speaker, so a corpus with forty voices is measured on
    forty voices rather than on whichever two come first in the file.
    """
    d = Path(corpus)
    man = Manifest.load(d / "manifest.json")
    if man.sample_rate != sample_rate:
        raise ValueError(f"{d} is {man.sample_rate} Hz, the codec is {sample_rate} Hz")
    samples = np.memmap(d / "audio.bin", dtype="<i2", mode="r")
    texts = load_transcripts(d)
    idx = range(man.n_clips - last, man.n_clips) if last else range(man.n_clips)
    taken: dict[str, int] = defaultdict(int)
    out = []
    for i in idx:
        name = man.names[i]
        text = plain_words(texts.get(name, ""))
        n = man.offsets[i + 1] - man.offsets[i]
        if not text or not (min_seconds * sample_rate <= n <= max_seconds * sample_rate):
            continue
        spk = speaker_of(name, d.name)
        if per_speaker and taken[spk] >= per_speaker:
            continue
        taken[spk] += 1
        wave = samples[man.offsets[i]:man.offsets[i + 1]].astype(np.float32) / 32768.0
        out.append({"name": name, "speaker": spk, "text": text, "wave": wave})
    return out


@torch.no_grad()
def reconstruct_rungs(codec, wave: np.ndarray, rungs) -> dict[int, np.ndarray]:
    """Encode once, decode at every rung — RVQ's prefix is itself a valid code."""
    device = next(codec.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0).to(device)
    codes = codec.encode(x)
    return {n: codec.decode(codes, n_codebooks=n)[0, : x.shape[-1]].float().cpu().numpy()
            for n in rungs}


def paired(before: list[float], after: list[float]) -> dict:
    """Per-clip counts of after > before (worse), < (better), == (same), + a sign test.

    Lower is better for both inputs (error rates). Clips are matched by position, so the two
    lists must describe the same clips in the same order.
    """
    from ..eval.versus import sign_test

    if len(before) != len(after):
        raise ValueError("paired lists must describe the same clips")
    worse = sum(a > b for b, a in zip(before, after))
    better = sum(a < b for b, a in zip(before, after))
    return {"worse": worse, "better": better, "same": len(before) - worse - better,
            "p": sign_test(worse, better)}


def measure_corpus(codec, clips: list[dict], transcribe_fn: Callable[[list], list[str]], *,
                   rungs=(1, 2, 4, 8), judged=(4, 8), keep=None, embed_fn=None) -> dict:
    """MCD at every rung and the judge's WER on the original and at each `judged` rung.

    `transcribe_fn(list_of_waves) -> list_of_str`. `keep(j, label, wave)` is called for each
    waveform so the caller can write a few out to listen to. `embed_fn(list_of_waves) ->
    (N, d) unit vectors` (a speaker model, audio/speaker.py) adds the voice-identity column:
    see `voice_columns`.
    """
    cfg = MelConfig(sample_rate=codec.cfg.sample_rate)
    rungs = tuple(r for r in rungs if r <= codec.cfg.n_codebooks)
    judged = tuple(r for r in judged if r in rungs)
    items, waves = [], {"original": [], **{f"codec{r}": [] for r in judged}}
    for j, c in enumerate(clips):
        recon = reconstruct_rungs(codec, c["wave"], rungs)
        orig = torch.from_numpy(c["wave"])
        items.append({"name": c["name"], "speaker": c["speaker"], "ref": c["text"],
                      "seconds": len(c["wave"]) / codec.cfg.sample_rate,
                      "mcd": {r: mcd(orig, torch.from_numpy(recon[r]), cfg) for r in rungs}})
        waves["original"].append(c["wave"])
        for r in judged:
            waves[f"codec{r}"].append(recon[r])
        if keep:
            keep(j, "original", c["wave"])
            for r in judged:
                keep(j, f"codec{r}", recon[r])
    for label, ws in waves.items():
        for it, hyp in zip(items, transcribe_fn(ws)):
            hyp = plain_words(hyp)
            it.setdefault("hyp", {})[label] = hyp
            it.setdefault("wer", {})[label] = word_error_rate(it["ref"], hyp)
            it.setdefault("cer", {})[label] = character_error_rate(it["ref"], hyp)
    out = {"items": items, **summarise(items, rungs, judged)}
    if embed_fn is not None:
        out.update(voice_columns(items, {k: embed_fn(ws) for k, ws in waves.items()}))
    return out


def voice_columns(items: list[dict], emb: dict) -> dict:
    """How much each reconstruction still sounds like its original, and the two numbers that
    give that a meaning.

    `voice[codecN]` is the mean cosine between a clip's original and its reconstruction.
    On its own a cosine of, say, 0.6 says nothing. So, from the originals alone and with the
    same model: `same_speaker` is the mean over pairs of DIFFERENT clips by the same person
    (what "the same voice" scores), and `different_speaker` over pairs by two people. A
    reconstruction near `same_speaker` or above it kept the voice; near `different_speaker`
    it did not. Each item gets its own `voice` per label too.
    """
    orig = emb["original"]
    voice = {}
    for label, e in emb.items():
        if label == "original":
            continue
        sims = (orig * e).sum(-1)
        for it, v in zip(items, sims.tolist()):
            it.setdefault("voice", {})[label] = v
        voice[label] = float(sims.mean())
    gram = orig @ orig.T
    spk = [it["speaker"] for it in items]
    same, diff = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (same if spk[i] == spk[j] else diff).append(float(gram[i, j]))
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"voice": voice, "voice_ref": {"same_speaker": mean(same),
                                          "different_speaker": mean(diff),
                                          "pairs": [len(same), len(diff)]}}


def summarise(items: list[dict], rungs, judged) -> dict:
    def mean(xs):
        xs = [x for x in xs if x == x]  # MCD is nan on an all-silent clip
        return sum(xs) / len(xs) if xs else float("nan")

    labels = ["original"] + [f"codec{r}" for r in judged]
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)
    top = max(rungs)
    return {
        "n": len(items), "speakers": len(by_spk),
        "seconds": sum(it["seconds"] for it in items),
        "mcd": {r: {"mean": mean([it["mcd"][r] for it in items]),
                    "median": statistics.median([it["mcd"][r] for it in items])}
                for r in rungs},
        "wer": {k: mean([it["wer"][k] for it in items]) for k in labels},
        "cer": {k: mean([it["cer"][k] for it in items]) for k in labels},
        # The reading the report exists for: did the codec make THIS clip harder to hear.
        "paired_wer": {f"codec{r}": paired([it["wer"]["original"] for it in items],
                                           [it["wer"][f"codec{r}"] for it in items])
                       for r in judged},
        "per_speaker": sorted(
            ({"speaker": s, "n": len(v), "mcd": mean([it["mcd"][top] for it in v]),
              "wer_original": mean([it["wer"]["original"] for it in v]),
              "wer_codec": mean([it["wer"][f"codec{top}"] for it in v])
              if f"codec{top}" in labels else None}
             for s, v in by_spk.items()), key=lambda r: -r["mcd"]),
    }
