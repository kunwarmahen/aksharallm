"""The numbers a recogniser is judged by — and the ones a single average hides.

**Corpus WER, not the mean of per-utterance WERs.** WER is total word errors over total
reference words. Averaging per-utterance rates instead weights a two-word utterance with one
error (50%) the same as a forty-word one with one error (2.5%), and reads several points worse
than the number every paper reports. `score` accumulates edits and words separately.

**The breakdown is the point** (day-two problem 2). A recogniser at 8% WER overall can be at
4% for most speakers and 25% for a few — and the few are usually the accents the training set
had least of. `score` returns per-speaker rates and reports the worst ones beside the mean,
the same argument as the per-domain loss in docs/13: a blend hides its parts.

**Silence is a test, not an absence of one** (day-two problem 1). `silence_check` feeds the
model audio with no speech in it — digital silence, hiss, low rumble, mains hum, a click
train — and counts every character it writes. The pass mark is zero. CTC makes this the
expected result rather than a hope (see `asr/ctc.py`), and measuring it is what turns that
argument into a fact about *this* checkpoint.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from . import vocab
from .ctc import greedy_decode


def edits(ref: list, hyp: list) -> int:
    """Levenshtein distance over any sequence — words for WER, characters for CER."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def score(pairs: list[tuple[str, str, str]]) -> dict:
    """`[(speaker, reference, hypothesis), ...]` -> corpus WER/CER, overall and per speaker."""
    tot = {"we": 0, "w": 0, "ce": 0, "c": 0}
    per = defaultdict(lambda: {"we": 0, "w": 0, "n": 0})
    for spk, ref, hyp in pairs:
        rw, hw = ref.split(), hyp.split()
        we = edits(rw, hw)
        tot["we"] += we
        tot["w"] += len(rw)
        tot["ce"] += edits(list(ref), list(hyp))
        tot["c"] += len(ref)
        per[spk]["we"] += we
        per[spk]["w"] += len(rw)
        per[spk]["n"] += 1
    speakers = {
        s: {"wer": v["we"] / max(v["w"], 1), "words": v["w"], "utts": v["n"]}
        for s, v in per.items()
    }
    worst = sorted(speakers.items(), key=lambda kv: -kv[1]["wer"])[:5]
    return {
        "wer": tot["we"] / max(tot["w"], 1),
        "cer": tot["ce"] / max(tot["c"], 1),
        "words": tot["w"],
        "utts": len(pairs),
        "speakers": speakers,
        "worst_speakers": [{"speaker": s, **v} for s, v in worst],
    }


def wer_interval(wer: float, words: int) -> float:
    """A rough ± (one standard error) for a WER, treating words as independent trials.

    Words are not independent — errors cluster within an utterance — so this is optimistic.
    It is printed anyway because a WER from 300 words quoted to three decimals is worse.
    """
    return math.sqrt(max(wer * (1 - wer), 0.0) / max(words, 1)) if wer < 1 else 0.0


@torch.no_grad()
def transcribe(model, waves: list[np.ndarray], device: str, batch: int = 16) -> list[str]:
    """Greedy transcripts for a list of float32 waveforms (16 kHz mono)."""
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(waves), batch):
        chunk = waves[i : i + batch]
        N = max(len(w) for w in chunk)
        x = torch.zeros(len(chunk), N)
        for r, w in enumerate(chunk):
            x[r, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        n = torch.tensor([len(w) for w in chunk])
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            lp, ln = model(x.to(device), n.to(device))
        out += [vocab.decode(ids) for ids in greedy_decode(lp, ln, vocab.BLANK)]
    model.train(was)
    return out


@torch.no_grad()
def log_probs(model, waves: list[np.ndarray], device: str, batch: int = 16) -> list[np.ndarray]:
    """The encoder's output for each waveform, trimmed to its own length, as float32 numpy.

    Separate from decoding on purpose: the encoder is the expensive part and runs once, and a
    beam search can then be re-run under any weights (tuning α and β) for the cost of the
    search alone.
    """
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(waves), batch):
        chunk = waves[i : i + batch]
        N = max(len(w) for w in chunk)
        x = torch.zeros(len(chunk), N)
        for r, w in enumerate(chunk):
            x[r, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        n = torch.tensor([len(w) for w in chunk])
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            lp, ln = model(x.to(device), n.to(device))
        lp = lp.float().cpu().numpy()
        out += [lp[r, : int(ln[r])] for r in range(len(chunk))]
    model.train(was)
    return out


def no_speech_clips(sample_rate: int = 16_000, seconds: float = 4.0, seed: int = 0) -> dict:
    """Five clips with nothing to transcribe. Named, because which one fails is the diagnosis."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    white = rng.standard_normal(n)
    # Brown noise: integrated white noise, i.e. a lot of rumble and little hiss — a fan, a car.
    brown = np.cumsum(white)
    brown = brown - np.convolve(brown, np.ones(4001) / 4001, mode="same")
    brown /= max(np.abs(brown).max(), 1e-9)
    clicks = np.zeros(n)
    clicks[:: sample_rate // 3] = 1.0  # a keyboard, roughly
    return {
        "digital silence": np.zeros(n),
        "room hiss (-50 dBFS)": 0.003 * white,
        "rumble (-26 dBFS)": 0.05 * brown,
        "mains hum 50 Hz": 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 150 * t),
        "keyboard clicks": 0.3 * clicks,
    }


def silence_check(model, device: str, sample_rate: int = 16_000) -> dict:
    """Every character written on audio with no speech in it. The pass mark is zero."""
    clips = no_speech_clips(sample_rate)
    hyps = transcribe(model, [c.astype(np.float32) for c in clips.values()], device)
    per = {name: h for name, h in zip(clips, hyps, strict=True)}
    return {
        "chars": sum(len(h.strip()) for h in hyps),
        "words": sum(len(h.split()) for h in hyps),
        "clips": len(hyps),
        "outputs": per,
    }


def name_recall(refs: list[str], hyps: list[str], targets: set[str]) -> dict:
    """Of the reference words in `targets` (names, usually), how many came out right — and how
    many times a target word was written where it was NOT said.

    The second number is what keeps a dictionary honest. A bias strong enough to write "shaun"
    every time it is said, and also in places nobody said it, has moved the error rather than
    fixed it; recall alone would call that a success.
    """
    from collections import Counter

    said = hit = false = 0
    for r, h in zip(refs, hyps, strict=True):
        rc = Counter(w for w in r.split() if w in targets)
        hc = Counter(w for w in h.split() if w in targets)
        said += sum(rc.values())
        hit += sum(min(n, hc[w]) for w, n in rc.items())
        false += sum(max(0, n - rc[w]) for w, n in hc.items())
    return {"said": said, "recalled": hit, "recall": hit / max(said, 1), "false_alarms": false}
