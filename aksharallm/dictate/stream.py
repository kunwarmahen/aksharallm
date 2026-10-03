"""Text while you are still talking: re-read everything so far, commit what two readings agree on.

Dictation without a live preview feels broken — you talk for twenty seconds into silence and
hope. There are two ways to show text early:

1. **A streaming encoder** — attention limited to chunks, convolutions that never look ahead
   — so each new 40 ms of audio costs a bounded amount. That is a different model, retrained.
2. **Re-encode the whole buffer every tick** and decide which words are settled. No new
   model, and the preview is decoded with *full* context, so it is as good as the final pass.

This file is (2), because the measurement says it is affordable here: the 20M encoder reads
5 s of audio in 36 ms on the CPU and 60 s in 357 ms (docs/23 § Streaming), so re-reading
everything twice a second costs a fraction of real time for any sentence a person dictates.
(1) is what you build when it stops being affordable — hour-long audio, a phone CPU — and is
written down as not built.

```mermaid
flowchart LR
    A["audio so far"] -->|"every tick"| E["encode + greedy<br/>(the whole buffer)"]
    E --> H["hypothesis n"]
    H --> L{"agrees with<br/>hypothesis n-1?"}
    L -->|"common prefix,<br/>minus the last word"| C["COMMITTED words<br/>never change"]
    L --> T["the rest: TENTATIVE<br/>(shown grey)"]
```

**Local agreement** (the rule Whisper-streaming calls LocalAgreement-2): a word is committed
once two consecutive readings agree on it and everything before it, and the newest word is
never committed — the audio may end in the middle of it. Committed words are shown solid and
never retracted while you speak; the tail is shown grey and changes freely. When you finish,
the full pipeline (beam search, your dictionary, cleanup) runs once on the whole recording
and replaces the preview — so the preview may spell a name the way greedy does and the final
text the way your dictionary does. The preview is a preview.

**What is measured** (`python -m aksharallm.dictate stream-eval`, written to `logs/asr/`):
how long after a word is *spoken* it is committed (its end is read from the final reading's
CTC alignment, 40 ms resolution); how many committed words turn out different from the final
reading (the flicker the rule exists to prevent); and what a tick costs.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..asr import vocab


def words_with_ends(lp: np.ndarray, frame_s: float) -> list[tuple[str, float]]:
    """Greedy CTC reading of `(frames, 29)` log-probs -> `[(word, end_seconds)]`. A word ends
    at the last frame that emitted one of its letters."""
    ids = lp.argmax(-1)
    words: list[tuple[str, float]] = []
    cur, end, prev = [], 0, vocab.BLANK
    for t, i in enumerate(ids.tolist()):
        if i != vocab.BLANK and i != prev:
            ch = vocab.ITOS[i]
            if ch == " ":
                if cur:
                    words.append(("".join(cur), (end + 1) * frame_s))
                cur = []
            else:
                cur.append(ch)
                end = t
        elif i != vocab.BLANK and i == prev and vocab.ITOS[i] != " ":
            end = t
        prev = i
    if cur:
        words.append(("".join(cur), (end + 1) * frame_s))
    return words


#: Measured on 200 test-clean utterances, ticks of 0.5 s (docs/23 § Streaming):
#:
#:     rule                              commit latency (median / p90)   committed then revised
#:     2 readings agree, hold back 1     0.85 s / 1.56 s                 3.66%
#:     3 readings agree, hold back 1     1.34 s / 2.18 s                 3.06%
#:     2 readings, hold back 2 / 3       0.95 / 1.20 s                   3.76 / 3.79%
#:     2 readings + 0.8 s edge guard     1.10 s                          3.56%
#:
#: The revisions are not an edge effect (a guard barely moves them): the encoder is
#: bidirectional, so later audio legitimately re-spells earlier words ("qushioned" ->
#: "cushioned"). Waiting longer buys little, and the final pass replaces the preview anyway,
#: so the defaults are the fastest row.
GUARD_S = 0.0
NEED = 2
HOLDBACK = 1


class LocalAgreement:
    """Commit what the last `need` hypotheses agree on, minus the newest `holdback` words and
    any word ending within `guard_s` of the audio's end."""

    def __init__(self, guard_s: float = GUARD_S, need: int = NEED, holdback: int = HOLDBACK):
        self.guard_s = guard_s
        self.need = need
        self.holdback = holdback
        self.history: list[list[str]] = []
        self.committed: list[str] = []

    def update(self, words: list[str], ends: list[float] | None = None,
               audio_s: float | None = None) -> tuple[list[str], list[str]]:
        self.history = (self.history + [words])[-self.need:]
        n = 0
        if len(self.history) == self.need:
            n = min(len(h) for h in self.history)
            for h in self.history[:-1]:
                k = 0
                while k < n and h[k] == words[k]:
                    k += 1
                n = k
        n = min(n, max(len(words) - self.holdback, 0))
        if ends is not None and audio_s is not None:
            while n and ends[n - 1] > audio_s - self.guard_s:
                n -= 1
        agreed = words[:n]
        # Only ever grow, and only along what is already committed: a committed word is never
        # retracted while the person is speaking (that is the flicker this exists to prevent).
        if len(agreed) > len(self.committed) and agreed[: len(self.committed)] == self.committed:
            self.committed = agreed
        k = len(self.committed)
        tail = words[k:] if words[:k] == self.committed else words[min(k, len(words)):]
        return list(self.committed), tail


class LiveTranscript:
    """One recording in progress: feed it the audio so far, get committed + tentative text."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.agree = LocalAgreement()
        self.audio = np.zeros(0, dtype=np.float32)
        self.last_ms = 0.0

    def add(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, np.asarray(chunk, dtype=np.float32)])

    @torch.no_grad()
    def tick(self) -> dict:
        from ..asr.measure import log_probs
        if len(self.audio) < 3200:      # under 0.2 s: nothing to read yet
            return {"stable": "", "tentative": "", "seconds": len(self.audio) / 16_000, "ms": 0.0}
        t0 = time.time()
        lp = log_probs(self.model, [self.audio], self.device)[0]
        frame_s = len(self.audio) / 16_000 / max(len(lp), 1)
        we = words_with_ends(lp, frame_s)
        stable, tail = self.agree.update([w for w, _ in we], [e for _, e in we],
                                         len(self.audio) / 16_000)
        self.last_ms = (time.time() - t0) * 1000
        return {"stable": " ".join(stable), "tentative": " ".join(tail),
                "seconds": round(len(self.audio) / 16_000, 2), "ms": round(self.last_ms, 1)}


def simulate(model, x: np.ndarray, device: str, tick_s: float = 0.5,
             guard_s: float = GUARD_S, need: int = NEED, holdback: int = HOLDBACK) -> dict:
    """Play one utterance in at `tick_s` steps and record when each word was committed."""
    import difflib
    from ..asr.measure import log_probs
    agree = LocalAgreement(guard_s, need, holdback)
    commit_time: dict[int, float] = {}
    costs = []
    n = len(x)
    step = int(tick_s * 16_000)
    t = step
    while True:
        cut = min(t, n)
        t0 = time.time()
        lp = log_probs(model, [x[:cut]], device)[0]
        costs.append(time.time() - t0)
        frame_s = cut / 16_000 / max(len(lp), 1)
        we = words_with_ends(lp, frame_s)
        committed, _ = agree.update([w for w, _ in we], [e for _, e in we], cut / 16_000)
        for i in range(len(committed)):
            commit_time.setdefault(i, cut / 16_000)
        if cut >= n:
            break
        t += step
    lp = log_probs(model, [x], device)[0]
    final = words_with_ends(lp, n / 16_000 / max(len(lp), 1))
    # At the end of speech everything left is committed by the final pass; those words have
    # no streaming latency to speak of, so only words committed DURING the recording count.
    # Align committed words to the final reading (not by position: one inserted word would
    # count every later word as revised). A committed word with no partner was revised.
    lat = []
    com = agree.committed
    sm = difflib.SequenceMatcher(a=com, b=[w for w, _ in final], autojunk=False)
    matched = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            matched += 1
            if i in commit_time:
                lat.append(commit_time[i] - final[j][1])
    wrong = len(com) - matched
    return {"latencies": lat, "committed": len(agree.committed), "revised": wrong,
            "final_words": len(final), "ticks": len(costs), "tick_cost": costs,
            "seconds": n / 16_000}


def evaluate(model, device: str, corpus, tick_s: float = 0.5, guard_s: float = GUARD_S,
             need: int = NEED, holdback: int = HOLDBACK) -> dict:
    lat, committed, revised, final, ticks, costs = [], 0, 0, 0, 0, []
    for u in corpus.utts:
        r = simulate(model, corpus.wave(u), device, tick_s, guard_s, need, holdback)
        lat += r["latencies"]
        committed += r["committed"]
        revised += r["revised"]
        final += r["final_words"]
        ticks += r["ticks"]
        costs += r["tick_cost"]
    a = np.array(lat) if lat else np.zeros(1)
    c = np.array(costs) if costs else np.zeros(1)
    return {"utts": len(corpus.utts), "tick_s": tick_s, "guard_s": guard_s, "need": need, "holdback": holdback, "words_final": final,
            "words_committed_live": len(lat),
            "latency_s": {"mean": float(a.mean()), "median": float(np.median(a)),
                          "p90": float(np.percentile(a, 90))},
            "revised": revised, "revision_rate": revised / max(committed, 1),
            "tick_ms": {"mean": float(c.mean() * 1000), "p90": float(np.percentile(c, 90) * 1000)}}
