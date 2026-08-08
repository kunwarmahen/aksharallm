"""Needle in a haystack, from scratch — can the model still *find* something back there?

`curve.py` asks whether the model is still fluent at position 6,000. That is necessary and
not sufficient: a model can predict the next token perfectly well from the last fifty and
have no access whatsoever to something it read 6,000 tokens ago. Fluency is local.
Retrieval is not, and only retrieval justifies the word "context".

So this hides one sentence in a lot of filler and asks for it back:

```
[ ......... 4,000 tokens of ordinary text ......... ]
                          ^
        "The secret code for Bengaluru is 7431."     <- the needle, at a chosen depth
[ ......... more ordinary text ......... ]

"Question: what is the secret code for Bengaluru?
 Answer: The secret code for Bengaluru is"           <- the probe
```

and sweeps **length x depth** into the grid everyone recognises: rows are how far back the
needle was, columns are how long the whole context was, cells are how often it came back.

Two design choices that make this work on a small model
--------------------------------------------------------
**Nothing is generated.** A 300M model asked an open question produces plausible text that
is a nightmare to grade. Instead the answer is *scored*: the true code and a handful of
distractor codes are each appended to the probe, the model's log-probability of each is
computed, and the trial is correct when the true one wins. Same machinery as the ARC and
PIQA suites, exactly reproducible, and it works on a model far too small to answer out
loud. Chance is `1/n_candidates` and is printed beside every number, because a grid of
25%s from a four-way choice is a grid of zeros.

**The filler is the model's own validation data.** Tokens are read straight from the run's
`val.bin` rather than invented, so the haystack is ordinary in-distribution text and a
failure means "could not retrieve", not "was confused by the noise we built".

What to expect, and why a bad grid is still a result
-----------------------------------------------------
A base model of a few hundred million parameters, never fine-tuned on retrieval and never
trained past 1k, is **expected to score near chance** — and it does; `docs/19` has our
numbers. That is not a broken test. It separates the two claims that get conflated
constantly: *the positions are legible* (which scaling fixes, and `curve.py` measures) and
*the model can use them* (which scaling does not fix, and this measures). Publishing the
second number honestly is the point of building it.

Read with: docs/19-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

#: Places for the needle to be about. Ordinary proper nouns, so the tokenizer handles them
#: as words rather than as byte soup, and unrelated to anything in the training blend.
CITIES = ("Bengaluru", "Trondheim", "Valparaiso", "Hokkaido", "Marrakesh",
          "Ljubljana", "Queenstown", "Salvador")

#: Depths as a fraction of the context: front, middle, back. The classic result is a "lost
#: in the middle" dip, and you cannot see it without sampling the middle.
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

NEEDLE = "\nThe secret code for {city} is {code}.\n"
PROBE = ("\nQuestion: what is the secret code for {city}?\n"
         "Answer: The secret code for {city} is")


@dataclass
class Trial:
    length: int
    depth: float
    city: str
    code: str
    correct: bool
    margin: float          # logprob(true) - best logprob(distractor); >0 means it won


def _codes(rng: np.random.Generator, n: int) -> list[str]:
    """`n` distinct four-digit codes. Distinct matters: a duplicated distractor would tie
    with the true answer and be scored as a loss for no reason."""
    out: list[str] = []
    while len(out) < n:
        c = str(int(rng.integers(1000, 10000)))
        if c not in out:
            out.append(c)
    return out


def build_context(tok, filler: np.ndarray, length: int, depth: float,
                  city: str, code: str) -> tuple[list[int], int]:
    """The full token sequence, and the token index the needle was actually placed at.

    The needle is inserted at a token boundary rather than a character one, because the
    caller asked for a *depth in tokens* and re-tokenizing spliced text moves it — by
    enough to matter when the whole question is "how far back was it".
    """
    needle_ids = tok.encode(NEEDLE.format(city=city, code=code))
    probe_ids = tok.encode(PROBE.format(city=city))
    room = length - len(needle_ids) - len(probe_ids)
    if room < 16:
        raise ValueError(f"length {length} leaves no room for the needle and the probe")

    at = int(round(depth * room))
    body = list(filler[:room].astype(np.int64))
    return body[:at] + needle_ids + body[at:] + probe_ids, at


@torch.no_grad()
def _score_candidates(model, tok, prefix: list[int], candidates: list[str],
                      device: str) -> list[float]:
    """Total log-probability of each candidate continuation, given one shared prefix.

    The prefix is run **once** and its KV cache reused for every candidate — the whole cost
    of this test is the haystack, and scoring four candidates by four full forward passes
    over 8,000 tokens would make the sweep four times slower for nothing.
    """
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False))
    scores: list[float] = []
    for cand in candidates:
        cand_ids = tok.encode(" " + cand)
        ids = torch.tensor([prefix + cand_ids], dtype=torch.long, device=device)
        with amp:
            logits, _ = model(ids, full_logits=True)
        # logits[t] predicts token t+1, so the candidate's first token is predicted by the
        # last prefix position. Off by one here scores every candidate on the wrong tokens
        # and still returns a plausible-looking grid.
        window = logits[0, len(prefix) - 1: -1].float()
        logp = F.log_softmax(window, dim=-1)
        tgt = torch.tensor(cand_ids, device=device)
        scores.append(float(logp.gather(-1, tgt[:, None]).sum()))
    return scores


def run(model, tok, bin_path: str, lengths: list[int], depths=DEFAULT_DEPTHS,
        trials: int = 3, n_candidates: int = 4, device: str = "cpu",
        seed: int = 7, progress=None) -> dict:
    """Sweep length x depth. Returns the grid plus every individual trial."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    results: list[Trial] = []

    total = len(lengths) * len(depths) * trials
    for length in lengths:
        for depth in depths:
            for t in range(trials):
                city = CITIES[int(rng.integers(len(CITIES)))]
                codes = _codes(rng, n_candidates)
                true = codes[0]
                start = int(rng.integers(0, max(1, len(data) - length - 8)))
                filler = np.asarray(data[start: start + length])
                try:
                    ids, _ = build_context(tok, filler, length, depth, city, true)
                except ValueError:
                    continue
                shuffled = list(codes)
                rng.shuffle(shuffled)
                scores = _score_candidates(model, tok, ids, shuffled, device)
                best = int(np.argmax(scores))
                true_i = shuffled.index(true)
                others = [s for i, s in enumerate(scores) if i != true_i]
                results.append(Trial(length, depth, city, true,
                                     shuffled[best] == true,
                                     scores[true_i] - max(others)))
                if progress:
                    progress(len(results), total, "haystack")

    return summarise(results, lengths, list(depths), n_candidates)


def summarise(trials: list[Trial], lengths: list[int], depths: list[float],
              n_candidates: int) -> dict:
    """Grid of accuracies, plus the per-length roll-up and the chance line."""
    grid = []
    for depth in depths:
        row = []
        for length in lengths:
            cells = [t for t in trials if t.length == length and t.depth == depth]
            row.append({
                "length": length, "depth": depth,
                "n": len(cells),
                "accuracy": (sum(t.correct for t in cells) / len(cells)) if cells else None,
                "margin": (sum(t.margin for t in cells) / len(cells)) if cells else None,
            })
        grid.append(row)

    by_length = []
    for length in lengths:
        cells = [t for t in trials if t.length == length]
        by_length.append({
            "length": length, "n": len(cells),
            "accuracy": (sum(t.correct for t in cells) / len(cells)) if cells else None,
        })

    return {
        "lengths": lengths, "depths": depths,
        "chance": 1.0 / n_candidates,
        "n_candidates": n_candidates,
        "grid": grid,
        "by_length": by_length,
        "accuracy": (sum(t.correct for t in trials) / len(trials)) if trials else None,
        "trials": [t.__dict__ for t in trials],
        # Standard error on the overall rate, so nobody reads 4 trials as a measurement.
        "stderr": _stderr([t.correct for t in trials]),
    }


def _stderr(hits: list[bool]) -> float | None:
    n = len(hits)
    if n < 2:
        return None
    p = sum(hits) / n
    return math.sqrt(max(p * (1 - p), 1e-12) / n)
