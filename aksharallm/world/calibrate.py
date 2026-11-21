"""Scoring a world model where the next frame is not a function: the `keys` game.

In `keys` (keys.py) a move slips 10% of the time and whether a door opens depends on a key
the frame does not show. "Is the predicted frame exact" is then the wrong question, because
the right answer is a DISTRIBUTION over frames, and `keys.transition` gives it exactly. The
model gives one too: the probability of a 64-cell frame is the product of its 64 teacher-
forced cell probabilities, so every candidate frame can be scored in one forward pass.

Three measurements, per transition, grouped by what the intended move does:

  mode         Is the model's greedy frame the MOST LIKELY true frame? (The same teacher-
               forced argmax trick as `evaluate.next_frame`.) Beside it, what copying the
               last frame would score.
  KL           KL(true || model) over next frames, in nats: 0 is a model whose probabilities
               are the world's. The mass the model puts on frames the world cannot produce is
               part of it (and is reported on its own as `off`).
  key belief   At every move into a door, the model's probability that the door OPENS,
               against it staying shut: two counterfactual frames, scored whatever actually
               happened. The truth is 1 if the agent holds a key and 0 if not. Split three
               ways, because that split is the whole test:
                   held, picked up INSIDE the context   seeing it is enough: should be ~1
                   held, picked up BEFORE the context   unknowable from the window
                   not held                              should be ~0
               A model that reads only the last frame scores the same in all three.
               `base_rate` is the share of held keys among doors with no pickup in view,
               and `unseen_belief` the model's mean belief over those doors: calibrated
               when they match. The frame can still say more (a map with no key left), so
               the base rate is a calibration target, not a ceiling on the middle row.

Episodes come from the aware policy (the one `world-keys` trained on) or the blind one
(`keys.policy_blind`), which never looks at what the agent carries; `policy` picks.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from math import exp, log

import numpy as np
import torch

from . import keys
from .data import FRAME, episode_tokens, ids_of, vocab_of

EVAL_SEED = 30_000
VOCAB = vocab_of("keys")


def _last_pickup(outcomes: list[str], t: int) -> int | None:
    for j in range(t - 1, -1, -1):
        if outcomes[j] == "took_key":
            return j
    return None


def transitions(n_episodes: int, steps: int, history: int, seed: int,
                slip: float = keys.SLIP, policy: str = "aware") -> list[dict]:
    """One record per transition t >= history: its context tokens, the merged true
    distribution over next frames, the bucket, and for a door move the two counterfactual
    frames (opens / stays shut)."""
    rng = np.random.default_rng(seed)
    out = []
    step_len = FRAME + 1
    for _ in range(n_episodes):
        e = keys.episode_full(rng, steps, slip, policy)
        toks = episode_tokens(e["frames"], e["actions"], VOCAB)[1:]
        for t in range(history, steps):
            s, a = e["states"][t], e["actions"][t]
            merged: dict[tuple, float] = {}
            for p, n, _, _ in keys.transition(s, a, slip):
                f = tuple(keys.render(n))
                merged[f] = merged.get(f, 0.0) + p
            intended = keys.apply(s, a)[1]
            rec = {"ctx": toks[(t - history) * step_len: t * step_len + FRAME + 1],
                   "frames": list(merged), "p": list(merged.values()),
                   "last": e["frames"][t], "outcome": intended, "door": None}
            if intended in ("locked", "unlocked"):
                opened = s.copy()
                opened.held = max(1, s.held)
                shut = s.copy()
                shut.held = 0
                j = _last_pickup(e["outcomes"], t)
                where = ("not held" if s.held == 0 else
                         "held, seen in context" if j is not None and j >= t - history else
                         "held, before the context")
                rec["door"] = {"held": s.held > 0, "where": where,
                               "seen": j is not None and j >= t - history,
                               "open": keys.render(keys.apply(opened, a)[0]),
                               "shut": keys.render(keys.apply(shut, a)[0])}
            out.append(rec)
    return out


@torch.no_grad()
def _frame_logprobs(model, ctxs: list[list[int]], frames: list[list[str]], device: str,
                    batch: int = 64) -> tuple[list[float], list[bool]]:
    """log P(frame | ctx) for each pair, and whether greedy decoding writes that frame."""
    lps, greedy = [], []
    for i in range(0, len(ctxs), batch):
        c = torch.tensor(ctxs[i:i + batch], device=device)
        f = torch.tensor([ids_of(x, VOCAB) for x in frames[i:i + batch]], device=device)
        logits, _ = model(torch.cat([c, f[:, :-1]], 1), full_logits=True)
        lp = torch.log_softmax(logits[:, c.shape[1] - 1:].float(), -1)
        lps += lp.gather(-1, f[..., None])[..., 0].sum(1).tolist()
        greedy += (lp.argmax(-1) == f).all(1).tolist()
    return lps, greedy


def score(model, *, episodes: int = 200, steps: int = 30, history: int = 5,
          seed: int = EVAL_SEED, slip: float = keys.SLIP, device: str = "cpu",
          policy: str = "aware") -> dict:
    recs = transitions(episodes, steps, history, seed, slip, policy)
    ctxs, frames, owner = [], [], []
    for i, r in enumerate(recs):
        for f in r["frames"]:
            ctxs.append(r["ctx"])
            frames.append(list(f))
            owner.append((i, "cand"))
        if r["door"]:
            for k in ("open", "shut"):
                ctxs.append(r["ctx"])
                frames.append(r["door"][k])
                owner.append((i, k))
    lps, greedy = _frame_logprobs(model, ctxs, frames, device)
    cand = [[] for _ in recs]
    door = [{} for _ in recs]
    for (i, kind), lp, g in zip(owner, lps, greedy):
        (cand[i].append((lp, g)) if kind == "cand" else door[i].__setitem__(kind, lp))

    rows = []
    for r, cs, d in zip(recs, cand, door):
        mode = int(np.argmax(r["p"]))
        kl = sum(p * (log(p) - lp) for p, (lp, _) in zip(r["p"], cs) if p > 0)
        row = {"outcome": r["outcome"], "mode_exact": cs[mode][1],
               "copy": list(r["frames"][mode]) == r["last"],
               "kl": kl, "off": max(0.0, 1.0 - sum(exp(lp) for lp, _ in cs)),
               "p_mode_true": r["p"][mode], "p_mode_model": exp(cs[mode][0])}
        if r["door"]:
            po, ps = exp(d["open"]), exp(d["shut"])
            row["door"] = {"where": r["door"]["where"], "held": r["door"]["held"],
                           "seen": r["door"]["seen"],
                           "belief": po / (po + ps) if po + ps > 0 else 0.5}
        rows.append(row)
    return summarise(rows) | {"episodes": episodes, "history": history, "seed": seed,
                              "slip": slip, "policy": policy}


def _mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def summarise(rows: list[dict]) -> dict:
    """Grouped by intended outcome, plus the door split. The `next_frame` block has the shape
    `evaluate.next_frame` writes (mode accuracy standing in for exact), so the World tab and
    `world results` read either game's file."""
    by = {}
    for w in sorted({r["outcome"] for r in rows}):
        g = [r for r in rows if r["outcome"] == w]
        by[w] = {"n": len(g), "exact": _mean(r["mode_exact"] for r in g),
                 "cells": None, "copy_baseline": _mean(r["copy"] for r in g),
                 "kl": _mean(r["kl"] for r in g), "off": _mean(r["off"] for r in g),
                 "p_mode_true": _mean(r["p_mode_true"] for r in g),
                 "p_mode_model": _mean(r["p_mode_model"] for r in g)}
    doors = {}
    for where in ("held, seen in context", "held, before the context", "not held"):
        g = [r["door"] for r in rows if r.get("door") and r["door"]["where"] == where]
        doors[where] = {"n": len(g), "belief": _mean(x["belief"] for x in g),
                        "truth": 1.0 if where.startswith("held") else 0.0,
                        "right": _mean((x["belief"] > 0.5) == x["held"] for x in g)}
    # Doors with no pickup in view: the share where a key is held anyway, and the model's
    # mean belief over the same doors. A calibrated model's mean matches the share. It is NOT
    # a ceiling on the "before the context" row: the frame itself carries evidence (a map with
    # no key left on it means one was probably taken: held 11.7% of the time against 4.0% with
    # a key still showing, on the blind agent), so a model that reads it can believe more on
    # held doors and less on empty-handed ones and still be calibrated.
    unseen = [r["door"] for r in rows if r.get("door") and not r["door"].get("seen", False)]
    base = _mean(x["held"] for x in unseen)
    unseen_belief = _mean(x["belief"] for x in unseen)
    n = len(rows)
    p = _mean(r["mode_exact"] for r in rows)
    return {"next_frame": {"n": n, "exact": p, "stderr": (p * (1 - p) / n) ** 0.5 if n else None,
                           "cells": None, "copy_baseline": _mean(r["copy"] for r in rows),
                           "by_outcome": by, "target": "most likely frame"},
            "calibration": {"kl": _mean(r["kl"] for r in rows),
                            "off": _mean(r["off"] for r in rows), "doors": doors,
                            "base_rate": base, "unseen_belief": unseen_belief}}
