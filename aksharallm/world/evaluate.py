"""How good is the world model? Checked exactly, because we wrote the game.

Three numbers, each against a baseline that says what it means:

  next frame        Given the last `history` steps and an action, is the predicted next frame
                    right in EVERY one of its 64 cells? Reported per outcome (`moved`,
                    `pushed`, `blocked`, `stayed`), because "copy the last frame" already
                    scores 100% on blocked and stayed and 0% on the rest. A model that only
                    learned to copy would look 30% right overall and 0% where it matters.
  cell accuracy     The same, per cell: a near miss and a garbled frame are not the same.
  rollout           Open loop: from one real frame, play `rollout` actions and feed the
                    model its OWN predictions back. A one-step error compounds here. Scored
                    as exact frames by step, plus how many predicted frames are not even a
                    possible state of the game (`game.parse` says None: two agents, a hole
                    in the border), which is the world-model version of "hallucination".

The next-frame check needs no sampling. Under teacher forcing the argmax of every cell is
what greedy decoding would write as long as every earlier cell was right, and a frame is
exact only if every cell is; so "all 64 teacher-forced argmaxes correct" equals "greedy
decoding writes this frame exactly". One forward pass per batch instead of 64.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch

from .data import FRAME, GAMES, VOCAB, episode_tokens, ids_of, vocab_of
from .game import episode

EVAL_SEED = 20_000


def _windows(n_episodes: int, steps: int, history: int, seed: int):
    """One context per transition t >= history: frames t-history..t with the actions between
    them and the action at t, then the true next frame. Episodes from their own seed stream."""
    rng = np.random.default_rng(seed)
    ctx, tgt, why, eps = [], [], [], []
    step_len = FRAME + 1
    for _ in range(n_episodes):
        frames, actions, outcomes = episode(rng, steps)
        toks = episode_tokens(frames, actions)[1:]          # drop the episode's <|endoftext|>
        eps.append((frames, actions, outcomes))
        for t in range(history, steps):
            a = (t - history) * step_len
            b = t * step_len + FRAME + 1                     # through frame t and action t
            ctx.append(toks[a:b])
            tgt.append(ids_of(frames[t + 1]))
            why.append(outcomes[t])
    return torch.tensor(ctx), torch.tensor(tgt), why, eps


@torch.no_grad()
def next_frame(model, *, episodes: int = 200, steps: int = 30, history: int = 5,
               batch: int = 64, seed: int = EVAL_SEED, device: str = "cpu") -> dict:
    ctx, tgt, why, _ = _windows(episodes, steps, history, seed)
    exact, cells = [], []
    for i in range(0, len(ctx), batch):
        c, t = ctx[i:i + batch].to(device), tgt[i:i + batch].to(device)
        x = torch.cat([c, t[:, :-1]], dim=1)
        logits, _ = model(x, full_logits=True)
        pred = logits[:, c.shape[1] - 1:].argmax(-1)         # the 64 positions of the frame
        ok = pred == t
        exact += ok.all(1).tolist()
        cells += ok.float().mean(1).tolist()
    copy = [w in ("blocked", "stayed") for w in why]         # what copying the last frame scores
    by = {}
    for w in sorted(set(why)):
        idx = [j for j, o in enumerate(why) if o == w]
        by[w] = {"n": len(idx), "exact": _mean([exact[j] for j in idx]),
                 "cells": _mean([cells[j] for j in idx]),
                 "copy_baseline": _mean([copy[j] for j in idx])}
    n = len(exact)
    p = _mean(exact)
    return {"n": n, "exact": p, "stderr": sqrt(p * (1 - p) / n) if n else float("nan"),
            "cells": _mean(cells), "copy_baseline": _mean(copy), "by_outcome": by,
            "history": history, "episodes": episodes, "seed": seed}


def most_likely_episode(game: str, rng, n_steps: int):
    """`(frames, actions, outcomes)` with no randomness in the world: what a rollout is
    scored against. Sokoban has none to remove; in `keys` every move goes where it was meant
    to, which is the single most likely game for those actions."""
    g = GAMES[game]
    return g.episode(rng, n_steps, slip=0.0) if game == "keys" else g.episode(rng, n_steps)


@torch.no_grad()
def rollout(model, *, episodes: int = 50, horizon: int = 10,
            seed: int = EVAL_SEED + 1, device: str = "cpu", game: str = "sokoban") -> dict:
    """Open loop from one real frame, greedy, the model's own frames fed back."""
    rng = np.random.default_rng(seed)
    vocab, parse = vocab_of(game), GAMES[game].parse
    exact = np.zeros(horizon)
    impossible = np.zeros(horizon)
    for _ in range(episodes):
        frames, actions, _ = most_likely_episode(game, rng, horizon)
        seq = ids_of(frames[0], vocab)
        for k in range(horizon):
            seq = seq + ids_of([actions[k]], vocab)
            out = []
            for _ in range(FRAME):                           # greedy, one cell at a time
                x = torch.tensor([(seq + out)[-model.cfg.max_seq_len:]], device=device)
                logits, _ = model(x, full_logits=True)
                out.append(int(logits[0, -1].argmax()))
            words = [vocab[i] for i in out]
            exact[k] += words == frames[k + 1]
            impossible[k] += parse(words) is None
            seq = seq + out
            seq = seq[-(model.cfg.max_seq_len - FRAME - 1):]
    return {"episodes": episodes, "horizon": horizon,
            "exact_by_step": (exact / episodes).tolist(),
            "impossible_by_step": (impossible / episodes).tolist(), "seed": seed}


def play(model, *, horizon: int = 5, seed: int = 0, device: str = "cpu",
         game: str = "sokoban") -> list[dict]:
    """One game, open loop: the model's frames fed back, each step beside the true frame.

    The same loop as `rollout`, for one episode, keeping the frames so a person can see where
    it goes wrong (`world play`, and the portal's World tab draw this)."""
    vocab, parse = vocab_of(game), GAMES[game].parse
    frames, actions, outcomes = most_likely_episode(game, np.random.default_rng(seed), horizon)
    seq = ids_of(frames[0], vocab)
    steps = [{"step": 0, "action": None, "outcome": None, "true": frames[0],
              "model": frames[0], "wrong": 0, "impossible": False}]
    with torch.no_grad():
        for k in range(horizon):
            seq = seq + ids_of([actions[k]], vocab)
            out = []
            for _ in range(FRAME):
                x = torch.tensor([(seq + out)[-model.cfg.max_seq_len:]], device=device)
                logits, _ = model(x, full_logits=True)
                out.append(int(logits[0, -1].argmax()))
            words = [vocab[i] for i in out]
            steps.append({"step": k + 1, "action": actions[k], "outcome": outcomes[k],
                          "true": frames[k + 1], "model": words,
                          "wrong": sum(a != b for a, b in zip(words, frames[k + 1])),
                          "impossible": parse(words) is None})
            seq = (seq + out)[-(model.cfg.max_seq_len - FRAME - 1):]
    return steps


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")
