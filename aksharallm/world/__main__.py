"""`python -m aksharallm.world prepare|eval|play` (docs/25).

  prepare   write data/world/{train,val}.bin + tokenizer.json from random games
  eval      next-frame accuracy by outcome, and open-loop rollouts, to logs/world/<run>-*.json
  play      one rollout printed beside the true game, to see where it goes wrong

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _device(args) -> str:
    return "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"


def _load(path, device):
    from ..infer.cli import load_model
    from .data import VOCAB
    model, ckpt = load_model(path, device)
    if model.cfg.vocab_size != len(VOCAB):
        raise SystemExit(f"{path} has a {model.cfg.vocab_size}-token vocabulary, not this "
                         f"game's {len(VOCAB)}: it is not a world model")
    return model, ckpt


def cmd_prepare(args) -> int:
    from .data import prepare
    meta = prepare(args.out, episodes=args.episodes, val_episodes=args.val_episodes,
                   steps=args.steps, seed=args.seed)
    print(json.dumps({k: meta[k] for k in ("tokens", "outcomes")}, indent=1))
    print(f"wrote {args.out}/train.bin, val.bin, tokenizer.json, meta.json")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import next_frame, rollout
    device = _device(args)
    model, ckpt = _load(args.checkpoint, device)
    t = time.time()
    nf = next_frame(model, episodes=args.episodes, history=args.history, device=device)
    print(f"next frame   {nf['exact'] * 100:.1f}% exact (± {nf['stderr'] * 100:.1f}, n={nf['n']}), "
          f"cells {nf['cells'] * 100:.2f}%, copying the last frame would score "
          f"{nf['copy_baseline'] * 100:.1f}%")
    print(f"{'outcome':>10} {'n':>6} {'exact':>8} {'cells':>8} {'copy':>7}")
    for w, r in nf["by_outcome"].items():
        print(f"{w:>10} {r['n']:>6} {r['exact'] * 100:>7.1f}% {r['cells'] * 100:>7.2f}% "
              f"{r['copy_baseline'] * 100:>6.0f}%")
    ro = rollout(model, episodes=args.rollouts, horizon=args.horizon, device=device)
    print("rollout      exact frame by step (own predictions fed back): "
          + " ".join(f"{x * 100:.0f}" for x in ro["exact_by_step"]))
    print("             impossible frames by step:                      "
          + " ".join(f"{x * 100:.0f}" for x in ro["impossible_by_step"]))
    run = Path(args.checkpoint).parent.name
    out = Path("logs/world") / f"{run}-step{ckpt.get('step')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "step": ckpt.get("step"),
                               "next_frame": nf, "rollout": ro,
                               "seconds": round(time.time() - t, 1)}, indent=1))
    print(f"wrote {out}")
    return 0


def cmd_play(args) -> int:
    from .data import FRAME, VOCAB, ids_of
    from .game import SIZE, episode
    device = _device(args)
    model, _ = _load(args.checkpoint, device)
    frames, actions, outcomes = episode(np.random.default_rng(args.seed), args.horizon)
    seq = ids_of(frames[0])
    with torch.no_grad():
        for k in range(args.horizon):
            seq += ids_of([actions[k]])
            out = []
            for _ in range(FRAME):
                logits, _ = model(torch.tensor([(seq + out)[-model.cfg.max_seq_len:]],
                                               device=device), full_logits=True)
                out.append(int(logits[0, -1].argmax()))
            pred, true = [VOCAB[i] for i in out], frames[k + 1]
            mark = "ok" if pred == true else f"{sum(a != b for a, b in zip(pred, true))} cells wrong"
            print(f"step {k + 1}: {actions[k]} ({outcomes[k]}) -> {mark}")
            print(f"  {'true':<{2 * SIZE}}   model")
            for r in range(SIZE):
                print("  " + " ".join(true[r * SIZE:(r + 1) * SIZE]) + "   "
                      + " ".join(pred[r * SIZE:(r + 1) * SIZE]))
            seq = (seq + out)[-(model.cfg.max_seq_len - FRAME - 1):]
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.world", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare", help="random games -> train.bin / val.bin / tokenizer.json")
    s.add_argument("--out", default="data/world")
    s.add_argument("--episodes", type=int, default=60_000)
    s.add_argument("--val-episodes", type=int, default=2_000)
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("eval", help="next-frame accuracy by outcome + open-loop rollouts")
    s.add_argument("checkpoint")
    s.add_argument("--episodes", type=int, default=200, help="episodes for next-frame accuracy")
    s.add_argument("--history", type=int, default=5, help="steps of context before the action")
    s.add_argument("--rollouts", type=int, default=50)
    s.add_argument("--horizon", type=int, default=10)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("play", help="one rollout beside the true game")
    s.add_argument("checkpoint")
    s.add_argument("--horizon", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_play)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
