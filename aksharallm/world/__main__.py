"""`python -m aksharallm.world prepare|eval|play` (docs/25).

  prepare   write data/world/{train,val}.bin + tokenizer.json from random games
  eval      next-frame accuracy by outcome, and open-loop rollouts, to logs/world/<run>-*.json
  play      one rollout printed beside the true game, to see where it goes wrong
  results   every saved evaluation, one line each

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    from .evaluate import play
    from .game import SIZE
    device = _device(args)
    model, _ = _load(args.checkpoint, device)
    steps = play(model, horizon=args.horizon, seed=args.seed, device=device)
    if args.json:
        print(json.dumps(steps))
        return 0
    for st in steps[1:]:
        mark = "ok" if not st["wrong"] else f"{st['wrong']} cells wrong"
        mark += " (impossible frame)" if st["impossible"] else ""
        print(f"step {st['step']}: {st['action']} ({st['outcome']}) -> {mark}")
        print(f"  {'true':<{2 * SIZE}}   model")
        for r in range(SIZE):
            print("  " + " ".join(st["true"][r * SIZE:(r + 1) * SIZE]) + "   "
                  + " ".join(st["model"][r * SIZE:(r + 1) * SIZE]))
    return 0


def cmd_results(args) -> int:
    rows = sorted(Path("logs/world").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not rows:
        print("no evaluations yet: python -m aksharallm.world eval <checkpoint>")
        return 0
    print(f"{'file':<40} {'step':>6} {'exact':>7} {'copy':>6} {'rollout@last':>13} {'impossible':>11}")
    for p in rows:
        d = json.loads(p.read_text())
        nf, ro = d["next_frame"], d["rollout"]
        print(f"{p.name:<40} {d.get('step') or 0:>6} {nf['exact'] * 100:>6.1f}% "
              f"{nf['copy_baseline'] * 100:>5.1f}% {ro['exact_by_step'][-1] * 100:>12.0f}% "
              f"{max(ro['impossible_by_step']) * 100:>10.0f}%")
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
    s.add_argument("--json", action="store_true", help="the frames as JSON, not a picture")
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("results", help="every saved evaluation, one line each")
    s.set_defaults(fn=cmd_results)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
