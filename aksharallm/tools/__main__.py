"""`python -m aksharallm.tools data|eval` (docs/06 § "Tool use").

  data   write the teaching conversations as JSONL, for `prepare_sft jsonl`
  eval   paired: the same held-out problems with the tool and without, by kind, including a
         kind never seen in training; JSON to logs/tools/<run>-<label>.json

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

EVAL_SEED = 99


def cmd_data(args) -> int:
    from .problems import TRAIN_KINDS, conversation, problems
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in problems(args.n, args.seed, TRAIN_KINDS):
            f.write(json.dumps({"messages": conversation(p), "kind": p.kind}) + "\n")
    print(f"wrote {args.n} conversations ({', '.join(TRAIN_KINDS)}) -> {out}")
    return 0


def cmd_eval(args) -> int:
    from ..infer.cli import load_model
    from ..tokenizer.tokenizer import Tokenizer
    from .agent import score
    from .problems import KINDS, problems
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, ckpt = load_model(args.checkpoint, device)
    tok = Tokenizer(args.tokenizer)
    probs = problems(args.per_kind * len(KINDS), EVAL_SEED, KINDS)
    t = time.time()
    rep = score(model, tok, probs, device=device)
    print(f"{args.checkpoint}: {rep['n']} problems, greedy")
    print(f"{'kind':>10} {'n':>4} {'tool':>7} {'direct':>7} {'called':>7}")
    for k, r in rep["by_kind"].items():
        print(f"{k:>10} {r['n']:>4} {r['tool'] * 100:>6.0f}% {r['direct'] * 100:>6.0f}% "
              f"{r['called'] * 100:>6.0f}%" + ("   (never trained on)" if k == "remainder" else ""))
    pr = rep["paired"]
    print(f"{'all':>10} {rep['n']:>4} {rep['tool'] * 100:>6.1f}% {rep['direct'] * 100:>6.1f}% "
          f"{rep['called'] * 100:>6.0f}%   paired: tool only {pr['tool_only']}, direct only "
          f"{pr['direct_only']}, sign test p {pr['p']:.3g}")
    run = Path(args.checkpoint).parent.name
    out = Path("logs/tools") / f"{run}-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "step": ckpt.get("step"),
                               "seed": EVAL_SEED, "seconds": round(time.time() - t, 1),
                               **rep}, indent=1))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.tools",
                                description="tool use: teach and score a Python tool call")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data", help="teaching conversations -> JSONL for prepare_sft")
    s.add_argument("--out", default="data/tools/sft.jsonl")
    s.add_argument("--n", type=int, default=30_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_data)
    s = sub.add_parser("eval", help="the same problems with the tool and without, paired")
    s.add_argument("checkpoint")
    s.add_argument("--tokenizer", default="data/blend/tokenizer.json")
    s.add_argument("--per-kind", type=int, default=20)
    s.add_argument("--label", default="tools")
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_eval)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
