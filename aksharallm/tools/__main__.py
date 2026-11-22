"""`python -m aksharallm.tools data|eval` (docs/06 § "Tool use").

  data   write the teaching conversations as JSONL, for `prepare_sft jsonl` (--set v2: twenty
         kinds, four held out)
  eval   paired: the same held-out problems with the tool and without, by kind, including a
         kind never seen in training; JSON to logs/tools/<run>-<label>.json
  ask    one question to a chat checkpoint, with or without the tool; prints the transcript
  compare  two saved reports on the same problems, paired per problem (before vs after)
  results  every saved report, one line each

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .problems import SETS

EVAL_SEED = 99


def cmd_data(args) -> int:
    from .problems import SETS, conversation, problems
    train = SETS[args.set]["train"]
    out = Path(args.out or ("data/tools/sft.jsonl" if args.set == "v1"
                            else f"data/tools/sft-{args.set}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in problems(args.n, args.seed, train):
            f.write(json.dumps({"messages": conversation(p), "kind": p.kind}) + "\n")
    print(f"wrote {args.n} conversations, set {args.set} ({len(train)} kinds: "
          f"{', '.join(train)}) -> {out}")
    return 0


def cmd_eval(args) -> int:
    from ..infer.cli import load_model
    from ..tokenizer.tokenizer import Tokenizer
    from .agent import score
    from .problems import SETS, eval_kinds, problems
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, ckpt = load_model(args.checkpoint, device)
    tok = Tokenizer(args.tokenizer)
    kinds, held = eval_kinds(args.set), SETS[args.set]["held_out"]
    probs = problems(args.per_kind * len(kinds), EVAL_SEED, kinds)
    t = time.time()
    rep = score(model, tok, probs, device=device)
    print(f"{args.checkpoint}: {rep['n']} problems, greedy")
    print(f"{'kind':>10} {'n':>4} {'tool':>7} {'direct':>7} {'called':>7}")
    for k, r in rep["by_kind"].items():
        print(f"{k:>10} {r['n']:>4} {r['tool'] * 100:>6.0f}% {r['direct'] * 100:>6.0f}% "
              f"{r['called'] * 100:>6.0f}%" + ("   (held out of this set's training)"
                                                 if k in held else ""))
    pr = rep["paired"]
    print(f"{'all':>10} {rep['n']:>4} {rep['tool'] * 100:>6.1f}% {rep['direct'] * 100:>6.1f}% "
          f"{rep['called'] * 100:>6.0f}%   paired: tool only {pr['tool_only']}, direct only "
          f"{pr['direct_only']}, sign test p {pr['p']:.3g}")
    run = Path(args.checkpoint).parent.name
    out = Path("logs/tools") / f"{run}-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "step": ckpt.get("step"),
                               "set": args.set, "held_out": list(held),
                               "seed": EVAL_SEED, "seconds": round(time.time() - t, 1),
                               **rep}, indent=1))
    print(f"wrote {out}")
    return 0


def cmd_ask(args) -> int:
    from ..infer.cli import load_model
    from ..tokenizer.tokenizer import Tokenizer
    from .agent import final_number, solve
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, _ = load_model(args.checkpoint, device)
    r = solve(model, Tokenizer(args.tokenizer), args.question, use_tool=not args.no_tool,
              device=device)
    if args.json:
        print(json.dumps(r))
        return 0
    for m in r["messages"]:
        print(f"--- {m['role']}\n{m['content']}")
    print(f"--- {r['calls']} tool call(s); final number: {final_number(r['final'])}")
    return 0


def cmd_compare(args) -> int:
    from .agent import compare
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    try:
        c = compare(a, b)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"a = {args.a}\nb = {args.b}\n{c['n']} problems, paired\n")
    print(f"{'kind':>10} {'a tool':>7} {'b tool':>7} {'a direct':>9} {'b direct':>9}")
    for k, r in c["by_kind"].items():
        print(f"{k:>10} {r['a_tool']:>4}/{r['n']:<2} {r['b_tool']:>4}/{r['n']:<2} "
              f"{r['a_direct']:>6}/{r['n']:<2} {r['b_direct']:>6}/{r['n']:<2}")
    print()
    for name, r in c["paired"].items():
        print(f"{name:>30}: b right where a was wrong {r['b_only']}, the reverse {r['a_only']}, "
              f"sign test p {r['p']:.3g}")
    return 0


def cmd_results(args) -> int:
    rows = sorted(Path("logs/tools").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not rows:
        print("no reports yet: python -m aksharallm.tools eval <checkpoint>")
        return 0
    print(f"{'file':<46} {'set':>4} {'n':>4} {'tool':>7} {'direct':>7} {'called':>7} {'held out, tool':>15}")
    for p in rows:
        d = json.loads(p.read_text())
        held = d.get("held_out") or ["remainder"]
        h = [d["by_kind"][k] for k in held if k in d.get("by_kind", {})]
        ho = sum(r["tool"] * r["n"] for r in h) / max(1, sum(r["n"] for r in h))
        print(f"{p.name:<46} {d.get('set', 'v1'):>4} {d['n']:>4} {d['tool'] * 100:>6.1f}% "
              f"{d['direct'] * 100:>6.1f}% {d['called'] * 100:>6.0f}% {ho * 100:>14.1f}%")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.tools",
                                description="tool use: teach and score a Python tool call")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data", help="teaching conversations -> JSONL for prepare_sft")
    s.add_argument("--set", choices=tuple(SETS), default="v1",
                   help="v1: 6 kinds trained, 1 held out; v2: 20 trained, 4 held out; "
                        "rl: GRPO's 12; v3: 32 trained (v2 + rl), the same 4 held out")
    s.add_argument("--out", default=None, help="default data/tools/sft.jsonl (v1), sft-<set>.jsonl")
    s.add_argument("--n", type=int, default=30_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_data)
    s = sub.add_parser("eval", help="the same problems with the tool and without, paired")
    s.add_argument("checkpoint")
    s.add_argument("--tokenizer", default="data/blend/tokenizer.json")
    s.add_argument("--per-kind", type=int, default=20)
    s.add_argument("--set", choices=tuple(SETS), default="v1",
                   help="which kinds to ask: every kind of the set, trained and held out")
    s.add_argument("--label", default="tools")
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("ask", help="one question, with the tool (or --no-tool), transcript printed")
    s.add_argument("checkpoint")
    s.add_argument("question")
    s.add_argument("--tokenizer", default="data/blend/tokenizer.json")
    s.add_argument("--no-tool", action="store_true", help="do not run the call it writes")
    s.add_argument("--json", action="store_true")
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("compare", help="two reports on the same problems, paired per problem")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("results", help="every saved report, one line each")
    s.set_defaults(fn=cmd_results)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
