"""Generating training data, from a terminal.

    python -m aksharallm.synth recipes                       # what can be generated, and how it is checked
    python -m aksharallm.synth gen python --name py-v1 --n 200
    python -m aksharallm.synth gen chat   --name chat-v1 --n 500 --teacher gemma4:31b
    python -m aksharallm.synth gen python --name py-v1 --n 2000 --stop-in 45m
    python -m aksharallm.synth list                          # every generated dataset
    python -m aksharallm.synth show py-v1 --samples 3        # what is actually in it
    python -m aksharallm.synth export py-v1                  # -> the shape prepare_sft reads

A run appends to the named dataset and can be stopped at any point — Ctrl-C, `echo > STOP`,
or `--stop-in 30m` — leaving a complete dataset with its provenance written. Starting again
with the same name carries on from where it stopped, walking new cells of the seed grid.

Everything the portal's Synth tab does, it does by running these commands.

Read with: docs/13-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..train import stopfile
from .dataset import Dataset, SynthError, list_datasets
from .prompts import grid_size
from .recipes import catalogue, get_recipe
from .run import GenerateOptions, generate, preflight
from .teacher import SynthConfig, Teacher

SUBCOMMANDS = ("gen", "recipes", "list", "show", "export")


def cmd_recipes(args) -> int:
    print("\n  What can be generated, and what checks it before it is kept.\n")
    for row in catalogue():
        check = "the tests are executed" if row["verified"] else "format + diversity only"
        print(f"  {row['name']:<12} -> {row['consumer'].upper()}   [{check}]")
        print(f"               {row['blurb']}")
        print(f"               seed grid: {row['grid']:,} distinct prompt cells "
              f"(template v{row['template_version']})")
        print()
    print("  The seed grid is where diversity comes from. A recipe asked for more samples\n"
          "  than it has cells wraps to a new shuffle rather than stopping.\n")
    return 0


def cmd_list(args) -> int:
    rows = list_datasets(args.root)
    if not rows:
        print("no generated datasets yet — try:  python -m aksharallm.synth gen python "
              "--name py-v1 --n 50")
        return 0
    print()
    print(f"  {'dataset':<20} {'recipe':<11} {'kept':>7} {'asked':>7} {'pass':>6}  teacher")
    for row in rows:
        rate = "—" if row["pass_rate"] is None else f"{100 * row['pass_rate']:.0f}%"
        print(f"  {row['name']:<20} {str(row['recipe']):<11} {row['kept']:>7,} "
              f"{row['asked']:>7,} {rate:>6}  {', '.join(row['teachers']) or '—'}")
    print()
    return 0


def cmd_show(args) -> int:
    ds = Dataset(args.name, root=args.root)
    if not ds.exists:
        raise SynthError(f"no dataset '{args.name}' under data/synth/")
    stats = ds.stats()
    print(f"\n  {stats['name']}  —  {stats['recipe']} from "
          f"{', '.join(stats['teachers']) or '?'}  (template v{stats['template_version']})")
    print(f"  {stats['dir']}")
    print(f"\n  asked {stats['asked']:,} → parsed {stats['parsed']:,} → "
          f"kept {stats['kept']:,}"
          + (f"   ({100 * stats['pass_rate']:.0f}% of asks survived)"
             if stats["pass_rate"] is not None else ""))
    if stats["rejected"]:
        from .filters import REJECT_REASONS

        print("\n  dropped:")
        for reason, n in sorted(stats["rejected"].items(), key=lambda kv: -kv[1]):
            share = 100 * n / max(1, stats["rejected_total"])
            print(f"    {reason:<18} {n:>6,}  {share:4.0f}%   "
                  f"{REJECT_REASONS.get(reason, '')}")
    for row in ds.samples(args.samples) if args.samples else []:
        print("\n  " + "-" * 76)
        _print_sample(row)
    if args.rejects:
        print("\n  rejected examples:")
        for row in ds.rejects(args.rejects):
            print(f"\n    [{row.get('reason')}] {row.get('detail', '')}")
            print("    " + (row.get("text", "")[:400].replace("\n", "\n    ")))
    print()
    return 0


def _print_sample(row: dict) -> None:
    kind = row.get("kind")
    if kind == "python":
        print(f"  {row['id']}  ({row.get('difficulty')}, {row.get('topic')})")
        print(f"\n  PROBLEM\n    " + row["problem"].replace("\n", "\n    "))
        print(f"\n  SOLUTION\n    " + row["solution"].replace("\n", "\n    "))
        print(f"\n  TESTS\n    " + row["tests"].replace("\n", "\n    "))
        v = row.get("verify") or {}
        print(f"\n  verified: {row.get('verified')}  —  {v.get('detail', '')}")
    elif kind == "chat":
        print(f"  {row['id']}  ({row.get('subject')} · {row.get('constraint')})")
        print(f"\n  PROMPT\n    " + row["prompt"].replace("\n", "\n    "))
        print(f"\n  ANSWER\n    " + row["answer"].replace("\n", "\n    "))
    else:
        print(f"  {row['id']}  (flaw: {row.get('flaw')})")
        print(f"\n  PROMPT\n    " + row["prompt"].replace("\n", "\n    "))
        print(f"\n  CHOSEN\n    " + row["chosen"].replace("\n", "\n    "))
        print(f"\n  REJECTED\n    " + row["rejected"].replace("\n", "\n    "))


def cmd_export(args) -> int:
    ds = Dataset(args.name, root=args.root)
    if not ds.exists:
        raise SynthError(f"no dataset '{args.name}' under data/synth/")
    out = ds.export(Path(args.out) if args.out else None)
    print(f"\n  wrote {out['rows']:,} rows to {out['path']}")
    print(f"\n  tokenize it with:\n    {out['next']}\n")
    return 0


def cmd_gen(args) -> int:
    recipe = get_recipe(args.recipe)
    cfg = SynthConfig.load(args.root)
    model = args.teacher or cfg.model_for(recipe.name)
    teacher = Teacher(cfg, model=model,
                      temperature=args.temperature if args.temperature is not None
                      else cfg.temperature_for(recipe.name))

    stop_file = Path(args.stop_file) if args.stop_file else None
    opts = GenerateOptions(
        n=args.n, seed=args.seed, max_asks=args.max_asks,
        max_attempts=args.max_attempts if args.max_attempts is not None else cfg.max_attempts,
        dedup=args.dedup, verify=not args.no_verify, mutate=not args.no_mutate,
        stop_file=stop_file,
        max_seconds=stopfile.parse_duration(args.stop_in) if args.stop_in else None)

    warnings = preflight(recipe, teacher, opts, root=args.root)
    print(f"\n  {args.name}: {args.n:,} × {recipe.name} from {model}")
    print(f"  seed grid: {grid_size(recipe.name):,} cells; "
          f"{'tests executed + mutation-checked' if recipe.verified and opts.verify else 'no execution'}")
    for note in warnings:
        print(f"  {note}")
    if opts.max_seconds:
        print(f"  budget: {stopfile.fmt_left(opts.max_seconds)} of wall clock")
    if stop_file:
        print(f"  stop file: {stop_file}   (empty = now, N = at N kept, @epoch = at a time)")
    print()
    sys.stdout.flush()

    ds = Dataset(args.name, root=args.root)
    t0 = time.time()

    def progress(stats):
        pct = int(100 * min(1.0, stats.kept / max(1, args.n)))
        per = (time.time() - t0) / max(1, stats.kept)
        left = (args.n - stats.kept) * per
        rate = "" if stats.pass_rate is None else f" · pass {100 * stats.pass_rate:.0f}%"
        # Parsed by the portal (see portal/synth.py `_PROGRESS_RE`) — keep the shape.
        print(f"[synth] {recipe.name} {stats.kept}/{args.n} ({pct}%) · "
              f"{stats.asked} asked{rate} · {per:.1f}s/sample · "
              f"eta {stopfile.fmt_left(left)}", flush=True)

    try:
        stats = generate(ds, recipe, teacher, opts, on_progress=progress, root=args.root)
    except KeyboardInterrupt:
        ds.close("interrupted")
        print("\n  interrupted — the dataset is complete as far as it got; run the same "
              "command again to add more.\n")
        return 130

    print(f"\n  stopped: {stats.stopped}")
    print(f"  asked {stats.asked:,} → answered {stats.answered:,} → "
          f"parsed {stats.parsed:,} → kept {stats.kept:,}"
          + (f"   ({100 * stats.pass_rate:.0f}%)" if stats.pass_rate else ""))
    if stats.rejected:
        print("\n  dropped:")
        for reason, n in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<18} {n:>6,}")
    if stats.last_error:
        print(f"\n  last teacher error: {stats.last_error}")
    print(f"\n  {ds.dir}\n  next:  python -m aksharallm.synth show {args.name} --samples 2\n")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"dataset": args.name, "recipe": recipe.name, "teacher": model,
             "stats": stats.as_dict(), "meta": ds.stats()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m aksharallm.synth",
                                 description="Generate training data with a local teacher.")
    ap.add_argument("--root", default=None, help="repo root (default: this checkout)")
    sub = ap.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen", help="generate samples into a dataset")
    gen.add_argument("recipe", choices=["python", "chat", "preference"])
    gen.add_argument("--name", required=True, help="dataset directory under data/synth/")
    gen.add_argument("--n", type=int, default=50, help="samples to KEEP (not to ask for)")
    gen.add_argument("--teacher", default=None, help="Ollama model (default: per recipe)")
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--seed", type=int, default=0, help="which shuffle of the seed grid")
    gen.add_argument("--max-asks", type=int, default=None,
                     help="ceiling on teacher calls (default: 6 per wanted sample)")
    gen.add_argument("--max-attempts", type=int, default=None,
                     help="re-asks per seed after a format failure")
    gen.add_argument("--dedup", type=float, default=0.6,
                     help="near-duplicate Jaccard threshold; lower is stricter")
    gen.add_argument("--no-verify", action="store_true",
                     help="do not run the generated tests (python recipe). Recorded on "
                          "every sample — this is a different dataset, not a faster one.")
    gen.add_argument("--no-mutate", action="store_true",
                     help="skip the stubbed re-run that proves the tests are not vacuous")
    gen.add_argument("--stop-file", default=None,
                     help="path watched for a stop (empty = now, N = at N kept, @epoch)")
    gen.add_argument("--stop-in", default=None, help="wall-clock budget, e.g. 45m")
    gen.add_argument("--json", default=None, help="write the run's stats here")

    sub.add_parser("recipes", help="what can be generated")
    sub.add_parser("list", help="every generated dataset")

    show = sub.add_parser("show", help="what is in a dataset")
    show.add_argument("name")
    show.add_argument("--samples", type=int, default=1)
    show.add_argument("--rejects", type=int, default=0)

    exp = sub.add_parser("export", help="write the shape prepare_sft / prepare_dpo read")
    exp.add_argument("name")
    exp.add_argument("--out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 2
    try:
        return {"gen": cmd_gen, "recipes": cmd_recipes, "list": cmd_list,
                "show": cmd_show, "export": cmd_export}[args.cmd](args)
    except SynthError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
