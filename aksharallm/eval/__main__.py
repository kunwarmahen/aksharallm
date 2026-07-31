"""The eval harness, from a terminal.

    python -m aksharallm.eval suites                       # what can be measured, and why
    python -m aksharallm.eval fetch --all                  # download the benchmarks, once
    python -m aksharallm.eval small-code                   # the default set, on the best ckpt
    python -m aksharallm.eval small-code --suite all --limit 0
    python -m aksharallm.eval small-code --suite judge --judge-model gemma4:31b
    python -m aksharallm.eval report --suite mmlu          # every score, across steps

The first positional is a checkpoint unless it names a subcommand, so the common case is
short. A checkpoint reference is a run name (`small-code` → its best checkpoint), an id
(`small-code/ckpt_last.pt`) or a path — the same resolution the Playground uses.

Everything writes to `logs/eval/<timestamp>-<run>-<label>.json`, which the portal's Eval tab
reads. A run started here appears there, and the other way round.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .report import Results, compare_table, summary_table
from .runner import Harness, Options, describe
from .sources import EvalError, SOURCES, fetch, status
from .suites import ALL_SUITES, DEFAULT_SUITES, SUITES, catalogue, resolve

SUBCOMMANDS = ("run", "fetch", "suites", "report")


def cmd_suites(args) -> int:
    print("\n  What can be measured, and what to expect at this project's scale.\n")
    cached = {row["name"] for row in status() if row["cached"]}
    for row in catalogue():
        tags = []
        if row["groups"]["default"]:
            tags.append("default")
        if row["groups"]["fast"]:
            tags.append("fast")
        head = f"  {row['name']}"
        print(f"{head:<16} [{row['kind']}]"
              + (f"  ({', '.join(tags)})" if tags else ""))
        print(f"                  {row['blurb']}")
        if row["source"]:
            state = "cached" if row["source"] in cached else "NOT downloaded"
            print(f"                  data: {row['source']} — {state}"
                  + (f", {row['shots']}-shot from {row['shot_source']}" if row["shot_source"] else ""))
        print(f"                  expect: {row['expect']}")
        print()
    print(f"  groups:  default = {', '.join(DEFAULT_SUITES)}")
    print(f"           all     = {', '.join(ALL_SUITES)}\n")
    return 0


def cmd_fetch(args) -> int:
    if args.list:
        print()
        for row in status():
            mark = "✓" if row["cached"] else " "
            size = f"{row['bytes'] / 1e6:.1f} MB" if row["cached"] else ""
            rows = f"{row['rows']:,} rows" if row.get("rows") else ""
            print(f"  {mark} {row['name']:<16} {rows:>12}  {size:>9}  {row['repo'] or row['repos'][0]}")
            print(f"     {row['note']}")
        print()
        return 0

    names = args.names or (list(SOURCES) if args.all else [])
    if not names:
        print("nothing to fetch. Name datasets, or use --all / --list.", file=sys.stderr)
        return 2
    for name in names:
        fetch(name, refresh=args.refresh)
    return 0


def cmd_report(args) -> int:
    results = Results(args.root)
    if args.suite:
        print(compare_table(results.compare(args.suite, run=args.run)))
        print()
        return 0
    rows = results.rows(limit=args.limit, run=args.run)
    if not rows:
        print("\n  no evaluations recorded yet. Run one:  python -m aksharallm.eval <run>\n")
        return 0
    names = [n for n in SUITES if any(n in r["scores"] for r in rows)]
    print()
    print("  " + f"{'when':<17}{'step':>9}  " + "".join(f"{n[:11]:>12}" for n in names))
    print("  " + "-" * (28 + 12 * len(names)))
    for row in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["when"]))
        step = f"{row['step']:,}" if row["step"] is not None else "–"
        cells = ""
        for name in names:
            entry = row["scores"].get(name)
            if not entry or entry.get("score") is None:
                cells += f"{'–':>12}"
            elif entry.get("kind") == "ppl":
                cells += f"{entry['score']:>12.3f}"
            else:
                cells += f"{entry['score'] * 100:>11.1f}%"
        print(f"  {when:<17}{step:>9}  {cells}   {row['checkpoint']}")
    print()
    print("  One suite across every step:  python -m aksharallm.eval report --suite mmlu\n")
    return 0


def cmd_run(args) -> int:
    harness = Harness(args.root)
    opts = Options(
        suites=resolve(args.suite), limit=args.limit, shots=args.shots,
        device=args.device, adapter=args.adapter, batch_tokens=args.batch_tokens,
        max_new_tokens=args.max_new_tokens, judge_model=args.judge_model,
        keep_items=not args.no_items, label=args.label)

    pre = harness.preflight(args.checkpoint, opts)
    if pre["missing"]:
        print(f"\n  Benchmark data is not downloaded: {', '.join(pre['missing'])}\n"
              f"  Fetch it once (it is cached under data/eval/ and reused forever):\n\n"
              f"      python -m aksharallm.eval fetch {' '.join(pre['missing'])}\n",
              file=sys.stderr)
        return 2
    for note in pre["notes"]:
        print(f"  note: {note}")

    last = [0.0]

    def progress(done: int, total: int, label: str):
        # A terminal line per second is enough. The portal reads the same stdout from a log
        # file, so this doubles as the browser's progress bar — hence the parseable shape.
        now = time.monotonic()
        if now - last[0] < 1.0 and done < total:
            return
        last[0] = now
        print(f"[eval] {label} {done}/{total} ({done / max(1, total) * 100:.0f}%)", flush=True)

    result = harness.run(args.checkpoint, opts, progress=progress, val_bin=args.val_bin)
    path = harness.save(result, args.json)
    print(summary_table(result))
    print(f"  wrote {path}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.eval",
        description="Measure a checkpoint on real benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="evaluate a checkpoint (the default)")
    run.add_argument("checkpoint", help="run name, run/name.pt, or a path")
    run.add_argument("--suite", default=None,
                     help="comma-separated, or a group: default, fast, mc, all. "
                          f"Suites: {', '.join(SUITES)}")
    run.add_argument("--limit", type=int, default=None,
                     help="items per suite; 0 for the whole cached split")
    run.add_argument("--shots", type=int, default=None, help="override the few-shot count")
    run.add_argument("--device", default=None, choices=("cuda", "cpu"),
                     help="default: the CPU while a run is training, else the GPU")
    run.add_argument("--adapter", default=None, help="a LoRA adapter to evaluate on top")
    run.add_argument("--batch-tokens", type=int, default=2048,
                     help="tokens per forward pass; lower it if scoring runs out of memory")
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--judge-model", default=None, help="Ollama model for the judge suite")
    run.add_argument("--val-bin", default=None,
                     help="perplexity data; defaults to the checkpoint's own val split")
    run.add_argument("--label", default=None, help="goes in the result filename")
    run.add_argument("--no-items", action="store_true",
                     help="do not keep per-question verdicts in the JSON")
    run.add_argument("--json", default=None, help="write the result here instead")
    run.add_argument("--root", default=None)

    fetch_p = sub.add_parser("fetch", help="download benchmark data into data/eval/")
    fetch_p.add_argument("names", nargs="*", metavar="NAME",
                         help=f"one or more of: {', '.join(SOURCES)}")
    fetch_p.add_argument("--all", action="store_true")
    fetch_p.add_argument("--list", action="store_true", help="what is downloaded already")
    fetch_p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    fetch_p.add_argument("--root", default=None)

    sub.add_parser("suites", help="what can be measured, and what to expect")

    rep = sub.add_parser("report", help="scores across every evaluation so far")
    rep.add_argument("--suite", default=None, help="one suite, across steps")
    rep.add_argument("--run", default=None, help="only this training run")
    rep.add_argument("--limit", type=int, default=25)
    rep.add_argument("--root", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `python -m aksharallm.eval small-code` should work without typing `run`. Anything that
    # is not a subcommand and not a flag is a checkpoint.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["run"] + argv
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    try:
        return {"run": cmd_run, "fetch": cmd_fetch, "suites": cmd_suites,
                "report": cmd_report}[args.cmd](args)
    except EvalError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  stopped.\n", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
