#!/usr/bin/env python3
"""Summarise a run's training *sessions* -- one row per launch.

A ~6-day run trained over evenings is many processes writing to one append-only
`checkpoints/<run>/train_log.jsonl`. `pretrain.py` brackets each launch with
`session_start` / `session_end` records; this reads them back so you can compare
sessions: did throughput drop last night, how much did loss actually move, where did a
session die.

    scripts/sessions.py                 # the only run that has a log, or list the options
    scripts/sessions.py small-code
    scripts/sessions.py small-code --steps    # also print every step line, grouped

The parsing lives in `aksharallm.train.runlog`, shared with the web portal
(`scripts/portal.sh`), so this table and the browser can never disagree about a run.

Sessions logged before the markers existed are still shown: a step number going backwards
is a resume, which is enough to split them.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from aksharallm.train.runlog import fmt_dur, load_sessions, summarise_sessions


def summarise(sessions: list[dict]) -> None:
    hdr = ("#", "started", "steps", "loss (ema)", "best val", "tok/s", "wall", "ended")
    rows = []
    for s in summarise_sessions(sessions):
        rows.append((
            str(s["index"]),
            s["started"] or "?",
            f"{s['first_step']} -> {s['last_step']}" if s["n_logged"] else "-",
            f"{s['ema_first']:.3f} -> {s['ema_last']:.3f}"
            if s["ema_first"] is not None and s["ema_last"] is not None else "-",
            f"{s['best_val']:.4f}" if s["best_val"] is not None else "-",
            f"{s['tok_per_sec'] / 1e3:.1f}k" if s["tok_per_sec"] else "-",
            fmt_dur(s["wall_s"]) if s["wall_s"] is not None else "?",
            s["ended"] or ("before session markers" if s["unmarked"]
                           else "no end record (killed, crashed, or still running)"),
        ))

    widths = [max(len(r[c]) for r in [hdr, *rows]) for c in range(len(hdr))]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))

    total_steps = sum(len(s["steps"]) for s in sessions)
    total_wall = sum(s["end"]["elapsed"] for s in sessions
                     if s["end"] and s["end"].get("elapsed") is not None)
    print(f"\n{len(sessions)} sessions, {total_steps} logged step lines, "
          f"{fmt_dur(total_wall)} of accounted wall-clock training.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", help="run name (a dir under checkpoints/)")
    ap.add_argument("--jsonl", help="path to a train_log.jsonl, instead of a run name")
    ap.add_argument("--steps", action="store_true", help="also list each session's steps")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.jsonl:
        path = Path(args.jsonl)
    else:
        found = sorted(root.glob("checkpoints/*/train_log.jsonl"))
        if args.run:
            path = root / "checkpoints" / args.run / "train_log.jsonl"
        elif len(found) == 1:
            path = found[0]
        else:
            names = " ".join(p.parent.name for p in found) or "(none)"
            raise SystemExit(f"which run? logs found for: {names}")
    if not path.exists():
        raise SystemExit(f"no log at {path}")

    print(f"{path}\n")
    sessions = load_sessions(path)
    if not sessions:
        raise SystemExit("no records in that log yet")
    summarise(sessions)

    if args.steps:
        for i, s in enumerate(sessions, 1):
            print(f"\n--- session {i} ---")
            for r in s["steps"]:
                when = (datetime.fromtimestamp(r["time"]).strftime("%m-%d %H:%M:%S")
                        if "time" in r else "  --  ")
                print(f"  {when}  step {r['step']:>6}  loss {r.get('loss', float('nan')):.4f}"
                      f"  ema {r.get('ema', float('nan')):.4f}"
                      f"  {r.get('tok_per_sec', 0)/1e3:>6.1f}k tok/s")


if __name__ == "__main__":
    main()
