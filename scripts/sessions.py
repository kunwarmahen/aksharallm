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

Sessions logged before the markers existed are still shown: a step number going backwards
is a resume, which is enough to split them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def fmt_dur(seconds: float) -> str:
    """Same compact form the training logs use (45.2s / 12m30s / 6h05m / 3d04h)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def load_sessions(path: Path) -> list[dict]:
    """Split the jsonl into sessions. Tolerates truncated/garbled lines -- a log that a
    `kill -9` cut mid-write should still be readable."""
    sessions: list[dict] = []
    cur: dict | None = None

    def new_session(**kw) -> dict:
        s = {"start_iso": None, "pid": None, "steps": [], "vals": [], "end": None, **kw}
        sessions.append(s)
        return s

    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = rec.get("event")
        if event == "session_start":
            cur = new_session(start_iso=rec.get("iso"), pid=rec.get("pid"),
                              start_step=rec.get("start_step"), t0=rec.get("time"))
        elif event == "session_end":
            if cur is None:
                cur = new_session()
            cur["end"] = rec
            cur = None
        elif "val_loss" in rec:
            if cur is None:
                cur = new_session()
            cur["vals"].append(rec)
        elif "step" in rec:
            # A step number that goes backwards means a resume happened without a marker
            # (a log from before this feature, or a session killed with -9).
            if cur is not None and cur["steps"] and rec["step"] < cur["steps"][-1]["step"]:
                cur = None
            if cur is None:
                cur = new_session(unmarked=True)
            cur["steps"].append(rec)
    return sessions


def summarise(sessions: list[dict]) -> None:
    hdr = ("#", "started", "steps", "loss (ema)", "best val", "tok/s", "wall", "ended")
    rows = []
    for i, s in enumerate(sessions, 1):
        steps, vals, end = s["steps"], s["vals"], s["end"]
        first, last = (steps[0], steps[-1]) if steps else (None, None)

        started = s["start_iso"]
        if started is None and first and "time" in first:
            started = datetime.fromtimestamp(first["time"]).strftime("%Y-%m-%d %H:%M:%S")

        if end and end.get("elapsed") is not None:
            wall = fmt_dur(end["elapsed"])
        elif first and last and "time" in first and "time" in last:
            wall = fmt_dur(last["time"] - first["time"])
        else:
            wall = "?"

        rate = [r["tok_per_sec"] for r in steps if "tok_per_sec" in r]
        rows.append((
            str(i),
            started or "?",
            f"{first['step']} -> {last['step']}" if steps else "-",
            f"{first['ema']:.3f} -> {last['ema']:.3f}"
            if steps and "ema" in first and "ema" in last else "-",
            f"{min(v['val_loss'] for v in vals):.4f}" if vals else "-",
            f"{sum(rate) / len(rate) / 1e3:.1f}k" if rate else "-",
            wall,
            (end.get("reason") or "?") if end else
            ("no end record (killed, crashed, or still running)"
             if not s.get("unmarked") else "before session markers"),
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
