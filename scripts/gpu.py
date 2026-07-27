#!/usr/bin/env python3
"""GPU telemetry from a terminal: what the card is doing now, and how that compares to idle.

The same samples the portal's GPU panel charts (`logs/gpu.jsonl`), so the two always agree.

    scripts/gpu.sh                    # now, plus a 1-hour summary and sparklines
    scripts/gpu.sh --window 6h        # 15m / 1h / 6h / 24h / all
    scripts/gpu.sh watch              # one line a second, like `nvidia-smi -l`
    scripts/gpu.sh daemon             # record samples without running the portal
    scripts/gpu.sh raw                # a single nvidia-smi reading, as JSON

`scripts/portal.sh` records samples by default, so the history is there as long as the
portal has been up. Nothing else needs to run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from aksharallm.portal import gpu as gpumod
    from aksharallm.portal.runs import RunStore
    from aksharallm.train.runlog import fmt_dur
except ImportError:  # not the venv's python — re-exec with it
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists() and Path(sys.executable).resolve() != venv.resolve():
        os.execv(str(venv), [str(venv), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise

SPARK = "▁▂▃▄▅▆▇█"


def parse_window(text: str) -> float | None:
    """`15m` / `6h` / `2d` / `all` -> seconds."""
    text = (text or "1h").strip().lower()
    if text in ("all", "0", ""):
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def spark(values: list[float | None], width: int = 48) -> str:
    """A one-line shape of the series. Not a chart — a hint that sends you to the portal."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "(no data)"
    if len(vals) > width:
        bucket = len(vals) / width
        vals = [sum(vals[int(i * bucket):int((i + 1) * bucket)] or [0])
                / max(len(vals[int(i * bucket):int((i + 1) * bucket)]), 1)
                for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(int((v - lo) / (hi - lo) * (len(SPARK) - 1)), len(SPARK) - 1)]
                   for v in vals)


def cmd_status(store: RunStore, args) -> int:
    snap = gpumod.snapshot(store, window_s=parse_window(args.window), index=args.index)
    if not snap["available"]:
        print(snap["reason"])
        return 1

    dev = snap["devices"][args.index] if args.index < len(snap["devices"]) else snap["devices"][0]
    cur = snap["current"] or {}
    print(f"GPU {dev['index']}  {dev['name']}"
          f"   {dev['mem_total'] / 1024:.0f} GB   {dev['power_limit']:.0f} W limit")
    if cur:
        print(f"now      {cur.get('util', 0):.0f}% util   "
              f"{(cur.get('mem_used') or 0) / 1024:.1f} GB   "
              f"{cur.get('temp', 0):.0f}°C   {cur.get('power') or 0:.0f} W"
              f"   ({fmt_dur(snap['current_age_s'])} ago)"
              + (f"   -- {snap['current_run']} is training" if snap["current_run"]
                 else "   -- nothing training"))
    else:
        print("now      no samples yet")

    if not snap["sampling"]:
        print("         NOT SAMPLING — no history is being recorded. Run scripts/portal.sh")
        print("         or scripts/gpu.sh daemon.")

    print(f"\nlast {args.window} ({snap['samples']} samples)")
    hdr = ("", "time", "util", "memory", "temp", "peak temp", "power")
    rows = []
    for key, label in (("training", "while training"), ("idle", "idle")):
        s = snap["summary"][key]
        if not s:
            continue
        rows.append((label, fmt_dur(s["seconds"]),
                     f"{s['util']:.0f}%" if s["util"] is not None else "-",
                     f"{s['mem_used'] / 1024:.1f} GB" if s["mem_used"] is not None else "-",
                     f"{s['temp']:.0f}°C" if s["temp"] is not None else "-",
                     f"{s['temp_max']:.0f}°C" if s["temp_max"] is not None else "-",
                     f"{s['power']:.0f} W" if s["power"] is not None else "-"))
    if rows:
        widths = [max(len(r[c]) for r in [hdr, *rows]) for c in range(len(hdr))]
        line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
        print(line)
        print("-" * len(line))
        for r in rows:
            print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    else:
        print("(nothing recorded in this window)")

    series = snap["series"]
    if series["time"]:
        print()
        for key, label, unit in (("util", "util  ", "%"), ("mem_used", "memory", "MB"),
                                 ("temp", "temp  ", "°C"), ("power", "power ", "W")):
            vals = [v for v in series[key] if v is not None]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            scale = 1024 if key == "mem_used" else 1
            print(f"{label} {spark(series[key])}  {lo / scale:.0f}-{hi / scale:.0f}"
                  f"{'GB' if key == 'mem_used' else unit}")
        print(f"\nthe same samples the portal charts: {gpumod.Sampler(store).path}")
    return 0


def cmd_watch(store: RunStore, args) -> int:
    print("util  memory      temp   power    run            (Ctrl-C to stop)")
    try:
        while True:
            rec = gpumod.sample(store)
            if rec is None:
                print("nvidia-smi is not answering", file=sys.stderr)
                return 1
            g = rec["gpus"][min(args.index, len(rec["gpus"]) - 1)]
            print(f"{g['util'] or 0:>3.0f}%  {(g['mem_used'] or 0) / 1024:>5.1f} GB   "
                  f"{g['temp'] or 0:>3.0f}°C  {g['power'] or 0:>5.0f} W  "
                  f"{rec.get('run') or '-'}")
            time.sleep(args.every)
    except KeyboardInterrupt:
        return 0


def cmd_daemon(store: RunStore, args) -> int:
    sampler = gpumod.Sampler(store, interval=args.every)
    if not sampler.devices():
        print("no NVIDIA GPU detected", file=sys.stderr)
        return 1
    if not sampler.lock():
        print(f"already sampling as pid {sampler.holder()} (the portal does this by "
              "default). Nothing to do.", file=sys.stderr)
        return 1
    print(f"sampling every {args.every:.0f}s into {sampler.path} (pid {os.getpid()}). "
          "Ctrl-C to stop.")
    try:
        sampler.run_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sampler.release()
    return 0


def cmd_raw(store: RunStore, args) -> int:
    rec = gpumod.sample(store)
    if rec is None:
        print("no NVIDIA GPU detected", file=sys.stderr)
        return 1
    print(json.dumps(rec, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/gpu.sh",
                                 description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("command", nargs="?", default="status",
                    choices=["status", "watch", "daemon", "raw"])
    ap.add_argument("--window", default="1h", help="15m | 1h | 6h | 24h | all (default 1h)")
    ap.add_argument("--index", type=int, default=0, help="which GPU (default 0)")
    ap.add_argument("--every", type=float, default=None, metavar="SECONDS",
                    help="sample interval for watch/daemon")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.every is None:
        args.every = 1.0 if args.command == "watch" else gpumod.SAMPLE_SECONDS

    store = RunStore(Path(args.root))
    return {"status": cmd_status, "watch": cmd_watch,
            "daemon": cmd_daemon, "raw": cmd_raw}[args.command](store, args)


if __name__ == "__main__":
    sys.exit(main())
