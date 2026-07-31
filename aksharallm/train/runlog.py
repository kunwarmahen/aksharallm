"""Reading a run's `train_log.jsonl` back.

A multi-day run trained over evenings is many processes appending to one file. This module
is the *reader* for that file — the counterpart to the writer in `pretrain.py` — and it is
deliberately the only place that knows the log's shape, so the CLI (`scripts/sessions.py`)
and the web portal (`aksharallm.portal`) can never drift apart in how they interpret a run.

Three record kinds are written:

    {"event": "session_start", ...}   one per launch: pid, start_step, max_steps,
                                      stop_at (step bound) and stop_by (time budget)
    {"step": N, "loss": ..., ...}     one per `log_every` steps (plus the step a stop lands on)
    {"step": N, "val_loss": ...}      one per eval
    {"event": "session_end", ...}     one per clean exit: reason, last_step, elapsed

Everything here tolerates damage. A `kill -9` can cut a line mid-write, and a log that is
one byte short must still be readable — losing the last line is fine, refusing to open the
file is not.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def fmt_dur(seconds: float | None) -> str:
    """Compact duration: 45.2s / 12m30s / 6h05m / 3d04h.

    Multi-day runs are read at a glance days later, so every timing we print goes through
    this: "2d04h" is instantly meaningful, "187214.6s" is not.
    """
    if seconds is None:
        return "?"
    if seconds < 0:
        return "-" + fmt_dur(-seconds)
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


def load_records(path: str | Path) -> list[dict]:
    """Every parseable record in the log, in file order. Unparseable lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from a kill -9, or a partial flush
        if isinstance(rec, dict):
            out.append(rec)
    return out


def split_sessions(records: Iterable[dict]) -> list[dict]:
    """Group records into one session per launch.

    Sessions logged before `session_start`/`session_end` markers existed are still
    recovered: a step number going *backwards* can only mean a resume happened, which is
    enough to split on.
    """
    sessions: list[dict] = []
    cur: dict | None = None

    def new_session(**kw) -> dict:
        s = {"start_iso": None, "pid": None, "steps": [], "vals": [], "end": None, **kw}
        sessions.append(s)
        return s

    for rec in records:
        event = rec.get("event")
        if event == "session_start":
            cur = new_session(start_iso=rec.get("iso"), pid=rec.get("pid"),
                              start_step=rec.get("start_step"), t0=rec.get("time"),
                              max_steps=rec.get("max_steps"), stop_at=rec.get("stop_at"),
                              stop_by=rec.get("stop_by"),
                              tokens_per_step=rec.get("tokens_per_step"))
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
            if cur is not None and cur["steps"] and rec["step"] < cur["steps"][-1]["step"]:
                cur = None  # a resume with no marker: pre-marker log, or a kill -9
            if cur is None:
                cur = new_session(unmarked=True)
            cur["steps"].append(rec)
    return sessions


def load_sessions(path: str | Path) -> list[dict]:
    """`split_sessions(load_records(path))` — the usual entry point."""
    return split_sessions(load_records(path))


def summarise_session(s: dict, index: int) -> dict:
    """One session as flat, JSON-friendly fields — the row both the CLI table and the
    portal render. Anything unknowable is None rather than a placeholder string, so the
    presentation layer decides how to show a gap."""
    steps, vals, end = s["steps"], s["vals"], s["end"]
    first, last = (steps[0], steps[-1]) if steps else (None, None)

    started = s["start_iso"]
    if started is None and first and "time" in first:
        started = datetime.fromtimestamp(first["time"]).strftime("%Y-%m-%d %H:%M:%S")

    if end and end.get("elapsed") is not None:
        wall = end["elapsed"]
    elif first and last and "time" in first and "time" in last:
        wall = last["time"] - first["time"]
    else:
        wall = None

    rate = [r["tok_per_sec"] for r in steps if "tok_per_sec" in r]
    return {
        "index": index,
        "started": started,
        "pid": s.get("pid"),
        "first_step": first["step"] if first else None,
        "last_step": last["step"] if last else None,
        "n_logged": len(steps),
        "ema_first": first.get("ema") if first else None,
        "ema_last": last.get("ema") if last else None,
        "best_val": min((v["val_loss"] for v in vals), default=None),
        "tok_per_sec": sum(rate) / len(rate) if rate else None,
        "wall_s": wall,
        "ended": (end.get("reason") if end else None),
        # No end record means the session was killed, crashed, or is running right now;
        # the caller knows which, because it knows whether the pid is alive.
        "open": end is None,
        "unmarked": bool(s.get("unmarked")),
    }


def summarise_sessions(sessions: Iterable[dict]) -> list[dict]:
    return [summarise_session(s, i) for i, s in enumerate(sessions, 1)]


#: Per-step numeric fields worth plotting, in the order the portal charts them.
SERIES_KEYS = ("loss", "ema", "lr", "grad_norm", "tok_per_sec", "mfu", "s_per_step")


def series(records: Iterable[dict], max_points: int = 2000) -> dict[str, Any]:
    """Columnar per-step series for charting, plus the validation curve.

    Columnar (parallel arrays) rather than a list of objects: it is a third the JSON of the
    row form, and it is exactly what a plotting loop wants. `max_points` strides the step
    series so a 40,000-step run at `log_every=1` can't hand the browser a megabyte — the
    *last* point is always kept, because the newest reading is the one being watched.
    """
    steps = [r for r in records if "step" in r and "loss" in r]
    vals = [r for r in records if "val_loss" in r]

    if max_points and len(steps) > max_points:
        stride = len(steps) // max_points + 1
        steps = steps[::stride] + ([steps[-1]] if (len(steps) - 1) % stride else [])

    out: dict[str, Any] = {"step": [r["step"] for r in steps]}
    for key in SERIES_KEYS:
        out[key] = [r.get(key) for r in steps]
    out["val_step"] = [r["step"] for r in vals]
    out["val_loss"] = [r["val_loss"] for r in vals]
    return out


def latest(records: Iterable[dict]) -> dict:
    """The newest reading of each thing worth a number on a dashboard.

    Read from the *end* of the log backwards, so a run that has just resumed still reports
    a best-known val loss from an earlier session instead of a blank.
    """
    records = list(records)
    step_recs = [r for r in records if "step" in r and "loss" in r]
    val_recs = [r for r in records if "val_loss" in r]
    last = step_recs[-1] if step_recs else {}
    starts = [r for r in records if r.get("event") == "session_start"]
    ends = [r for r in records if r.get("event") == "session_end"]

    return {
        "step": last.get("step"),
        "loss": last.get("loss"),
        "ema": last.get("ema"),
        "lr": last.get("lr"),
        "grad_norm": last.get("grad_norm"),
        "tok_per_sec": last.get("tok_per_sec"),
        "mfu": last.get("mfu"),
        "s_per_step": last.get("s_per_step"),
        "elapsed": last.get("elapsed"),
        "eta_s": last.get("eta_s"),
        "step_time": last.get("time"),
        "val_loss": val_recs[-1]["val_loss"] if val_recs else None,
        "val_step": val_recs[-1]["step"] if val_recs else None,
        "best_val": min((v["val_loss"] for v in val_recs), default=None),
        "max_steps": starts[-1].get("max_steps") if starts else None,
        "tokens_per_step": starts[-1].get("tokens_per_step") if starts else None,
        "session_start": starts[-1] if starts else None,
        "session_end": ends[-1] if ends else None,
        "n_sessions": len(starts),
    }
