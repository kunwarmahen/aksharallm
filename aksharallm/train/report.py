"""The end-of-run report: everything one training run did, on one page a person can read.

A finished run leaves a lot of evidence behind — 40,000 step lines, a dozen session markers,
an energy ledger, a folder of benchmark JSON — and no single place that says *how it went*.
Reading that evidence is a skill; a run should not require one. So when a trainer exits it
writes `<out_dir>/report.md`: the budget it spent, what the loss did, how fast it ran, what
it cost in electricity, what it scored, and — the part worth the module — **the things worth
knowing**, computed rather than eyeballed: crashed sessions, loss spikes, a validation loss
that stopped improving two thirds of the way in, a throughput regression between sessions,
an expert that died.

Three rules it is built on:

1. **Derived, never authoritative.** Everything here is recomputed from `train_log.jsonl`
   and the files around it, so the report can be regenerated at any time
   (`python -m aksharallm.train.report <run>`) and deleting it loses nothing. That is why
   it is safe to overwrite on every exit.
2. **A gap is a gap.** Anything unknowable prints as `–`, never as zero. A report that
   invents a number is worse than one that admits it was not measured — which is the same
   rule the cost panel's `coverage` exists for.
3. **It must never take the run down.** A trainer calls `write_quietly`, which swallows
   everything: a report that cannot be written is a shrug, not a lost day of training.

It reads the other three trainers' logs too (`sft_log.jsonl`, `dpo_log.jsonl`,
`grpo_log.jsonl`). Those carry no session markers and no throughput, so their reports are
shorter — the shape is the same and the missing rows say so.

Read with: docs/09-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from . import runlog
from .runlog import fmt_dur

#: The log each trainer writes, and what to call that kind of run in prose. Order matters:
#: it is the search order when a run directory holds more than one.
LOGS: dict[str, str] = {
    "train_log.jsonl": "pretraining",
    "sft_log.jsonl": "SFT",
    "dpo_log.jsonl": "DPO",
    "grpo_log.jsonl": "GRPO",
}

#: Per-step numbers that only some trainers write, with the direction that counts as better.
#: DPO's preference accuracy and GRPO's reward are the headline of their runs the way
#: validation loss is the headline of a pretraining run, so a report that only knew about
#: loss would leave out the one number those runs are for.
EXTRAS: dict[str, tuple[str, str]] = {
    "acc": ("train preference accuracy", "max"),
    "val_acc": ("val preference accuracy", "max"),
    "reward": ("mean reward", "max"),
    "solved": ("fraction solved", "max"),
}

_SPARK = "▁▂▃▄▅▆▇█"


def repo_root() -> Path:
    """The repo root, inferred from this file's location (aksharallm/train/report.py)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# small formatters — the report is read by a person, so every number is shaped for one
# --------------------------------------------------------------------------------------

def num(v, digits: int = 4) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def integer(v) -> str:
    return "–" if v is None else f"{int(v):,}"


def compact(n) -> str:
    """1.2B / 340M / 12.5K — for token counts, which are otherwise unreadable."""
    if n is None:
        return "–"
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}".replace(".00", "")
        n /= 1000
    return f"{n:.2f}P"


def bytes_(n) -> str:
    if n is None:
        return "–"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def wh(v) -> str:
    return "–" if v is None else (f"{v / 1000:.2f} kWh" if v >= 1000 else f"{v:.0f} Wh")


def clock(ts) -> str:
    return "–" if not ts else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def dur(seconds) -> str:
    """`fmt_dur`, but an unknown duration is the same em-dash every other gap uses. The
    trainers' logs did not always carry timestamps, and "?" in one column beside "–" in the
    next reads as two different kinds of missing."""
    return "–" if seconds is None else fmt_dur(seconds)


def score(entry: dict) -> str:
    """One benchmark cell. Perplexity is not a percentage and a judge score is out of five —
    formatting all three as `score * 100` is how a report claims the model scored 433% on
    perplexity, which it did in the first version of this file."""
    if not entry or entry.get("score") is None:
        return "–"
    kind = entry.get("kind")
    if kind == "ppl":
        return f"{entry['score']:.3f}"
    if kind == "judge":
        return f"{entry['score']:.2f}/5"
    return f"{entry['score'] * 100:.1f}%"


def ppl(loss) -> str:
    """Perplexity beside every loss. A loss of 2.65 means nothing on its own; "14.2 — it is
    choosing between about fourteen equally likely words" is a sentence a person can hold."""
    import math
    return "–" if loss is None else f"{math.exp(min(loss, 20)):,.2f}"


def spark(values, width: int = 48) -> str:
    """A one-line loss curve in block characters.

    Not a chart — the portal has real ones. This is for the terminal and for the top of a
    file, where the *shape* (still falling / flat for the last third / a spike at 12k) is
    the whole message and takes one line to deliver.
    """
    pts = [v for v in values if v is not None and v == v]      # v == v drops NaN
    if len(pts) < 2:
        return ""
    if len(pts) > width:
        stride = len(pts) / width
        pts = [pts[min(int(i * stride), len(pts) - 1)] for i in range(width)]
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-12:
        return _SPARK[0] * len(pts)
    return "".join(_SPARK[min(int((v - lo) / (hi - lo) * len(_SPARK)), len(_SPARK) - 1)]
                   for v in pts)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


# --------------------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------------------

def find_log(out_dir: Path, log: str | None = None) -> tuple[Path | None, str | None]:
    """The log to report on, and what kind of run wrote it."""
    out_dir = Path(out_dir)
    if log:
        path = out_dir / log
        return (path, LOGS.get(path.name, "training")) if path.exists() else (None, None)
    for name, kind in LOGS.items():
        if (out_dir / name).exists():
            return out_dir / name, kind
    return None, None


def _config(root: Path, run: str | None) -> dict:
    """The run's YAML, read with the project's own loader so the defaults and derived values
    are the ones the trainer would have used. A run without a config (every SFT and DPO run,
    which are driven by flags) simply has no config block."""
    if not run:
        return {}
    path = root / "configs" / f"{run}.yaml"
    if not path.exists():
        return {}
    try:
        from ..config import load_config
        cfg = load_config(str(path))
    except Exception as exc:                 # a half-edited YAML must not lose the report
        return {"error": f"{type(exc).__name__}: {exc}"}
    m, t, o = cfg.model, cfg.train, cfg.optim
    return {
        "path": str(path.relative_to(root)),
        "arch": f"d={m.d_model} L={m.n_layers} H={m.n_heads} KV={m.n_kv_heads} "
                f"ff={m.d_ff} ctx={m.max_seq_len}",
        "vocab_size": m.vocab_size,
        "batch": f"{t.batch_size} x {t.grad_accum} accum x {t.seq_len} ctx",
        "tokens_per_step": t.batch_size * t.grad_accum * t.seq_len,
        "max_steps": t.max_steps,
        "lr": o.lr, "schedule": o.schedule, "warmup": o.warmup_steps,
        "grad_clip": o.grad_clip, "weight_decay": o.weight_decay,
        "eval_every": t.eval_every, "ckpt_every": t.ckpt_every, "seed": t.seed,
        "sources": [s.get("bin") for s in (cfg.data.train_sources or [])]
                   or [cfg.data.train_bin],
    }


def _energy(root: Path, run: str | None, wall_s: float | None) -> dict | None:
    """What this run drew, from the portal's energy ledger.

    Optional in every direction: no ledger, no portal, no rate configured — each of those
    is a missing row rather than a failure. `coverage` is the honest caveat: the sampler
    only runs while the portal is up, so a run trained from a terminal all weekend can show
    a real figure covering a third of it.
    """
    if not run:
        return None
    try:
        from ..portal.cost import CostConfig, Ledger
    except Exception:
        return None
    ledger = Ledger(root / "logs" / "energy.jsonl")
    rows = [e for e in ledger.entries()
            if e.get("label") == run and (e.get("kind") or "") == "training"]
    if not rows:
        return None
    cfg = CostConfig.load(root)
    watt_hours = sum(r.get("wh") or 0.0 for r in rows)
    seconds = sum(r.get("seconds") or 0.0 for r in rows)
    priced = cfg.price(watt_hours, seconds)
    coverage = min(seconds / wall_s, 1.0) if wall_s else None
    return {
        "wh": watt_hours, "seconds": seconds, "coverage": coverage,
        "money": priced.get("money"), "currency": cfg.currency,
        "configured": cfg.configured, "basis": cfg.basis(),
        # Scaling the measured part up to the whole run is the only way to answer "what did
        # this cost", and it lives under `estimated_` because it assumes the unwatched hours
        # looked like the watched ones.
        "estimated_wh": (watt_hours / coverage) if coverage else None,
        "estimated_money": (priced["money"] / coverage
                            if (coverage and priced.get("money") is not None) else None),
    }


def _evals(root: Path, run: str | None) -> list[dict]:
    """Benchmark results recorded for this run, oldest first. Torch is imported by the eval
    package, so this is deliberately behind a try: a report must be producible on a machine
    that cannot load a model."""
    if not run:
        return []
    try:
        from ..eval.report import Results
        rows = Results(root).rows(limit=200, run=run)
    except Exception:
        return []
    rows.sort(key=lambda r: (r.get("step") is None, r.get("step") or 0, r.get("when") or 0))
    return rows


def build(out_dir: str | Path, run: str | None = None, log: str | None = None,
          root: str | Path | None = None) -> dict:
    """Every number the report shows, as plain JSON-friendly data.

    Split from `render` so the portal can serve the numbers, a test can assert on them, and
    the markdown stays a presentation detail rather than the only place a fact exists.
    """
    out_dir = Path(out_dir)
    root = Path(root) if root else repo_root()
    run = run or out_dir.name
    log_path, kind = find_log(out_dir, log)

    data: dict = {
        "run": run,
        "kind": kind or "training",
        "dir": str(out_dir),
        "log": str(log_path.relative_to(root)) if log_path and _under(log_path, root)
               else (str(log_path) if log_path else None),
        "generated": time.time(),
        "config": _config(root, run),
        "checks": [],
    }
    records = runlog.load_records(log_path) if log_path else []
    if not records:
        data["empty"] = True
        data["checks"] = [{"level": "warn",
                           "text": f"No training log in {out_dir} — nothing to report on yet."}]
        return data

    steps = [r for r in records if "step" in r and "loss" in r]
    vals = [r for r in records if "val_loss" in r]
    last = runlog.latest(records)
    sessions = runlog.summarise_sessions(runlog.split_sessions(records))
    starts = [r for r in records if r.get("event") == "session_start"]

    cfg = data["config"]
    max_steps = last.get("max_steps") or cfg.get("max_steps")
    reached = last.get("trained_to")
    tokens_per_step = last.get("tokens_per_step") or cfg.get("tokens_per_step")
    wall = sum(s["wall_s"] for s in sessions if s.get("wall_s")) or None
    first_t = next((r.get("time") for r in records if r.get("time")), None)
    last_t = next((r.get("time") for r in reversed(records) if r.get("time")), None)

    data.update({
        "step": reached,
        "max_steps": max_steps,
        "complete": bool(max_steps and reached is not None and reached + 1 >= max_steps),
        "remaining": (max_steps - reached - 1) if (max_steps and reached is not None) else None,
        "progress": ((reached + 1) / max_steps) if (max_steps and reached is not None) else None,
        "sessions": sessions,
        "n_sessions": len(sessions),
        "wall_s": wall,
        "span_s": (last_t - first_t) if (first_t and last_t) else None,
        "started": first_t,
        "ended": last_t,
        "tokens_per_step": tokens_per_step,
        "tokens": tokens_per_step * (reached + 1) if (tokens_per_step and reached is not None)
                  else None,
        # Parameter counts are recorded by the trainer at session start (older logs predate
        # that, and fall back to whatever an evaluation recorded). Never computed here: a
        # second implementation of "how big is this model" is a second thing to get wrong.
        "params": next((s.get("params") for s in reversed(starts) if s.get("params")), None),
        "params_active": next((s.get("params_active") for s in reversed(starts)
                               if s.get("params_active")), None),
    })

    # ---- what it learned ---------------------------------------------------------------
    best_val = min((v["val_loss"] for v in vals), default=None)
    best_at = next((v["step"] for v in vals if v["val_loss"] == best_val), None) if vals else None
    data["loss"] = {
        "best_val": best_val,
        "best_val_step": best_at,
        "final_val": vals[-1]["val_loss"] if vals else None,
        "first_val": vals[0]["val_loss"] if vals else None,
        "ema_first": steps[0].get("ema") if steps else None,
        "ema_last": steps[-1].get("ema") if steps else None,
        "loss_last": steps[-1].get("loss") if steps else None,
        "n_evals": len(vals),
    }
    data["curve"] = _curve(steps, vals)
    data["spark"] = {
        "train": spark([r.get("ema", r.get("loss")) for r in steps]),
        "val": spark([v["val_loss"] for v in vals]),
    }
    data["extras"] = _extras(steps, vals)

    # ---- how fast it ran ----------------------------------------------------------------
    data["speed"] = {
        "tok_per_sec": _mean(r.get("tok_per_sec") for r in steps),
        "tok_per_sec_best": max((r["tok_per_sec"] for r in steps if r.get("tok_per_sec")),
                                default=None),
        "mfu": _mean(r.get("mfu") for r in steps),
        "s_per_step": _median(r.get("s_per_step") for r in steps),
    }
    data["energy"] = _energy(root, run, wall)
    data["evals"] = _evals(root, run)
    data["moe"] = _moe(steps)
    data["files"] = _files(out_dir)
    data["checks"] = checks(data, steps, vals, sessions)
    return data


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _curve(steps: list[dict], vals: list[dict], points: int = 6) -> list[dict]:
    """The loss at a handful of evenly spaced places, with the validation loss measured
    nearest each one. Six rows say "still falling" or "flat since halfway" at a glance; the
    full series is in the log and on the portal's charts."""
    if not steps:
        return []
    out = []
    for i in range(points):
        rec = steps[min(int(i / (points - 1) * (len(steps) - 1)), len(steps) - 1)]
        step = rec["step"]
        near = min(vals, key=lambda v: abs(v["step"] - step)) if vals else None
        out.append({
            "step": step,
            "train": rec.get("ema", rec.get("loss")),
            "val": near["val_loss"] if near else None,
            "val_step": near["step"] if near else None,
            "elapsed": rec.get("elapsed"),
        })
    # Successive rows can land on the same logged step on a very short run.
    return [r for i, r in enumerate(out) if i == 0 or r["step"] != out[i - 1]["step"]]


def _extras(steps: list[dict], vals: list[dict]) -> list[dict]:
    """First/last/best for the metrics only some trainers write (DPO's accuracy, GRPO's
    reward). Absent keys produce no row, so a pretraining report never mentions them."""
    out = []
    for key, (label, direction) in EXTRAS.items():
        src = vals if key.startswith("val_") else steps
        series = [r[key] for r in src if isinstance(r.get(key), (int, float))]
        if not series:
            continue
        out.append({"key": key, "label": label, "first": series[0], "last": series[-1],
                    "best": (max if direction == "max" else min)(series), "n": len(series)})
    return out


def _moe(steps: list[dict]) -> dict | None:
    """Routing health for a mixture of experts: the balance at the end, the worst it got,
    and any expert that stopped being chosen. Absent for a dense run."""
    routed = [r["moe"] for r in steps if isinstance(r.get("moe"), dict)]
    if not routed:
        return None
    shares = routed[-1].get("shares") or []
    balances = [r.get("balance") for r in routed if r.get("balance") is not None]
    return {
        "experts": len(shares),
        "balance": routed[-1].get("balance"),
        "balance_min": min(balances) if balances else None,
        "dead": routed[-1].get("dead"),
        "dead_ever": max((r.get("dead") or 0 for r in routed), default=0),
        "min_share": routed[-1].get("min_share"),
        "max_share": routed[-1].get("max_share"),
        "shares": shares,
    }


def _files(out_dir: Path) -> list[dict]:
    out = []
    for p in sorted(out_dir.glob("*.pt")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


# --------------------------------------------------------------------------------------
# the part worth the module: what a person should be told without having to look
# --------------------------------------------------------------------------------------

def checks(data: dict, steps: list[dict], vals: list[dict],
           sessions: list[dict]) -> list[dict]:
    """Findings, computed from the log rather than noticed by a human reading it.

    Every one of these is something this project has actually been bitten by or built
    against: a session killed with `-9` leaves no end record, a spike can undo a day, a
    validation loss that stopped moving means the remaining budget is being spent for
    nothing, and a collapsed router looks exactly like a healthy loss curve. Each finding
    says what was measured and what to do, because "warning: grad norm" helps nobody.

    Levels: `warn` (look at this), `note` (worth knowing), `good` (checked, and fine — a
    report with no `good` lines reads as though nothing was verified).
    """
    out: list[dict] = []
    cfg = data.get("config") or {}
    loss = data.get("loss") or {}

    # -- did it finish? -------------------------------------------------------------------
    if data.get("complete"):
        out.append({"level": "good", "text":
                    f"Ran its full budget: {integer(data['max_steps'])} steps."})
    elif data.get("remaining"):
        out.append({"level": "note", "text":
                    f"Stopped {integer(data['remaining'])} steps short of the "
                    f"{integer(data['max_steps'])}-step budget "
                    f"({num(data.get('progress', 0) * 100, 1)}% done). Rerunning the same "
                    f"command resumes at step {integer((data['step'] or 0) + 1)}."})

    # -- non-finite loss ------------------------------------------------------------------
    bad = [r["step"] for r in steps if r.get("loss") is not None and r["loss"] != r["loss"]]
    if bad:
        out.append({"level": "warn", "text":
                    f"The loss was NaN at {len(bad)} logged step(s), first at "
                    f"{integer(bad[0])}. A checkpoint from after that point is not usable — "
                    f"see docs/08-troubleshooting.md."})

    # -- spikes ---------------------------------------------------------------------------
    # Against the EMA at that moment, not a global mean: a spike is a step that is bad
    # *compared with how the run was going*, which early in training is a much larger number
    # than it is at the end.
    spikes = [r for r in steps
              if r.get("ema") and r.get("loss") and r["loss"] > 1.5 * r["ema"]]
    if spikes:
        worst = max(spikes, key=lambda r: r["loss"] / r["ema"])
        out.append({"level": "note" if len(spikes) < 5 else "warn", "text":
                    f"{len(spikes)} loss spike(s) above 1.5x the running average — worst at "
                    f"step {integer(worst['step'])} ({num(worst['loss'], 3)} against "
                    f"{num(worst['ema'], 3)}). Gradient clipping absorbs these; a cluster of "
                    f"them is a data problem."})

    # -- clipping -------------------------------------------------------------------------
    clip = cfg.get("grad_clip")
    norms = [r["grad_norm"] for r in steps if r.get("grad_norm") is not None]
    if clip and norms:
        over = sum(1 for n in norms if n > clip)
        frac = over / len(norms)
        if frac > 0.5:
            out.append({"level": "note", "text":
                        f"The gradient norm was above the clip threshold ({clip}) on "
                        f"{num(frac * 100, 0)}% of logged steps, so the effective learning "
                        f"rate was set by the clip rather than the schedule. Not wrong — "
                        f"worth knowing before reading the LR curve."})

    # -- did the val loss stop moving? ----------------------------------------------------
    best_at, reached = loss.get("best_val_step"), data.get("step")
    if best_at is not None and reached and loss.get("n_evals", 0) >= 4:
        frac = best_at / reached
        if frac < 0.7:
            out.append({"level": "warn", "text":
                        f"The best validation loss was at step {integer(best_at)} — "
                        f"{num(frac * 100, 0)}% of the way in — and nothing since beat it. "
                        f"`ckpt_best.pt` is that step, not the last one. Overfitting, too "
                        f"high an LR floor, or a val set too small to resolve the difference."})
        else:
            out.append({"level": "good", "text":
                        f"Validation loss was still improving near the end (best at step "
                        f"{integer(best_at)} of {integer(reached)}) — the budget was not wasted."})
    elif not vals:
        out.append({"level": "warn", "text":
                    "No validation loss was ever recorded, so there is nothing here that "
                    "says whether the model generalised. Set train.eval_every."})

    # -- sessions that did not end cleanly -------------------------------------------------
    # The last session is excluded: it is either the one that just wrote this report, or a
    # run that is training right now, and neither is a crash.
    crashed = [s for s in sessions[:-1] if s.get("open") and not s.get("unmarked")]
    if crashed:
        out.append({"level": "warn", "text":
                    f"{len(crashed)} session(s) ended without a session_end record — killed "
                    f"with -9, or crashed (#{', #'.join(str(s['index']) for s in crashed)}). "
                    f"Work since that session's last checkpoint was lost and retrained."})

    # -- throughput between sessions --------------------------------------------------------
    # Against the MEDIAN session, not the fastest one. A single session can report an
    # inflated rate — a short one, or one from before the partial-window fix in `pretrain`
    # (gotcha: a resumed run's first log window covers fewer steps than `log_every`) — and
    # comparing against the maximum makes that one session the yardstick for every run after
    # it, which is how a health check becomes noise.
    rates = [s["tok_per_sec"] for s in sessions if s.get("tok_per_sec")]
    if len(rates) >= 3:
        typical, last_r = _median(rates), rates[-1]
        if typical and last_r < 0.8 * typical:
            out.append({"level": "note", "text":
                        f"The last session ran at {last_r / 1000:.1f}k tokens/s against a "
                        f"median session's {typical / 1000:.1f}k — "
                        f"{num((1 - last_r / typical) * 100, 0)}% slower. Something else was "
                        f"using the card, or the run was sharing it with an evaluation."})

    # -- routing ----------------------------------------------------------------------------
    moe = data.get("moe")
    if moe:
        if moe.get("dead_ever"):
            out.append({"level": "warn", "text":
                        f"Up to {moe['dead_ever']} expert(s) were receiving no tokens. That is "
                        f"router collapse, and the loss curve does not show it — the model is "
                        f"quietly a smaller dense one. See docs/14-moe.md."})
        elif moe.get("balance_min") is not None and moe["balance_min"] < 0.6:
            out.append({"level": "note", "text":
                        f"Expert balance fell to {num(moe['balance_min'], 2)} (1.0 is even). "
                        f"Not collapse, but the load-balancing loss is working hard."})
        else:
            out.append({"level": "good", "text":
                        f"The router stayed balanced ({num(moe.get('balance'), 2)} at the end, "
                        f"never below {num(moe.get('balance_min'), 2)}), so every expert trained."})

    # -- energy -----------------------------------------------------------------------------
    energy = data.get("energy")
    if energy and energy.get("coverage") is not None and energy["coverage"] < 0.8:
        out.append({"level": "note", "text":
                    f"The energy figure covers {num(energy['coverage'] * 100, 0)}% of the "
                    f"run — the sampler only records while the portal is up. The whole-run "
                    f"number below is scaled from the measured part, and says so."})

    # -- data reuse ---------------------------------------------------------------------------
    # Tokens seen against the corpus is the difference between "trained on 10B tokens" and
    # "read the same 2B five times", which are not the same run.
    return out


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

_ICON = {"good": "✅", "note": "•", "warn": "⚠️"}


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    return (["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def verdict(d: dict) -> str:
    """The paragraph at the top: what happened, in sentences, before any table.

    Written from the same fields the tables use — someone who reads only this should not be
    misled by it, which is why the completion status and the loudest warning both appear.
    """
    kind, loss = d.get("kind", "training"), d.get("loss") or {}
    bits = []
    if d.get("complete"):
        bits.append(f"This {kind} run **finished its budget** of "
                    f"{integer(d.get('max_steps'))} steps")
    elif d.get("remaining"):
        bits.append(f"This {kind} run **stopped {integer(d['remaining'])} steps short** of "
                    f"its {integer(d.get('max_steps'))}-step budget")
    else:
        bits.append(f"This {kind} run trained {integer(d.get('step'))} steps")
    if d.get("wall_s"):
        span = (f", spread over {fmt_dur(d['span_s'])} of calendar time"
                if d.get("span_s") and d["span_s"] > 1.5 * d["wall_s"] else "")
        bits.append(f"in {fmt_dur(d['wall_s'])} of training across "
                    f"{d.get('n_sessions', 1)} session(s){span}")
    if d.get("tokens"):
        bits.append(f"on {compact(d['tokens'])} tokens")
    line = " ".join(bits).rstrip(".") + "."

    if loss.get("best_val") is not None:
        line += (f" Its best validation loss was **{num(loss['best_val'])}** "
                 f"(perplexity {ppl(loss['best_val'])}) at step "
                 f"{integer(loss.get('best_val_step'))}")
        if loss.get("first_val") is not None and loss["first_val"] > loss["best_val"]:
            line += f", down from {num(loss['first_val'])} at the first evaluation"
        line += "."
    warns = [c for c in d.get("checks", []) if c["level"] == "warn"]
    if warns:
        line += f" **Worth looking at:** {warns[0]['text']}"
    return line


def render(d: dict) -> str:
    """The report as markdown — read in a terminal, in the portal, or in an editor."""
    run, cfg = d.get("run"), d.get("config") or {}
    lines = [f"# {run} — run report", ""]
    if d.get("empty"):
        lines += [f"No training log under `{d.get('dir')}`, so there is nothing to report "
                  f"on yet.", ""]
        return "\n".join(lines)

    lines += [
        f"*Generated {clock(d.get('generated'))} from `{d.get('log')}`. Everything here is "
        f"recomputed from that log — regenerate with "
        f"`python -m aksharallm.train.report {run}`.*", "",
        verdict(d), "",
        "## At a glance", "",
    ]
    loss, speed = d.get("loss") or {}, d.get("speed") or {}
    glance = [
        ["steps", f"{integer(d.get('step'))}"
                  + (f" of {integer(d['max_steps'])}" if d.get("max_steps") else "")
                  + (f" ({num((d.get('progress') or 0) * 100, 1)}%)" if d.get("progress") else "")],
        ["tokens seen", compact(d.get("tokens"))
                        + (f" · {compact(d.get('tokens_per_step'))} per step"
                           if d.get("tokens_per_step") else "")],
        ["parameters", (f"{compact(d['params'])}"
                        + (f" total, {compact(d['params_active'])} active per token"
                           if d.get("params_active") and d["params_active"] != d["params"]
                           else "")) if d.get("params") else "–"],
        ["architecture", cfg.get("arch", "–")],
        ["best val loss", f"{num(loss.get('best_val'))} (ppl {ppl(loss.get('best_val'))})"
                          f" at step {integer(loss.get('best_val_step'))}"
                          if loss.get("best_val") is not None else "– (never evaluated)"],
        ["final train loss", f"{num(loss.get('ema_last'), 3)} ema"
                             f" (ppl {ppl(loss.get('ema_last'))})"
                             if loss.get("ema_last") is not None else "–"],
        ["training time", f"{dur(d.get('wall_s'))} across {d.get('n_sessions')} session(s)"],
        ["started / ended", f"{clock(d.get('started'))} → {clock(d.get('ended'))}"],
        ["throughput", (f"{speed['tok_per_sec'] / 1000:.1f}k tok/s mean"
                        + (f" · MFU {num((speed.get('mfu') or 0) * 100, 1)}%"
                           if speed.get("mfu") else "")
                        + (f" · {num(speed.get('s_per_step'), 2)}s/step"
                           if speed.get("s_per_step") else ""))
                       if speed.get("tok_per_sec") else "–"],
    ]
    energy = d.get("energy")
    if energy:
        cost = ""
        if energy.get("estimated_money") is not None:
            cost = f" · about {energy['currency']}{energy['estimated_money']:,.2f}"
        elif not energy.get("configured"):
            cost = " · no price set (cost: {per_kwh: …} in configs/portal.yaml)"
        glance.append(["energy", f"{wh(energy.get('estimated_wh') or energy.get('wh'))}"
                                 f"{cost} · measured over "
                                 f"{num((energy.get('coverage') or 0) * 100, 0)}% of the run"])
    lines += _table(["", ""], glance) + [""]

    # ---- the curve -----------------------------------------------------------------------
    lines += ["## What it learned", ""]
    sp = d.get("spark") or {}
    if sp.get("train"):
        lines += [f"    train  `{sp['train']}`  {num(loss.get('ema_first'), 3)} → "
                  f"{num(loss.get('ema_last'), 3)}"]
    if sp.get("val"):
        lines += [f"    val    `{sp['val']}`  {num(loss.get('first_val'), 3)} → "
                  f"{num(loss.get('final_val'), 3)}"]
    lines += [""]
    curve = d.get("curve") or []
    # "session up", not "elapsed": the trainer's `elapsed` is uptime within the session that
    # logged the line, so on a run trained over evenings it goes back to zero every night.
    lines += _table(["step", "train (ema)", "perplexity", "val", "session up"],
                    [[integer(c["step"]), num(c["train"], 4), ppl(c["train"]),
                      num(c["val"]), dur(c.get("elapsed"))] for c in curve]) + [""]
    for x in d.get("extras") or []:
        lines += [f"- **{x['label']}**: {num(x['first'], 3)} → {num(x['last'], 3)} "
                  f"(best {num(x['best'], 3)}, over {x['n']} readings)"]
    if d.get("extras"):
        lines += [""]

    # ---- sessions -------------------------------------------------------------------------
    sessions = d.get("sessions") or []
    if sessions:
        lines += ["## Sessions", "",
                  "One row per launch: a run trained over evenings is many processes "
                  "appending to one log.", ""]
        lines += _table(["#", "started", "steps", "loss (ema)", "best val", "tok/s", "wall",
                         "ended"],
                        [[f"#{s['index']}", s.get("started") or "?",
                          "–" if s.get("first_step") is None
                          else f"{integer(s['first_step'])} → {integer(s['last_step'])}",
                          "–" if s.get("ema_first") is None
                          else f"{num(s['ema_first'], 3)} → {num(s['ema_last'], 3)}",
                          num(s.get("best_val")),
                          "–" if not s.get("tok_per_sec") else f"{s['tok_per_sec'] / 1000:.1f}k",
                          dur(s.get("wall_s")),
                          s.get("ended") or ("no end record (killed or crashed)"
                                             if not s.get("unmarked")
                                             else "before session markers")]
                         for s in sessions]) + [""]

    # ---- routing ---------------------------------------------------------------------------
    moe = d.get("moe")
    if moe:
        lines += ["## Expert routing", "",
                  f"{moe['experts']} experts. Balance is 1.0 when every expert gets an equal "
                  f"share and 1/N when one takes everything.", "",
                  f"- balance at the end **{num(moe.get('balance'), 2)}**, lowest "
                  f"{num(moe.get('balance_min'), 2)}",
                  f"- share range {num((moe.get('min_share') or 0) * 100, 1)}% – "
                  f"{num((moe.get('max_share') or 0) * 100, 1)}%",
                  f"- experts receiving nothing: {moe.get('dead') or 0} at the end, "
                  f"{moe.get('dead_ever') or 0} at worst", ""]

    # ---- benchmarks --------------------------------------------------------------------------
    evals = d.get("evals") or []
    if evals:
        suites = sorted({s for row in evals for s in (row.get("scores") or {})})
        lines += ["## Benchmarks", "",
                  f"{len(evals)} evaluation(s) recorded for this run "
                  f"(`python -m aksharallm.eval {run} --suite fast`). Chance is in brackets — "
                  f"a score inside its error bars of chance is not a result.", ""]
        # The chance baseline belongs in the header, next to the number it makes sense of.
        # Perplexity has none, which is why the column is built per suite rather than assumed.
        head = ["step"]
        for s in suites:
            base = next((e["baseline"] for r in evals
                         if (e := (r.get("scores") or {}).get(s)) and e.get("baseline")), None)
            head.append(f"{s} ({base * 100:.0f}%)" if base else s)
        rows = [[integer(row.get("step"))]
                + [score((row.get("scores") or {}).get(s) or {}) for s in suites]
                for row in evals[-6:]]
        lines += _table(head, rows) + [""]

    # ---- checks -------------------------------------------------------------------------------
    lines += ["## Things worth knowing", "",
              "Computed from the log, not noticed by a person reading it.", ""]
    for c in d.get("checks") or []:
        lines += [f"- {_ICON.get(c['level'], '•')} {c['text']}"]
    lines += [""]

    # ---- files and what to do next ---------------------------------------------------------
    files = d.get("files") or []
    if files:
        lines += ["## Files", ""]
        lines += _table(["checkpoint", "size", "written"],
                        [[f"`{f['name']}`", bytes_(f["size"]), clock(f["mtime"])]
                         for f in files]) + [""]
    lines += ["## What to do with it", "", "```bash"]
    if not d.get("complete") and d.get("max_steps"):
        lines += [f"scripts/phase2.sh                      # resumes at step "
                  f"{integer((d.get('step') or 0) + 1)}"]
    lines += [f"python -m aksharallm.eval {run} --suite fast     # perplexity, arc-easy, piqa",
              f"python -m aksharallm.infer.cli checkpoints/{run}/ckpt_best.pt   # talk to it",
              "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------

def write(out_dir: str | Path, run: str | None = None, log: str | None = None,
          root: str | Path | None = None, path: str | Path | None = None) -> Path:
    """Build, render, and write `<out_dir>/report.md`. Returns the path written.

    Overwriting the previous report is deliberate: it is derived data, the log it is built
    from is append-only, and a folder of report-2026-08-05T14-22.md files is a mess nobody
    reads. The JSON beside it is the same data unrendered, for anything that would rather
    not parse markdown.
    """
    out_dir = Path(out_dir)
    data = build(out_dir, run=run, log=log, root=root)
    target = Path(path) if path else out_dir / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(data))
    target.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str))
    return target


def write_quietly(out_dir: str | Path, run: str | None = None, log: str | None = None,
                  echo=print) -> Path | None:
    """`write`, but a failure is a printed shrug rather than an exception.

    This is what the trainers call, on the last line of a run that may have taken six days.
    Nothing about summarising a run is worth risking the run's own clean exit.
    """
    try:
        path = write(out_dir, run=run, log=log)
    except Exception as exc:                                  # noqa: BLE001 - deliberate
        echo(f"[report] could not write the run report: {type(exc).__name__}: {exc}")
        return None
    echo(f"report       {path}  (regenerate: python -m aksharallm.train.report "
         f"{run or Path(out_dir).name})")
    return path


def main(argv: list[str] | None = None) -> int:
    """`python -m aksharallm.train.report <run|dir> [--stdout] [--json]`

    The trainers write this file themselves when they exit. This exists to read a report
    for a run that is still going, to regenerate one after more evaluations have been
    recorded, and to print it in a terminal over ssh.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.train.report",
        description="Summarise a training run: budget, loss, throughput, cost, benchmarks, "
                    "and the things worth knowing about how it went.")
    ap.add_argument("run", help="a run name (checkpoints/<run>) or a path to a run directory")
    ap.add_argument("--log", default=None,
                    help=f"which log to read ({', '.join(LOGS)}); default is the first present")
    ap.add_argument("--stdout", action="store_true", help="print it instead of writing it")
    ap.add_argument("--json", action="store_true", help="print the underlying data as JSON")
    ap.add_argument("--root", default=None, help="repo root (default: this checkout)")
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else repo_root()
    given = Path(args.run)
    out_dir = given if given.is_dir() else root / "checkpoints" / args.run
    run = out_dir.name
    if not out_dir.is_dir():
        print(f"no run directory at {out_dir}")
        return 1

    if args.json:
        print(json.dumps(build(out_dir, run=run, log=args.log, root=root), indent=1,
                         default=str))
        return 0
    if args.stdout:
        print(render(build(out_dir, run=run, log=args.log, root=root)))
        return 0
    path = write(out_dir, run=run, log=args.log, root=root)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
