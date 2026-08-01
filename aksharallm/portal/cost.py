"""What a run cost to produce, in money.

The GPU panel answers "how hard is the card working?". This module answers the question a
person asks after the third evening session: *what has this model cost me so far, and which
run spent it?* Electricity is the real bill on a machine you own, and the sampler already
records the only quantity it needs — `power.draw`, every five seconds, tagged with whatever
was running at that moment. Energy is the integral of that; money is energy times a rate you
supply in `configs/portal.yaml`.

Three pieces, in the order a watt travels through them:

* **:class:`Ledger`** — the durable half. `logs/gpu.jsonl` is a *rolling* buffer (8 MB, oldest
  half dropped), which is right for charts and catastrophic for a total: a three-week run
  would quietly get cheaper as its early samples were deleted. So every sample is folded, as
  it arrives, into a ten-minute bucket in `logs/energy.jsonl` — append-only, ~144 short lines
  a day, never trimmed. Trimming the telemetry can no longer lose money.
* **:func:`integrate`** — the arithmetic, on raw samples. Trapezoidal, gap-aware, and it
  refuses to bridge a hole: if the portal was down for an hour, that hour has no energy
  reading and is reported as uncovered rather than estimated from the sample either side.
* **:class:`CostConfig`** — the rate. Two independent ones, added together if you set both:
  `per_kwh` bills the electricity you actually pay for, and `per_hour` prices the same hours
  as if you had rented the card, which is the number to put next to a cloud quote.

**What the meter would say.** `nvidia-smi` measures the card, not the wall. The rest of the
machine (CPU, drives, fans, the monitor you left on) draws its own 60-120 W, and the PSU
wastes ~10% of everything on the way in. `host_watts` and `psu_efficiency` exist so the
total can be reconciled against a plug meter; both default to "off", and when they are off
the report says *card only* rather than pretending otherwise.

**Attribution** is whatever the sampler tagged: a training run (pretrain, SFT, DPO, GRPO), a
portal job (eval, quantize, fine-tune), or idle. An interval takes the label of the sample
that *starts* it, so a transition can misplace at most one interval — five seconds.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

#: Ledger resolution. Ten minutes is far finer than any question about money and keeps the
#: file at ~15 KB a day; a whole year of it is still smaller than one training log.
BUCKET_SECONDS = 600.0

#: Fallback sampling interval, for a hand-made list of records. Real callers pass the
#: sampler's own interval — this module deliberately imports nothing from `gpu`, because
#: `gpu` imports *it* (the sampler folds every tick into the ledger).
SAMPLE_SECONDS = 5.0

#: A gap wider than this many intervals is a hole in the record, not a long sample. Same
#: rule and the same reason as `gpu.training_spans`: bridging it would invent energy for
#: hours nothing was watching.
GAP_INTERVALS = 4

WH_PER_KWH = 1000.0


def _pick(rec: dict, index: int) -> dict | None:
    for g in rec.get("gpus", ()):
        if g.get("index") == index:
            return g
    return None


def label_of(rec: dict) -> tuple[str | None, str]:
    """(label, kind) for one sample: a training run, a portal job, or idle.

    `run` and `job` are separate fields on purpose — a trainer and an eval job are both
    worth billing, but only one of them is training, and the charts band the other one
    differently.
    """
    if rec.get("run"):
        return str(rec["run"]), "training"
    if rec.get("job"):
        return str(rec["job"]), "job"
    return None, "idle"


def integrate(records: list[dict], index: int = 0, interval: float = SAMPLE_SECONDS,
              t0: float | None = None, t1: float | None = None) -> dict:
    """Watt-hours from raw samples, grouped by what was running.

    Trapezoidal — the energy between two readings is their mean power times the time between
    them, not either endpoint held flat. At five-second spacing the difference is invisible;
    on a card that ramps 30 W → 350 W in one interval it is the difference between the two
    honest answers and one wrong one.

    Missing power (`nvidia-smi` printed `[N/A]`) contributes no energy *and* no covered time,
    so a card that never reports watts produces zeros and a coverage of nothing — rather than
    a confident zero-watt bill.
    """
    gap = max(interval, 0.5) * GAP_INTERVALS
    groups: dict[tuple[str | None, str], dict] = {}
    covered = uncovered = 0.0
    prev: tuple[float, float | None, str | None, str] | None = None

    for rec in records:
        t = rec.get("time")
        g = _pick(rec, index)
        if t is None or g is None:
            continue
        if t0 is not None and t < t0:
            prev = None                     # do not integrate across the window edge
            continue
        if t1 is not None and t > t1:
            break
        power = g.get("power")
        label, kind = label_of(rec)
        if prev is not None:
            dt = t - prev[0]
            if 0 < dt <= gap and power is not None and prev[1] is not None:
                key = (prev[2], prev[3])
                acc = groups.setdefault(key, {"label": key[0], "kind": key[1], "wh": 0.0,
                                              "seconds": 0.0, "samples": 0,
                                              "first": prev[0], "last": t})
                acc["wh"] += (power + prev[1]) / 2 * dt / 3600.0
                acc["seconds"] += dt
                acc["samples"] += 1
                acc["last"] = t
                covered += dt
            elif dt > 0:
                uncovered += dt
        prev = (t, power, label, kind)

    return {"entries": sorted(groups.values(), key=lambda e: -e["wh"]),
            "wh": sum(e["wh"] for e in groups.values()),
            "seconds": covered,
            "uncovered_s": uncovered}


# --------------------------------------------------------------------------------------
# the durable ledger
# --------------------------------------------------------------------------------------

class Ledger:
    """Energy that survives the telemetry being trimmed.

    One record per (ten-minute bucket, label), appended when the bucket closes:

        {"start": 1785080400.0, "seconds": 597.4, "label": "small-code", "kind": "training",
         "index": 0, "wh": 51.83, "samples": 119}

    The open bucket lives in memory and is written on close, so a `kill -9` of the portal
    costs at most ten minutes of *durable* record — and those ten minutes are still in
    `logs/gpu.jsonl`, which is where the live view reads them from anyway.
    """

    def __init__(self, path: Path | str, interval: float = SAMPLE_SECONDS,
                 bucket: float = BUCKET_SECONDS, index: int = 0):
        self.path = Path(path)
        self.interval = interval
        self.bucket = bucket
        self.index = index
        self._open: dict | None = None
        self._prev: tuple[float, float | None, str | None, str] | None = None
        self._cache: tuple[float, int, list[dict]] | None = None

    # ---- writing ---------------------------------------------------------------------
    def fold(self, rec: dict) -> None:
        """Add one sample. Called by the sampler on every tick; cheap and allocation-free
        in the common case (same bucket, same label, one float add)."""
        t = rec.get("time")
        g = _pick(rec, self.index)
        if t is None or g is None:
            return
        power = g.get("power")
        label, kind = label_of(rec)
        prev = self._prev
        self._prev = (t, power, label, kind)
        if prev is None:
            return
        dt = t - prev[0]
        if not (0 < dt <= max(self.interval, 0.5) * GAP_INTERVALS):
            self._close_open()          # a hole: never let it be spanned by one bucket
            return
        if power is None or prev[1] is None:
            return
        start = (prev[0] // self.bucket) * self.bucket
        cur = self._open
        if cur is None or cur["start"] != start or cur["label"] != prev[2]:
            self._close_open()
            cur = self._open = {"start": start, "seconds": 0.0, "label": prev[2],
                                "kind": prev[3], "index": self.index, "wh": 0.0,
                                "samples": 0}
        cur["wh"] += (power + prev[1]) / 2 * dt / 3600.0
        cur["seconds"] += dt
        cur["samples"] += 1

    def _emit(self, rec: dict) -> None:
        """Append one closed bucket. Replaceable per-instance, which is how `backfill`
        collects buckets instead of writing them."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass                        # a full disk must not kill the sampler thread

    def _close_open(self) -> None:
        if not self._open:
            return
        rec, self._open = self._open, None
        rec["wh"] = round(rec["wh"], 4)
        rec["seconds"] = round(rec["seconds"], 1)
        self._emit(rec)

    def backfill(self, records: list[dict]) -> dict:
        """Fold samples that were written before there was a ledger.

        `logs/gpu.jsonl` holds up to a week of history, and on the day this module was added
        that week was real, measured, and about to be trimmed away unrecorded. A bucket
        already in the ledger is skipped rather than added, so running this twice — or
        running it on a file the sampler is still appending to — cannot double-bill.
        """
        known = {(e.get("start"), e.get("label")) for e in self.entries(include_open=False)}
        scratch = Ledger(self.path, self.interval, self.bucket, self.index)
        collected: list[dict] = []
        scratch._emit = collected.append          # type: ignore[method-assign]
        for rec in records:
            scratch.fold(rec)
        scratch.close()
        fresh = [r for r in collected if (r["start"], r["label"]) not in known]
        for rec in fresh:
            self._emit(rec)
        self._cache = None
        return {"buckets": len(collected), "added": len(fresh),
                "skipped": len(collected) - len(fresh),
                "wh": sum(r["wh"] for r in fresh),
                "seconds": sum(r["seconds"] for r in fresh)}

    def close(self) -> None:
        """Flush the open bucket — called when the sampler stops, so a clean shutdown
        loses nothing."""
        self._close_open()

    # ---- reading ---------------------------------------------------------------------
    def entries(self, include_open: bool = True) -> list[dict]:
        """Every bucket, in the order written (which is chronological, except for whatever
        `backfill` appended). Cached against the file's mtime and size: the panel asks for
        this on every poll and the file only grows by one line per bucket."""
        out: list[dict] = []
        try:
            st = self.path.stat()
        except OSError:
            st = None
        if st is not None:
            if self._cache and self._cache[0] == st.st_mtime and self._cache[1] == st.st_size:
                out = self._cache[2]
            else:
                out = []
                try:
                    for line in self.path.read_text(errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue      # a torn last line is one bucket, not a failure
                        if isinstance(rec, dict) and rec.get("start") is not None:
                            out.append(rec)
                except OSError:
                    out = []
                self._cache = (st.st_mtime, st.st_size, out)
        if include_open and self._open:
            out = out + [dict(self._open)]
        return out


# --------------------------------------------------------------------------------------
# the rate
# --------------------------------------------------------------------------------------

@dataclass
class CostConfig:
    """What a kilowatt-hour costs you, and what else to count.

    Read from `configs/portal.yaml` (`cost:`), overridable from the environment for one
    session (`AKSHARALLM_COST_PER_KWH=8.0`). Both rates default to unset, and an unset rate
    is reported as unset: the panel then shows energy and hours and says what to fill in,
    which is the honest answer to "what did this cost" when nobody has said what power costs.
    """

    SECTION = "cost"
    ENV_PREFIX = "AKSHARALLM_COST"

    #: Prefix, not a currency code — it is only ever concatenated in front of a number.
    currency: str = "$"
    #: What you pay for a kilowatt-hour, in `currency`.
    per_kwh: float | None = None
    #: What an hour of this machine would cost rented, for the cloud comparison. Added to
    #: the electricity if both are set — they answer different questions and neither is a
    #: correction of the other.
    per_hour: float | None = None
    #: Everything the card is not: CPU, drives, fans, the monitor. Applied to covered time,
    #: including idle time.
    host_watts: float = 0.0
    #: Wall-plug efficiency of the PSU (0.9 = 10% is lost as heat before the card sees it).
    psu_efficiency: float = 1.0
    note: str | None = None
    path: Path | None = field(default=None, repr=False)
    _mtime: float = field(default=0.0, repr=False)

    @classmethod
    def load(cls, root: Path | None = None) -> "CostConfig":
        from .runs import repo_root
        root = Path(root).resolve() if root else repo_root()
        cfg = cls(path=root / "configs" / "portal.yaml")
        cfg.reload()
        return cfg

    def refresh(self) -> "CostConfig":
        """Re-read if the file changed, so editing the rate does not need a restart."""
        try:
            if self.path and self.path.stat().st_mtime != self._mtime:
                self.reload()
        except OSError:
            pass
        return self

    def reload(self) -> "CostConfig":
        data: dict = {}
        self.note = None
        if self.path and self.path.is_file():
            try:
                self._mtime = self.path.stat().st_mtime
                loaded = yaml.safe_load(self.path.read_text()) or {}
                data = (loaded.get(self.SECTION) or {}) if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError) as exc:
                self.note = f"{self.path.name} could not be read ({exc}); no rate applied."
                data = {}
        if data.get("currency") is not None:
            self.currency = str(data["currency"])
        for key in ("per_kwh", "per_hour", "host_watts", "psu_efficiency"):
            if data.get(key) is not None:
                try:
                    setattr(self, key, float(data[key]))
                except (TypeError, ValueError):
                    self.note = f"cost.{key} is not a number; ignored."
        for key in ("PER_KWH", "PER_HOUR", "HOST_WATTS", "PSU_EFFICIENCY"):
            env = os.environ.get(f"{self.ENV_PREFIX}_{key}")
            if env:
                try:
                    setattr(self, key.lower(), float(env))
                except ValueError:
                    pass
        if os.environ.get(f"{self.ENV_PREFIX}_CURRENCY"):
            self.currency = os.environ[f"{self.ENV_PREFIX}_CURRENCY"]
        # A zero or negative efficiency would divide the bill into nonsense (or by zero).
        if not (0 < self.psu_efficiency <= 1):
            self.note = (f"cost.psu_efficiency must be >0 and <=1, got "
                         f"{self.psu_efficiency}; using 1.0 (card only).")
            self.psu_efficiency = 1.0
        if self.host_watts < 0:
            self.host_watts = 0.0
        return self

    @property
    def configured(self) -> bool:
        return self.per_kwh is not None or self.per_hour is not None

    def basis(self) -> str:
        """One line saying what the number does and does not include — printed next to
        every total, because "what did it cost" has a different answer per assumption."""
        parts = ["GPU card"]
        if self.host_watts:
            parts.append(f"+{self.host_watts:.0f} W for the rest of the machine")
        if self.psu_efficiency < 1:
            parts.append(f"at {self.psu_efficiency * 100:.0f}% PSU efficiency")
        return " ".join(parts) if len(parts) > 1 else "GPU card only — the wall socket draws more"

    def price(self, gpu_wh: float, seconds: float) -> dict:
        """Energy and money for one bundle of measured time.

        `energy` is what the meter would show; `rental` is the same hours priced as if the
        card were rented. `money` is their sum, or None when no rate is set — never 0.0,
        which would read as "this was free".
        """
        hours = max(seconds, 0.0) / 3600.0
        host_wh = self.host_watts * hours
        wall_wh = (max(gpu_wh, 0.0) + host_wh) / self.psu_efficiency
        energy = (wall_wh / WH_PER_KWH) * self.per_kwh if self.per_kwh is not None else None
        rental = hours * self.per_hour if self.per_hour is not None else None
        money = None if energy is None and rental is None else (energy or 0.0) + (rental or 0.0)
        return {"gpu_wh": gpu_wh, "host_wh": host_wh, "wall_wh": wall_wh, "hours": hours,
                "energy": energy, "rental": rental, "money": money}


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------

def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _midnight(now: float) -> float:
    d = datetime.fromtimestamp(now)
    return datetime(d.year, d.month, d.day).timestamp()


def _bundle(cfg: CostConfig, rows: list[dict], **extra) -> dict:
    wh = sum(r.get("wh") or 0.0 for r in rows)
    seconds = sum(r.get("seconds") or 0.0 for r in rows)
    out = cfg.price(wh, seconds)
    out.update({"wh": wh, "seconds": seconds, "buckets": len(rows)}, **extra)
    return out


def report(ledger: Ledger, cfg: CostConfig, store=None, now: float | None = None,
           days: int = 14) -> dict:
    """Everything the cost panel shows: the grand total, each run, and the last fortnight.

    `store` is optional and only used for the two things the ledger cannot know — how many
    tokens a run bought with its energy, and how long that run has actually been alive, which
    is what turns "3.1 kWh" into "3.1 kWh over 88% of the run" and stops a partial record
    from being read as a complete bill.
    """
    now = now or time.time()
    cfg.refresh()
    entries = ledger.entries()

    by_label: dict[tuple[str | None, str], list[dict]] = {}
    by_day: dict[str, list[dict]] = {}
    for e in entries:
        by_label.setdefault((e.get("label"), e.get("kind") or "idle"), []).append(e)
        by_day.setdefault(_day_key(e.get("start") or now), []).append(e)

    def window(since: float) -> dict:
        rows = [e for e in entries if (e.get("start") or 0) + (e.get("seconds") or 0) >= since]
        return _bundle(cfg, rows)

    runs = []
    for (label, kind), rows in by_label.items():
        item = _bundle(cfg, rows, label=label, kind=kind,
                       first=min(r.get("start") or now for r in rows),
                       last=max((r.get("start") or 0) + (r.get("seconds") or 0)
                                for r in rows))
        if store is not None and kind == "training" and label:
            item.update(_run_extras(store, label, item))
        runs.append(item)
    runs.sort(key=lambda r: (-(r["wh"] or 0.0), str(r["label"])))

    day_rows = sorted(by_day.items(), reverse=True)[:days]
    daily = [dict(_bundle(cfg, rows), day=day) for day, rows in reversed(day_rows)]

    return {
        "configured": cfg.configured,
        "currency": cfg.currency,
        "per_kwh": cfg.per_kwh,
        "per_hour": cfg.per_hour,
        "host_watts": cfg.host_watts,
        "psu_efficiency": cfg.psu_efficiency,
        "basis": cfg.basis(),
        "note": cfg.note,
        "hint": None if cfg.configured else
                "no rate set — put `cost: {per_kwh: ...}` in configs/portal.yaml "
                "(or AKSHARALLM_COST_PER_KWH=...) and every number below gains a price.",
        "total": _bundle(cfg, entries),
        "today": window(_midnight(now)),
        "week": window(now - 7 * 86400),
        "runs": runs,
        "daily": daily,
        "since": min((e.get("start") for e in entries if e.get("start")), default=None),
        "bucket_s": ledger.bucket,
        "server_time": now,
    }


def _fmt_money(cfg: CostConfig, v: float | None) -> str:
    if v is None:
        return "–"
    return f"{cfg.currency}{v:,.2f}" if abs(v) >= 0.1 else f"{cfg.currency}{v:.4f}"


def _fmt_wh(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v / 1000:.2f} kWh" if v >= 1000 else f"{v:.0f} Wh"


def main(argv: list[str] | None = None) -> int:
    """`python -m aksharallm.portal.cost [backfill]` — the same numbers the panel shows.

    The portal is the nice way to read this; a terminal is the way to read it over ssh
    while the run it is billing is still going.
    """
    import argparse

    from ..train.runlog import fmt_dur
    from .gpu import read_records
    from .runs import RunStore

    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.portal.cost",
        description="What each run has cost, from the GPU sampler's energy ledger.")
    ap.add_argument("action", nargs="?", default="report", choices=["report", "backfill"],
                    help="report (default), or fold an existing logs/gpu.jsonl into the "
                         "ledger — for samples taken before the ledger existed")
    ap.add_argument("--root", default=None, help="repo root (default: this checkout)")
    ap.add_argument("--days", type=int, default=14, help="how many days to break out")
    args = ap.parse_args(argv)

    store = RunStore(args.root)
    cfg = CostConfig.load(store.root)
    ledger = Ledger(store.root / "logs" / "energy.jsonl")

    if args.action == "backfill":
        samples = read_records(store.root / "logs" / "gpu.jsonl", window_s=None)
        out = ledger.backfill(samples)
        print(f"{len(samples):,} samples → {out['buckets']:,} buckets, "
              f"{out['added']:,} added ({_fmt_wh(out['wh'])}, {fmt_dur(out['seconds'])}), "
              f"{out['skipped']:,} already recorded")
        return 0

    rep = report(ledger, cfg, store=store, days=args.days)
    if not rep["total"]["buckets"]:
        print("Nothing measured yet. The portal's GPU sampler fills the ledger; "
              "`python -m aksharallm.portal.cost backfill` folds in any old telemetry.")
        return 0

    print(f"measures the {rep['basis']}")
    if rep["hint"]:
        print(rep["hint"])
    for key, label in (("total", "all time"), ("week", "last 7 days"), ("today", "today")):
        b = rep[key]
        print(f"  {label:<12} {_fmt_wh(b['wh']):>10}  {fmt_dur(b['seconds']):>8}  "
              f"{_fmt_money(cfg, b['money']):>12}")
    print(f"\n  {'what':<22} {'kind':<9} {'time':>9} {'energy':>10} {'cost':>12} "
          f"{'covered':>8} {'whole run':>11} {'per 1M tok':>11}")
    for r in rep["runs"]:
        cov = "–" if r.get("coverage") is None else f"{r['coverage'] * 100:.0f}%"
        # Without a rate there is still something to say per million tokens: the energy.
        per_m = (_fmt_money(cfg, r["per_mtoken"]) if r.get("per_mtoken") is not None
                 else _fmt_wh(r.get("wh_per_mtoken")))
        est = (_fmt_money(cfg, r["estimated_money"]) if r.get("estimated_money") is not None
               else _fmt_wh(r.get("estimated_wh")))
        print(f"  {(r['label'] or 'idle'):<22} {r['kind']:<9} {fmt_dur(r['seconds']):>9} "
              f"{_fmt_wh(r['wh']):>10} {_fmt_money(cfg, r['money']):>12} {cov:>8} "
              f"{est:>11} {per_m:>11}")
    return 0


def _run_extras(store, run: str, item: dict) -> dict:
    """Tokens bought and record coverage, from the run's own training log.

    Coverage is the honest caveat on every one of these numbers: the sampler only records
    while the portal is up, so a run trained from a terminal all weekend can show a real
    energy figure that covers a fraction of it. Showing the fraction is the difference
    between a number and a lie.
    """
    from ..train import runlog
    out: dict = {}
    try:
        if run not in store.runs():
            return out
        records = runlog.load_records(store.run_dir(run) / "train_log.jsonl")
        last = runlog.latest(records)
        sessions = runlog.summarise_sessions(runlog.split_sessions(records))
    except (OSError, ValueError):
        return out
    wall = sum(s["wall_s"] for s in sessions if s.get("wall_s"))
    step, per_step = last.get("step"), last.get("tokens_per_step")
    tokens = per_step * (step + 1) if (per_step and step is not None) else None
    coverage = min(item["seconds"] / wall, 1.0) if wall else None
    out["wall_s"] = wall or None
    out["coverage"] = coverage
    out["tokens"] = tokens

    # Everything below scales the measured part up to the whole run, and says so by living
    # under `estimated_`. Half a run's energy against *all* of its tokens would halve the
    # cost per token — a number that looks precise and is wrong by exactly the fraction of
    # the run nobody was watching. The assumption is that the unmeasured hours drew power
    # and produced tokens at the same rate as the measured ones, which holds for a steady
    # training run and is stated rather than hidden.
    frac = coverage if (coverage and coverage > 0) else None
    out["estimated_wh"] = (item["wh"] / frac) if frac else None
    out["estimated_money"] = (item["money"] / frac
                              if (frac and item.get("money") is not None) else None)
    measured_tokens = (tokens * frac if frac else tokens) if tokens else None
    if measured_tokens:
        if item.get("money") is not None:
            out["per_mtoken"] = item["money"] / (measured_tokens / 1e6)
        out["wh_per_mtoken"] = item["wh"] / (measured_tokens / 1e6)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
