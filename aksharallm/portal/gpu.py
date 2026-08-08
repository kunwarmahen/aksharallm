"""GPU telemetry: what the card is doing, during a run and between runs.

`nvidia-smi` answers "right now". The interesting questions are shaped differently —
*did the GPU actually stay busy all night?*, *is it thermally throttling at hour forty?*,
*what does it draw when nothing is training?* — and those need history. So a small sampler
appends one reading every few seconds to `logs/gpu.jsonl`, and the portal plots it.

Each sample records whether a trainer was alive at that moment, which is what makes the
comparison possible: the chart bands the training periods, and the summary splits every
average into *while training* and *idle*. On a machine you also use for other things, that
split is the difference between "my run is slow" and "something else is on the GPU".

    {"time": 1785080103.4, "run": "small-code",
     "gpus": [{"index": 0, "util": 98, "mem_used": 19140, "temp": 71, "power": 310.5}]}

`run` is a trainer — pretraining, or one of the post-training stages. A portal job (an
evaluation, a quantization, a fine-tune) is tagged `job` instead: both are worth attributing
power to, but only one of them is a training run, and conflating them would band the charts
wrong. Neither field is present when the machine is idle.

Every sample is also folded into `cost.Ledger` as it is written. That is not decoration:
this file is a rolling buffer (see `MAX_BYTES`), so anything that must survive being trimmed
— and a run's total energy bill must — cannot be computed by reading it back later.

No dependency: `nvidia-smi --query-gpu=… --format=csv` is always there if the driver is,
and a machine without it degrades to an honest "no NVIDIA GPU detected" rather than an
exception. (NVML bindings would be cheaper per sample, but they are a package to install,
and one subprocess every five seconds is not a real cost.)

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in. See also docs/08-scaling.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .cost import CostConfig, Ledger, integrate
from .runs import RunStore, _alive, _cmdline, _read_int

#: Sampled every tick. Order matters — it is the `--query-gpu` order.
FIELDS = ["index", "utilization.gpu", "memory.used", "temperature.gpu", "power.draw"]
KEYS = ["index", "util", "mem_used", "temp", "power"]

#: Static per-card facts, read once (they do not change while the machine is up).
STATIC_FIELDS = ["index", "name", "memory.total", "power.limit"]
STATIC_KEYS = ["index", "name", "mem_total", "power_limit"]

SAMPLE_SECONDS = 5.0
#: ~130 bytes a sample, so 8 MB is roughly a week at 5s. When it grows past this the oldest
#: half is dropped: GPU telemetry is a rolling picture, not a permanent record — the
#: training log is the thing that has to survive.
MAX_BYTES = 8 * 1024 * 1024
BYTES_PER_SAMPLE = 160  # generous, for sizing tail reads


def _run_smi(fields: list[str], timeout: float = 5.0) -> list[list[str]] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.strip().splitlines():
        if line.strip():
            rows.append([c.strip() for c in line.split(",")])
    return rows or None


def _num(text: str) -> float | None:
    """nvidia-smi prints `[N/A]` for anything the card doesn't report (power on some
    laptops, fan speed on blower-less cards). That is missing data, not zero."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def devices() -> list[dict]:
    """The cards present, with the facts that don't change. Empty if there is no GPU."""
    rows = _run_smi(STATIC_FIELDS)
    if not rows:
        return []
    out = []
    for row in rows:
        d = dict(zip(STATIC_KEYS, row))
        out.append({"index": int(_num(d["index"]) or 0), "name": d["name"],
                    "mem_total": _num(d["mem_total"]), "power_limit": _num(d["power_limit"])})
    return out


#: The portal's detached jobs, as (label, pid file under `logs/`, what the process's command
#: line must contain). The predicate is the one each job's own `status()` uses: a stale pid
#: file left by a killed job must never bill somebody else's process to it.
JOB_PIDS = (
    ("eval", "eval/eval.pid", "aksharallm.eval"),
    ("quantize", "quant/quant.pid", "aksharallm.quant"),
    ("finetune", "finetune/finetune.pid", "aksharallm.train.sft"),
    # A server is not a bounded job — it sits there for hours holding the weights — but it
    # is unambiguously using the card, and leaving it out put every watt it drew in the
    # *idle* column. Same class of mistake as the SFT stages above, found the same way.
    ("serve", "serve/serve.pid", "aksharallm.serve"),
)


def activity(store: RunStore) -> tuple[str | None, str | None]:
    """`(run, job)` — what is using the card right now, by name.

    Trainers win over jobs, because the portal will not start a job on the GPU while a run
    is training; if both are somehow alive, the run is the thing holding 21 GB.
    """
    for run in store.runs():
        if store.trainer_pid(run):
            return run, None

    # Post-training stages (`<base>-sft`, `-dpo`, `-grpo`) have no `configs/<name>.yaml` and
    # write `sft_log.jsonl` rather than `train_log.jsonl`, so `store.runs()` has never heard
    # of them — but they are trainers, they hold the whole card for hours, and being unknown
    # here is what made an SFT run's power land in the *idle* column.
    ckpt = store.root / "checkpoints"
    try:
        dirs = sorted(p for p in ckpt.iterdir() if p.is_dir()) if ckpt.is_dir() else []
    except OSError:
        dirs = []
    for d in dirs:
        pid = _read_int(d / "train.pid")
        if pid and _alive(pid):
            args = _cmdline(pid)
            if "aksharallm.train." in args and "aksharallm_smoke" not in args:
                return d.name, None

    for label, rel, needle in JOB_PIDS:
        pid = _read_int(store.root / "logs" / rel)
        if pid and _alive(pid) and needle in _cmdline(pid):
            return None, label
    return None, None


def sample(store: RunStore | None = None) -> dict | None:
    """One reading of every card, tagged with whatever was running at that moment."""
    rows = _run_smi(FIELDS)
    if not rows:
        return None
    gpus = []
    for row in rows:
        d = dict(zip(KEYS, row))
        gpus.append({"index": int(_num(d["index"]) or 0),
                     "util": _num(d["util"]), "mem_used": _num(d["mem_used"]),
                     "temp": _num(d["temp"]), "power": _num(d["power"])})
    rec: dict = {"time": round(time.time(), 1), "gpus": gpus}
    if store is not None:
        run, job = activity(store)
        if run:
            rec["run"] = run
        elif job:
            rec["job"] = job
    return rec


class Sampler:
    """The polling loop. One per machine, like the scheduler — see `lock()`."""

    def __init__(self, store: RunStore, interval: float = SAMPLE_SECONDS):
        self.store = store
        self.interval = interval
        self.path = store.root / "logs" / "gpu.jsonl"
        self.pid_path = store.root / "logs" / "gpu.pid"
        # The permanent half of the record. `logs/gpu.jsonl` is trimmed; this is not.
        self.ledger = Ledger(store.root / "logs" / "energy.jsonl", interval=interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._devices: list[dict] | None = None

    # ---- one sampler at a time ---------------------------------------------------------
    def holder(self) -> int | None:
        pid = _read_int(self.pid_path)
        if pid and pid != os.getpid() and _alive(pid) and "aksharallm" in _cmdline(pid):
            return pid
        return None

    def lock(self) -> bool:
        if self.holder():
            return False
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(f"{os.getpid()}\n")
        return True

    def release(self) -> None:
        if _read_int(self.pid_path) == os.getpid():
            self.pid_path.unlink(missing_ok=True)

    # ---- running -----------------------------------------------------------------------
    def devices(self) -> list[dict]:
        if self._devices is None:
            self._devices = devices()
        return self._devices

    def tick(self) -> dict | None:
        rec = sample(self.store)
        if rec is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        # Fold *before* anything can trim the file this sample was written to.
        self.ledger.fold(rec)
        return rec

    def trim(self) -> None:
        """Drop the oldest half once the file passes MAX_BYTES."""
        try:
            if self.path.stat().st_size <= MAX_BYTES:
                return
            with open(self.path, "rb") as fh:
                fh.seek(-MAX_BYTES // 2, os.SEEK_END)
                keep = fh.read().split(b"\n", 1)[-1]  # drop the partial first line
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(keep)
            tmp.replace(self.path)
        except OSError:
            pass

    def run_forever(self) -> None:
        last_trim = 0.0
        try:
            while not self._stop.is_set():
                self.tick()
                if time.time() - last_trim > 3600:
                    self.trim()
                    last_trim = time.time()
                self._stop.wait(self.interval)
        finally:
            self.ledger.close()     # a clean shutdown loses no energy
            self.release()

    def start(self) -> bool:
        if not self.devices():
            return False          # no GPU: nothing to sample, and say so honestly
        if not self.lock():
            return False
        self._thread = threading.Thread(target=self.run_forever, name="gpu-sampler",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)


# ---- reading it back ---------------------------------------------------------------------

def read_records(path: Path, window_s: float | None = 3600,
                 interval: float = SAMPLE_SECONDS) -> list[dict]:
    """Samples from the last `window_s` seconds (None = the whole file).

    Reads only the tail it needs. A week of samples is megabytes, and the page asks for
    this every couple of seconds — re-parsing all of it to draw the last hour would be the
    one genuinely wasteful thing in the portal.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    with open(path, "rb") as fh:
        if window_s is not None:
            want = int(window_s / max(interval, 0.5) * BYTES_PER_SAMPLE) + 4096
            if want < size:
                fh.seek(size - want)
                fh.readline()  # discard the partial line
        text = fh.read().decode(errors="replace")

    cutoff = time.time() - window_s if window_s is not None else None
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff is None or rec.get("time", 0) >= cutoff:
            out.append(rec)
    return out


def _pick(rec: dict, index: int) -> dict | None:
    for g in rec.get("gpus", []):
        if g.get("index") == index:
            return g
    return None


def series(records: list[dict], index: int = 0, max_points: int = 900) -> dict:
    """Columnar series for one card, bucket-averaged down to `max_points`.

    Averaged rather than strided: at a 24-hour window one point covers ~two minutes, and
    the mean of those is the honest summary. Striding would show whichever instant happened
    to land on the stride and make a steady 70% look ragged.
    """
    rows = [(r["time"], g) for r in records if (g := _pick(r, index))]
    if not rows:
        return {k: [] for k in ("time", "util", "mem_used", "temp", "power")}

    if len(rows) > max_points:
        bucket = len(rows) / max_points
        merged = []
        for b in range(max_points):
            chunk = rows[int(b * bucket):int((b + 1) * bucket)] or None
            if not chunk:
                continue
            avg = {"time": sum(t for t, _ in chunk) / len(chunk)}
            for key in ("util", "mem_used", "temp", "power"):
                vals = [g[key] for _, g in chunk if g.get(key) is not None]
                avg[key] = sum(vals) / len(vals) if vals else None
            merged.append(avg)
        return {k: [m[k] for m in merged] for k in
                ("time", "util", "mem_used", "temp", "power")}

    return {"time": [t for t, _ in rows],
            **{k: [g.get(k) for _, g in rows] for k in ("util", "mem_used", "temp", "power")}}


def training_spans(records: list[dict], gap: float = SAMPLE_SECONDS * 4) -> list[dict]:
    """Contiguous stretches during which a trainer was alive.

    A gap longer than a few sample intervals ends the span rather than bridging it —
    otherwise a portal restart would draw one continuous band across the hours it wasn't
    watching, which is exactly the kind of quiet lie a monitoring page must not tell.
    """
    spans: list[dict] = []
    for rec in records:
        run, t = rec.get("run"), rec.get("time")
        if not run or t is None:
            continue
        if spans and spans[-1]["run"] == run and t - spans[-1]["end"] <= gap:
            spans[-1]["end"] = t
        else:
            spans.append({"run": run, "start": t, "end": t})
    return spans


def summarise(records: list[dict], index: int = 0) -> dict:
    """Averages and peaks, split into *while training* and *idle*.

    This is the comparison the charts are for, as numbers: 98% and 310W under load against
    2% and 25W at rest tells you the card is working; 40% under load tells you something
    else is wrong.
    """
    groups: dict[str, list[dict]] = {"training": [], "idle": []}
    for rec in records:
        g = _pick(rec, index)
        if g:
            groups["training" if rec.get("run") else "idle"].append(g)

    def stats(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        out: dict = {"samples": len(rows), "seconds": len(rows) * SAMPLE_SECONDS}
        for key in ("util", "mem_used", "temp", "power"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            out[key] = sum(vals) / len(vals) if vals else None
            out[f"{key}_max"] = max(vals) if vals else None
        return out

    return {"training": stats(groups["training"]), "idle": stats(groups["idle"])}


def snapshot(store: RunStore, window_s: float | None = 3600, index: int = 0,
             max_points: int = 900, sampler: Sampler | None = None,
             cost: CostConfig | None = None) -> dict:
    """Everything the GPU panel shows, in one call.

    `cost` is optional: with a rate configured, the window also carries what it cost. It is
    priced here rather than in the browser so there is exactly one implementation of the
    host-watts-and-PSU arithmetic, and the panel and the cost totals can never disagree.
    """
    sampler = sampler or Sampler(store)
    devs = sampler.devices()
    if not devs:
        return {"available": False, "devices": [],
                "reason": "no NVIDIA GPU detected (nvidia-smi is missing or reports none)"}
    records = read_records(sampler.path, window_s)
    latest = records[-1] if records else None
    holder = sampler.holder()
    energy = integrate(records, index, sampler.interval)
    if cost is not None:
        energy["price"] = cost.refresh().price(energy["wh"], energy["seconds"])
        energy["currency"] = cost.currency
    return {
        "available": True,
        "devices": devs,
        "index": index,
        "window_s": window_s,
        "sampling": bool(holder or sampler._thread),
        "sampler_pid": holder or (os.getpid() if sampler._thread else None),
        "interval_s": sampler.interval,
        # `current` is the newest sample rather than a fresh nvidia-smi call, so the tiles
        # and the right-hand end of the chart can never disagree.
        "current": _pick(latest, index) if latest else None,
        "current_age_s": (time.time() - latest["time"]) if latest else None,
        "current_run": latest.get("run") if latest else None,
        "current_job": latest.get("job") if latest else None,
        # Energy for exactly the window the charts above are drawing.
        "energy": energy,
        "series": series(records, index, max_points),
        "spans": training_spans(records),
        "summary": summarise(records, index),
        "samples": len(records),
    }
