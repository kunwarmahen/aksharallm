"""Quantization jobs, for the portal.

Like everything else here, this panel is a **view over the CLI, not a second
implementation**: pressing Run shells out to

    python -m aksharallm.quant <ckpt> [flags] --json logs/quant/<id>.json

which is exactly what you would type. So a job started in the browser and one started in a
terminal produce the same artifacts, write the same log, and can be stopped by either.

Why a subprocess rather than a thread
-------------------------------------
Quantizing the 300M model with GPTQ takes minutes and allocates over a gigabyte of Hessians
on the GPU. Running that inside the portal process would make a page that is supposed to
stay responsive share an address space (and a CUDA context) with a heavy job, and a crash
in the job would take the portal with it — including the scheduler that starts training at
22:00. A separate process fails alone.

One job at a time, for the same reason the trainer allows one run at a time: two
simultaneous GPTQ jobs would fight over the card and both would be slower than running them
in sequence.

Sharing the GPU with a training run
-----------------------------------
Quantization wants the GPU and so does the trainer. Unlike the playground — where the model
is small and the fallback is a slow tab — a quantization job can allocate a lot, and the
downside of getting it wrong is the *training run* dying at 3am. So when a run is live the
default device becomes the CPU and the panel says why. `device: cuda` overrides it, which
is the right choice when nothing is training (and the panel says that too).

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..infer.checkpoints import CheckpointStore, InferError
from ..train import stopfile
from .runs import MAX_STOP_SECONDS, RunError, _alive, _cmdline, _read_int, repo_root

#: method -> (needs calibration data, human blurb)
METHODS = {
    "rtn": (False, "round to nearest — no data, seconds, the baseline everything is measured against"),
    "awq": (True, "scale the channels that matter up before rounding, fold the inverse into the previous op"),
    "gptq": (True, "quantize column by column, pushing each error into the columns not yet done"),
    "qat": (False, "fine-tune with the rounding in the loop — needs training data and GPU time"),
}

#: The group sizes worth offering. 128 is included but flagged: it does not divide
#: d_ff=2752 on small-code, so that layer silently regroups to 64 (and says so).
GROUPS = (
    (64, "64 — divides every layer in both configs"),
    (128, "128 — regroups to 64 on the SwiGLU down-projection (d_ff=2752)"),
    (-1, "per-channel — one scale per row, smallest and worst"),
)

_JOB_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_.-]+$")


class QuantJobs:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()
        self.store = CheckpointStore(self.root)

    # ---- paths ------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.root / "logs" / "quant"

    @property
    def pid_file(self) -> Path:
        return self.dir / "quant.pid"

    @property
    def current_file(self) -> Path:
        return self.dir / "current.json"

    def log_path(self, job: str) -> Path:
        return self.dir / f"{job}.log"

    def json_path(self, job: str) -> Path:
        return self.dir / f"{job}.json"

    @property
    def stop_file(self) -> Path:
        """Where a bounded stop for the running QAT job is queued (see `stop`)."""
        return self.dir / "STOP"

    # ---- what can be quantized --------------------------------------------------------
    def checkpoints(self) -> list[dict]:
        """Float checkpoints, newest first, with any quantized siblings noted.

        Already-quantized checkpoints are listed but not offered as *sources*: quantizing a
        quantized model compounds the error, and the CLI refuses anyway.
        """
        out = []
        for ck in self.store.list():
            quantized = bool(re.search(r"-(rtn|gptq|awq|qat)-int[48]-", ck.name))
            out.append({
                "id": ck.rel, "run": ck.run, "name": ck.name, "rel": ck.rel,
                "size": ck.size, "step": ck.step, "best_val": ck.best_val,
                "params": ck.params, "stage": ck.stage,
                "quantized": quantized, "can_quantize": not quantized,
            })
        return out

    # ---- process state ----------------------------------------------------------------
    def _pid(self) -> int | None:
        pid = _read_int(self.pid_file)
        if pid and _alive(pid) and "aksharallm.quant" in _cmdline(pid):
            return pid
        return None

    def _current(self) -> dict:
        try:
            return json.loads(self.current_file.read_text())
        except (OSError, ValueError):
            return {}

    def training(self) -> list[str]:
        """Runs currently training — quantizing would share the card with them."""
        live = []
        for run in self.store.dirs():
            pid = _read_int(run / "train.pid")
            if pid and _alive(pid):
                live.append(run.name)
        return live

    def plan_device(self, requested: str | None = None) -> dict:
        busy = self.training()
        if requested in ("cuda", "cpu"):
            return {"device": requested, "training": busy, "forced": True,
                    "reason": f"you chose {requested}" + (
                        f"; note {', '.join(busy)} is training" if busy and requested == "cuda" else "")}
        if busy:
            return {"device": "cpu", "training": busy, "forced": False,
                    "reason": f"{', '.join(busy)} is training — quantizing on the CPU so a "
                              "GPTQ job cannot take the run down. Slower, same result."}
        return {"device": "cuda", "training": [], "forced": False,
                "reason": "nothing is training, so the GPU is free"}

    def status(self, tail: int = 200) -> dict:
        pid = self._pid()
        cur = self._current()
        running = pid is not None
        log = []
        if cur.get("job"):
            log = self._tail(self.log_path(cur["job"]), tail)
        if not running and cur and cur.get("state") == "running":
            # The process is gone but nobody wrote the ending: decide from the artifacts.
            cur = {**cur, "state": "done" if self.json_path(cur["job"]).exists() else "failed"}
        return {
            "running": running,
            "pid": pid,
            "current": cur or None,
            "stop": self.stop_request() if running else None,
            "can_bound": self.can_bound(),
            "log": log,
            "device": self.plan_device(),
            "methods": [{"id": k, "needs_calibration": v[0], "blurb": v[1]}
                        for k, v in METHODS.items()],
            "groups": [{"value": g, "label": lbl} for g, lbl in GROUPS],
            "results": self.results(),
        }

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    # ---- results ----------------------------------------------------------------------
    def results(self, limit: int = 40) -> list[dict]:
        """Every finished job's measurements, newest first, flattened into table rows.

        Reads the same `--json` payload the CLI writes, so a job run in a terminal with
        `--json logs/quant/<name>.json` shows up here too.
        """
        rows = []
        if not self.dir.is_dir():
            return rows
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:limit]:
            if path.name == "current.json":
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            entries = data.get("bench") or []
            rows.append({
                "job": path.stem,
                "when": path.stat().st_mtime,
                "checkpoint": data.get("checkpoint"),
                "device": data.get("device"),
                "bench": entries,
                "report": data.get("report"),
                "totals": (data.get("report") or {}).get("totals"),
                "awq": data.get("awq"),
                "qat": data.get("qat"),
                "out": data.get("out"),
            })
        return rows

    # ---- actions ----------------------------------------------------------------------
    def start(self, spec: dict) -> dict:
        if self._pid():
            raise RunError("a quantization job is already running — stop it first.")

        ref = str(spec.get("checkpoint") or "").strip()
        if not ref:
            raise RunError("pick a checkpoint")
        try:
            info = self.store.get(self.store.identify(ref))
        except (InferError, Exception) as exc:  # noqa: BLE001
            raise RunError(f"unknown checkpoint {ref!r}: {exc}")
        if re.search(r"-(rtn|gptq|awq|qat)-int[48]-", info.name):
            raise RunError(
                f"{info.rel} is already quantized. Quantizing it again compounds the "
                "error — start from the float checkpoint.")

        method = str(spec.get("method") or "rtn")
        if method not in METHODS:
            raise RunError(f"unknown method: {method}")
        compare = bool(spec.get("compare"))
        bits = int(spec.get("bits") or 4)
        if bits not in (4, 8):
            raise RunError("bits must be 4 or 8")
        group = int(spec.get("group") if spec.get("group") is not None else 64)
        plan = self.plan_device(spec.get("device"))

        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)  # a leftover would end this job at step 1
        label = "compare" if compare else f"{method}-int{bits}"
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{info.run}-{label}"
        if not _JOB_RE.match(job):
            raise RunError(f"bad job name: {job}")

        cmd = [sys.executable, "-u", "-m", "aksharallm.quant", str(info.path),
               "--device", plan["device"], "--json", str(self.json_path(job))]
        if compare:
            cmd += ["--compare"]
        else:
            cmd += ["--method", method, "--bits", str(bits), "--group", str(group)]
            if spec.get("bench"):
                cmd += ["--bench"]
            if not spec.get("save"):
                cmd += ["--no-save"]
            if method == "qat":
                cmd += ["--qat-steps", str(int(spec.get("qat_steps") or 800)),
                        "--stop-file", str(self.stop_file)]
        if spec.get("backend") in ("torch", "triton", "auto"):
            cmd += ["--backend", str(spec["backend"])]
        if spec.get("calib_seqs"):
            cmd += ["--calib-seqs", str(int(spec["calib_seqs"]))]

        log = self.log_path(job)
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                cmd, cwd=self.root, env={**os.environ}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid_file.write_text(str(proc.pid))
        current = {
            "job": job, "state": "running", "pid": proc.pid,
            "checkpoint": info.rel, "method": "compare" if compare else method,
            "bits": bits, "group": group, "device": plan["device"],
            "device_reason": plan["reason"], "started": time.time(),
            "cmd": " ".join(cmd[2:]),  # drop the interpreter path; keep it readable
        }
        self.current_file.write_text(json.dumps(current))
        return {"ok": True, "action": "start", **current}

    def stop(self, mode: str = "now", steps: int | None = None,
             seconds: int | None = None) -> dict:
        """Stop the running quantization job.

        Only QAT can be *bounded*: it is the one method that is a training loop, so ending
        it at a step or a time leaves a partly-recovered model that exports normally. RTN,
        GPTQ and AWQ are single passes over the weights — there is no useful halfway point
        to stop at, so for those the only mode is `now`, which is a kill and writes nothing.
        """
        if mode not in ("now", "at", "in", "cancel"):
            raise RunError(f"unknown stop mode: {mode!r}")
        pid = self._pid()
        if not pid:
            raise RunError("no quantization job is running.")

        if mode == "cancel":
            if not self.stop_request():
                raise RunError("no stop is queued for this job.")
            self.stop_file.unlink(missing_ok=True)
            return {"ok": True, "action": "stop:cancel", "pid": pid,
                    "note": "queued stop withdrawn; QAT runs to its full step count."}

        if mode in ("at", "in"):
            if not self.can_bound():
                raise RunError(
                    f"{self._current().get('method', 'this method')} is a single pass over "
                    "the weights, not a training loop — there is no step to stop at. Only "
                    "QAT can be bounded; stop this one now, or let it finish.")
            if mode == "at":
                if not steps or steps < 1:
                    raise RunError("a bounded stop needs a positive step number.")
                request = stopfile.StopRequest(step=int(steps))
                note = f"queued: QAT stops at step {steps}, then exports and measures."
            else:
                if not seconds or seconds < 1:
                    raise RunError("a timed stop needs a duration of at least one second.")
                if seconds > MAX_STOP_SECONDS:
                    raise RunError(f"a timed stop is capped at {MAX_STOP_SECONDS // 3600} "
                                   "hours — bound it by steps instead.")
                request = stopfile.StopRequest(deadline=time.time() + int(seconds))
                note = (f"queued: {stopfile.fmt_left(seconds)} more QAT, then export and "
                        "measure.")
            stopfile.write(self.stop_file, request)
            return {"ok": True, "action": f"stop:{mode}", "pid": pid, "note": note}

        os.kill(pid, 15)
        cur = self._current()
        if cur:
            self.current_file.write_text(json.dumps({**cur, "state": "stopped"}))
        return {"ok": True, "action": "stop", "pid": pid,
                "note": "killed; a job stopped this way writes no quantized checkpoint."}

    def can_bound(self) -> bool:
        """Whether a step- or time-bounded stop means anything for the running job."""
        return bool(self._pid()) and self._current().get("method") == "qat"

    def stop_request(self) -> dict | None:
        """The stop queued for the running job, in the same shape the run panel uses."""
        req = stopfile.read(self.stop_file)
        if req is None:
            return None
        return {"target": req.step, "deadline": req.deadline, "now": req.now,
                "label": req.describe()}
