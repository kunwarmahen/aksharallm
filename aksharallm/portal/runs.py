"""What a training run *is*, from the outside: its state, and the two things you can do to it.

The portal never reimplements starting or stopping. `scripts/phase2.sh` and
`scripts/stop.sh` remain the only things that launch a trainer or ask one to stop; this
module shells out to them exactly as a human would, and reads back the same files they
write (`train.pid`, `STOP`, `run.meta`, `train_log.jsonl`, `logs/<run>/*.log`). That is
what keeps the button and the terminal honest about each other.

State lives on disk, never in this process, so the portal can be restarted, or run twice,
or not run at all, without a training run noticing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ..train import runlog

#: Run names come off the wire and end up in paths and a subprocess argument, so they are
#: whitelisted rather than escaped: letters, digits, dash, underscore, dot, no leading dot.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: How each run is launched. `scripts/phase2.sh` is the Phase-2 launcher and knows two
#: recipes; a run that isn't in here (e.g. `tiny`) is still fully *visible* in the portal,
#: it just has no start button, because there is no script that knows how to build its data.
LAUNCHERS: dict[str, dict[str, str]] = {
    "small-code": {},              # the blended 85/15 general+Python base (the default)
    "small": {"PURE": "1"},        # the non-blended FineWeb-Edu-only fallback
}

PHASE_IDLE = "idle"
PHASE_LAUNCHING = "launching"   # phase2.sh is in pre-flight/data/smoke, no trainer yet
PHASE_TRAINING = "training"
PHASE_STOPPING = "stopping"     # a stop was requested and the trainer is still alive


def repo_root() -> Path:
    """The repo root, inferred from this file's location (aksharallm/portal/runs.py)."""
    return Path(__file__).resolve().parents[2]


class RunError(Exception):
    """A request that cannot be honoured (unknown run, already training, nothing to stop).

    The server turns these into a 4xx with the message, so every refusal the UI shows is
    the same sentence this module would print on a terminal.
    """


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)  # alive, but owned by someone else
    return True


def _cmdline(pid: int) -> str:
    """The process's command line, or "" if it can't be read. Used to make sure a recycled
    pid is not mistaken for our trainer — signalling a stranger would be unforgivable."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace")
    except OSError:
        try:
            return subprocess.run(["ps", "-p", str(pid), "-o", "args="],
                                  capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return ""


def _read_int(path: Path) -> int | None:
    try:
        digits = re.sub(r"\D", "", path.read_text())
        return int(digits) if digits else None
    except (OSError, ValueError):
        return None


class RunStore:
    """Every run under one repo root."""

    #: The pgrep fallback below costs a process spawn. The page polls every couple of
    #: seconds and asks about every run, so the answer is cached for a moment — well under
    #: the time it takes a trainer to start or die, and it keeps a dashboard left open
    #: overnight from forking thousands of times for nothing.
    _PGREP_TTL = 2.0

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or repo_root()).resolve()
        self._pgrep_cache: dict[str, tuple[float, int | None]] = {}

    # ---- paths -------------------------------------------------------------------------
    def run_dir(self, run: str) -> Path:
        return self.root / "checkpoints" / run

    def log_dir(self, run: str) -> Path:
        return self.root / "logs" / run

    def config_path(self, run: str) -> Path:
        return self.root / "configs" / f"{run}.yaml"

    # ---- discovery ---------------------------------------------------------------------
    def runs(self) -> list[str]:
        """Every run the portal knows: one per config, plus any checkpoint dir with a log.

        A checkpoint dir with no config still shows up — a run whose YAML was renamed is
        exactly when you want to read its history, not when you want it to vanish.
        """
        names = {p.stem for p in (self.root / "configs").glob("*.yaml")}
        ckpt = self.root / "checkpoints"
        if ckpt.is_dir():
            names |= {p.name for p in ckpt.iterdir()
                      if p.is_dir() and (p / "train_log.jsonl").exists()}
        return sorted(n for n in names if RUN_NAME_RE.match(n))

    def check(self, run: str) -> str:
        if not RUN_NAME_RE.match(run or ""):
            raise RunError(f"invalid run name: {run!r}")
        if run not in self.runs():
            raise RunError(f"no such run: {run} (known: {', '.join(self.runs()) or 'none'})")
        return run

    # ---- process state -----------------------------------------------------------------
    def trainer_pid(self, run: str) -> int | None:
        """The live trainer for this run, or None.

        Trusts `train.pid` only after confirming the process is both alive and actually a
        trainer; falls back to the command line, the same way `scripts/stop.sh` does, so a
        run started by hand is still adopted by the portal.
        """
        pid = _read_int(self.run_dir(run) / "train.pid")
        if pid and _alive(pid) and "aksharallm.train" in _cmdline(pid):
            return pid

        cached = self._pgrep_cache.get(run)
        if cached and time.time() - cached[0] < self._PGREP_TTL:
            return cached[1] if _alive(cached[1]) else None
        try:
            found = subprocess.run(
                ["pgrep", "-f", f"aksharallm.train.pretrain configs/{run}.yaml"],
                capture_output=True, text=True, timeout=5).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None
        pid = int(found[0]) if found else None
        self._pgrep_cache[run] = (time.time(), pid)
        return pid

    def launcher(self, run: str) -> dict | None:
        """The `phase2.sh` process the portal started, while it is still pre-flighting."""
        path = self.run_dir(run) / "portal_launch.json"
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not _alive(info.get("pid")):
            return None
        return info

    def stop_request(self, run: str) -> dict | None:
        """The pending stop, read out of the STOP file the trainer polls.

        Empty file == stop after the current step. A number == stop on reaching that step.
        (`pretrain.stop_file_target` is the trainer's half of this contract.)
        """
        path = self.run_dir(run) / "STOP"
        if not path.exists():
            return None
        target = _read_int(path)
        return {"target": target, "now": target is None}

    # ---- status ------------------------------------------------------------------------
    def status(self, run: str, max_points: int = 2000) -> dict:
        """Everything the dashboard shows for one run, in one read of the disk."""
        self.check(run)
        rdir, ldir = self.run_dir(run), self.log_dir(run)
        records = runlog.load_records(rdir / "train_log.jsonl")
        last = runlog.latest(records)
        sessions = runlog.summarise_sessions(runlog.split_sessions(records))

        pid = self.trainer_pid(run)
        launcher = self.launcher(run)
        stop = self.stop_request(run)
        if pid:
            # A queued stop at a step we haven't reached yet is not "stopping" — the run is
            # training normally and merely has a finish line. Only an imminent stop is.
            imminent = stop and (stop["now"] or (last["step"] is not None
                                                 and stop["target"] <= last["step"]))
            phase = PHASE_STOPPING if imminent else PHASE_TRAINING
        elif launcher:
            phase = PHASE_LAUNCHING
        else:
            phase = PHASE_IDLE

        # Uptime comes from the session_start record rather than the process table: it is
        # the moment this session began training, which is what the number means to a human.
        started = last.get("session_start") or {}
        uptime = time.time() - started["time"] if pid and started.get("time") else None

        max_steps = last["max_steps"] or self._config_max_steps(run)
        step = last["step"]
        cfg = self._config_summary(run)
        tokens_per_step = last["tokens_per_step"] or cfg.get("tokens_per_step")

        return {
            "run": run,
            "phase": phase,
            "pid": pid,
            "launcher": launcher,
            "stop": stop,
            "uptime_s": uptime,
            "step": step,
            "max_steps": max_steps,
            "progress": ((step + 1) / max_steps) if (step is not None and max_steps) else None,
            "tokens_seen": (tokens_per_step * (step + 1)
                            if tokens_per_step and step is not None else None),
            "tokens_per_step": tokens_per_step,
            "last": last,
            "config": cfg,
            "sessions": sessions,
            "series": runlog.series(records, max_points=max_points),
            "checkpoints": self._checkpoints(run),
            "logs": self._logs(run),
            "can_start": run in LAUNCHERS and phase == PHASE_IDLE,
            "can_stop": phase in (PHASE_TRAINING, PHASE_STOPPING),
            "start_hint": (None if run in LAUNCHERS else
                           f"no launcher for '{run}' — scripts/phase2.sh builds the Phase-2 "
                           f"runs ({', '.join(LAUNCHERS)}); start this one from a terminal"),
            "meta": self._text(rdir / "run.meta"),
            "server_time": time.time(),
        }

    def summary(self, run: str) -> dict:
        """The short form for the run switcher: no series, no sessions."""
        full = self.status(run, max_points=0)
        keep = ("run", "phase", "pid", "step", "max_steps", "progress", "can_start", "can_stop")
        out = {k: full[k] for k in keep}
        out["ema"] = full["last"]["ema"]
        out["best_val"] = full["last"]["best_val"]
        out["updated"] = full["last"].get("step_time")
        return out

    # ---- actions -----------------------------------------------------------------------
    def start(self, run: str, stop_after: int | None = None, skip_smoke: bool = False) -> dict:
        """Launch `scripts/phase2.sh` detached, with its output going to a launch log.

        Detached (`start_new_session`) for the same reason phase2.sh nohups the trainer: the
        run must outlive whatever started it. Killing the portal mid-launch does not kill
        the launch, and the trainer it spawns is nobody's child.
        """
        self.check(run)
        if run not in LAUNCHERS:
            raise RunError(f"no launcher for '{run}': scripts/phase2.sh only builds "
                           f"{', '.join(LAUNCHERS)}. Start it from a terminal.")
        if (pid := self.trainer_pid(run)):
            raise RunError(f"'{run}' is already training as pid {pid}.")
        if (live := self.launcher(run)):
            raise RunError(f"a launch of '{run}' is already in pre-flight (pid {live['pid']}).")
        if stop_after is not None and stop_after < 1:
            raise RunError("stop_after must be at least 1 step.")

        script = self.root / "scripts" / "phase2.sh"
        if not script.exists():
            raise RunError(f"missing launcher: {script}")

        env = {**os.environ, **LAUNCHERS[run]}
        if stop_after is not None:
            env["STOP_AFTER"] = str(stop_after)
        if skip_smoke:
            env["SKIP_SMOKE"] = "1"

        ldir = self.log_dir(run)
        ldir.mkdir(parents=True, exist_ok=True)
        self.run_dir(run).mkdir(parents=True, exist_ok=True)
        log = ldir / f"launch_{datetime.now():%Y%m%d-%H%M%S}.log"
        with open(log, "wb") as fh:
            proc = subprocess.Popen(["bash", str(script)], cwd=self.root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        info = {"pid": proc.pid, "log": str(log.relative_to(self.root)),
                "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stop_after": stop_after, "skip_smoke": bool(skip_smoke)}
        (self.run_dir(run) / "portal_launch.json").write_text(json.dumps(info, indent=2))
        return {"ok": True, "action": "start", **info,
                "note": "pre-flight runs tests, checks the data, then a 50-step smoke test "
                        "before the real run starts — expect several minutes of log first."
                        if not skip_smoke else
                        "smoke test skipped: resuming a config that has already trained."}

    def stop(self, run: str, mode: str = "now", steps: int | None = None) -> dict:
        """Ask a live run to stop, via `scripts/stop.sh`.

        `now` finishes the step in flight and saves; `after`/`at` queue a bounded finish and
        return immediately; `cancel` withdraws a queued one. Stopping is always safe — the
        trainer saves `ckpt_last.pt` at the exact step it stops on and the resume continues
        with no loss spike.
        """
        self.check(run)
        if mode not in ("now", "after", "at", "cancel"):
            raise RunError(f"unknown stop mode: {mode!r}")
        if mode in ("after", "at"):
            if not steps or steps < 1:
                raise RunError(f"--{mode} needs a positive step "
                               f"{'count' if mode == 'after' else 'number'}.")
            if mode == "at":
                cur = runlog.latest(
                    runlog.load_records(self.run_dir(run) / "train_log.jsonl"))["step"]
                if cur is not None and steps <= cur:
                    raise RunError(f"step {steps} is already behind this run "
                                   f"(it is at {cur}) — use 'stop now'.")
        if mode != "cancel" and not self.trainer_pid(run):
            raise RunError(f"'{run}' is not training.")
        if mode == "cancel" and not self.stop_request(run):
            raise RunError(f"no stop is queued for '{run}'.")

        args = ["bash", str(self.root / "scripts" / "stop.sh"), run]
        if mode == "after":
            args += ["--after", str(steps)]
        elif mode == "at":
            args += ["--at", str(steps)]
        elif mode == "cancel":
            args += ["--cancel"]

        ldir = self.log_dir(run)
        ldir.mkdir(parents=True, exist_ok=True)
        log = ldir / f"stop_{datetime.now():%Y%m%d-%H%M%S}.log"
        # `stop now` waits for the checkpoint to land (up to WAIT seconds), which is far
        # longer than an HTTP request should take — so this is detached too, and the UI
        # watches the phase go stopping -> idle instead of waiting on the response.
        with open(log, "wb") as fh:
            proc = subprocess.Popen(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        return {"ok": True, "action": f"stop:{mode}", "pid": proc.pid,
                "log": str(log.relative_to(self.root)),
                "note": {"now": "saving a checkpoint at the current step, then exiting "
                                "(~30s for a 300M model).",
                         "after": f"queued: {steps} more steps, then save and exit.",
                         "at": f"queued: finish step {steps}, then save and exit.",
                         "cancel": "queued stop withdrawn; the run continues to its budget."}[mode]}

    # ---- logs --------------------------------------------------------------------------
    def _logs(self, run: str) -> list[dict]:
        """Every session log for this run, newest first. Symlinks are skipped: the stable
        `train_<run>.log` link points at one of these and would show up twice."""
        ldir = self.log_dir(run)
        if not ldir.is_dir():
            return []
        out = []
        for p in ldir.glob("*.log"):
            if p.is_symlink() or not p.is_file():
                continue
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime,
                        "kind": p.name.split("_")[0]})
        return sorted(out, key=lambda d: d["mtime"], reverse=True)

    def log_tail(self, run: str, name: str | None = None, lines: int = 300) -> dict:
        """The tail of one log. `name=None` means the most recently written one, which is
        the trainer's log while training and the launcher's during pre-flight."""
        self.check(run)
        logs = self._logs(run)
        if not logs:
            return {"run": run, "file": None, "lines": [], "size": 0,
                    "text": "(no logs yet for this run)"}
        if name is None:
            name = logs[0]["name"]
        elif name not in {d["name"] for d in logs}:
            raise RunError(f"no such log: {name}")

        path = self.log_dir(run) / name
        size = path.stat().st_size
        lines = max(1, min(int(lines), 5000))
        # Read only the tail: a multi-day session log is tens of MB and re-reading it every
        # poll would be silly. 400 bytes/line is generous for these log lines.
        with open(path, "rb") as fh:
            back = min(size, lines * 400 + 4096)
            fh.seek(size - back)
            chunk = fh.read()
        text = chunk.decode(errors="replace")
        if back < size:
            text = text.split("\n", 1)[-1]  # drop the partial first line
        tail = text.splitlines()[-lines:]
        return {"run": run, "file": name, "size": size, "truncated": back < size,
                "lines": tail, "files": logs}

    # ---- odds and ends -----------------------------------------------------------------
    def _checkpoints(self, run: str) -> list[dict]:
        rdir = self.run_dir(run)
        if not rdir.is_dir():
            return []
        out = []
        for p in sorted(rdir.glob("*.pt")):
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
        return out

    def _text(self, path: Path, limit: int = 4000) -> str | None:
        try:
            return path.read_text(errors="replace")[:limit]
        except OSError:
            return None

    def _config_summary(self, run: str) -> dict:
        """The handful of config fields worth showing. Parsed with the project's own loader
        so defaults and `d_ff`-style derived values are the ones the trainer would use."""
        path = self.config_path(run)
        if not path.exists():
            return {}
        try:
            from ..config import load_config
            cfg = load_config(str(path))
        except Exception as exc:  # a half-edited YAML must not blank the whole dashboard
            return {"error": f"{type(exc).__name__}: {exc}"}
        m, t, o = cfg.model, cfg.train, cfg.optim
        return {
            "path": str(path.relative_to(self.root)),
            "arch": f"d={m.d_model} L={m.n_layers} H={m.n_heads} KV={m.n_kv_heads} "
                    f"ff={m.d_ff} ctx={m.max_seq_len}",
            "vocab_size": m.vocab_size,
            "batch": f"{t.batch_size} x {t.grad_accum} accum x {t.seq_len} ctx",
            "tokens_per_step": t.batch_size * t.grad_accum * t.seq_len,
            "max_steps": t.max_steps,
            "lr": o.lr,
            "schedule": o.schedule,
            "grad_clip": o.grad_clip,
            "eval_every": t.eval_every,
            "ckpt_every": t.ckpt_every,
            "sources": [s.get("bin") for s in (cfg.data.train_sources or [])] or
                       [cfg.data.train_bin],
        }

    def _config_max_steps(self, run: str) -> int | None:
        return self._config_summary(run).get("max_steps")
