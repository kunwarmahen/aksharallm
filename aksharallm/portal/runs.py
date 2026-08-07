"""What a training run *is*, from the outside: its state, and the two things you can do to it.

The portal never reimplements starting or stopping. `scripts/phase2.sh` and
`scripts/stop.sh` remain the only things that launch a trainer or ask one to stop; this
module shells out to them exactly as a human would, and reads back the same files they
write (`train.pid`, `STOP`, `run.meta`, `train_log.jsonl`, `logs/<run>/*.log`). That is
what keeps the button and the terminal honest about each other.

State lives on disk, never in this process, so the portal can be restarted, or run twice,
or not run at all, without a training run noticing.

Read with: docs/09-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..train import runlog, stopfile

#: Run names come off the wire and end up in paths and a subprocess argument, so they are
#: whitelisted rather than escaped: letters, digits, dash, underscore, dot, no leading dot.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: How each run is launched: which script, and what environment it needs. Two launchers,
#: because the runs are two different shapes — `phase2.sh` builds 20 GB of tokens and
#: pre-flights for a six-day base-model run, `experiment.sh` starts a Phase-1-scale
#: experiment on data that already exists. A run that is in neither (a config someone added
#: by hand) is still fully *visible* in the portal; it just has no Start button, because
#: nothing here knows how to build its data.
#:
#: `script` and `env` are both optional so a test can register a run with `{}` and get the
#: Phase-2 launcher, which is what every caller meant before there were two.
#: Every module in this repo that trains something into a `checkpoints/<run>/` directory
#: and writes `train.pid` there. `trainer_pid` validates a pid file against this list, so a
#: trainer missing from it reports its run as **idle while it is training** — the pid is
#: read, the process is alive, and it is rejected as somebody else's.
TRAINERS: tuple[str, ...] = (
    "aksharallm.train.pretrain",
    "aksharallm.train.sft",
    "aksharallm.train.dpo",
    "aksharallm.train.grpo",
    "aksharallm.audio.train_codec",
    "aksharallm.audio.train_lm",
    "aksharallm.vision.train",
)

#: Every shell script that pre-flights a run and publishes `launch.pid` / `launch.meta`.
#: A launcher missing from this tuple makes its run read as **idle while it is pre-flighting**,
#: with the Start button still enabled -- which invites a second launch on top of the first.
LAUNCH_SCRIPTS: tuple[str, ...] = ("phase2.sh", "experiment.sh", "audio.sh")

LAUNCHERS: dict[str, dict] = {
    "small-code": {},                                        # blended 85/15 base (default)
    "small": {"env": {"PURE": "1"}},                         # FineWeb-Edu only fallback
    "tiny-moe": {"script": "scripts/experiment.sh", "args": ["tiny-moe"]},
    "tiny": {"script": "scripts/experiment.sh", "args": ["tiny"]},
    "tiny-diffusion": {"script": "scripts/experiment.sh",
                       "args": ["tiny-diffusion"]},   # docs/19
    # Audio and vision: a different launcher, the same pid/meta/log contract. docs/20, 21.
    "codec-synth": {"script": "scripts/audio.sh", "args": ["codec-synth"]},
    "codec-lj": {"script": "scripts/audio.sh", "args": ["codec-lj"]},
    "audiolm-synth": {"script": "scripts/audio.sh", "args": ["audiolm-synth"]},
}


def launcher_for(run: str) -> tuple[str, list[str], dict[str, str]]:
    """(script, args, env) for a startable run."""
    spec = LAUNCHERS[run]
    return (spec.get("script", "scripts/phase2.sh"),
            list(spec.get("args", [])),
            dict(spec.get("env", {})))

PHASE_IDLE = "idle"
PHASE_LAUNCHING = "launching"   # phase2.sh is in pre-flight/data/smoke, no trainer yet
PHASE_TRAINING = "training"
PHASE_STOPPING = "stopping"     # a stop was requested and the trainer is still alive

#: Longest timed stop the buttons will queue. A deadline is a promise the trainer keeps for
#: as long as it runs, so there is no technical limit — but "stop this in 14 hours" is a
#: schedule, and saying so points at the panel that survives a reboot and repeats itself.
MAX_STOP_SECONDS = 12 * 3600


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


#: A top-level section that some trainer in this repo would recognise — the cheap, text-only
#: test for "this is a run and not a settings file". `model:` is a language model, `codec:`
#: and `audiolm:` are docs/20, `vision:` is docs/21. Deliberately not a YAML parse: `runs()`
#: is called on every poll of every open page.
_RUN_CONFIG_RE = re.compile(r"^(model|codec|audiolm|vision):", re.MULTILINE)


def _is_run_config(path: Path) -> bool:
    try:
        return bool(_RUN_CONFIG_RE.search(path.read_text(errors="replace")))
    except OSError:
        return False


def _read_meta(path: Path) -> dict[str, str]:
    """Parse the `key   value` files the shell scripts write (`run.meta`, `launch.meta`).

    Deliberately the format a human reads with `cat`, not JSON: these files exist to be
    understood from a terminal at 2am, and shell can write them without a helper.
    """
    out: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            key, _, value = line.partition(" ")
            if key:
                out[key.strip()] = value.strip()
    except OSError:
        pass
    return out


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

    def is_archived(self, run: str) -> bool:
        """A snapshot of a finished run, set aside so a new one can start.

        Decided by a marker file the archiver writes, not by "has no config" — a run whose
        YAML was renamed or never committed is not an archive, it is a run with no config,
        and conflating the two would label half of somebody's checkpoints directory. Not by
        the name either: `RUN_NAME_RE` allows dots, so a legitimately dotted run name would
        be misread.

        Archives are read-only by construction rather than by a flag: the launcher table is
        keyed on config names, so there is nothing to start. They stay in the picker because
        being able to open one and read what it did is the entire point of keeping it.
        """
        return (self.run_dir(run) / "archive.meta").exists()

    def dir_bytes(self, run: str) -> int:
        total = 0
        for base in (self.run_dir(run), self.log_dir(run)):
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    continue
        return total

    # ---- discovery ---------------------------------------------------------------------
    def runs(self) -> list[str]:
        """Every run the portal knows: one per config, plus any checkpoint dir with a log.

        A checkpoint dir with no config still shows up — a run whose YAML was renamed is
        exactly when you want to read its history, not when you want it to vanish.

        Not every YAML under `configs/` is a run: `portal.yaml` configures the portal's own
        code explainer. A run config is one *some* trainer could read — a `model:` section
        (a language model), a `codec:` or `audiolm:` one (docs/20), or `vision:` (docs/21).
        Anything else is a settings file that happens to live next door, and a phantom run in
        the picker with no log and no launcher is the kind of thing you waste an evening on.
        """
        names = {p.stem for p in (self.root / "configs").glob("*.yaml")
                 if _is_run_config(p)}
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

        `train.pid` is written by the trainer itself into its own `out_dir`, so it answers
        "who is training into this directory" — the question that matters. The command-line
        fallback (for a run launched before that existed) is anchored with `$` on purpose:
        the 50-step smoke test inside `phase2.sh` runs the identical command with `-o
        train.out_dir=/tmp/...` appended, and an unanchored match reports the smoke test as
        the run — which is how a stop request ends up aimed at a process that never reads it.
        `scripts/stop.sh` uses the same anchor and the same smoke-test guard.

        **`TRAINERS` is why this is a list and not one string.** The validation used to be
        `"aksharallm.train" in cmdline`, which silently excluded every trainer that does not
        live in `aksharallm/train/`: a codec run's command line is
        `aksharallm.audio.train_codec`, so its pid file was read, rejected, and the run
        reported **idle while it was training** — with the log tail still advancing and no
        Stop button, which is the most confusing possible combination. Adding a trainer means
        adding it here.
        """
        pid = _read_int(self.run_dir(run) / "train.pid")
        if pid and _alive(pid):
            args = _cmdline(pid)
            if any(m in args for m in TRAINERS) and "aksharallm_smoke" not in args:
                return pid

        cached = self._pgrep_cache.get(run)
        if cached and time.time() - cached[0] < self._PGREP_TTL:
            return cached[1] if _alive(cached[1]) else None
        try:
            # The fallback stays **pretrain-only**, and widening it to `TRAINERS` is a bug:
            # several trainers share one config and write to *different* directories, so
            # `aksharallm.train.sft configs/demo.yaml` writes into `checkpoints/demo-sft/`
            # and matching on the config name alone would report it as run `demo`. The pid
            # file has no such ambiguity — it lives in the directory it describes — which is
            # exactly why it exists and why only its check needed widening.
            found = subprocess.run(
                ["pgrep", "-f", rf"aksharallm\.train\.pretrain configs/{run}\.yaml$"],
                capture_output=True, text=True, timeout=5).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None
        pid = int(found[0]) if found else None
        self._pgrep_cache[run] = (time.time(), pid)
        return pid

    def launcher(self, run: str) -> dict | None:
        """A live launch script for this run — pre-flight, before any trainer exists.

        Read from the files the script itself writes (`launch.pid` + `launch.meta`), not
        from anything the portal remembers. So a pre-flight started in a terminal shows up
        here as `pre-flight` too, and the portal's own launches are visible to
        `scripts/stop.sh --status`. One record, both directions.

        **`LAUNCH_SCRIPTS`, not `"phase2.sh"`.** This check used to name one script, so a run
        launched by `experiment.sh` or `audio.sh` reported **idle during its whole pre-flight**
        — with the Start button still enabled, inviting a second launch on top of the first.
        Same shape of mistake as `TRAINERS` above, and it needs the same discipline: a new
        launcher goes in this tuple.
        """
        pid = _read_int(self.run_dir(run) / "launch.pid")
        if not pid or not _alive(pid) or not any(s in _cmdline(pid) for s in LAUNCH_SCRIPTS):
            return None
        meta = _read_meta(self.run_dir(run) / "launch.meta")
        return {"pid": pid, "stage": meta.get("stage"), "started": meta.get("started"),
                "config": meta.get("config"), "log": meta.get("log"), "meta": meta}

    def stop_request(self, run: str) -> dict | None:
        """The pending stop, read out of the STOP file the trainer polls.

        Empty file == stop after the current step; a number == stop on reaching that step;
        `@<epoch>` == stop at that wall-clock time. `aksharallm.train.stopfile` is the
        trainer's half of this contract, and the only place the three forms are defined.
        """
        req = stopfile.read(self.run_dir(run) / "STOP")
        if req is None:
            return None
        return {"target": req.step, "deadline": req.deadline, "now": req.now,
                "label": req.describe()}

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
            # A queued stop at a step (or a time) we haven't reached yet is not "stopping" —
            # the run is training normally and merely has a finish line. Only an imminent
            # stop is. A deadline counts as imminent once the clock passes it, which is the
            # window between the deadline and the trainer finishing the step it is on.
            imminent = bool(stop) and (
                stop["now"]
                or (stop["target"] is not None and last["step"] is not None
                    and stop["target"] <= last["step"])
                or (stop["deadline"] is not None and time.time() >= stop["deadline"]))
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

        # A run that has trained its whole budget. Starting it again is not harmful — the
        # trainer resumes, sees there is nothing to do and exits without touching the
        # checkpoint — but it runs a full pre-flight (tests, data checks, a smoke test) to
        # get there, so from the outside it looks like a launch that silently failed. Say so
        # instead of offering the button.
        # `trained_to` rather than `step`: the last *logged* step lags the last *trained*
        # one by up to log_every, so a completed 8,000-step run whose log ends at 7,980
        # would otherwise read as 20 steps short of its budget forever.
        reached = last.get("trained_to")
        if reached is None:
            reached = step
        finished = bool(max_steps and reached is not None and reached + 1 >= max_steps)
        archived = self.is_archived(run)

        return {
            "run": run,
            "phase": phase,
            "finished": finished,
            "archived": archived,
            # Whether deleting this run leaves a recipe behind. The delete dialog says so,
            # and saying it wrongly is worse than not saying it.
            "has_config": self.config_path(run).exists(),
            # Deleting is refused for anything alive; everything else is fair game, and the
            # confirmation is the human's, not the code's, to give.
            "can_delete": phase == PHASE_IDLE and (self.run_dir(run).is_dir()
                                                   or self.log_dir(run).is_dir()),
            # A finished run cannot resume — there is nothing left in its budget — but it can
            # be set aside and started again from step 0, which is what "run it again" means
            # for an experiment. Archiving keeps the old one readable in the picker.
            "can_restart": (run in LAUNCHERS and phase == PHASE_IDLE and finished
                            and not archived),
            "size_bytes": self.dir_bytes(run),
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
            "can_start": (run in LAUNCHERS and phase == PHASE_IDLE and not finished
                          and not archived),
            # A pre-flight is stoppable too — that aborts the launch. Bounded stops are not:
            # there is no step count to count from until the trainer exists.
            "can_stop": phase in (PHASE_TRAINING, PHASE_STOPPING, PHASE_LAUNCHING),
            "can_bound": phase in (PHASE_TRAINING, PHASE_STOPPING),
            "start_hint": (
                (f"'{run}' is an archive of a finished run — read-only. Its config is gone, "
                 "so there is nothing to start.") if archived else
                (f"'{run}' has trained its whole budget: {max_steps:,} of {max_steps:,} "
                 f"steps. Start fresh to archive it and begin again from step 0, or raise "
                 f"train.max_steps in configs/{run}.yaml to carry on training this one."
                 ) if finished and run in LAUNCHERS else
                None if run in LAUNCHERS else
                           f"no launcher for '{run}' — the portal can start "
                           f"{', '.join(sorted(LAUNCHERS))} (scripts/phase2.sh for the base "
                           "model, scripts/experiment.sh for the Phase-1 experiments); "
                           "start this one from a terminal"),
            "meta": self._text(rdir / "run.meta"),
            "server_time": time.time(),
        }

    def report(self, run: str, save: bool = False) -> dict:
        """The run report: the same markdown a trainer writes to `report.md` when it exits.

        Built fresh on every request rather than served from the file, because the file is a
        snapshot from the last exit and this panel is often opened *during* a run — a report
        that silently showed last Tuesday's numbers would be the most confidently wrong thing
        on the page. `save=True` writes it to disk, which is the button beside it.
        """
        self.check(run)
        from ..train import report as run_report
        out_dir = self.run_dir(run)
        data = run_report.build(out_dir, run=run, root=self.root)
        saved = run_report.write(out_dir, run=run, root=self.root) if save else None
        on_disk = out_dir / "report.md"
        return {
            "run": run,
            "markdown": run_report.render(data),
            "generated": data.get("generated"),
            "complete": data.get("complete"),
            "checks": data.get("checks"),
            "saved": str(saved.relative_to(self.root)) if saved else None,
            # What is on disk, and when — so "Save" can say whether it changed anything.
            "file": str(on_disk.relative_to(self.root)) if on_disk.exists() else None,
            "file_mtime": on_disk.stat().st_mtime if on_disk.exists() else None,
        }

    def summary(self, run: str) -> dict:
        """The short form for the run switcher: no series, no sessions."""
        full = self.status(run, max_points=0)
        keep = ("run", "phase", "pid", "step", "max_steps", "progress", "can_start",
                "can_stop", "archived", "finished")
        out = {k: full[k] for k in keep}
        out["ema"] = full["last"]["ema"]
        out["best_val"] = full["last"]["best_val"]
        out["updated"] = full["last"].get("step_time")
        return out

    # ---- archive and delete --------------------------------------------------------------
    def _idle_or_raise(self, run: str, verb: str) -> None:
        if (pid := self.trainer_pid(run)):
            raise RunError(f"'{run}' is training as pid {pid} — stop it before you {verb} it.")
        if (live := self.launcher(run)):
            raise RunError(f"'{run}' is in pre-flight (pid {live['pid']}) — stop it before "
                           f"you {verb} it.")

    def _owned(self, path: Path) -> Path:
        """A path this store is allowed to move or remove.

        Every name has already been through `RUN_NAME_RE`, so this cannot fail for any input
        the API accepts — which is exactly why it is here. The one operation in this file
        that removes data should not depend on a regex three functions away still being
        right, so it resolves symlinks and re-checks containment immediately before acting.
        """
        resolved = path.resolve()
        for base in (self.root / "checkpoints", self.root / "logs"):
            if resolved.parent == base.resolve():
                return resolved
        raise RunError(f"refusing to touch {path}: outside checkpoints/ and logs/")

    def archive(self, run: str) -> dict:
        """Set a finished run aside under a timestamped name, keeping all of it.

        `tiny-moe` becomes `tiny-moe.20260801-105843` — a rename, not a copy, so a 3 GB run
        is set aside instantly and nothing is duplicated. The new name passes `RUN_NAME_RE`
        (dots are legal) and sorts next to the original in the picker, and because no
        `configs/<name>.yaml` exists for it, it is read-only from then on.
        """
        self.check(run)
        self._idle_or_raise(run, "archive")
        if self.is_archived(run):
            raise RunError(f"'{run}' is already an archive.")
        if not self.run_dir(run).is_dir():
            raise RunError(f"'{run}' has nothing to archive yet.")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{run}.{stamp}"
        if not RUN_NAME_RE.match(name):
            raise RunError(f"bad archive name: {name}")
        moved = []
        for src, dst in ((self.run_dir(run), self.run_dir(name)),
                         (self.log_dir(run), self.log_dir(name))):
            if src.is_dir():
                if dst.exists():
                    raise RunError(f"{dst} already exists — archive by hand.")
                self._owned(src).rename(dst)
                moved.append(str(dst.relative_to(self.root)))
        # The marker is what makes this an archive rather than a directory with a date in
        # its name, and it records what it was archived from — which is the question asked
        # of it a month later.
        (self.run_dir(name) / "archive.meta").write_text(
            f"archived {datetime.now():%Y-%m-%d %H:%M:%S}\nfrom    {run}\n"
            f"config  configs/{run}.yaml (as it was then)\n")
        # The stable `train_<run>.log` symlink points into the directory that just moved.
        link = self.root / f"train_{run}.log"
        if link.is_symlink():
            link.unlink()
        return {"ok": True, "action": "archive", "run": run, "archive": name,
                "moved": moved,
                "note": f"'{run}' is now '{name}' and is read-only. Starting '{run}' again "
                        "begins a new run from step 0."}

    def delete(self, run: str, confirm: str | None = None) -> dict:
        """Remove a run's checkpoints and logs. Its config, if any, is left alone.

        The config is source and is committed; the artifacts are hundreds of megabytes of
        reproducible output. Deleting the two together would mean a mis-click loses the
        recipe as well as the result.

        `confirm` must repeat the run's name. The browser asks the human first, but the
        API is the thing that actually removes files, so it does not rely on a dialog it
        cannot see having been shown.
        """
        self.check(run)
        if confirm != run:
            raise RunError(f"deleting '{run}' needs confirm='{run}' — nothing was removed.")
        self._idle_or_raise(run, "delete")

        freed = self.dir_bytes(run)
        removed = []
        for base in (self.run_dir(run), self.log_dir(run)):
            if base.is_dir():
                shutil.rmtree(self._owned(base))
                removed.append(str(base.relative_to(self.root)))
        link = self.root / f"train_{run}.log"
        if link.is_symlink():
            link.unlink()
        kept = self.config_path(run)
        return {"ok": True, "action": "delete", "run": run, "removed": removed,
                "freed": freed,
                "note": (f"removed {', '.join(removed) or 'nothing'} ({freed / 1e9:.2f} GB)."
                         + (f" configs/{run}.yaml was kept — start it again for a fresh run."
                            if kept.exists() else ""))}

    # ---- actions -----------------------------------------------------------------------
    def start(self, run: str, stop_after: int | None = None, skip_smoke: bool = False,
              stop_after_s: int | None = None, fresh: bool = False) -> dict:
        """Launch `scripts/phase2.sh` detached, with its output going to a launch log.

        Detached (`start_new_session`) for the same reason phase2.sh nohups the trainer: the
        run must outlive whatever started it. Killing the portal mid-launch does not kill
        the launch, and the trainer it spawns is nobody's child.
        """
        self.check(run)
        if run not in LAUNCHERS:
            raise RunError(f"no launcher for '{run}': the portal can start "
                           f"{', '.join(sorted(LAUNCHERS))}. Start it from a terminal.")
        archived_as = None
        if fresh:
            # Set the finished run aside FIRST, so what launches next sees an empty
            # directory and starts at step 0 instead of resuming into a spent budget.
            if self.run_dir(run).is_dir():
                archived_as = self.archive(run)["archive"]
        if (pid := self.trainer_pid(run)):
            raise RunError(f"'{run}' is already training as pid {pid}.")
        if (live := self.launcher(run)):
            raise RunError(f"a launch of '{run}' is already in pre-flight (pid {live['pid']}).")
        if stop_after is not None and stop_after < 1:
            raise RunError("stop_after must be at least 1 step.")
        if stop_after_s is not None and stop_after_s < 1:
            raise RunError("a time budget must be at least one second.")
        if stop_after is not None and stop_after_s is not None:
            # Both would work — the trainer honours whichever lands first — but a session
            # bounded two ways at once is a session nobody can predict the end of.
            raise RunError("bound this session by steps or by time, not both.")

        rel, args, extra_env = launcher_for(run)
        script = self.root / rel
        if not script.exists():
            raise RunError(f"missing launcher: {script}")

        ldir = self.log_dir(run)
        ldir.mkdir(parents=True, exist_ok=True)
        self.run_dir(run).mkdir(parents=True, exist_ok=True)
        log = ldir / f"launch_{datetime.now():%Y%m%d-%H%M%S}.log"

        env = {**os.environ, **extra_env}
        if stop_after is not None:
            env["STOP_AFTER"] = str(stop_after)
        if stop_after_s is not None:
            env["STOP_IN"] = f"{stop_after_s}s"
        if skip_smoke:
            env["SKIP_SMOKE"] = "1"
        # Both launchers record this path in launch.meta, so `scripts/stop.sh --status` can
        # point at the same log the portal is streaming. The launch record itself is written
        # by the script (launch.pid + launch.meta) — the portal keeps no private copy.
        env["LAUNCH_LOG"] = str(log.relative_to(self.root))

        with open(log, "wb") as fh:
            proc = subprocess.Popen(["bash", str(script), *args], cwd=self.root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        info = {"pid": proc.pid, "log": str(log.relative_to(self.root)),
                "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stop_after": stop_after, "stop_after_s": stop_after_s,
                "skip_smoke": bool(skip_smoke), "archived": archived_as}
        return {"ok": True, "action": "start", **info,
                "note": (f"'{archived_as}' keeps the previous run; this one starts at step 0. "
                         if archived_as else "")
                        + "pre-flight runs tests, checks the data, then a 50-step smoke test "
                        "before the real run starts — expect several minutes of log first."
                        if not skip_smoke else
                        "smoke test skipped: resuming a config that has already trained."}

    def stop(self, run: str, mode: str = "now", steps: int | None = None,
             seconds: int | None = None) -> dict:
        """Ask a live run to stop, via `scripts/stop.sh`.

        `now` finishes the step in flight and saves; `after`/`at` queue a bounded finish in
        steps and `in` queues one in wall-clock, all returning immediately; `cancel`
        withdraws a queued one. Stopping is always safe — the trainer saves `ckpt_last.pt`
        at the exact step it stops on and the resume continues with no loss spike.
        """
        self.check(run)
        if mode not in ("now", "after", "at", "in", "cancel"):
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
        if mode == "in":
            if not seconds or seconds < 1:
                raise RunError("a timed stop needs a duration of at least one second.")
            if seconds > MAX_STOP_SECONDS:
                raise RunError(f"a timed stop is capped at {MAX_STOP_SECONDS // 3600} hours "
                               "— past that, bound the run by steps or use the schedule.")
        launching = None if mode == "cancel" else self.launcher(run)
        if mode != "cancel" and not self.trainer_pid(run):
            # Nothing is training, but a pre-flight may be minutes from starting one.
            # `stop.sh` aborts it; that is the only sensible reading of "stop" right now.
            if not launching:
                raise RunError(f"'{run}' is not training.")
            if mode != "now":
                raise RunError(
                    f"'{run}' is still in pre-flight ({launching.get('stage')}) — there is no "
                    "step count to bound yet. Use 'stop now' to abort the launch, or wait "
                    "for training to start.")
            if launching.get("stage") == "launching":
                raise RunError("the launcher is starting the trainer right now — give it a "
                               "few seconds, then stop the run itself.")
        if mode == "cancel" and not self.stop_request(run):
            raise RunError(f"no stop is queued for '{run}'.")

        args = ["bash", str(self.root / "scripts" / "stop.sh"), run]
        if mode == "after":
            args += ["--after", str(steps)]
        elif mode == "at":
            args += ["--at", str(steps)]
        elif mode == "in":
            args += ["--in", f"{int(seconds)}s"]
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
                "note": {"now": "aborting the launch — nothing has trained yet, so nothing "
                                "is lost." if launching else
                                "saving a checkpoint at the current step, then exiting "
                                "(~30s for a 300M model).",
                         "after": f"queued: {steps} more steps, then save and exit.",
                         "at": f"queued: finish step {steps}, then save and exit.",
                         "in": (f"queued: {stopfile.fmt_left(seconds or 0)} more training, "
                                f"until about "
                                f"{datetime.fromtimestamp(time.time() + (seconds or 0)):%H:%M}"
                                ", then save and exit."),
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


# --------------------------------------------------------------------------------------
# a terminal for the two housekeeping actions
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """`python -m aksharallm.portal.runs list|archive|delete <run>`

    The portal's buttons call the functions above directly rather than shelling out here —
    unlike start and stop, these are file operations, not processes to supervise. This exists
    so the same two actions are available without a browser, and so `delete` can be typed
    deliberately by someone who wants to think about it first.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m aksharallm.portal.runs",
                                 description="List, archive or delete a run's artifacts.")
    ap.add_argument("action", choices=("list", "archive", "delete"))
    ap.add_argument("run", nargs="?")
    ap.add_argument("--root", default=None)
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt for delete")
    args = ap.parse_args(argv)

    store = RunStore(args.root)
    try:
        if args.action == "list":
            print()
            print(f"  {'run':<28} {'kind':<9} {'step':>8} {'size':>9}  phase")
            for name in store.runs():
                st = store.status(name, max_points=0)
                kind = "archive" if st["archived"] else "run"
                step = "–" if st["step"] is None else f"{st['step']:,}"
                print(f"  {name:<28} {kind:<9} {step:>8} "
                      f"{st['size_bytes'] / 1e9:>8.2f}G  {st['phase']}")
            print()
            return 0

        if not args.run:
            print(f"{args.action} needs a run name", file=sys.stderr)
            return 2

        if args.action == "archive":
            print(store.archive(args.run)["note"])
            return 0

        st = store.status(args.run, max_points=0)
        if not args.yes:
            print(f"\n  delete '{args.run}'?")
            print(f"    checkpoints/{args.run}/ and logs/{args.run}/  "
                  f"({st['size_bytes'] / 1e9:.2f} GB)")
            if st["step"] is not None:
                print(f"    {st['step']:,} steps trained"
                      + (f", best val {st['last']['best_val']:.4f}"
                         if st["last"].get("best_val") else ""))
            if store.config_path(args.run).exists():
                print(f"    configs/{args.run}.yaml is kept")
            print("\n  This cannot be undone. Type the run's name to confirm: ", end="")
            if input().strip() != args.run:
                print("  not deleted.")
                return 1
        print(store.delete(args.run, confirm=args.run)["note"])
        return 0
    except RunError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
