"""Generation jobs, for the portal.

The fourth panel built this way, and for the fourth time the rule is that the browser is a
*view over the CLI*: pressing Generate shells out to

    python -m aksharallm.synth gen <recipe> --name <dataset> --n <n> ...

which is exactly what you would type. A job started in the browser and one started in a
terminal append to the same dataset, write the same `meta.json`, print to the same log, and
either can stop the other through the same STOP file.

What is different here from the quantize and eval panels
-------------------------------------------------------
**The GPU problem cannot be solved by falling back to the CPU.** Everywhere else the portal
loads *our* model and can quietly choose a device. Here the model is loaded by Ollama, in
another process, and this panel has exactly two levers: which teacher is asked for, and
`synth.num_gpu`. So it reports instead of deciding — the panel says what a 19 GB teacher
would do to a live training run and offers the small one, and the choice stays with the
person who can see the whole machine.

**A stop is not a loss.** Quantization stopped halfway writes nothing and an evaluation
stopped halfway is worse than nothing. A generation run stopped halfway is a smaller
dataset, complete, with its provenance written and its funnel counted — so this panel, alone
among the three, offers the same bounded stops the trainer does: now, at N samples, or in
twenty minutes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..synth.dataset import list_datasets, synth_root, Dataset, SynthError
from ..synth.recipes import catalogue
from ..synth.teacher import DEFAULT_MODELS, SynthConfig, contention
from ..train import stopfile
from .explain import Ollama
from .runs import MAX_STOP_SECONDS, RunError, _alive, _cmdline, _read_int, repo_root

#: A dataset name is a directory name and reaches the command line, so it is checked rather
#: than escaped: letters, digits, dot, dash, underscore, and nothing else.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")


class SynthJobs:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()
        self.cfg = SynthConfig.load(self.root)

    # ---- paths --------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.root / "logs" / "synth"

    @property
    def pid_file(self) -> Path:
        return self.dir / "synth.pid"

    @property
    def current_file(self) -> Path:
        return self.dir / "current.json"

    @property
    def stop_file(self) -> Path:
        """Watched by the running generator. Deliberately not a dataset's own directory:
        the STOP belongs to the *job*, and two jobs never run at once."""
        return self.dir / "STOP"

    def log_path(self, job: str) -> Path:
        return self.dir / f"{job}.log"

    def json_path(self, job: str) -> Path:
        return self.dir / f"{job}.json"

    # ---- process state -------------------------------------------------------------------
    def _pid(self) -> int | None:
        pid = _read_int(self.pid_file)
        if pid and _alive(pid) and "aksharallm.synth" in _cmdline(pid):
            return pid
        return None

    def _current(self) -> dict:
        try:
            return json.loads(self.current_file.read_text())
        except (OSError, ValueError):
            return {}

    # ---- what the panel needs to draw itself ---------------------------------------------
    def teachers(self) -> dict:
        """Which models are pulled, and what each recipe would use by default.

        Asked of Ollama, so the picker offers what exists rather than a hardcoded list. If
        Ollama is not running this returns the reason instead of raising: the rest of the
        panel — the datasets, their funnels, the samples — is worth showing to somebody
        whose Ollama is off.
        """
        cfg = self.cfg.reload_if_changed()
        defaults = {name: cfg.model_for(name) for name in DEFAULT_MODELS}
        try:
            models = Ollama(cfg).models()
        except RunError as exc:
            return {"models": [], "defaults": defaults, "error": str(exc), "host": cfg.host}
        return {"models": models, "defaults": defaults, "error": None, "host": cfg.host,
                "num_gpu": cfg.num_gpu}

    def contention(self, model: str | None = None) -> dict:
        return contention(self.root, model)

    def datasets(self) -> list[dict]:
        return list_datasets(self.root)

    def dataset(self, name: str, samples: int = 5, rejects: int = 5) -> dict:
        """One dataset in full: its funnel, some kept samples, and some rejected ones.

        Both halves matter. The kept samples are what will be trained on; the rejects are
        the only way to tell *why* a pass rate is what it is, and reading three of them is
        usually enough to know whether the prompt or the teacher is the thing to change.
        """
        if not _NAME_RE.match(name or ""):
            raise RunError(f"not a dataset name: {name!r}")
        try:
            ds = Dataset(name, root=self.root)
        except SynthError as exc:
            raise RunError(str(exc))
        if not ds.exists:
            raise RunError(f"no dataset '{name}' under data/synth/")
        return {**ds.stats(),
                "samples": ds.samples(max(0, samples)),
                "rejects": ds.rejects(max(0, rejects))}

    def status(self, tail: int = 200) -> dict:
        pid = self._pid()
        cur = self._current()
        running = pid is not None
        log = self._tail(self.log_path(cur["job"]), tail) if cur.get("job") else []
        if not running and cur and cur.get("state") == "running":
            # No artifact decides this one: the dataset exists either way. The log's last
            # line does — the CLI prints "stopped: <reason>" whenever it ends normally.
            ended = any(line.strip().startswith("stopped:") for line in log[-12:])
            cur = {**cur, "state": "done" if ended else "failed"}
        return {
            "running": running,
            "pid": pid,
            "current": cur or None,
            "progress": self._progress(log) if running else None,
            "stop": self.stop_request() if running else None,
            "log": log,
            "recipes": catalogue(),
            "teachers": self.teachers(),
            "contention": self.contention((cur or {}).get("teacher")),
            "datasets": self.datasets(),
            "root": str(synth_root(self.root)),
        }

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    #: The CLI's own progress line, parsed rather than invented — the bar in the browser can
    #: only ever show a number the job itself printed.
    _PROGRESS_RE = re.compile(
        r"^\[synth\] (\S+) (\d+)/(\d+) \((\d+)%\) · (\d+) asked(?: · pass (\d+)%)?")

    def _progress(self, log: list[str]) -> dict | None:
        for line in reversed(log):
            found = self._PROGRESS_RE.match(line)
            if found:
                return {"recipe": found.group(1), "kept": int(found.group(2)),
                        "total": int(found.group(3)), "pct": int(found.group(4)),
                        "asked": int(found.group(5)),
                        "pass_rate": (int(found.group(6)) / 100
                                      if found.group(6) is not None else None),
                        "line": line}
        return None

    # ---- actions --------------------------------------------------------------------------
    def start(self, spec: dict) -> dict:
        if self._pid():
            raise RunError("a generation job is already running — stop it first.")

        recipe = str(spec.get("recipe") or "").strip()
        known = {r["name"]: r for r in catalogue()}
        if recipe not in known:
            raise RunError(f"unknown recipe {recipe!r}. Known: {', '.join(known)}")

        name = str(spec.get("name") or "").strip()
        if not _NAME_RE.match(name):
            raise RunError("a dataset name is letters, digits, dot, dash and underscore — "
                           "it becomes a directory under data/synth/.")
        # Appending to a dataset of another recipe would produce a file that cannot be
        # exported. Refused here as well as in the library so the panel can say it in time.
        existing = Dataset(name, root=self.root)
        if existing.exists and existing.meta.get("recipe") not in (None, recipe):
            raise RunError(f"'{name}' already holds {existing.meta['recipe']} samples — a "
                           f"dataset is one recipe. Pick another name for {recipe}.")

        # `or 50` would read a posted 0 as "unset" and quietly start a 50-sample job.
        try:
            n = 50 if spec.get("n") in (None, "") else int(spec["n"])
        except (TypeError, ValueError):
            raise RunError(f"not a sample count: {spec.get('n')!r}")
        if not 1 <= n <= 100_000:
            raise RunError("ask for between 1 and 100,000 samples.")

        cfg = self.cfg.reload_if_changed()
        teacher = str(spec.get("teacher") or "").strip() or cfg.model_for(recipe)
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,80}$", teacher):
            raise RunError(f"not a model name: {teacher!r}")

        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)   # a leftover would end this job immediately
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{recipe}"

        cmd = [sys.executable, "-u", "-m", "aksharallm.synth", "gen", recipe,
               "--name", name, "--n", str(n), "--teacher", teacher,
               "--stop-file", str(self.stop_file), "--json", str(self.json_path(job))]
        if spec.get("seed") not in (None, ""):
            cmd += ["--seed", str(int(spec["seed"]))]
        if spec.get("max_asks"):
            cmd += ["--max-asks", str(int(spec["max_asks"]))]
        if spec.get("dedup") not in (None, ""):
            cmd += ["--dedup", str(float(spec["dedup"]))]
        if spec.get("temperature") not in (None, ""):
            cmd += ["--temperature", str(float(spec["temperature"]))]
        if spec.get("stop_in_s"):
            seconds = int(spec["stop_in_s"])
            if not 1 <= seconds <= MAX_STOP_SECONDS:
                raise RunError(f"a time budget is capped at {MAX_STOP_SECONDS // 3600} hours.")
            cmd += ["--stop-in", f"{seconds}s"]
        if spec.get("no_verify"):
            cmd += ["--no-verify"]
        if spec.get("no_mutate"):
            cmd += ["--no-mutate"]

        log = self.log_path(job)
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                cmd, cwd=self.root, env={**os.environ}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid_file.write_text(str(proc.pid))
        current = {
            "job": job, "state": "running", "pid": proc.pid, "dataset": name,
            "recipe": recipe, "teacher": teacher, "n": n, "started": time.time(),
            "verify": not spec.get("no_verify"),
            "contention": self.contention(teacher),
            "cmd": " ".join(cmd[2:]),
        }
        self.current_file.write_text(json.dumps(current))
        return {"ok": True, "action": "start", **current}

    def stop(self, mode: str = "now", samples: int | None = None,
             seconds: int | None = None) -> dict:
        """Stop the running job now, at a sample count, at a time, or withdraw a queued stop.

        Bounded stops mean something here, unlike quantization's single pass: the generator
        checks the file between samples and every sample already written is already
        verified, deduplicated and recorded. Stopping is how these runs are *meant* to end.
        """
        if mode not in ("now", "at", "in", "cancel"):
            raise RunError(f"unknown stop mode: {mode!r}")
        pid = self._pid()
        if not pid:
            raise RunError("no generation job is running.")

        if mode == "cancel":
            if not self.stop_request():
                raise RunError("no stop is queued for this job.")
            self.stop_file.unlink(missing_ok=True)
            return {"ok": True, "action": "stop:cancel", "pid": pid,
                    "note": "queued stop withdrawn; generation runs to its full count."}

        if mode == "at":
            if not samples or samples < 1:
                raise RunError("a bounded stop needs a positive number of samples.")
            request = stopfile.StopRequest(step=int(samples))
            note = f"queued: generation stops once the dataset holds {samples} samples."
        elif mode == "in":
            if not seconds or seconds < 1:
                raise RunError("a timed stop needs a duration of at least one second.")
            if seconds > MAX_STOP_SECONDS:
                raise RunError(f"a timed stop is capped at {MAX_STOP_SECONDS // 3600} hours.")
            request = stopfile.StopRequest(deadline=time.time() + int(seconds))
            note = (f"queued: {stopfile.fmt_left(seconds)} more generation, then the "
                    "dataset is closed and written.")
        else:
            request = stopfile.StopRequest(now=True)
            note = ("stopping after the sample in flight. Everything written so far is a "
                    "complete dataset — run it again to add more.")
        stopfile.write(self.stop_file, request)
        return {"ok": True, "action": f"stop:{mode}", "pid": pid, "note": note}

    def stop_request(self) -> dict | None:
        req = stopfile.read(self.stop_file)
        if req is None:
            return None
        return {"target": req.step, "deadline": req.deadline, "now": req.now,
                "label": req.describe()}

    def export(self, name: str) -> dict:
        """Write the JSONL `prepare_sft` / `prepare_dpo` read, and hand back the command.

        The panel deliberately stops here rather than tokenizing: tokenizing needs a
        tokenizer path and an output directory and belongs to the data pipeline, which has
        its own command and its own checks. Generating data and preparing it for a trainer
        are two decisions, and the second one should be typed.
        """
        if not _NAME_RE.match(name or ""):
            raise RunError(f"not a dataset name: {name!r}")
        try:
            return Dataset(name, root=self.root).export()
        except SynthError as exc:
            raise RunError(str(exc))
