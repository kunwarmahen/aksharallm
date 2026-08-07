"""Evaluation jobs, for the portal.

Same arrangement as the quantize and finetune panels, and for the same reasons: pressing
Run shells out to

    python -m aksharallm.eval run <ckpt> --suite ... --json logs/eval/<id>.json

which is exactly what you would type. A job started in the browser and one started in a
terminal write the same file, appear in the same table, and either can stop the other.

Why a subprocess rather than a thread
-------------------------------------
An evaluation holds a model — 1.2 GB for the 300M — for as long as it runs, which can be
half an hour. Doing that inside the portal process would mean the web server that also runs
the scheduler is carrying a model and, on the GPU, a CUDA context. The scheduler is what
starts training at 22:00; it must not be able to die because HumanEval ran out of memory.

Sharing the machine with a training run
---------------------------------------
The harness's device choice is the Playground's (`aksharallm/infer/engine.py`): the CPU
whenever a run is training, automatically, with the reason stated. This panel does not
re-implement that — it asks for it and shows the answer, so the browser and the CLI can
never disagree about where a job will run.

There is **no bounded stop** here, unlike the run panel and QAT. An evaluation is not a
training loop: stopping it halfway leaves a partial score, and a partial score is worse
than no score because it looks like a real one in a table. Stop means stop, and it writes
nothing. Bound the *work* instead, with a smaller `--limit`.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..eval import report as report_mod
from ..eval import sources
from ..eval import suites as suites_mod
from ..infer.checkpoints import CheckpointStore, InferError
from ..infer.engine import InferConfig, plan_device
from .runs import RunError, _alive, _cmdline, _read_int, repo_root

_JOB_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_.-]+$")
#: A suite name from the browser is interpolated into a command line, so it is checked
#: against the registry rather than merely escaped. Nothing that is not a known suite runs.
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


class EvalJobs:
    def __init__(self, root: Path | None = None, cfg: InferConfig | None = None):
        self.root = Path(root) if root else repo_root()
        self.store = CheckpointStore(self.root)
        self.cfg = cfg or InferConfig.load(self.root)
        self.results = report_mod.Results(self.root)

    # ---- paths -------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.root / "logs" / "eval"

    @property
    def pid_file(self) -> Path:
        return self.dir / "eval.pid"

    @property
    def current_file(self) -> Path:
        return self.dir / "current.json"

    def log_path(self, job: str) -> Path:
        return self.dir / f"{job}.log"

    def json_path(self, job: str) -> Path:
        return self.dir / f"{job}.json"

    # ---- what can be evaluated ---------------------------------------------------------
    def checkpoints(self) -> list[dict]:
        """Every checkpoint, newest first. Unlike the quantize panel nothing is excluded:
        a quantized checkpoint is one of the more interesting things to evaluate, because
        "what did int4 cost on MMLU" is a question perplexity cannot answer."""
        out = []
        for ck in self.store.list():
            out.append({
                "id": ck.rel, "run": ck.run, "name": ck.name, "rel": ck.rel,
                "size": ck.size, "step": ck.step, "best_val": ck.best_val,
                "params": ck.params, "stage": ck.stage,
                "has_val": bool(ck.val_bin), "error": ck.error,
            })
        return out

    def adapters(self) -> list[dict]:
        from ..infer.checkpoints import AdapterStore

        return [a.as_dict() for a in AdapterStore(self.root).list()]

    # ---- process state -----------------------------------------------------------------
    def _pid(self) -> int | None:
        pid = _read_int(self.pid_file)
        if pid and _alive(pid) and "aksharallm.eval" in _cmdline(pid):
            return pid
        return None

    def _current(self) -> dict:
        try:
            return json.loads(self.current_file.read_text())
        except (OSError, ValueError):
            return {}

    def training(self) -> list[str]:
        live = []
        for run in self.store.dirs():
            pid = _read_int(run / "train.pid")
            if pid and _alive(pid):
                live.append(run.name)
        return live

    def device(self) -> dict:
        """Where a job would load the model, and why — the engine's own answer."""
        plan = plan_device(self.cfg.reload_if_changed(), self.training())
        return {**plan.as_dict(), "training": self.training()}

    def datasets(self) -> list[dict]:
        return sources.status(self.root)

    def status(self, tail: int = 200, results: int = 25) -> dict:
        pid = self._pid()
        cur = self._current()
        running = pid is not None
        log = self._tail(self.log_path(cur["job"]), tail) if cur.get("job") else []
        if not running and cur and cur.get("state") == "running":
            # The process is gone and nobody wrote the ending. A fetch job leaves no JSON,
            # so it is judged by its exit rather than by an artifact.
            if cur.get("kind") == "fetch":
                cur = {**cur, "state": "done"}
            else:
                cur = {**cur, "state": "done" if self.json_path(cur["job"]).exists()
                       else "failed"}
        return {
            "running": running,
            "pid": pid,
            "current": cur or None,
            "progress": self._progress(log) if running else None,
            "log": log,
            "device": self.device(),
            "suites": suites_mod.catalogue(),
            "groups": {"default": list(suites_mod.DEFAULT_SUITES),
                       "fast": list(suites_mod.FAST_SUITES),
                       "all": list(suites_mod.ALL_SUITES)},
            "datasets": self.datasets(),
            "results": self.results.rows(limit=results),
        }

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    #: The CLI's progress line. Parsed rather than invented, so the bar in the browser can
    #: only ever show what the job itself printed.
    _PROGRESS_RE = re.compile(r"^\[eval\] (\S+) (\d+)/(\d+) \((\d+)%\)")

    def _progress(self, log: list[str]) -> dict | None:
        for line in reversed(log):
            found = self._PROGRESS_RE.match(line)
            if found:
                return {"label": found.group(1), "done": int(found.group(2)),
                        "total": int(found.group(3)), "pct": int(found.group(4))}
        return None

    def compare(self, suite: str, run: str | None = None) -> dict:
        suites_mod.get(suite)               # raises with the list of known names
        return self.results.compare(suite, run=run)

    def result(self, name: str) -> dict:
        """One result file in full, including the per-item verdicts.

        The table is the summary; this is how you find out *which* HumanEval problems
        passed and what the model actually wrote.
        """
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise RunError(f"not a result file: {name!r}")
        path = self.dir / name
        if not path.is_file() or path.name in report_mod.NOT_RESULTS:
            raise RunError(f"no such result: {name}")
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RunError(f"{name} could not be read: {exc}")

    # ---- actions -----------------------------------------------------------------------
    def start(self, spec: dict) -> dict:
        if self._pid():
            raise RunError("an evaluation is already running — stop it first.")

        ref = str(spec.get("checkpoint") or "").strip()
        if not ref:
            raise RunError("pick a checkpoint")
        try:
            info = self.store.get(self.store.identify(ref))
        except (InferError, Exception) as exc:  # noqa: BLE001
            raise RunError(f"unknown checkpoint {ref!r}: {exc}")
        if info.error:
            raise RunError(f"{info.rel} cannot be loaded: {info.error}")

        names = suites_mod.resolve(spec.get("suites") or None)
        missing = [n for n in suites_mod.datasets_for(names)
                   if not sources.is_cached(n, self.root)]
        if missing:
            raise RunError(
                f"benchmark data for {', '.join(names)} is not downloaded yet "
                f"({', '.join(missing)}). Press Download data first — it is fetched once "
                "and reused for every evaluation after.")

        limit = spec.get("limit")
        limit = None if limit in (None, "") else max(0, int(limit))
        label = str(spec.get("label") or "eval")
        if not _SAFE_LABEL.match(label):
            raise RunError("the label may only contain letters, digits, dot, dash and "
                           "underscore.")

        self.dir.mkdir(parents=True, exist_ok=True)
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{info.run}-{label}"
        if not _JOB_RE.match(job):
            raise RunError(f"bad job name: {job}")

        cmd = [sys.executable, "-u", "-m", "aksharallm.eval", "run", str(info.path),
               "--suite", ",".join(names), "--json", str(self.json_path(job)),
               "--label", label]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if spec.get("shots") not in (None, ""):
            cmd += ["--shots", str(int(spec["shots"]))]
        if spec.get("device") in ("cuda", "cpu"):
            cmd += ["--device", str(spec["device"])]
        if spec.get("adapter"):
            cmd += ["--adapter", str(spec["adapter"])]
        if spec.get("judge_model"):
            cmd += ["--judge-model", str(spec["judge_model"])]
        if spec.get("max_new_tokens"):
            cmd += ["--max-new-tokens", str(int(spec["max_new_tokens"]))]
        if spec.get("batch_tokens"):
            cmd += ["--batch-tokens", str(int(spec["batch_tokens"]))]

        return self._launch(cmd, job, {
            "kind": "eval", "checkpoint": info.rel, "suites": names,
            "limit": limit, "adapter": spec.get("adapter") or None,
            "device": spec.get("device") or self.device().get("device"),
            "device_reason": self.device().get("reason"),
        })

    def fetch(self, names: list[str] | None = None) -> dict:
        """Download benchmark data, as a job, so the panel can show it happening.

        Fetching MMLU and HellaSwag is ~19 MB and a minute of the Hub's time. Doing it in
        the request thread would block the page; doing it as a job means the same log pane
        shows it, and the Run button is correctly disabled while it happens.
        """
        if self._pid():
            raise RunError("a job is already running — wait for it to finish.")
        wanted = [n for n in (names or list(sources.SOURCES)) if n in sources.SOURCES]
        if not wanted:
            raise RunError("nothing to fetch")
        self.dir.mkdir(parents=True, exist_ok=True)
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-data-fetch"
        cmd = [sys.executable, "-u", "-m", "aksharallm.eval", "fetch", *wanted]
        return self._launch(cmd, job, {"kind": "fetch", "datasets": wanted})

    def start_audit(self, spec: dict) -> dict:
        """The two checks that measure the *benchmark* rather than the model.

        They share the eval panel's one-job-at-a-time lock deliberately. A contamination
        scan streams ten billion tokens and a per-domain split runs the model; neither
        wants to be doing that while an evaluation is trying to produce a number.
        """
        kind = str(spec.get("kind") or "")
        if kind not in ("contaminate", "domains", "calibrate", "dedup"):
            raise RunError(f"unknown audit {kind!r}")
        if self._pid():
            raise RunError("a job is already running — wait for it to finish.")
        self.dir.mkdir(parents=True, exist_ok=True)
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"

        if kind == "contaminate":
            cfg = str(spec.get("config") or "configs/small-code.yaml")
            # A config name, not a path: this decides which .bin files get opened.
            if "/" in cfg.replace("configs/", "") or not cfg.endswith(".yaml"):
                raise RunError("pick one of the run configs")
            if not (self.root / cfg).is_file():
                raise RunError(f"{cfg} does not exist")
            cmd = [sys.executable, "-u", "-m", "aksharallm.eval", "contaminate",
                   "--config", cfg,
                   "--suite", ",".join(suites_mod.resolve(spec.get("suites") or "mc"))]
            if spec.get("max_tokens"):
                cmd += ["--max-tokens", str(int(spec["max_tokens"]))]
            if spec.get("verify"):
                cmd.append("--verify")
            if spec.get("against"):
                cmd += ["--against", str(self.json_path(str(spec["against"])))]
            meta = {"kind": "contaminate", "config": cfg}
        else:
            ref = str(spec.get("checkpoint") or "").strip()
            if not ref:
                raise RunError("pick a checkpoint")
            info = self.store.get(self.store.identify(ref))
            if info.error:
                raise RunError(f"{info.rel} cannot be loaded: {info.error}")
            cmd = [sys.executable, "-u", "-m", "aksharallm.eval", "domains", str(info.path),
                   "--device", self.device().get("device", "cpu")]
            if spec.get("batches"):
                cmd += ["--batches", str(int(spec["batches"]))]
            meta = {"kind": "domains", "checkpoint": info.rel}

        if kind == "calibrate":
            ref = str(spec.get("checkpoint") or "").strip()
            if not ref:
                raise RunError("pick a checkpoint")
            info = self.store.get(self.store.identify(ref))
            if info.error:
                raise RunError(f"{info.rel} cannot be loaded: {info.error}")
            cmd = [sys.executable, "-u", "-m", "aksharallm.eval", "calibrate", str(info.path),
                   "--device", self.device().get("device", "cpu"),
                   # Deliberately modest. Calibration keeps the FULL logit vector per
                   # position, because temperature scaling needs the whole distribution --
                   # see eval/calibration.py's `collect`.
                   "--batches", str(int(spec.get("batches") or 24)),
                   "--batch", str(int(spec.get("batch") or 2))]
            meta = {"kind": "calibrate", "checkpoint": info.rel}

        if kind == "dedup":
            source = str(spec.get("source") or "")
            # A path into `data/`, resolved and contained: this opens a file on disk.
            path = (self.root / source).resolve()
            if not source.endswith(".bin") or not str(path).startswith(
                    str((self.root / "data").resolve())):
                raise RunError("pick a tokenized .bin under data/")
            if not path.is_file():
                raise RunError(f"{source} does not exist")
            out = self.dir / f"dedup-{Path(source).stem}-{job}.json"
            cmd = [sys.executable, "-u", "-m", "aksharallm.data.dedup", str(path),
                   "--limit", str(int(spec.get("limit") or 60_000)),
                   "--start-token", str(int(spec.get("start_token") or 0)),
                   "--out", str(out)]
            meta = {"kind": "dedup", "source": source,
                    "start_token": int(spec.get("start_token") or 0)}

        return self._launch(cmd, job, meta)

    def audits(self, limit: int = 10) -> dict:
        """The latest contamination report, and where to find the rest.

        Read from the JSON the CLI wrote, never recomputed — the terminal and the browser
        have to be looking at the same measurement or one of them is lying.
        """
        files = sorted(self.dir.glob("contamination-*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        latest = None
        if files:
            try:
                latest = {**json.loads(files[0].read_text()),
                          "name": files[0].name, "when": files[0].stat().st_mtime}
            except (json.JSONDecodeError, OSError):
                latest = None
        return {"latest": latest, "history": [f.name for f in files]}

    def _latest(self, pattern: str, limit: int) -> dict:
        """The newest JSON matching a glob, read from disk and never recomputed.

        The terminal and the browser have to be looking at the same measurement, or one of
        them is lying — the same rule `audits` follows for contamination.
        """
        files = sorted(self.dir.glob(pattern), key=lambda p: p.stat().st_mtime,
                       reverse=True)[:limit]
        latest = None
        if files:
            try:
                latest = {**json.loads(files[0].read_text()),
                          "name": files[0].name, "when": files[0].stat().st_mtime}
            except (json.JSONDecodeError, OSError):
                latest = None
        return {"latest": latest, "history": [f.name for f in files]}

    def calibration(self, limit: int = 10) -> dict:
        """Is the model's confidence honest? See `eval/calibration.py`."""
        return self._latest("calibration-*.json", limit)

    def dedup(self, limit: int = 10) -> dict:
        """How much of the corpus is a near-duplicate of the rest of it. See `data/dedup.py`.

        `history` matters more here than elsewhere: a dedup number is quoted per *offset*,
        and the honest way to read one is beside another taken somewhere else in the file.
        """
        return self._latest("dedup-*.json", limit)

    def corpora(self) -> list[dict]:
        """Tokenized `.bin` files worth scanning, with their size — the picker's contents."""
        out = []
        for path in sorted((self.root / "data").rglob("*.bin")):
            if path.name in ("val.bin",) or "audio" in path.parts or "vision" in path.parts:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({"rel": str(path.relative_to(self.root)),
                        "tokens": size // 2, "gb": round(size / 1e9, 2)})
        return out

    def _launch(self, cmd: list[str], job: str, meta: dict) -> dict:
        log = self.log_path(job)
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                cmd, cwd=self.root, env={**os.environ}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid_file.write_text(str(proc.pid))
        current = {"job": job, "state": "running", "pid": proc.pid,
                   "started": time.time(), "cmd": " ".join(cmd[2:]), **meta}
        self.current_file.write_text(json.dumps(current))
        return {"ok": True, "action": "start", **current}

    def stop(self) -> dict:
        pid = self._pid()
        if not pid:
            raise RunError("no evaluation is running.")
        os.kill(pid, 15)
        cur = self._current()
        if cur:
            self.current_file.write_text(json.dumps({**cur, "state": "stopped"}))
        return {"ok": True, "action": "stop", "pid": pid,
                "note": "stopped. An evaluation writes its result at the end, so a "
                        "stopped job records nothing — that is deliberate: half a "
                        "benchmark looks like a whole one in a table."}
