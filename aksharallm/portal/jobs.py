"""One detached CLI job at a time, judged by its exit code — shared by the newer tabs.

The same contract `AsrJobs` (portal/dictate.py) established, pulled out so a tab that only
needs "run this CLI command from a button" does not grow its own copy:

* **The job IS the CLI command.** The browser picks arguments from a fixed menu, the tab
  builds the argv, and the panel prints `python -m ...` so the terminal and the browser never
  disagree. There is no second implementation to drift.
* **Detached** (`start_new_session`), so closing the page or restarting the portal does not
  kill it. One job per folder (`<dir>/current.json`).
* **Success is the exit code**, written by a wrapper shell into `<job>.rc` — never "an output
  file appeared" (gotcha 20: that check reported every Eval audit as failed on success).
* **The identity is computed from the argv, not read from /proc** right after launch, because
  a child that has not exec'd yet shows the parent's command line (gotcha 26).

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .runs import exec_cmdline


class JobError(RuntimeError):
    """Something to show as a message rather than a stack trace."""


class CliJobs:
    def __init__(self, root: Path, folder: str, module: str):
        self.root = Path(root)
        self.folder = folder        # e.g. "logs/world/jobs"
        self.module = module        # e.g. "aksharallm.world"

    @property
    def dir(self) -> Path:
        p = self.root / self.folder
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _current(self) -> dict:
        try:
            return json.loads((self.dir / "current.json").read_text())
        except (OSError, ValueError):
            return {}

    def _pid(self) -> int | None:
        cur = self._current()
        pid = cur.get("pid")
        if not pid:
            return None
        try:
            os.kill(int(pid), 0)
            live = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except (OSError, ValueError):
            return None
        # A recycled pid would not reproduce the command line recorded at launch.
        return int(pid) if live == cur.get("cmdline") else None

    def _rc(self, job: str) -> int | None:
        try:
            return int((self.dir / f"{job}.rc").read_text().strip())
        except (OSError, ValueError):
            return None

    def status(self, tail: int = 30) -> dict:
        cur = self._current()
        pid = self._pid()
        if cur and cur.get("state") == "running" and pid is None:
            rc = self._rc(cur.get("job", ""))
            cur = {**cur, "state": "done" if rc == 0 else ("failed" if rc is not None else "lost"),
                   "rc": rc}
        log = []
        if cur.get("job"):
            try:
                text = (self.dir / f"{cur['job']}.log").read_text(errors="replace")
                # Progress lines rewrite themselves with \r; keep the last state of each.
                log = [ln.split("\r")[-1] for ln in text.splitlines()][-tail:]
            except OSError:
                pass
        return {"running": pid is not None, "current": cur or None, "log": log}

    def start(self, argv: list[str], label: str) -> dict:
        if self._pid() is not None:
            raise JobError("a job is already running here — wait for it or stop it")
        job = time.strftime("%Y%m%d-%H%M%S") + "-" + argv[0]
        cli = [sys.executable, "-u", "-m", self.module, *argv]
        rc = self.dir / f"{job}.rc"
        script = " ".join(shlex.quote(a) for a in cli) + f"; echo $? > {shlex.quote(str(rc))}"
        with open(self.dir / f"{job}.log", "wb") as fh:
            proc = subprocess.Popen(["/bin/sh", "-c", script], cwd=self.root,
                                    stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        cur = {"job": job, "kind": argv[0], "label": label, "state": "running",
               "pid": proc.pid, "started": time.time(),
               "cmdline": exec_cmdline(["/bin/sh", "-c", script]),
               "command": f"python -m {self.module} " + " ".join(shlex.quote(a) for a in argv)}
        (self.dir / "current.json").write_text(json.dumps(cur))
        return {"ok": True, **cur}

    def stop(self) -> dict:
        pid = self._pid()
        if pid is None:
            raise JobError("no job is running")
        os.killpg(pid, 15)   # the shell and the CLI under it
        cur = self._current()
        (self.dir / "current.json").write_text(json.dumps({**cur, "state": "stopped"}))
        return {"ok": True, "stopped": pid}


def training(root: Path) -> str | None:
    """A run holding the card right now (a live `train.pid`), or None — the device policy:
    a job started from the browser goes to the CPU while anything trains."""
    for pid_file in sorted((Path(root) / "checkpoints").glob("*/train.pid")):
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            return pid_file.parent.name
        except (OSError, ValueError):
            continue
    return None
