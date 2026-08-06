"""The Serve panel: start, stop and watch the HTTP server from the browser.

The server is a *separate process* with its own lifetime — that is the point of it — so this
module holds none of its state. It does what the rest of the portal does with anything that
outlives a page load: **shells out to the script** (`scripts/serve.sh`), reads the pid file
the script wrote, and asks the server itself how it is doing through the `/health` endpoint it
already serves to everyone else.

Which means there is exactly one way to start a server, whether you typed it or clicked it,
and a server started in a terminal shows up in the panel — the same contract `phase2.sh` and
the dashboard have had since the beginning.

Read with: docs/16-serving.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .runs import RunError, repo_root

DEFAULT_PORT = 8770
#: How long to wait for `/health`. The server answers in milliseconds when it is up; the
#: timeout only matters while it is still loading a checkpoint, and a panel that blocks for
#: five seconds on every poll is worse than one that says "starting".
HEALTH_TIMEOUT = 1.5
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


class ServeJobs:
    """One server at a time, driven through `scripts/serve.sh`."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or repo_root()).resolve()
        self.dir = self.root / "logs" / "serve"

    # ---- reading -----------------------------------------------------------------------
    def _meta(self) -> dict:
        path = self.dir / "serve.meta"
        out: dict = {}
        try:
            for line in path.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key:
                    out[key.strip()] = value.strip()
        except OSError:
            pass
        return out

    def _pid(self) -> int | None:
        try:
            pid = int((self.dir / "serve.pid").read_text().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return pid

    def _health(self, port: int) -> dict | None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=HEALTH_TIMEOUT) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _log(self, lines: int = 60) -> list[str]:
        try:
            return (self.dir / "serve.log").read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return []

    def status(self, tail: int = 60) -> dict:
        meta = self._meta()
        port = int(meta.get("port") or DEFAULT_PORT)
        pid = self._pid()
        health = self._health(port) if pid else None
        return {
            "running": bool(pid),
            # A pid with no health is the window between launch and the checkpoint finishing
            # loading — a 1.2 GB file, so several seconds. Saying "starting" beats saying
            # "running" and having every number below it be blank.
            "phase": "running" if health else ("starting" if pid else "idle"),
            "pid": pid,
            "port": port,
            "url": f"http://127.0.0.1:{port}/v1",
            "meta": meta,
            "health": health,
            "log": self._log(tail),
            "hint": (f"curl -s http://127.0.0.1:{port}/v1/completions "
                     f"-d '{{\"prompt\": \"def quicksort(arr):\", \"max_tokens\": 64}}'"),
        }

    # ---- writing -------------------------------------------------------------------------
    def start(self, checkpoint: str | None = None, port: int | None = None,
              max_batch: int | None = None, device: str | None = None,
              speculate: int | None = None) -> dict:
        if self._pid():
            raise RunError("a server is already running — stop it first.")
        ckpt = (checkpoint or "small-code").strip()
        if not RUN_NAME_RE.match(ckpt):
            raise RunError(f"invalid checkpoint: {ckpt!r}")
        if device and device not in ("auto", "cuda", "cpu"):
            raise RunError(f"unknown device: {device!r}")

        script = self.root / "scripts" / "serve.sh"
        if not script.exists():
            raise RunError(f"{script} is missing")
        env = {**os.environ, "PORT": str(port or DEFAULT_PORT),
               "MAX_BATCH": str(max_batch or 32), "SPECULATE": str(speculate or 0)}
        if device and device != "auto":
            env["DEVICE"] = device
        self.dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([str(script), ckpt, "--bg"], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RunError((proc.stderr or proc.stdout or "serve.sh failed").strip())
        return {"ok": True, "note": (proc.stdout or "").strip().splitlines()[:1],
                **self.status()}

    def stop(self) -> dict:
        script = self.root / "scripts" / "serve.sh"
        proc = subprocess.run([str(script), "--stop"], cwd=self.root, capture_output=True,
                              text=True, timeout=60)
        if proc.returncode != 0:
            raise RunError((proc.stderr or proc.stdout or "could not stop").strip())
        return {"ok": True, "note": (proc.stdout or "").strip(), **self.status()}
