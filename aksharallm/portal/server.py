"""The portal's HTTP layer: one page, a small JSON API, no dependencies.

    .venv/bin/python -m aksharallm.portal            # then open http://127.0.0.1:8765

Design notes worth knowing before changing anything here:

* **stdlib only.** `ThreadingHTTPServer` is plenty for one viewer polling every two seconds,
  and the whole point of this project is that the machinery is legible. Threading (not the
  single-threaded default) matters: a `status` call that stats a few files must not block
  the page's other requests.
* **Loopback by default.** The POST endpoints start and stop processes on this machine, so
  the server binds `127.0.0.1` and refuses any other address without `--allow-remote`.
* **Writes are guarded.** Every mutating request must carry `X-Portal: 1`; a browser cannot
  attach a custom header to a cross-site form post, so a random page you visit cannot stop
  your training run through your own localhost server.
* **The API is read-mostly.** Two POSTs (`/start`, `/stop`) and everything else is a GET, so
  refreshing the page can never do anything.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import json
import mimetypes
import os
import signal
import socket
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .gpu import Sampler, snapshot
from .runs import LAUNCHERS, RunError, RunStore, repo_root
from .schedule import Rule, Schedule, Scheduler, parse_days

STATIC = Path(__file__).resolve().parent / "static"
GUARD_HEADER = "X-Portal"
MAX_BODY = 64 * 1024


def asdict_rule(rule: Rule) -> dict:
    d = dataclasses.asdict(rule)
    d["describe"] = rule.describe()
    nxt = rule.next_fire()
    d["next_fire"] = nxt.isoformat(timespec="minutes") if nxt else None
    return d


class Handler(BaseHTTPRequestHandler):
    """Routes. `store` is injected by `serve()` via `functools.partial`."""

    server_version = "aksharallm-portal"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, store: RunStore, scheduler: Scheduler, sampler: Sampler,
                 quiet: bool = True, **kw):
        self.store = store
        self.scheduler = scheduler
        self.sampler = sampler
        self.quiet = quiet
        super().__init__(*args, **kw)

    # ---- plumbing ----------------------------------------------------------------------
    def log_message(self, fmt, *args):  # noqa: D102 - stdlib hook
        if not self.quiet:
            sys.stderr.write(f"[portal] {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # A dashboard that shows a cached step number is worse than one that shows nothing.
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        # allow_nan=False would raise on a NaN loss; default=str keeps a stray Path or
        # datetime from taking the whole response down.
        body = json.dumps(obj, default=str).encode()
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, msg: str):
        self._json({"ok": False, "error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise RunError("request body too large")
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise RunError("body is not valid JSON")
        if not isinstance(data, dict):
            raise RunError("body must be a JSON object")
        return data

    def _int(self, data: dict, key: str) -> int | None:
        val = data.get(key)
        if val in (None, "", False):
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            raise RunError(f"{key} must be a whole number, got {val!r}")

    # ---- routing -----------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = parse_qs(url.query)
        try:
            if not parts:
                return self._static("index.html")
            if parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if parts == ["favicon.ico"]:
                return self._send(204, b"", "image/x-icon")
            if parts[0] == "api":
                return self._api_get(parts[1:], query)
        except RunError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # never let one bad read kill the poll loop silently
            self.log_message("error: %s", exc)
            return self._error(500, f"{type(exc).__name__}: {exc}")
        self._error(404, f"no such path: {url.path}")

    do_HEAD = do_GET

    def do_POST(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        # See the module docstring: a custom header is the cheap, effective guard against a
        # cross-site page poking a localhost server.
        if self.headers.get(GUARD_HEADER) != "1":
            return self._error(403, f"missing {GUARD_HEADER}: 1 header — "
                                    "the portal's own page sends it")
        try:
            data = self._body()
            if parts[:2] == ["api", "schedule"] and len(parts) == 3:
                return self._json(self._schedule_post(parts[2], data))
            if len(parts) == 4 and parts[:2] == ["api", "run"]:
                run, action = parts[2], parts[3]
                if action == "start":
                    return self._json(self.store.start(
                        run, stop_after=self._int(data, "stop_after"),
                        skip_smoke=bool(data.get("skip_smoke"))))
                if action == "stop":
                    return self._json(self.store.stop(
                        run, mode=str(data.get("mode", "now")),
                        steps=self._int(data, "steps")))
        except RunError as exc:
            return self._error(409, str(exc))
        except Exception as exc:
            self.log_message("error: %s", exc)
            return self._error(500, f"{type(exc).__name__}: {exc}")
        self._error(404, f"no such path: {url.path}")

    def _api_get(self, parts: list[str], query: dict):
        if parts == ["runs"]:
            return self._json({"root": str(self.store.root),
                               "runs": [self.store.summary(r) for r in self.store.runs()]})
        if len(parts) == 2 and parts[0] == "run":
            points = int((query.get("max_points") or [2000])[0])
            return self._json(self.store.status(parts[1], max_points=max(0, points)))
        if len(parts) == 3 and parts[0] == "run" and parts[2] == "log":
            name = (query.get("file") or [None])[0]
            lines = int((query.get("lines") or [300])[0])
            return self._json(self.store.log_tail(parts[1], name=name, lines=lines))
        if parts == ["gpu"]:
            window = (query.get("window") or ["3600"])[0]
            return self._json(snapshot(
                self.store,
                window_s=None if window in ("all", "0") else float(window),
                index=int((query.get("index") or [0])[0]),
                sampler=self.sampler))
        if parts == ["schedule"]:
            sched = self.scheduler.schedule.reload_if_changed()
            holder = self.scheduler.holder()
            return self._json({
                **sched.as_dict(),
                # "running" is about the machine, not this process: a scheduler started by
                # scripts/schedule.sh counts, and the page says which pid owns it.
                "running": bool(holder or self.scheduler._thread),
                "holder": holder or (os.getpid() if self.scheduler._thread else None),
                "in_portal": bool(self.scheduler._thread),
                "events": self.scheduler.recent(40),
                "startable": sorted(LAUNCHERS),
            })
        return self._error(404, "no such api path")

    def _schedule_post(self, action: str, data: dict) -> dict:
        """Edit the rule file. Every path here validates before it writes: a schedule that
        cannot fire is worse than no schedule, because you stop watching."""
        sched = self.scheduler.schedule.reload_if_changed()

        def check_run(run: str, act: str) -> str:
            self.store.check(run)
            if act == "start" and run not in LAUNCHERS:
                raise RunError(f"'{run}' has no launcher, so a scheduled start could never "
                               f"work. Startable runs: {', '.join(sorted(LAUNCHERS))}.")
            return run

        if action == "enable":               # the master switch
            sched.enabled = bool(data.get("enabled", True))
            sched.save()
            return {"ok": True, "enabled": sched.enabled,
                    "note": "schedule armed." if sched.enabled else
                            "schedule paused — rules are kept, nothing will fire."}

        if action == "window":
            run = check_run(str(data.get("run", "")), "start")
            rules = sched.add_window(
                run, str(data.get("start_at", "")), str(data.get("stop_at", "")),
                parse_days(data.get("days")),
                stop_after=(int(data["stop_after"]) if data.get("stop_after") else None),
                skip_smoke=bool(data.get("skip_smoke", True)))
            crosses = rules[1].days != rules[0].days
            return {"ok": True, "rules": [asdict_rule(r) for r in rules],
                    "note": f"{rules[0].describe()}; {rules[1].describe()}."
                            + (" The window crosses midnight, so the stops land on the "
                               "following day." if crosses else "")}

        if action == "rule":
            act = str(data.get("action", ""))
            rule = Rule(run=check_run(str(data.get("run", "")), act), action=act,
                        at=str(data.get("at", "")), days=parse_days(data.get("days")),
                        stop_after=(int(data["stop_after"]) if data.get("stop_after") else None),
                        skip_smoke=bool(data.get("skip_smoke", True)),
                        note=data.get("note") or None)
            sched.add(rule)
            return {"ok": True, "rule": asdict_rule(rule), "note": f"added: {rule.describe()}."}

        if action == "remove":
            rule = sched.remove(str(data.get("id", "")))
            return {"ok": True, "note": f"removed: {rule.describe()}."}

        if action == "toggle":
            rule = sched.set_enabled(str(data.get("id", "")), bool(data.get("enabled")))
            return {"ok": True, "note": f"{'enabled' if rule.enabled else 'paused'}: "
                                        f"{rule.describe()}."}

        raise RunError(f"unknown schedule action: {action}")

    def _static(self, name: str):
        # Serve only the files that ship with the package, by exact name: no traversal, no
        # surprises about what a local web server is exposing.
        path = STATIC / name
        if "/" in name or "\\" in name or not path.is_file():
            return self._error(404, f"no such file: {name}")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or name.endswith((".js", ".css")):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)


def lan_addresses() -> list[str]:
    """This machine's addresses on the local network, best effort.

    Opening a UDP socket toward a public address doesn't send anything; it just makes the
    kernel pick the interface it *would* route through, which is the address a phone on the
    same wifi should type. Falls back to whatever the hostname resolves to.
    """
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    return found or [socket.gethostname()]


def serve(root: Path | None = None, host: str = "127.0.0.1", port: int = 8765,
          quiet: bool = True) -> ThreadingHTTPServer:
    """Build a server (not yet serving). Port 0 picks a free port — used by the tests.

    The scheduler object is created but *not* started; `main()` starts it. That keeps a
    test server from firing anybody's rules.
    """
    store = RunStore(root)
    scheduler = Scheduler(store, Schedule(store.root))
    sampler = Sampler(store)
    handler = partial(Handler, store=store, scheduler=scheduler, sampler=sampler,
                      quiet=quiet)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.scheduler = scheduler
    httpd.sampler = sampler
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.portal",
        description="Local web portal: start/stop a training run and watch it.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default=None,
                    help="bind address (default 127.0.0.1, or 0.0.0.0 with --lan)")
    ap.add_argument("--lan", action="store_true",
                    help="serve on every interface so other machines on your network can "
                         "reach it, and print the address to use. The API starts and stops "
                         "training and has no login — only do this on a network you trust.")
    ap.add_argument("--root", default=None, help="repo root (default: this checkout)")
    ap.add_argument("--allow-remote", action="store_true",
                    help="permit a non-loopback --host (implied by --lan)")
    ap.add_argument("--open", action="store_true", help="open a browser at the portal")
    ap.add_argument("--no-gpu", action="store_true",
                    help="don't sample GPU telemetry in this process")
    ap.add_argument("--no-schedule", action="store_true",
                    help="don't run the scheduler in this process (rules still show, but "
                         "nothing fires unless scripts/schedule.sh --daemon is running)")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    # Same reason as the trainer: redirected to a file (nohup, a service), block buffering
    # would swallow the banner — including the LAN address you started it to read.
    sys.stdout.reconfigure(line_buffering=True)

    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    loopback = host in ("127.0.0.1", "::1", "localhost")
    if not loopback and not (args.lan or args.allow_remote):
        ap.error(f"refusing to bind {host}: the portal can start and stop training runs "
                 "and has no login. Pass --lan (or --allow-remote) if you mean it.")

    root = Path(args.root).resolve() if args.root else repo_root()
    try:
        httpd = serve(root, host, args.port, quiet=not args.verbose)
    except OSError as exc:
        print(f"cannot bind {host}:{args.port} — {exc}", file=sys.stderr)
        print("is a portal already running?  (try --port 8766)", file=sys.stderr)
        return 1

    # A pid file so `scripts/portal.sh --stop/--restart/--status` can find this process,
    # the same convention the trainer and the launcher use. SIGTERM is routed through the
    # Ctrl-C path so a restart shuts the scheduler down cleanly and releases its lock.
    pid_file = root / "logs" / "portal.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")

    def _release_pid():
        try:
            if int(pid_file.read_text().strip()) == os.getpid():
                pid_file.unlink()
        except (OSError, ValueError):
            pass

    atexit.register(_release_pid)

    def _terminate(signum, frame):
        raise KeyboardInterrupt   # the same clean path as Ctrl-C

    signal.signal(signal.SIGTERM, _terminate)

    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"aksharallm portal  ->  {url}  (pid {os.getpid()})")
    print(f"    repo   {root}")
    print(f"    runs   {', '.join(RunStore(root).runs()) or '(none found)'}")
    if not loopback:
        for addr in lan_addresses():
            print(f"    on your network:  http://{addr}:{httpd.server_port}/")
        print("    Anyone who can reach that address can start and stop training — there is")
        print("    no login. Fine on a home LAN; do not expose it to the internet.")
    print("    Ctrl-C to stop the portal, or scripts/portal.sh --stop / --restart from")
    print("    any terminal. Stopping the portal never touches a training run.")
    # The scheduler lives here by default: the portal is the process you leave running, and
    # a schedule nobody is watching for is a trap. One per machine — if scripts/schedule.sh
    # already holds the lock, this says so instead of double-firing every rule.
    scheduler = httpd.scheduler
    rules = len(scheduler.schedule.rules)
    if args.no_schedule:
        print(f"    schedule  not running here (--no-schedule), {rules} rules on file")
    elif scheduler.start():
        print(f"    schedule  running here, {rules} rules"
              f"{'' if scheduler.schedule.enabled else ' (PAUSED — nothing will fire)'}")
        for rule in sorted(scheduler.schedule.rules, key=lambda r: r.at):
            nxt = rule.next_fire()
            print(f"              {rule.describe()}"
                  f"{'  ->  next ' + nxt.strftime('%a %H:%M') if nxt else '  (paused)'}")
    else:
        print(f"    schedule  already running as pid {scheduler.holder()} — not starting a "
              "second one")

    sampler = httpd.sampler
    if args.no_gpu:
        print("    gpu       not sampling here (--no-gpu)")
    elif sampler.start():
        names = ", ".join(f"{d['index']}: {d['name']}" for d in sampler.devices())
        print(f"    gpu       sampling every {sampler.interval:.0f}s — {names}")
    elif not sampler.devices():
        print("    gpu       no NVIDIA GPU detected — the GPU panel will say so")
    else:
        print(f"    gpu       already sampled by pid {sampler.holder()}")

    if args.open:
        threading.Timer(0.4, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nportal stopped (training runs are unaffected).")
    finally:
        scheduler.stop()
        sampler.stop()
        httpd.server_close()
    return 0
