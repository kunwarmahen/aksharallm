"""Night queues: the multi-night job lists in `logs/queue/*.sh`, as the Schedule panel sees them.

The portal's own scheduler fires *rules* (start this run at 00:30, stop it at 05:30) from
`schedule.json`. A night queue is a different thing: a bash loop that sleeps until a window
opens, runs a list of jobs — train until 05:00, then `eval domains`, then HumanEval — marks
each one done in `logs/queue/<name>/`, and sleeps again. Phase 3's scoring and Phase 4's
continued pretraining both ran this way (`logs/queue/nights.sh`, `py-nights.sh`).

Until this module the portal could not see one at all, so the Schedule panel read "nothing
scheduled" on a machine that was training every night.

Nothing here asks a queue to describe itself. Everything is read from what a queue already
leaves behind, so a queue started before this module existed shows up too:

* **alive** — a process whose command line names the script (`/proc/*/cmdline`). A queue
  killed by a reboot leaves a log ending "sleeping until Sat 00:30" forever; without this
  check the panel would repeat that confidently. Dead reads "not running".
* **state** — the last `=== <time> <message>` line of `logs/queue/<name>.log`, the lines
  the queue's own `say()` writes.
* **window** — the `START=${START:-00:30}` / `END=` defaults in the script.
* **done** — the `*.done` markers in the directory the script's `DONE=` names, oldest
  first. `ALL.done` means the queue finished its list.

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

_DEFAULT = re.compile(r"^(START|END|TRAIN_END)=\$\{\1:-(\d{1,2}:\d{2})\}", re.M)
_DONE = re.compile(r"^DONE=(\S+)", re.M)
_SAY = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)$")


def _pids_running(script: Path) -> list[int]:
    """Processes running exactly this script file.

    A relative argv (`bash logs/queue/py-nights.sh`, which is how queues are started) is
    resolved against *that process's* working directory. Matching the relative path alone
    made the one live queue on this machine "alive" in every checkout and temp directory
    that had a script of the same name -- caught by the tests, which run in tmp dirs.
    """
    want = script.resolve()
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().split(b"\0")[:3]
            if not any(a.endswith(script.name.encode()) for a in argv):
                continue
            cwd = Path(proc.joinpath("cwd").resolve())
        except OSError:
            continue
        for a in argv:
            p = Path(a.decode("utf-8", "replace"))
            if p.name == script.name and (p if p.is_absolute() else cwd / p).resolve() == want:
                out.append(int(proc.name))
                break
    return sorted(out)


def _last_say(log: Path) -> tuple[str | None, str | None]:
    """(time, message) of the last `=== ...` line the queue wrote."""
    try:
        with log.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16384))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None, None
    for line in reversed(lines):
        m = _SAY.match(line)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _next_at(hhmm: str, now: datetime) -> datetime:
    h, m = map(int, hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def describe(script: Path, root: Path, now: datetime | None = None) -> dict | None:
    """One queue, or None if it has never run (no log)."""
    now = now or datetime.now()
    name = script.stem
    log = script.with_suffix(".log")
    if not log.is_file():
        return None
    text = script.read_text(errors="replace")
    defaults = dict(_DEFAULT.findall(text))
    m = _DONE.search(text)
    done_dir = root / m.group(1) if m else script.with_suffix("")
    done = ([p.name[: -len(".done")] for p in
             sorted(done_dir.glob("*.done"), key=lambda p: p.stat().st_mtime)]
            if done_dir.is_dir() else [])
    finished = "ALL" in done
    pids = _pids_running(script)
    alive = bool(pids)
    said_at, said = _last_say(log)
    if alive:
        state = said or "running"
    else:
        state = "finished" if finished else "not running"
    nxt = None
    if alive and said and "sleeping until" in said and defaults.get("START"):
        nxt = _next_at(defaults["START"], now)
    return {
        "name": name,
        "script": str(script.relative_to(root)),
        "log": str(log.relative_to(root)),
        "alive": alive,
        "pid": pids[0] if pids else None,
        "state": state,
        "said_at": said_at,
        "start": defaults.get("START"),
        "end": defaults.get("END"),
        "train_end": defaults.get("TRAIN_END"),
        "finished": finished,
        "done": [d for d in done if d != "ALL"],
        "next_window": nxt.timestamp() if nxt else None,
        "next_window_in_s": (nxt - now).total_seconds() if nxt else None,
    }


def queues(root: Path | str, now: datetime | None = None) -> list[dict]:
    """Every queue that has run here: live ones first, then by last activity."""
    root = Path(root)
    found = [q for s in sorted((root / "logs" / "queue").glob("*.sh"))
             if (q := describe(s, root, now))]
    return sorted(found, key=lambda q: (not q["alive"], -(Path(root / q["log"]).stat().st_mtime)))
