"""Announce a terminal-started evaluation so the portal can see it.

The rule everywhere in this repo is that the browser and the terminal never disagree: the
portal shells out to the same commands and reads the same files, so it is a *view* and not a
second system. Evaluation was only half-holding that up. A finished result appeared in the
Eval tab either way -- both paths write `logs/eval/<stamp>-<run>-<label>.json` -- but only a
portal-launched job wrote `eval.pid` and `current.json`, so while a `python -m aksharallm.eval`
was running in a terminal the tab said "nothing running", offered a Start button beside it,
and showed no progress at all.

This is `pretrain.claim_pid_file` for evaluation, and for the same reason. The pid belongs to
the **directory**, not to a command line, so whoever started the job -- the portal, a script,
or a bare command -- the readers get one unambiguous answer.

Two things it is careful about:

- **It never steals a live job's slot.** If `eval.pid` names a process that is alive and is
  an eval, this publishes nothing and says so. A CLI run is still allowed to proceed (that is
  the user's business), but it must not overwrite another job's state and make the portal
  describe the wrong work.
- **It closes out its own state.** The portal infers done-vs-failed for an abandoned job by
  looking for the artifact, which only works when the job name matches the artifact name. A
  terminal job writes its own ending instead, so that guess is never needed.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

from .report import results_dir


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _claimed_by(current: Path, pid: int) -> bool:
    """Does `current.json` say this pid owns the slot? Belt and braces beside `_is_eval`,
    for a job invoked in a shape that does not mention the module by name."""
    try:
        return json.loads(current.read_text()).get("pid") == pid
    except (OSError, ValueError, AttributeError):
        return False


def _own_cmdline() -> str:
    """This process's command line, in the same shape the portal reads for any other pid."""
    try:
        return Path(f"/proc/{os.getpid()}/cmdline").read_bytes().replace(
            b"\0", b" ").decode(errors="replace")
    except OSError:
        return " ".join(sys.argv)


def _is_eval(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "aksharallm.eval" in cmdline


class _Tee:
    """Write to the terminal and to the job log at once.

    The portal tails `logs/eval/<job>.log`, which a portal-launched job gets for free from
    `Popen(stdout=fh)`. A terminal job prints to a terminal, so without this the tab knows
    something is running and can show nothing about it. Flushed per write: a progress line
    that arrives in the file a minute late is worse than none, because the reader concludes
    the job has hung.
    """

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text: str) -> int:
        n = self._stream.write(text)
        try:
            self._fh.write(text)
            self._fh.flush()
        except (OSError, ValueError):
            pass          # the log is a convenience; never let it break the command
        return n

    def flush(self) -> None:
        self._stream.flush()
        with contextlib.suppress(OSError, ValueError):
            self._fh.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def announced(kind: str, meta: dict | None = None, root: Path | str | None = None):
    """Publish "this terminal is evaluating" for the duration of the block.

    Yields the job name, or None when another live job already owns the slot -- in which
    case nothing is written and the caller simply runs unannounced.
    """
    directory = results_dir(root)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield None
        return

    pid_file = directory / "eval.pid"
    current = directory / "current.json"
    owner = _read_int(pid_file)
    if owner and _alive(owner) and (owner == os.getpid() or _is_eval(owner)
                                    or _claimed_by(current, owner)):
        # Either another evaluation holds the slot, or this process already announced one
        # (a nested block). Both must leave the existing state alone: overwriting it would
        # make the portal describe work that is not the work it is showing progress for.
        if owner != os.getpid():
            print(f"[eval] another evaluation is running (pid {owner}); the portal will "
                  f"keep showing that one.", file=sys.stderr)
        yield None
        return

    job = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    state = {"job": job, "state": "running", "pid": os.getpid(), "started": time.time(),
             "kind": kind, "source": "terminal",
             # The real command line, recorded so a reader can tell this process from a
             # recycled pid without having to pattern-match on how it was invoked.
             "cmdline": _own_cmdline(),
             "cmd": " ".join(sys.argv[1:]) or kind, **(meta or {})}

    def write_state() -> None:
        with contextlib.suppress(OSError):
            current.write_text(json.dumps(state))

    try:
        pid_file.write_text(f"{os.getpid()}\n")
    except OSError:
        yield None
        return
    write_state()

    log = directory / f"{job}.log"
    saved = sys.stdout
    try:
        with open(log, "w", buffering=1) as fh:
            sys.stdout = _Tee(saved, fh)
            try:
                yield job
            except BaseException:
                state["state"] = "failed"
                raise
            else:
                state["state"] = "done"
            finally:
                sys.stdout = saved
                state["ended"] = time.time()
                write_state()
    finally:
        sys.stdout = saved
        # Only ever release our own claim; another process may have taken over since.
        with contextlib.suppress(OSError, ValueError):
            if _read_int(pid_file) == os.getpid():
                pid_file.unlink()
