"""The STOP file: one small contract that every long loop in this repo obeys.

A run is stopped by writing a file, not by signalling a process. The trainer reads
`<out_dir>/STOP` fresh on every step, so a stop can be queued from a terminal
(`echo 20000 > checkpoints/small-code/STOP`), from `scripts/stop.sh`, or from the portal's
buttons, and all three mean exactly the same thing to the process.

The file holds one line, in one of three forms::

    (empty)       stop after the current step
    20000         stop on reaching absolute step 20000   (inclusive: 20000 is trained)
    @1753985400   stop on the first step at or after this epoch time

The third form is what makes "stop in twenty minutes" possible without anything watching
the clock on the trainer's behalf. It matters that the *trainer* owns the deadline: a
timer living in the portal dies with the portal, and a duration converted to a step count
at the moment you press the button drifts as soon as throughput changes -- an eval pass, a
thermal throttle, another process on the GPU. A deadline in the file is still true after a
portal restart, and still true if the run slows to half speed.

Anything unreadable or unparseable is read as "stop now". That is the safe reading of an
ambiguous stop, and it also means an older trainer handed a `@`-deadline stops promptly
rather than ignoring it.

The trainers' half of the contract is `reached()`: give it the current step and it returns
the reason string to print, or None to keep going.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: "30m", "90s", "2h", "1h30m", or a bare number of minutes ("30"). Durations are how
#: people say this out loud; steps are how the loop counts. Both end up in the same file.
_DURATION_RE = re.compile(r"^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?\s*$", re.I)
_CLOCK_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


@dataclass(frozen=True)
class StopRequest:
    """What a STOP file is asking for. Exactly one of the three fields is meaningful."""

    now: bool = False
    step: int | None = None           # stop on reaching this absolute step (inclusive)
    deadline: float | None = None     # stop on the first step at/after this epoch time

    @property
    def bounded(self) -> bool:
        """A finish line in the future, rather than a stop happening right now."""
        return not self.now

    def text(self) -> str:
        """The line to write into a STOP file. Round-trips through `parse`."""
        if self.deadline is not None:
            return f"@{int(self.deadline)}"
        if self.step is not None:
            return str(self.step)
        return ""

    def describe(self, step: int | None = None) -> str:
        """One human phrase for logs, `--status` output and the portal."""
        if self.deadline is not None:
            left = self.deadline - time.time()
            when = datetime.fromtimestamp(self.deadline).strftime("%H:%M")
            return f"stop at {when}" + (f" ({fmt_left(left)} from now)" if left > 0 else " (due)")
        if self.step is not None:
            ahead = f" ({self.step - step} steps from now)" if step is not None else ""
            return f"stop after step {self.step}{ahead}"
        return "stop after the current step"


def fmt_left(seconds: float) -> str:
    """A rough "how much longer", for prose rather than for a table."""
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{round(seconds / 60)}m"
    h, m = divmod(round(seconds / 60), 60)
    return f"{h}h{m:02d}m"


def parse_duration(text: str) -> int:
    """Seconds from "30m", "90s", "2h", "1h30m", or a bare number read as minutes.

    Bare numbers are minutes because that is what every use of this reads as out loud --
    "give it another 30" is half an hour, never half a minute.
    """
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty duration")
    if raw.isdigit():
        return int(raw) * 60
    h, m, s = _DURATION_RE.match(raw).groups() if _DURATION_RE.match(raw) else (None,) * 3
    if not any((h, m, s)):
        raise ValueError(f"cannot read {text!r} as a duration -- try 30m, 90s, 2h or 1h30m")
    total = int(h or 0) * 3600 + int(m or 0) * 60 + int(s or 0)
    if total < 1:
        raise ValueError("a duration must be at least one second")
    return total


def deadline_from_clock(clock: str, now: float | None = None) -> float:
    """Epoch time for the next occurrence of a local "HH:MM" -- tomorrow if it has passed."""
    match = _CLOCK_RE.match(str(clock).strip())
    if not match:
        raise ValueError(f"time must be HH:MM (24-hour), not {clock!r}")
    base = datetime.fromtimestamp(now if now is not None else time.time())
    when = base.replace(hour=int(match.group(1)), minute=int(match.group(2)),
                        second=0, microsecond=0)
    stamp = when.timestamp()
    return stamp + 86400 if stamp <= base.timestamp() else stamp


def parse(text: str | None) -> StopRequest:
    """Read a STOP file's contents. Anything ambiguous means "stop now"."""
    raw = (text or "").strip()
    if not raw:
        return StopRequest(now=True)
    if raw.startswith("@"):
        try:
            return StopRequest(deadline=float(raw[1:]))
        except ValueError:
            return StopRequest(now=True)
    try:
        return StopRequest(step=int(raw))
    except ValueError:
        return StopRequest(now=True)


def read(path: Path) -> StopRequest | None:
    """The pending stop for a run, or None if no STOP file exists.

    A file that exists but cannot be read is a stop: a trainer that cannot tell whether it
    was asked to stop should stop.
    """
    try:
        return parse(path.read_text())
    except FileNotFoundError:
        return None
    except OSError:
        return StopRequest(now=True)


def write(path: Path, request: StopRequest) -> None:
    """Queue a stop. Written whole, so a reader never sees half a number."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(request.text())
    tmp.replace(path)


def reached(request: StopRequest | None, step: int, now: float | None = None) -> str | None:
    """Why this step should be the last one, or None to keep going.

    The string is printed and recorded, so it says which of the three forms fired -- "STOP
    file" and "reached stop time 12:35" send you to very different places when you are
    working out why a run ended earlier than you expected.
    """
    if request is None:
        return None
    if request.now:
        return "STOP file"
    if request.step is not None and step >= request.step:
        return f"STOP file asked for step {request.step}"
    if request.deadline is not None and (now if now is not None else time.time()) >= request.deadline:
        when = datetime.fromtimestamp(request.deadline).strftime("%H:%M")
        return f"reached stop time {when}"
    return None
