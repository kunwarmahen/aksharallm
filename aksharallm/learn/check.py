"""Running a lesson's check: one pytest node, and an answer a person can read.

The check is deliberately the *existing* test suite rather than anything written for the
lessons. Two reasons, and the second is the important one:

* a lesson's exercise is to break real code, so the thing that should notice is the test that
  already guards that code;
* a bespoke checker would be a second copy of the project's assumptions, free to drift from
  the first. There is nothing to keep in sync here — `verify` is a node id, and if it stops
  existing the suite says so.

What comes back matters as much as pass/fail. `pytest -q` on a failure prints the assertion,
the values on both sides, and the test's own docstring — which in this repo is usually a
sentence explaining *why* the thing being asserted matters. That is the lesson, so the last
lines of it are handed back rather than a red X.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: A node id reaches a command line. It is matched, not escaped: `tests/x.py::test_y[param]`
#: and nothing else — no shell metacharacters, no paths outside `tests/`.
NODE_RE = re.compile(r"^tests/[A-Za-z0-9_./-]+\.py(::[A-Za-z0-9_\[\]{}.,:<>+/= -]+)?$")

#: Long enough for the slowest single test here (a few seconds), short enough that a hung
#: check does not hold a browser request open forever.
TIMEOUT_S = 300


class CheckError(Exception):
    pass


@dataclass
class CheckResult:
    node: str
    passed: bool
    duration_s: float
    summary: str
    output: str

    def as_dict(self) -> dict:
        return {"node": self.node, "passed": self.passed,
                "duration_s": round(self.duration_s, 2),
                "summary": self.summary, "output": self.output}


def run(node: str, root: Path | None = None, timeout_s: int = TIMEOUT_S) -> CheckResult:
    """Run one pytest node id from the repo root."""
    from ..portal.runs import repo_root

    node = (node or "").strip()
    if not NODE_RE.match(node):
        raise CheckError(f"not a pytest node id: {node!r} — expected "
                         "tests/<file>.py::<test name>")
    base = Path(root) if root else repo_root()
    if not (base / node.split("::")[0]).exists():
        raise CheckError(f"{node.split('::')[0]} does not exist — this lesson has drifted "
                         "from the code it describes.")

    t0 = time.monotonic()
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", node, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=base, capture_output=True, text=True, timeout=timeout_s,
            stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return CheckResult(node, False, timeout_s,
                           f"the check was still running after {timeout_s}s and was stopped",
                           "")
    duration = time.monotonic() - t0
    out = (done.stdout or "") + (done.stderr or "")
    passed = done.returncode == 0
    return CheckResult(node, passed, duration, summarise(out, passed), tail(out))


def summarise(output: str, passed: bool) -> str:
    """One line: pytest's own summary, or the assertion that failed.

    The assertion is worth more than the count. "1 failed" tells you the exercise worked;
    `assert 0.51 == approx(1.0)` tells you *what the broken code now believes*, which is the
    thing the lesson is about.
    """
    if passed:
        found = re.search(r"^(\d+ passed.*)$", output, re.MULTILINE)
        return found.group(1).strip() if found else "passed"
    for line in output.splitlines():
        if line.startswith("E   ") and line.strip() != "E":
            return line[4:].strip()[:300]
    found = re.search(r"^(\d+ failed.*)$", output, re.MULTILINE)
    if found:
        return found.group(1).strip()
    return "the check did not pass"


def tail(output: str, lines: int = 60) -> str:
    kept = [ln for ln in output.splitlines() if ln.strip()]
    return "\n".join(kept[-lines:])


def collectable(node: str, root: Path | None = None, timeout_s: int = 120) -> bool:
    """Does this node id still exist? The drift check, without running anything.

    Used by `tests/test_lessons.py` over every lesson at once — one collection pass for the
    whole path rather than one per lesson, which turns a rename of a test into a failing
    suite instead of a lesson that mysteriously cannot be completed.
    """
    from ..portal.runs import repo_root

    if not NODE_RE.match((node or "").strip()):
        return False
    done = subprocess.run(
        [sys.executable, "-m", "pytest", node, "--collect-only", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=Path(root) if root else repo_root(), capture_output=True, text=True,
        timeout=timeout_s, stdin=subprocess.DEVNULL)
    return done.returncode == 0
