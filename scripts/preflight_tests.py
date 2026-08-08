#!/usr/bin/env python
"""Run the test suite as a launch gate, and *show that it is moving*.

The gate itself is not in question: a six-day run should not start on code that does not
pass its own tests, and it has caught real breakage. What was wrong was the display.
`pytest -q` prints one dot per test, so 1,250 tests are 1,250 bare dots over ninety silent
seconds with no file names, no counts and no indication of how far along it is. That reads
exactly like a hang, and the reasonable response to a hang is to cancel it — which is what
happened here, twice in one evening, to a launch that was working perfectly.

So: **one line per test file**, printed as each file finishes, with a running percentage.
Same suite, same gate, same exit code; it just says what it is doing.

    tests/test_audio.py                       29 passed        [  2%]
    tests/test_audiolm.py                     29 passed        [  5%]
    tests/test_codec.py                       41 passed        [  8%]

Quiet when green, loud when red: on failure the raw pytest output is replayed in full, so
nothing is hidden by the summarising. The tracebacks are the reason you ran it.

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter

#: `pytest -v` writes one of these per test. The percentage is pytest's own, so it stays
#: correct however the suite is filtered or reordered.
LINE = re.compile(
    r"^(?P<file>\S+\.py)::(?P<node>\S+)\s+(?P<verdict>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r".*?(?:\[\s*(?P<pct>\d+)%\])?\s*$"
)
#: Verdicts that mean the gate should not open. SKIPPED is deliberately not one of them —
#: but see the note in `summarise`, because a suite that is mostly skips is its own problem.
BAD = {"FAILED", "ERROR"}


def summarise(counts: Counter) -> str:
    """`29 passed`, or `27 passed, 2 skipped` — in a fixed order, so columns line up."""
    parts = []
    for verdict, label in (("PASSED", "passed"), ("FAILED", "failed"), ("ERROR", "error"),
                           ("SKIPPED", "skipped"), ("XFAIL", "xfail"), ("XPASS", "xpass")):
        if counts[verdict]:
            parts.append(f"{counts[verdict]} {label}")
    return ", ".join(parts) or "no tests"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["tests/"])
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    cmd = [args.python, "-m", "pytest", *(args.paths or ["tests/"]), "-v", "--no-header",
           "--tb=short", "-p", "no:cacheprovider"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    raw: list[str] = []
    current: str | None = None
    counts: Counter = Counter()
    totals: Counter = Counter()
    failures: list[str] = []

    def flush(pct: str | None):
        if current is None:
            return
        # Padded to a fixed width so the verdicts form a column and a slow file is obvious
        # by where the line stops rather than by reading it.
        print(f"    {current:<44}{summarise(counts):<22}"
              f"{f'[{int(pct):>3}%]' if pct else ''}", flush=True)

    pct = None
    for line in proc.stdout:
        raw.append(line)
        m = LINE.match(line.rstrip())
        if not m:
            continue
        if m["file"] != current:
            flush(pct)
            current, counts = m["file"], Counter()
        counts[m["verdict"]] += 1
        totals[m["verdict"]] += 1
        pct = m["pct"] or pct
        if m["verdict"] in BAD:
            failures.append(f"{m['file']}::{m['node']}")
    flush(pct)
    code = proc.wait()

    print(f"\n    {summarise(totals)}")
    if code != 0:
        # Everything, verbatim. The summarising above exists to make a *passing* run
        # legible; a failing one is the case where you want the raw output.
        print("\n" + "=" * 78)
        print("the suite did not pass — full output follows")
        print("=" * 78)
        sys.stdout.writelines(raw)
        if failures:
            print(f"\n{len(failures)} failing node(s):")
            for node in failures[:40]:
                print(f"    {node}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
