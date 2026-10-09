"""portal/queues.py: the Schedule panel's view of the night queues in logs/queue/*.sh.

The panel showed "nothing scheduled" while a queue trained every night, because the portal
only knew about schedule.json rules. These pin what it reads instead -- and above all that a
dead queue is never reported as the "sleeping until 00:30" its log ended on.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from aksharallm.portal.queues import queues

SCRIPT = """#!/usr/bin/env bash
DONE=logs/queue/{name}
START=${{START:-00:30}}
END=${{END:-05:30}}
TRAIN_END=${{TRAIN_END:-05:00}}
sleep 30
"""


def make(root: Path, name: str, log: str, done=()) -> Path:
    q = root / "logs" / "queue"
    (q / name).mkdir(parents=True, exist_ok=True)
    script = q / f"{name}.sh"
    script.write_text(SCRIPT.format(name=name))
    (q / f"{name}.log").write_text(log)
    for d in done:
        (q / name / f"{d}.done").touch()
    return script


LOG = ("=== 2026-10-09 05:00:18 small-code-py stopped for the night\n"
       "[eval] humaneval 3/164 (2%)\n"
       "=== 2026-10-09 05:36:38 outside the window; sleeping until Sat 00:30\n")


def test_a_dead_queue_is_not_reported_as_sleeping(tmp_path):
    make(tmp_path, "py-nights", LOG, done=["train-2026-10-09"])
    (q,) = queues(tmp_path)
    assert not q["alive"] and q["state"] == "not running"
    assert q["next_window"] is None, "a dead queue has no next window"
    assert q["done"] == ["train-2026-10-09"]
    assert (q["start"], q["end"], q["train_end"]) == ("00:30", "05:30", "05:00")


def test_a_live_queue_reports_its_last_line_and_next_window(tmp_path):
    script = make(tmp_path, "py-nights", LOG)
    proc = subprocess.Popen(["bash", "logs/queue/py-nights.sh"], cwd=tmp_path)
    try:
        time.sleep(0.3)
        (q,) = queues(tmp_path, now=datetime(2026, 10, 9, 6, 43))
        assert q["alive"] and q["pid"] == proc.pid
        assert q["state"] == "outside the window; sleeping until Sat 00:30"
        assert q["next_window_in_s"] == (17 * 60 + 47) * 60   # 06:43 -> 00:30 tomorrow
    finally:
        proc.kill()
        proc.wait()
    assert script.exists()


def test_one_queue_name_does_not_match_another(tmp_path):
    """`nights.sh` must not be "alive" because `py-nights.sh` is running."""
    make(tmp_path, "nights", "=== 2026-10-06 01:24:10 ALL DONE\n", done=["ALL", "x"])
    make(tmp_path, "py-nights", LOG)
    proc = subprocess.Popen(["bash", "logs/queue/py-nights.sh"], cwd=tmp_path)
    try:
        time.sleep(0.3)
        by = {q["name"]: q for q in queues(tmp_path)}
        assert by["py-nights"]["alive"] and not by["nights"]["alive"]
        assert by["nights"]["state"] == "finished" and by["nights"]["done"] == ["x"]
        assert list(by) == ["py-nights", "nights"], "live queues are listed first"
    finally:
        proc.kill()
        proc.wait()


def test_a_script_that_never_ran_is_not_listed(tmp_path):
    q = tmp_path / "logs" / "queue"
    q.mkdir(parents=True)
    (q / "draft.sh").write_text(SCRIPT.format(name="draft"))
    assert queues(tmp_path) == []
