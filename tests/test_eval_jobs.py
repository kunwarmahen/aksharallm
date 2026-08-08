"""A terminal-started evaluation has to be visible in the portal.

The standing rule is that the browser and the terminal never disagree. Evaluation only half
held it up: the *result* appeared in the Eval tab whichever way it was started, because both
write into `logs/eval/`, but the *running* state was published only by the portal's own
launcher. So a `python -m aksharallm.eval` in a terminal left the tab saying "nothing
running", with a Start button beside it and no progress at all.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import os

import pytest

from aksharallm.eval.jobs import announced
from aksharallm.portal.evals import EvalJobs


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "logs" / "eval").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    return tmp_path


def test_the_portal_sees_a_terminal_job_while_it_runs(repo):
    """The whole point: `running` is true and `current` describes the work, for a job the
    portal did not launch."""
    jobs = EvalJobs(repo)
    assert jobs.status()["running"] is False

    with announced("run", {"checkpoint": "small-code/ckpt_best.pt", "label": "base"},
                   root=repo) as job:
        assert job
        st = jobs.status()
        assert st["running"] is True
        assert st["pid"] == os.getpid()
        cur = st["current"]
        assert cur["kind"] == "run" and cur["state"] == "running"
        assert cur["checkpoint"] == "small-code/ckpt_best.pt"
        assert cur["source"] == "terminal", "the tab should say where the job came from"


def test_the_claim_is_released_and_the_ending_recorded(repo):
    """A finished job must not leave the tab claiming something is running, and must say
    how it ended rather than leaving the portal to guess from an artifact whose name a
    terminal job does not control."""
    with announced("domains", {"checkpoint": "tiny/ckpt_best.pt"}, root=repo):
        pass
    assert not (repo / "logs" / "eval" / "eval.pid").exists()
    st = EvalJobs(repo).status()
    assert st["running"] is False
    assert st["current"]["state"] == "done"


def test_a_crash_is_recorded_as_failed_not_as_still_running(repo):
    with pytest.raises(RuntimeError):
        with announced("calibrate", {}, root=repo):
            raise RuntimeError("boom")
    state = json.loads((repo / "logs" / "eval" / "current.json").read_text())
    assert state["state"] == "failed"
    assert not (repo / "logs" / "eval" / "eval.pid").exists()


def test_it_refuses_to_steal_a_live_jobs_slot(repo):
    """Two evaluations at once must not make the portal describe the wrong one. The second
    runs unannounced rather than overwriting the first's state."""
    with announced("run", {"label": "first"}, root=repo) as first:
        assert first
        with announced("contaminate", {"label": "second"}, root=repo) as second:
            assert second is None, "the second job must not publish"
        state = json.loads((repo / "logs" / "eval" / "current.json").read_text())
        assert state["label"] == "first" and state["kind"] == "run"


def test_a_stale_pid_does_not_block_a_new_job(repo):
    """A `kill -9` leaves the file behind. A dead pid must not lock the portal out of ever
    showing a terminal job again — liveness is checked, not mere existence."""
    (repo / "logs" / "eval" / "eval.pid").write_text("999999\n")
    with announced("run", {"label": "after-a-crash"}, root=repo) as job:
        assert job is not None
        assert EvalJobs(repo).status()["running"] is True


def test_the_terminals_output_reaches_the_portals_log(repo, capsys):
    """The tab tails `logs/eval/<job>.log`. A portal job gets that from Popen; a terminal
    job has to tee it, or the browser knows a job exists and can show nothing about it —
    and the terminal must still see its own output."""
    with announced("run", {}, root=repo) as job:
        print("[eval] mmlu 40/160 (25%)")
    assert "[eval] mmlu 40/160 (25%)" in capsys.readouterr().out
    assert "40/160" in (repo / "logs" / "eval" / f"{job}.log").read_text()
