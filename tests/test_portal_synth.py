"""Tests for the portal's Synth panel.

Like the quantize and eval panels, this one starts a subprocess from whatever the browser
posted, so the tests are mostly about the *refusals* and about the exact command line — that
command line is the contract between the browser and `python -m aksharallm.synth`, and if it
drifts, a job started in the browser stops meaning what a job started in a terminal means.

Nothing here runs a real generation: `Popen` is stubbed. Nothing here talks to Ollama
either — the teacher listing is stubbed, because "Ollama is not running" must degrade the
panel to a warning rather than an exception.
"""

import json
from pathlib import Path

import pytest

from aksharallm.portal.runs import RunError
from aksharallm.portal.synth import SynthJobs


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


class FakeProc:
    def __init__(self, pid=515151):
        self.pid = pid


@pytest.fixture
def spy_popen(monkeypatch):
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return FakeProc()

    monkeypatch.setattr("aksharallm.portal.synth.subprocess.Popen", fake)
    return seen


@pytest.fixture
def no_ollama(monkeypatch):
    """Ollama is not running — the common case on a machine that is only training."""
    def boom(self):
        raise RunError("cannot reach Ollama at http://127.0.0.1:11434")

    monkeypatch.setattr("aksharallm.portal.synth.Ollama.models", boom)


# ---- the command line ------------------------------------------------------------------

def test_builds_the_cli_command_a_person_would_type(repo, spy_popen):
    jobs = SynthJobs(repo)
    res = jobs.start({"recipe": "python", "name": "py-v1", "n": 40, "teacher": "coder:3b"})
    cmd = spy_popen["cmd"]
    assert cmd[2:6] == ["-m", "aksharallm.synth", "gen", "python"]
    assert "--name" in cmd and cmd[cmd.index("--name") + 1] == "py-v1"
    assert cmd[cmd.index("--n") + 1] == "40"
    assert cmd[cmd.index("--teacher") + 1] == "coder:3b"
    # Always passed: it is how the browser's Stop button reaches the running job.
    assert cmd[cmd.index("--stop-file") + 1] == str(jobs.stop_file)
    assert res["state"] == "running" and res["dataset"] == "py-v1"


def test_a_time_budget_becomes_stop_in(repo, spy_popen):
    SynthJobs(repo).start({"recipe": "chat", "name": "chat-v1", "n": 10,
                           "teacher": "big:31b", "stop_in_s": 1800})
    cmd = spy_popen["cmd"]
    assert cmd[cmd.index("--stop-in") + 1] == "1800s"


def test_turning_verification_off_is_passed_through(repo, spy_popen):
    SynthJobs(repo).start({"recipe": "python", "name": "py-fast", "n": 5,
                           "teacher": "coder:3b", "no_verify": True})
    assert "--no-verify" in spy_popen["cmd"]


def test_the_default_teacher_is_the_recipes_own(repo, spy_popen):
    """Not the section default: a code model for code and a big model for chat differ by an
    order of magnitude in quality-per-hour, in opposite directions."""
    jobs = SynthJobs(repo)
    jobs.start({"recipe": "python", "name": "py-d", "n": 1})
    assert spy_popen["cmd"][spy_popen["cmd"].index("--teacher") + 1] == "qwen2.5:14b"
    jobs.pid_file.unlink()
    jobs.start({"recipe": "chat", "name": "chat-d", "n": 1})
    assert spy_popen["cmd"][spy_popen["cmd"].index("--teacher") + 1] == "gemma4:31b"


def test_job_runs_detached_so_the_portal_can_restart_under_it(repo, spy_popen):
    SynthJobs(repo).start({"recipe": "chat", "name": "c", "n": 1, "teacher": "m:1b"})
    assert spy_popen["kw"]["start_new_session"] is True
    assert spy_popen["kw"]["stdin"] is not None


# ---- refusals ---------------------------------------------------------------------------

def test_refuses_an_unknown_recipe(repo, spy_popen):
    with pytest.raises(RunError):
        SynthJobs(repo).start({"recipe": "poetry", "name": "x", "n": 1})


def test_refuses_a_name_that_is_not_one_path_segment(repo, spy_popen):
    """The name becomes a directory *and* reaches a command line."""
    for bad in ("../etc", "a/b", "", "-rf", "x;rm -rf /"):
        with pytest.raises(RunError):
            SynthJobs(repo).start({"recipe": "chat", "name": bad, "n": 1})


def test_refuses_a_model_name_that_is_not_a_model_name(repo, spy_popen):
    with pytest.raises(RunError):
        SynthJobs(repo).start({"recipe": "chat", "name": "c", "n": 1,
                               "teacher": "m:1b; rm -rf /"})


def test_refuses_mixing_two_recipes_in_one_dataset(repo, spy_popen):
    """A dataset is one recipe: two sample shapes in one file cannot be exported."""
    from aksharallm.synth.dataset import Dataset

    ds = Dataset("mixed", root=repo)
    ds.open("chat", "m:1b", "h", {}, 1)
    with pytest.raises(RunError) as exc:
        SynthJobs(repo).start({"recipe": "python", "name": "mixed", "n": 1})
    assert "one recipe" in str(exc.value)


def test_refuses_a_second_concurrent_job(repo, spy_popen, monkeypatch):
    jobs = SynthJobs(repo)
    jobs.start({"recipe": "chat", "name": "c", "n": 1, "teacher": "m:1b"})
    monkeypatch.setattr("aksharallm.portal.synth._alive", lambda pid: True)
    monkeypatch.setattr("aksharallm.portal.synth._cmdline", lambda pid: "python -m aksharallm.synth gen")
    with pytest.raises(RunError):
        jobs.start({"recipe": "chat", "name": "d", "n": 1, "teacher": "m:1b"})


def test_refuses_an_absurd_sample_count(repo, spy_popen):
    with pytest.raises(RunError):
        SynthJobs(repo).start({"recipe": "chat", "name": "c", "n": 0})


# ---- stopping -----------------------------------------------------------------------------

def test_stop_refuses_when_nothing_is_running(repo):
    with pytest.raises(RunError):
        SynthJobs(repo).stop()


def test_every_stop_mode_is_bounded_here_unlike_quantization(repo, spy_popen, monkeypatch):
    """A generation run stopped halfway is a smaller *complete* dataset, so all three
    bounded stops mean something — which is not true of a single-pass quantization."""
    jobs = SynthJobs(repo)
    jobs.start({"recipe": "python", "name": "p", "n": 100, "teacher": "m:1b"})
    monkeypatch.setattr("aksharallm.portal.synth._alive", lambda pid: True)
    monkeypatch.setattr("aksharallm.portal.synth._cmdline", lambda pid: "aksharallm.synth")
    monkeypatch.setattr("aksharallm.portal.synth.os.kill", lambda pid, sig: None)

    jobs.stop(mode="at", samples=40)
    assert jobs.stop_file.read_text() == "40"
    jobs.stop(mode="in", seconds=600)
    assert jobs.stop_file.read_text().startswith("@")
    jobs.stop(mode="cancel")
    assert not jobs.stop_file.exists()
    jobs.stop(mode="now")
    assert jobs.stop_file.read_text() == ""


def test_a_stale_stop_file_is_cleared_before_a_new_job(repo, spy_popen):
    """A leftover STOP would end the new job at its first sample — the same trap the
    quantize panel had to fix."""
    jobs = SynthJobs(repo)
    jobs.dir.mkdir(parents=True, exist_ok=True)
    jobs.stop_file.write_text("")
    jobs.start({"recipe": "chat", "name": "c", "n": 5, "teacher": "m:1b"})
    assert not jobs.stop_file.exists()


# ---- status --------------------------------------------------------------------------------

def test_status_survives_ollama_being_down(repo, no_ollama):
    """The datasets, their funnels and their samples are all worth reading on a machine
    where the teacher is not running."""
    st = SynthJobs(repo).status()
    assert st["running"] is False
    assert st["teachers"]["error"]
    assert st["teachers"]["defaults"]["python"] == "qwen2.5:14b"
    assert [r["name"] for r in st["recipes"]] == ["python", "chat", "preference"]


def test_progress_is_parsed_from_the_jobs_own_output(repo):
    """The bar in the browser can only show a number the job printed."""
    jobs = SynthJobs(repo)
    got = jobs._progress([
        "[synth] python 3/50 (6%) · 9 asked · pass 33% · 5.1s/sample · eta 4m",
        "[synth] python 4/50 (8%) · 11 asked · pass 36% · 5.0s/sample · eta 4m"])
    assert got["kept"] == 4 and got["total"] == 50 and got["asked"] == 11
    assert got["pass_rate"] == 0.36


def test_a_finished_job_reads_as_done_from_its_last_line(repo, spy_popen, monkeypatch):
    jobs = SynthJobs(repo)
    cur = jobs.start({"recipe": "chat", "name": "c", "n": 1, "teacher": "m:1b"})
    jobs.log_path(cur["job"]).write_text("[synth] chat 1/1 (100%) · 1 asked\n"
                                         "  stopped: done\n")
    monkeypatch.setattr("aksharallm.portal.synth.Ollama.models", lambda self: [])
    st = jobs.status()
    assert st["running"] is False and st["current"]["state"] == "done"


def test_a_job_that_died_reads_as_failed(repo, spy_popen, monkeypatch):
    jobs = SynthJobs(repo)
    cur = jobs.start({"recipe": "chat", "name": "c", "n": 1, "teacher": "m:1b"})
    jobs.log_path(cur["job"]).write_text("Traceback (most recent call last):\n")
    monkeypatch.setattr("aksharallm.portal.synth.Ollama.models", lambda self: [])
    assert jobs.status()["current"]["state"] == "failed"


def test_contention_is_reported_not_silently_resolved(repo, monkeypatch):
    """Unlike the Playground and Quantize panels, this one cannot fall back to the CPU: the
    teacher is loaded by Ollama in another process. So it warns, and the choice stays with
    the person who can see the whole machine."""
    monkeypatch.setattr("aksharallm.portal.runs._alive", lambda pid: True)
    run = repo / "checkpoints" / "small-code"
    run.mkdir(parents=True)
    (run / "train.pid").write_text("321")
    (run / "ckpt_last.pt").write_bytes(b"x")
    jobs = SynthJobs(repo)
    assert jobs.contention("gemma4:31b")["safe"] is False
    assert jobs.contention("starcoder2:3b")["safe"] is True


# ---- datasets --------------------------------------------------------------------------------

def test_dataset_returns_kept_and_rejected_together(repo):
    from aksharallm.synth.dataset import Dataset

    ds = Dataset("py-x", root=repo)
    ds.open("python", "m:1b", "h", {}, 1)
    ds.append({"kind": "python", "id": "a", "problem": "p", "solution": "s", "tests": "t"})
    ds.reject("tests_failed", "seed-1", detail="AssertionError", text="### PROBLEM…")
    ds.save()

    got = SynthJobs(repo).dataset("py-x")
    assert got["kept"] == 1
    assert got["rejected"]["tests_failed"] == 1
    assert got["samples"][0]["id"] == "a"
    assert got["rejects"][0]["reason"] == "tests_failed"


def test_dataset_refuses_a_path_instead_of_a_name(repo):
    with pytest.raises(RunError):
        SynthJobs(repo).dataset("../../etc/passwd")


def test_export_hands_back_the_tokenizing_command(repo):
    from aksharallm.synth.dataset import Dataset

    ds = Dataset("py-e", root=repo)
    ds.open("python", "m:1b", "h", {}, 1)
    ds.append({"kind": "python", "id": "a", "problem": "Write a thing.",
               "solution": "def f():\n    return 1", "tests": "assert f() == 1"})
    ds.save()
    out = SynthJobs(repo).export("py-e")
    assert out["rows"] == 1 and out["consumer"] == "sft"
    assert "prepare_sft" in out["next"]
    rows = [json.loads(line) for line in Path(out["path"]).read_text().splitlines()]
    assert rows[0]["messages"][0]["role"] == "user"
