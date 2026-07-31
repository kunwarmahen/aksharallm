"""Tests for the portal's Quantize panel.

The panel starts a subprocess that can allocate gigabytes and hold the GPU, driven by
whatever the browser posts. So these tests are mostly about the *refusals*: an already
quantized checkpoint, a second concurrent job, a bad method, and — the one with real
consequences — quietly moving to the CPU when a training run owns the card.

Nothing here launches a real job; `start()` is exercised with a stubbed Popen so the
command line it would run is asserted directly. That command line is the actual contract
between the panel and the CLI.
"""

import json
from pathlib import Path

import pytest

from aksharallm.portal.quantize import METHODS, QuantJobs
from aksharallm.portal.runs import RunError


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "checkpoints" / "tiny").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "configs").mkdir()
    return tmp_path


def _ckpt(repo: Path, run: str, name: str, step: int = 100):
    import torch

    d = repo / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    torch.save({
        "model": {"tok_emb.weight": torch.zeros(8, 4)},
        "model_config": {"vocab_size": 8, "d_model": 4, "n_layers": 1, "n_heads": 2,
                         "max_seq_len": 8, "tie_embeddings": True},
        "config": {"data": {"tokenizer": "t.json", "val_bin": "v.bin"},
                   "train": {"seq_len": 8}},
        "step": step, "best_val": 1.5,
    }, path)
    return path


class FakeProc:
    def __init__(self, pid=424242):
        self.pid = pid


@pytest.fixture
def spy_popen(monkeypatch):
    """Capture the command instead of running it."""
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return FakeProc()

    monkeypatch.setattr("aksharallm.portal.quantize.subprocess.Popen", fake)
    return seen


# ---- listing ------------------------------------------------------------------------

def test_lists_float_checkpoints_as_quantizable(repo):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    cks = q.checkpoints()
    assert len(cks) == 1
    assert cks[0]["can_quantize"] and not cks[0]["quantized"]


def test_quantized_checkpoints_are_listed_but_not_offered_as_sources(repo):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _ckpt(repo, "tiny", "ckpt_best-gptq-int4-g64-asym.pt")
    q = QuantJobs(repo)
    by_name = {c["name"]: c for c in q.checkpoints()}
    assert by_name["ckpt_best.pt"]["can_quantize"]
    assert by_name["ckpt_best-gptq-int4-g64-asym.pt"]["quantized"]
    assert not by_name["ckpt_best-gptq-int4-g64-asym.pt"]["can_quantize"]


# ---- device policy -------------------------------------------------------------------

def test_defaults_to_the_gpu_when_nothing_is_training(repo):
    q = QuantJobs(repo)
    plan = q.plan_device()
    assert plan["device"] == "cuda" and plan["training"] == []


def test_falls_back_to_cpu_while_a_run_is_training(repo, monkeypatch):
    """The consequential one. A GPTQ job allocates over a gigabyte of Hessians; taking the
    card out from under a training run loses hours of work, and the run is the thing that
    matters. Slower is the correct trade."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    monkeypatch.setattr(q, "training", lambda: ["small-code"])
    plan = q.plan_device()
    assert plan["device"] == "cpu"
    assert "small-code" in plan["reason"]


def test_explicit_device_overrides_the_fallback(repo, monkeypatch):
    q = QuantJobs(repo)
    monkeypatch.setattr(q, "training", lambda: ["small-code"])
    plan = q.plan_device("cuda")
    assert plan["device"] == "cuda" and plan["forced"]
    assert "training" in plan["reason"]     # still warned about


# ---- refusals ------------------------------------------------------------------------

def test_refuses_an_unknown_checkpoint(repo):
    with pytest.raises(RunError, match="unknown checkpoint"):
        QuantJobs(repo).start({"checkpoint": "nope/nope.pt"})


def test_refuses_to_quantize_a_quantized_checkpoint(repo):
    """Compounding the error would look like a much worse method rather than a mistake."""
    _ckpt(repo, "tiny", "ckpt_best-rtn-int4-g64-asym.pt")
    q = QuantJobs(repo)
    with pytest.raises(RunError, match="already quantized"):
        q.start({"checkpoint": "tiny/ckpt_best-rtn-int4-g64-asym.pt"})


def test_refuses_an_unknown_method(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    with pytest.raises(RunError, match="unknown method"):
        QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "method": "magic"})


def test_refuses_bad_bit_widths(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    with pytest.raises(RunError, match="bits must be"):
        QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "bits": 3})


def test_refuses_a_second_concurrent_job(repo, spy_popen, monkeypatch):
    """Two GPTQ jobs would fight over the card and both finish slower than running them
    in sequence."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt"})
    monkeypatch.setattr(q, "_pid", lambda: 424242)
    with pytest.raises(RunError, match="already running"):
        q.start({"checkpoint": "tiny/ckpt_best.pt"})


def test_stop_refuses_when_nothing_is_running(repo):
    with pytest.raises(RunError, match="no quantization job"):
        QuantJobs(repo).stop()


def test_only_qat_can_be_stopped_at_a_step_or_a_time(repo, spy_popen, monkeypatch):
    """RTN, GPTQ and AWQ are one pass over the weights. Offering "stop in 10 minutes" for
    them would be a promise about a loop that has no steps to check the clock between."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt", "method": "gptq"})
    monkeypatch.setattr(q, "_pid", lambda: 424242)
    assert q.can_bound() is False
    with pytest.raises(RunError, match="single pass"):
        q.stop("in", seconds=600)


def test_a_qat_job_takes_a_stop_file_and_honours_a_bounded_stop(repo, spy_popen, monkeypatch):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt", "method": "qat", "qat_steps": 800})
    assert f"--stop-file {q.stop_file}" in " ".join(spy_popen["cmd"])
    monkeypatch.setattr(q, "_pid", lambda: 424242)
    assert q.can_bound() is True
    q.stop("at", steps=300)
    assert q.stop_file.read_text() == "300"
    assert q.stop_request()["target"] == 300


# ---- the command line it builds ------------------------------------------------------

def test_builds_the_cli_command_the_docs_describe(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt", "method": "gptq", "bits": 4,
             "group": 64, "bench": True, "save": True, "device": "cpu"})
    cmd = spy_popen["cmd"]
    assert "aksharallm.quant" in cmd
    assert "--method" in cmd and cmd[cmd.index("--method") + 1] == "gptq"
    assert cmd[cmd.index("--bits") + 1] == "4"
    assert cmd[cmd.index("--group") + 1] == "64"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert "--bench" in cmd
    assert "--no-save" not in cmd          # save was requested
    assert "--json" in cmd


def test_not_saving_passes_no_save(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "save": False})
    assert "--no-save" in spy_popen["cmd"]


def test_compare_ignores_the_single_scheme_flags(repo, spy_popen):
    """`--compare` runs its own list; passing --method alongside it would be misleading."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "compare": True,
                           "method": "gptq"})
    cmd = spy_popen["cmd"]
    assert "--compare" in cmd
    assert "--method" not in cmd and "--bits" not in cmd


def test_qat_passes_its_step_budget(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "method": "qat",
                           "qat_steps": 300})
    cmd = spy_popen["cmd"]
    assert cmd[cmd.index("--qat-steps") + 1] == "300"


def test_job_runs_detached_from_the_portal(repo, spy_popen):
    """The portal must survive its jobs, and its jobs must survive the portal — the
    scheduler that starts training at 22:00 lives in this process."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    QuantJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt"})
    assert spy_popen["kw"]["start_new_session"] is True


# ---- status and results --------------------------------------------------------------

def test_status_reports_the_running_job(repo, spy_popen, monkeypatch):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt", "method": "rtn"})
    monkeypatch.setattr(q, "_pid", lambda: 424242)
    st = q.status()
    assert st["running"] and st["current"]["method"] == "rtn"
    assert st["current"]["checkpoint"] == "tiny/ckpt_best.pt"


def test_a_dead_job_with_results_reads_as_done(repo, spy_popen):
    """The CLI does not write an 'I finished' marker, so completion is inferred from the
    artifact. Without this a finished job would show as running forever."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt"})
    job = json.loads(q.current_file.read_text())["job"]
    q.json_path(job).write_text(json.dumps({"checkpoint": "x", "bench": []}))
    st = q.status()
    assert not st["running"]
    assert st["current"]["state"] == "done"


def test_a_dead_job_without_results_reads_as_failed(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt"})
    st = q.status()
    assert st["current"]["state"] == "failed"


def test_results_are_read_back_from_the_cli_json(repo):
    """A job run in a terminal with --json into logs/quant shows up in the panel too —
    the panel is a view over the CLI's output, not a separate record."""
    q = QuantJobs(repo)
    q.dir.mkdir(parents=True)
    (q.dir / "20260730-120000-tiny-rtn-int4.json").write_text(json.dumps({
        "checkpoint": "checkpoints/tiny/ckpt_best.pt", "device": "cuda",
        "bench": [{"label": "bf16 (baseline)", "nbytes": 100, "perplexity": 4.3},
                  {"label": "rtn-int4-g64-asym", "nbytes": 40, "perplexity": 4.4}],
    }))
    rows = q.results()
    assert len(rows) == 1
    assert len(rows[0]["bench"]) == 2
    assert rows[0]["device"] == "cuda"


def test_current_json_is_not_mistaken_for_a_result(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    q = QuantJobs(repo)
    q.start({"checkpoint": "tiny/ckpt_best.pt"})
    assert q.current_file.exists()
    assert all(r["job"] != "current" for r in q.results())


def test_a_corrupt_result_file_does_not_break_the_panel(repo):
    q = QuantJobs(repo)
    q.dir.mkdir(parents=True)
    (q.dir / "20260730-120000-broken.json").write_text("{not json")
    assert q.results() == []


def test_status_offers_every_method_and_group(repo):
    st = QuantJobs(repo).status()
    assert {m["id"] for m in st["methods"]} == set(METHODS)
    assert [g["value"] for g in st["groups"]] == [64, 128, -1]
    # 128 must be labelled with the d_ff caveat, or someone will pick it and wonder.
    g128 = next(g for g in st["groups"] if g["value"] == 128)
    assert "2752" in g128["label"]
