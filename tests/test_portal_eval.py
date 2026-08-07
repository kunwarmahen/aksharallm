"""Tests for the portal's Eval panel.

Same shape as the Quantize panel's tests, and mostly for the same reason: the panel builds
a command line out of whatever the browser posts, and that command line is the contract
between the tab and the CLI. Nothing here launches a real job — `Popen` is stubbed and the
command is asserted directly.

The refusals matter more than the happy path. A job started with data that is not
downloaded fails ten minutes later inside a loader; the panel has to say so before the
button does anything.
"""

import json
from pathlib import Path

import pytest

from aksharallm.portal.evals import EvalJobs
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
        "config": {"data": {"tokenizer": "t.json", "val_bin": "data/val.bin"},
                   "train": {"seq_len": 8}},
        "step": step, "best_val": 1.5,
    }, path)
    return path


def _cache(repo: Path, *names: str):
    d = repo / "data" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / f"{name}.jsonl").write_text('{"a": 1}\n')
        (d / f"{name}.meta.json").write_text(json.dumps({"name": name, "rows": 1}))


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

    monkeypatch.setattr("aksharallm.portal.evals.subprocess.Popen", fake)
    return seen


# ---- listing --------------------------------------------------------------------------

def test_lists_every_checkpoint_including_quantized_ones(repo):
    """Unlike the Quantize panel, nothing is excluded here. Evaluating a quantized
    checkpoint is the *point* — "what did int4 cost on MMLU" is the question perplexity
    cannot answer."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _ckpt(repo, "tiny", "ckpt_best-gptq-int4-g64-asym.pt")
    names = {c["name"] for c in EvalJobs(repo).checkpoints()}
    assert names == {"ckpt_best.pt", "ckpt_best-gptq-int4-g64-asym.pt"}


def test_status_carries_the_suite_catalogue_and_the_data_state(repo):
    _cache(repo, "piqa")
    st = EvalJobs(repo).status()
    assert {s["name"] for s in st["suites"]} >= {"mmlu", "gsm8k", "humaneval", "judge"}
    assert st["groups"]["fast"]
    by_name = {d["name"]: d for d in st["datasets"]}
    assert by_name["piqa"]["cached"] and not by_name["mmlu"]["cached"]
    assert st["running"] is False


def test_every_suite_offered_to_the_browser_carries_its_expect_line(repo):
    """The tab prints it under each suite. It is the sentence that stops a reader
    concluding a 25% MMLU means the model is broken."""
    for suite in EvalJobs(repo).status()["suites"]:
        assert suite["expect"]


# ---- device policy ---------------------------------------------------------------------

def test_the_device_answer_comes_from_the_engine_not_from_this_panel(repo, monkeypatch):
    """The panel must not have its own opinion about the GPU. If it did, the browser and
    the CLI could disagree about where a job runs, and only one of them would be right."""
    jobs = EvalJobs(repo)
    monkeypatch.setattr(jobs, "training", lambda: ["small-code"])
    plan = jobs.device()
    assert plan["device"] == "cpu"
    assert "small-code" in plan["reason"]


# ---- refusals ----------------------------------------------------------------------------

def test_refuses_an_unknown_checkpoint(repo):
    with pytest.raises(RunError, match="unknown checkpoint"):
        EvalJobs(repo).start({"checkpoint": "nope/nope.pt"})


def test_refuses_when_the_benchmark_data_is_not_downloaded(repo):
    """The consequential refusal. Without it the job starts, runs the suites that *are*
    cached, and dies partway through — after the expensive part."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    with pytest.raises(RunError, match="not downloaded"):
        EvalJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "suites": "mmlu"})


def test_refuses_an_unknown_suite(repo):
    from aksharallm.eval.sources import EvalError

    _ckpt(repo, "tiny", "ckpt_best.pt")
    with pytest.raises((RunError, EvalError), match="hellswag"):
        EvalJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "suites": "hellswag"})


def test_refuses_a_label_that_would_escape_the_filename(repo, spy_popen):
    """The label goes into a filename and onto a command line. It is validated against a
    pattern rather than escaped, so there is nothing to get wrong later."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _cache(repo, "piqa")
    with pytest.raises(RunError, match="label"):
        EvalJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "suites": "piqa",
                              "label": "../../etc/passwd"})


def test_refuses_a_second_concurrent_job(repo, spy_popen, monkeypatch):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _cache(repo, "piqa")
    jobs = EvalJobs(repo)
    jobs.start({"checkpoint": "tiny/ckpt_best.pt", "suites": "piqa"})
    monkeypatch.setattr(jobs, "_pid", lambda: 515151)
    with pytest.raises(RunError, match="already running"):
        jobs.start({"checkpoint": "tiny/ckpt_best.pt", "suites": "piqa"})


def test_stopping_nothing_says_so(repo):
    with pytest.raises(RunError, match="no evaluation is running"):
        EvalJobs(repo).stop()


# ---- the command line it builds --------------------------------------------------------------

def test_the_command_is_the_one_you_would_type(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _cache(repo, "piqa", "arc-easy")
    res = EvalJobs(repo).start({
        "checkpoint": "tiny/ckpt_best.pt", "suites": "piqa,arc-easy", "limit": 200,
        "device": "cpu", "label": "nightly"})
    cmd = spy_popen["cmd"]
    assert cmd[2:5] == ["-m", "aksharallm.eval", "run"]
    assert "--suite" in cmd and cmd[cmd.index("--suite") + 1] == "piqa,arc-easy"
    assert cmd[cmd.index("--limit") + 1] == "200"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--json") + 1].endswith(f"{res['job']}.json")
    assert res["state"] == "running"


def test_the_quick_look_reaches_the_scan_and_not_the_probe(repo, spy_popen):
    """The contamination panel offers scan depth (`--max-tokens`) and deliberately not item
    count (`--limit`).

    They are not interchangeable. The cost of the check is the corpus — 10B tokens streamed
    past a sorted hash array — so cutting items saves almost nothing (the whole `mc` probe
    builds in 19s) while silently leaving most of the benchmark unchecked. Cutting the scan
    buys time proportionally and degrades *evenly* across every item. So one is exposed and
    the other is not, and this asserts that on the command line where it is decided.
    """
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / "small-code.yaml").write_text("model: {}\n")
    EvalJobs(repo).start_audit({"kind": "contaminate", "config": "configs/small-code.yaml",
                               "suites": "mc", "max_tokens": 500_000_000, "verify": True})
    cmd = spy_popen["cmd"]
    assert cmd[2:5] == ["-m", "aksharallm.eval", "contaminate"]
    assert cmd[cmd.index("--max-tokens") + 1] == "500000000"
    assert "--verify" in cmd
    assert "--limit" not in cmd


def test_the_full_scan_passes_no_bound_at_all(repo, spy_popen):
    """`max_tokens: null` from the browser must mean "read everything", not "read 0"."""
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / "small-code.yaml").write_text("model: {}\n")
    EvalJobs(repo).start_audit({"kind": "contaminate", "config": "configs/small-code.yaml",
                               "suites": "mc", "max_tokens": None, "verify": True})
    assert "--max-tokens" not in spy_popen["cmd"]


def test_a_group_alias_is_expanded_before_it_reaches_the_command_line(repo, spy_popen):
    """`fast` is a portal-side convenience; the recorded command has to name the actual
    suites, or a result file cannot say what was measured."""
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _cache(repo, "piqa", "arc-easy")
    res = EvalJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "suites": "fast"})
    assert "fast" not in spy_popen["cmd"]
    assert set(res["suites"]) == {"perplexity", "arc-easy", "piqa"}


def test_an_adapter_is_passed_through(repo, spy_popen):
    _ckpt(repo, "tiny", "ckpt_best.pt")
    _cache(repo, "piqa")
    EvalJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "suites": "piqa",
                          "adapter": "tiny/sft_best.lora.pt"})
    cmd = spy_popen["cmd"]
    assert cmd[cmd.index("--adapter") + 1] == "tiny/sft_best.lora.pt"


def test_fetch_runs_the_same_cli(repo, spy_popen):
    res = EvalJobs(repo).fetch(["piqa", "gsm8k"])
    cmd = spy_popen["cmd"]
    assert cmd[2:5] == ["-m", "aksharallm.eval", "fetch"]
    assert cmd[5:] == ["piqa", "gsm8k"]
    assert res["kind"] == "fetch"


def test_fetch_ignores_a_name_that_is_not_a_known_dataset(repo, spy_popen):
    EvalJobs(repo).fetch(["piqa", "; rm -rf /"])
    assert spy_popen["cmd"][5:] == ["piqa"]


# ---- results -------------------------------------------------------------------------------

def _result(repo: Path, name: str, step: int, scores: dict, run: str = "tiny"):
    d = repo / "logs" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({
        "checkpoint": f"{run}/ckpt_best.pt",
        "provenance": {"run": run, "step": step, "best_val": 1.5},
        "stage": "base", "device": "cpu", "started": 1, "seconds": 10,
        "options": {"label": "eval"},
        "suites": {k: {"score": v, "kind": "mc", "n": 100, "stderr": 0.04,
                       "baseline": 0.25} for k, v in scores.items()},
    }))


def test_the_running_job_state_is_not_mistaken_for_a_result(repo):
    """`current.json` lives beside the results, the same way the quantize panel's does.
    Without the exclusion the running job appears in the table as an empty evaluation."""
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.6})
    (repo / "logs" / "eval" / "current.json").write_text('{"job": "x", "state": "running"}')
    rows = EvalJobs(repo).status()["results"]
    assert len(rows) == 1 and rows[0]["step"] == 100


def _audits(repo: Path):
    """The three audit shapes that really share `logs/eval/`, keys copied from files the
    CLI wrote. Contamination reusing the key `suites` for a *list* is the whole point."""
    d = repo / "logs" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "contamination-1786042785.json").write_text(json.dumps({
        "n": 13, "dirty_ids": [],
        "suites": [{"suite": "piqa", "parts": {"question": {"checkable": 10, "dirty": 0,
                                                            "rate": 0.0}}}],
    }))
    (d / "calibration-tiny-20260806-193614.json").write_text(json.dumps({
        "n_total": 100, "n_fit": 50, "n_scored": 50, "temperature": 1.02,
        "before": {"ece": 0.0135}, "after": {"ece": 0.0141}, "checkpoint": "tiny/ckpt_best.pt",
    }))
    (d / "dedup-tinystories-20260806-233309.json").write_text(json.dumps({
        "documents": 1000, "tokens": 50000, "clusters": 3, "duplicate_documents": 7,
        "duplicate_token_share": 0.014,
    }))


def test_an_audit_beside_the_results_does_not_take_the_eval_tab_down(repo):
    """A contamination report calls its list of per-suite overlaps `suites`, and `rows()`
    wants `suites` to be a name->score map. Iterating one as the other raised
    `AttributeError: 'list' object has no attribute 'items'` *inside* `/api/eval`, so the
    whole tab got `{"ok": false}`: no suite checkboxes were built and Evaluate stayed
    disabled forever. The crash was three layers from the symptom, so this asserts the
    symptom — status() answers at all, and it answers with the suites the browser ticks."""
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.6})
    _audits(repo)
    st = EvalJobs(repo).status()
    assert st["suites"] and st["groups"]["all"]


def test_audits_are_not_counted_as_evaluations(repo):
    """Calibration and dedup carry no `suites` key at all, so they never crashed — they
    quietly became rows with no score, no checkpoint and no step. A phantom evaluation in
    the trend table is the failure that does not announce itself."""
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.6})
    _audits(repo)
    rows = EvalJobs(repo).status()["results"]
    assert [r["step"] for r in rows] == [100]
    assert EvalJobs(repo).compare("piqa")["points"][0]["step"] == 100


def test_a_result_is_identified_by_shape_not_by_its_filename(repo):
    """`NOT_RESULTS` is an exact-name set: it excludes the one filename someone thought of,
    and every audit added later has to remember to add itself. Naming an audit anything at
    all must still keep it out of the table — and a real result named nothing in particular
    must still land in it."""
    from aksharallm.eval.report import is_result

    assert is_result({"suites": {"piqa": {"score": 0.6}}})
    assert not is_result({"suites": [{"suite": "piqa"}]})
    assert not is_result({"n_total": 100, "temperature": 1.02})
    assert not is_result([1, 2, 3])

    _result(repo, "whatever-i-called-it.json", 100, {"piqa": 0.6})
    (repo / "logs" / "eval" / "20260101-000000-tiny-eval.json").write_text(
        json.dumps({"suites": [{"suite": "piqa", "parts": {}}]}))
    rows = EvalJobs(repo).status()["results"]
    assert [r["file"] for r in rows] == ["whatever-i-called-it.json"]


def test_compare_orders_by_training_step_not_by_when_it_was_run(repo):
    """Checkpoints get evaluated out of order — you go back and measure an older one. The
    trend is against training progress, so it has to sort by step."""
    _result(repo, "20260102-000000-tiny-eval.json", 9000, {"piqa": 0.7})
    _result(repo, "20260101-000000-tiny-eval.json", 1000, {"piqa": 0.55})
    points = EvalJobs(repo).compare("piqa")["points"]
    assert [p["step"] for p in points] == [1000, 9000]
    assert [p["score"] for p in points] == [0.55, 0.7]


def test_compare_carries_the_chance_line(repo):
    """A score with no baseline beside it is not a measurement — the chart draws this as a
    rule, and a flat line at chance has to be readable as 'still guessing'."""
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.5})
    assert EvalJobs(repo).compare("piqa")["baseline"] == 0.5
    assert EvalJobs(repo).compare("mmlu")["baseline"] == 0.25


def test_compare_can_be_filtered_to_one_run(repo):
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.5}, run="tiny")
    _result(repo, "20260102-000000-small-eval.json", 200, {"piqa": 0.6}, run="small-code")
    assert len(EvalJobs(repo).compare("piqa")["points"]) == 2
    assert len(EvalJobs(repo).compare("piqa", run="tiny")["points"]) == 1


def test_reading_one_result_refuses_a_path(repo):
    _result(repo, "20260101-000000-tiny-eval.json", 100, {"piqa": 0.5})
    jobs = EvalJobs(repo)
    assert jobs.result("20260101-000000-tiny-eval.json")["stage"] == "base"
    with pytest.raises(RunError):
        jobs.result("../../etc/passwd")
    with pytest.raises(RunError):
        jobs.result("current.json")


def test_progress_is_parsed_from_the_jobs_own_output(repo):
    """The bar in the browser can only ever show what the job printed. Inventing a
    percentage from elapsed time is how a progress bar comes to lie."""
    jobs = EvalJobs(repo)
    assert jobs._progress(["[eval] mmlu 40/160 (25%)", "[eval] mmlu 80/160 (50%)"]) == {
        "label": "mmlu", "done": 80, "total": 160, "pct": 50}
    assert jobs._progress(["nothing useful"]) is None
