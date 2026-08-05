"""Tests for the end-of-run report.

The report is read by a person once, at the end of a run that may have taken six days, and
it is the only place several of those numbers are ever stated. So the tests here are mostly
about the two ways it could quietly lie: a **formatting** lie (perplexity rendered as a
percentage, an unknown duration printed as zero) and a **finding** lie (a check that never
fires, or one that fires on a healthy run and teaches the reader to ignore the section).

Nothing here trains anything — a report is a view over a log file, so a hand-written log
exercises all of it.
"""

from __future__ import annotations

import json

import pytest

from aksharallm.portal.runs import RunStore
from aksharallm.train import report

CONFIG = """
name: demo
model:
  vocab_size: 512
  d_model: 64
  n_layers: 2
  n_heads: 4
  max_seq_len: 128
optim:
  lr: 3.0e-4
  grad_clip: 1.0
train:
  out_dir: checkpoints/demo
  batch_size: 2
  grad_accum: 4
  seq_len: 128
  max_steps: 400
"""


def step(n, loss, ema=None, **kw):
    rec = {"step": n, "loss": loss, "ema": ema if ema is not None else loss,
           "lr": 3e-4, "grad_norm": 0.9, "tok_per_sec": 26000.0, "mfu": 0.7,
           "time": 1785000000.0 + n * 9, "s_per_step": 9.0, "elapsed": n * 9.0}
    rec.update(kw)
    return rec


def session(start, **kw):
    return {"event": "session_start", "time": 1785000000.0 + start * 9,
            "iso": "2026-07-25 09:35:41", "run": "demo", "pid": 4242, "start_step": start,
            "max_steps": 400, "tokens_per_step": 245760, "params": 13_800_000,
            "params_active": 13_800_000, **kw}


def ended(last, reason="max_steps", **kw):
    return {"event": "session_end", "time": 1785000000.0 + last * 9, "run": "demo",
            "reason": reason, "last_step": last, "steps": last, "elapsed": last * 9.0, **kw}


def write_log(root, lines, name="train_log.jsonl", run="demo"):
    d = root / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(json.dumps(x) if isinstance(x, dict) else x
                                    for x in lines) + "\n")
    return d


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "demo.yaml").write_text(CONFIG)
    (tmp_path / "logs").mkdir()
    return tmp_path


#: A healthy short run: one session, four evaluations, still improving at the end.
HEALTHY = [
    session(0),
    step(0, 10.6), {"step": 50, "val_loss": 8.0}, step(50, 8.3),
    {"step": 150, "val_loss": 6.2}, step(150, 6.4),
    {"step": 250, "val_loss": 5.4}, step(250, 5.5),
    {"step": 399, "val_loss": 4.9}, step(399, 5.0),
    ended(399),
]


# ---- what the report is built from -----------------------------------------------------

def test_a_run_with_no_log_is_reported_as_such_rather_than_crashing(repo):
    d = repo / "checkpoints" / "empty"
    d.mkdir(parents=True)
    data = report.build(d, run="empty", root=repo)
    assert data["empty"] is True
    text = report.render(data)
    assert "nothing to report" in text
    # Still a well-formed page with the run's name on it, not a stack trace.
    assert text.startswith("# empty — run report")


def test_a_finished_run_says_so_and_a_short_one_says_how_short(repo):
    d = write_log(repo, HEALTHY)
    done = report.build(d, run="demo", root=repo)
    assert done["complete"] is True and done["remaining"] == 0
    assert "finished its budget" in report.verdict(done)

    d = write_log(repo, HEALTHY[:-3] + [ended(250, reason="signal")], run="short")
    (repo / "configs" / "short.yaml").write_text(CONFIG.replace("demo", "short"))
    part = report.build(d, run="short", root=repo)
    assert part["complete"] is False and part["remaining"] == 149
    assert "stopped 149 steps short" in report.verdict(part).lower()


def test_parameter_counts_come_from_the_log_not_from_a_second_implementation(repo):
    """The trainer records them at session start. Nothing in the report recomputes a
    parameter count from a config — two implementations of that number is two chances to
    print a wrong one, and the wrong one would look completely plausible."""
    d = write_log(repo, HEALTHY)
    data = report.build(d, run="demo", root=repo)
    assert data["params"] == 13_800_000
    assert "13.80M" in report.render(data)

    # An older log without the field reports a gap rather than a guess.
    d2 = write_log(repo, [r for r in HEALTHY if r.get("event") != "session_start"],
                   run="old")
    assert report.build(d2, run="old", root=repo)["params"] is None


def test_the_log_of_any_trainer_is_found_and_named(repo):
    """SFT, DPO and GRPO write differently named logs with no session markers. The report
    reads them too; what it must not do is claim they were pretraining runs."""
    d = write_log(repo, [step(0, 2.1, acc=0.51), step(10, 1.2, acc=0.74),
                         {"step": 10, "val_loss": 1.3}], name="dpo_log.jsonl", run="demo-dpo")
    data = report.build(d, run="demo-dpo", root=repo)
    assert data["kind"] == "DPO"
    assert any(x["key"] == "acc" and x["best"] == 0.74 for x in data["extras"])
    assert "train preference accuracy" in report.render(data)


def test_tokens_seen_uses_the_trainers_own_tokens_per_step(repo):
    d = write_log(repo, HEALTHY)
    data = report.build(d, run="demo", root=repo)
    assert data["tokens"] == 245760 * 400


# ---- the findings ----------------------------------------------------------------------

def levels(data, needle):
    return [c for c in data["checks"] if needle in c["text"]]


def test_a_crashed_session_is_found_but_the_last_one_is_not_accused(repo):
    """A session with no end record was killed with -9 or crashed. The *last* session has no
    end record for a much more ordinary reason — it is the one running right now, or the one
    that just called this report — and flagging it would make the check noise."""
    lines = [session(0), step(0, 10.6), step(50, 8.3),          # no end: crashed
             session(51), step(100, 7.0), ended(100),
             session(101), step(150, 6.4)]                       # no end: still going
    d = write_log(repo, lines)
    data = report.build(d, run="demo", root=repo)
    found = levels(data, "without a session_end record")
    assert len(found) == 1 and found[0]["level"] == "warn"
    assert "#1" in found[0]["text"] and "#3" not in found[0]["text"]


def test_a_validation_loss_that_stopped_improving_is_called_out(repo):
    lines = [session(0), step(0, 10.6),
             {"step": 20, "val_loss": 4.0},          # the best, one twentieth of the way in
             {"step": 100, "val_loss": 4.4}, {"step": 200, "val_loss": 4.5},
             {"step": 399, "val_loss": 4.6}, step(399, 5.0), ended(399)]
    d = write_log(repo, lines)
    data = report.build(d, run="demo", root=repo)
    warn = levels(data, "best validation loss was at step")
    assert warn and warn[0]["level"] == "warn"
    assert "ckpt_best.pt" in warn[0]["text"]        # says which checkpoint that means


def test_a_healthy_run_gets_good_findings_not_silence(repo):
    """A section that only ever prints warnings teaches the reader to skip it when it is
    empty, which is exactly when they should trust it."""
    d = write_log(repo, HEALTHY)
    data = report.build(d, run="demo", root=repo)
    assert [c for c in data["checks"] if c["level"] == "good"]
    assert not [c for c in data["checks"] if c["level"] == "warn"]
    assert levels(data, "still improving near the end")


def test_never_evaluating_is_itself_a_finding(repo):
    d = write_log(repo, [session(0), step(0, 10.6), step(399, 5.0), ended(399)])
    data = report.build(d, run="demo", root=repo)
    assert levels(data, "No validation loss")


def test_nan_and_spikes_are_measured_against_the_running_average(repo):
    """A spike is a step that is bad *compared with how the run was going*. Early in
    training a loss of 8 is normal and late in training it is a disaster, so the threshold
    has to move with the EMA rather than being a constant."""
    lines = [session(0), step(0, 10.6), step(50, 8.3),
             step(100, 14.0, ema=6.0),                  # a real spike: 2.3x the average
             step(150, 6.1, ema=6.0),                   # normal
             step(200, float("nan"), ema=6.0),
             {"step": 200, "val_loss": 6.0}, ended(200)]
    d = write_log(repo, lines, name="train_log.jsonl")
    # NaN is not valid JSON to every reader, so write it the way Python's json does.
    data = report.build(d, run="demo", root=repo)
    spikes = levels(data, "loss spike")
    assert spikes and "step 100" in spikes[0]["text"]
    # Exactly one. Step 0's loss of 10.6 is higher than the spike at step 100 in absolute
    # terms and is not a spike at all — it is where a run starts. Any threshold that does not
    # move with the EMA counts it, which is the version of this check that cries wolf.
    assert "1 loss spike" in spikes[0]["text"]
    assert levels(data, "NaN")


def test_a_throughput_regression_is_measured_against_a_typical_session(repo):
    """Against the median session, not the fastest. One session can report an inflated rate
    — a short one, or one from before the partial-window fix — and making that the yardstick
    turns the check into noise for every session after it."""
    lines = [session(0), step(0, 10.6, tok_per_sec=99000.0), ended(0),      # the outlier
             session(1), step(100, 8.0, tok_per_sec=26000.0), ended(100),
             session(101), step(150, 7.5, tok_per_sec=26000.0), ended(150),
             session(151), step(200, 7.0, tok_per_sec=24000.0), ended(200)]
    d = write_log(repo, lines)
    data = report.build(d, run="demo", root=repo)
    # 24k against a median of 26k is 8% down — inside the noise, and against the *maximum*
    # it would have read as 76% slower and been reported every time.
    assert not levels(data, "slower")

    lines[-2] = step(200, 7.0, tok_per_sec=12000.0)
    d = write_log(repo, lines)
    slower = levels(report.build(d, run="demo", root=repo), "slower")
    assert slower and "54%" in slower[0]["text"]


def test_router_collapse_is_reported_for_a_mixture_of_experts(repo):
    lines = [session(0),
             step(0, 10.6, moe={"balance": 0.97, "shares": [0.5, 0.5], "dead": 0,
                                "min_share": 0.49, "max_share": 0.51}),
             step(50, 8.3, moe={"balance": 0.51, "shares": [1.0, 0.0], "dead": 1,
                                "min_share": 0.0, "max_share": 1.0}),
             ended(50)]
    d = write_log(repo, lines)
    data = report.build(d, run="demo", root=repo)
    assert data["moe"]["experts"] == 2 and data["moe"]["dead_ever"] == 1
    collapse = levels(data, "router collapse")
    assert collapse and collapse[0]["level"] == "warn"
    assert "Expert routing" in report.render(data)


# ---- formatting: the quiet way a report lies -------------------------------------------

def test_perplexity_is_not_rendered_as_a_percentage():
    """The first version of this file multiplied every benchmark score by 100, which turned
    a perplexity of 4.337 into "433.7%" — a number that is wrong, plausible and never
    questioned, because the column beside it really is a percentage."""
    assert report.score({"score": 4.337, "kind": "ppl"}) == "4.337"
    assert report.score({"score": 0.55, "kind": "mc"}) == "55.0%"
    assert report.score({"score": 3.4, "kind": "judge"}) == "3.40/5"
    assert report.score({"score": None, "kind": "mc"}) == "–"


def test_everything_unknowable_prints_as_a_gap_never_as_zero():
    assert report.num(None) == "–"
    assert report.integer(None) == "–"
    assert report.compact(None) == "–"
    assert report.dur(None) == "–"          # fmt_dur's own answer is "?", which reads as a bug
    assert report.clock(None) == "–"
    assert report.wh(None) == "–"
    assert report.bytes_(None) == "–"
    assert report.ppl(None) == "–"


def test_the_sparkline_is_bounded_and_survives_a_flat_or_empty_series():
    assert report.spark([]) == "" and report.spark([1.0]) == ""
    assert report.spark([2.0, 2.0, 2.0]) == "▁▁▁"
    line = report.spark(list(range(500)), width=40)
    assert len(line) == 40 and line[0] == "▁" and line[-1] == "█"
    # A falling loss curve should read as falling, which is the only thing anyone reads it for.
    assert report.spark([9.0, 6.0, 3.0, 1.0])[0] == "█"


# ---- writing it ------------------------------------------------------------------------

def test_write_leaves_markdown_and_the_same_data_as_json(repo):
    d = write_log(repo, HEALTHY)
    path = report.write(d, run="demo", root=repo)
    assert path.name == "report.md" and path.read_text().startswith("# demo — run report")
    data = json.loads((d / "report.json").read_text())
    assert data["run"] == "demo" and data["complete"] is True


def test_writing_twice_overwrites_rather_than_accumulating(repo):
    d = write_log(repo, HEALTHY)
    report.write(d, run="demo", root=repo)
    report.write(d, run="demo", root=repo)
    assert sorted(p.name for p in d.glob("report*")) == ["report.json", "report.md"]


def test_a_report_failure_can_never_take_a_run_down(repo, monkeypatch):
    """The last line of a six-day run is not the place to raise. `write_quietly` swallows
    everything and says so on one line."""
    said = []
    monkeypatch.setattr(report, "write",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert report.write_quietly(repo / "checkpoints" / "demo", run="demo",
                                echo=said.append) is None
    assert said and "disk full" in said[0]


# ---- through the portal ----------------------------------------------------------------

def test_the_portal_builds_the_report_live_and_can_save_it(repo):
    """Built on request rather than served from disk: the panel is usually opened *during* a
    run, and a snapshot from the last exit would be the most confidently wrong thing on the
    page."""
    write_log(repo, HEALTHY)
    store = RunStore(repo)
    live = store.report("demo")
    assert live["file"] is None                       # nothing written yet
    assert "# demo — run report" in live["markdown"]

    saved = store.report("demo", save=True)
    assert saved["saved"] == "checkpoints/demo/report.md"
    assert (repo / "checkpoints" / "demo" / "report.md").exists()


def test_the_portal_still_whitelists_the_run_name(repo):
    from aksharallm.portal.runs import RunError
    with pytest.raises(RunError):
        RunStore(repo).report("../../etc")
