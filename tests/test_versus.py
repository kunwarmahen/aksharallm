"""Head-to-head judging (`aksharallm/eval/versus.py`).

The two properties that make a pairwise verdict trustworthy are tested against fake judges
whose behaviour is known exactly: one that is fair, and one that only ever prefers whichever
answer it read first. A position-biased judge must produce ties, never wins — if it produced
wins, every comparison would be decided by the order the answers were listed in.
"""

from __future__ import annotations

import json

import pytest

from aksharallm.eval import judge as judge_mod
from aksharallm.eval import report, suites, versus


def test_a_win_needs_both_orders_to_agree():
    assert versus.verdict("A", "B") == "win"        # model 1 shown first, then second
    assert versus.verdict("B", "A") == "loss"
    assert versus.verdict("A", "A") == "tie"        # preferred whatever came first
    assert versus.verdict("B", "B") == "tie"        # preferred whatever came second
    assert versus.verdict("tie", "B") == "tie"
    assert versus.verdict(None, "B") == "ungraded"


def test_the_sign_test_is_exact():
    assert versus.sign_test(0, 0) is None
    assert versus.sign_test(5, 5) == 1.0
    assert versus.sign_test(9, 5) == pytest.approx(0.4240, abs=1e-4)   # sounds decisive; is not
    assert versus.sign_test(10, 0) == pytest.approx(2 / 1024)


def test_the_judges_reply_is_read_whatever_its_shape():
    assert versus.parse_choice('{"better": "A", "reason": "x"}') == "A"
    assert versus.parse_choice('Sure. {"better": "b"}') == "B"
    assert versus.parse_choice('{"better": "tie"}') == "tie"
    assert versus.parse_choice("I cannot decide") is None


def _side(ckpt, answers, suite="judge48"):
    items = [{"id": i.id, "group": i.group, "answer": answers(i)}
             for i in suites.JUDGE_SETS[suite]]
    return {"checkpoint": ckpt, "options": {"label": ckpt.split("/")[0]},
            "suites": {suite: {"items": items}}}


class FakeOllama:
    """Answers the pairwise question by a fixed rule over the two answers it is shown."""
    rule = None

    def __init__(self, cfg):
        pass

    def chat(self, messages, model=None):
        body = messages[-1]["content"]
        first = body.split("ANSWER A:\n")[1].split("\n\nANSWER B:")[0]
        second = body.split("ANSWER B:\n")[1].split("\n\nWhich")[0]
        yield "delta", json.dumps({"better": FakeOllama.rule(first, second)})


@pytest.fixture
def fake_judge(monkeypatch):
    monkeypatch.setattr(judge_mod, "Ollama", FakeOllama)
    monkeypatch.setattr(judge_mod, "available", lambda cfg: (True, ""))
    released = []
    monkeypatch.setattr(judge_mod, "release", lambda cfg, model: released.append(model))
    return released


def test_a_fair_judge_finds_the_better_model(fake_judge):
    FakeOllama.rule = staticmethod(lambda a, b: "A" if a == "good" else "B" if b == "good"
                                   else "tie")
    res = versus.compare(judge_mod.default_config(), _side("m1/x.pt", lambda i: "bad"),
                         _side("m2/y.pt", lambda i: "good"), "judge48")
    o = res["overall"]
    assert (o["win"], o["loss"], o["tie"]) == (0, 48, 0)
    assert o["p_value"] < 1e-10 and o["win_rate"] == 0.0
    assert set(res["groups"]) == {i.group for i in suites.JUDGE48_PROMPTS}
    assert fake_judge, "the judge must be unloaded when the comparison ends"


def test_a_judge_that_prefers_whatever_it_read_first_decides_nothing(fake_judge):
    FakeOllama.rule = staticmethod(lambda a, b: "A")
    res = versus.compare(judge_mod.default_config(), _side("m1/x.pt", lambda i: "bad"),
                         _side("m2/y.pt", lambda i: "good"), "judge48")
    o = res["overall"]
    assert (o["win"], o["loss"], o["tie"]) == (0, 0, 48)
    assert o["inconsistent"] == 48 and o["p_value"] is None


def test_a_comparison_file_is_never_read_as_a_benchmark_result(tmp_path, fake_judge):
    FakeOllama.rule = staticmethod(lambda a, b: "tie")
    res = versus.compare(judge_mod.default_config(), _side("m1/x.pt", lambda i: "a"),
                         _side("m2/y.pt", lambda i: "b"), "judge48")
    (tmp_path / "logs" / "eval").mkdir(parents=True)
    out = versus.write(res, tmp_path)
    assert out.name.startswith("versus-m1-vs-m2-")
    assert not report.is_result(json.loads(out.read_text()))


def test_the_newest_result_for_a_checkpoint_is_the_one_compared(tmp_path, monkeypatch):
    from aksharallm.infer import checkpoints
    monkeypatch.setattr(checkpoints.CheckpointStore, "identify", lambda self, ref: "run/a.pt")
    d = tmp_path / "logs" / "eval"
    d.mkdir(parents=True)
    for stamp, ans in (("20260101-000000", "old"), ("20260201-000000", "new")):
        (d / f"{stamp}-run-x.json").write_text(json.dumps(_side("run/a.pt", lambda i: ans)))
    (d / "20260301-000000-run-y.json").write_text(json.dumps(
        {"checkpoint": "run/a.pt", "suites": {"mmlu": {"score": 0.3}}}))   # no judge48
    side = versus.load_side("run", "judge48", tmp_path)
    assert side["suites"]["judge48"]["items"][0]["answer"] == "new"


def test_every_judge48_prompt_has_a_rubric_and_a_unique_id():
    ids = [i.id for i in suites.JUDGE48_PROMPTS + suites.JUDGE_PROMPTS]
    assert len(ids) == len(set(ids))
    for item in suites.JUDGE48_PROMPTS:
        assert item.rubric and len(item.rubric) > 5 and item.group, item.id
