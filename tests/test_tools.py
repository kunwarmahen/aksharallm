"""Tool use (docs/06 § "Tool use", aksharallm/tools/): the problems' answers, what the model is
trained on, and the agent loop."""

from __future__ import annotations

import pytest

from aksharallm.infer import sandbox
from aksharallm.tools import agent
from aksharallm.tools.problems import (
    HELD_OUT_KIND,
    KINDS,
    TRAIN_KINDS,
    conversation,
    problems,
)

needs_sandbox = pytest.mark.skipif(not sandbox.available()[0], reason="no sandbox here")


@needs_sandbox
def test_every_reference_call_prints_its_answer():
    """The teaching data says "the tool printed X". If a reference call printed anything
    else, the model would be taught to read a result the tool never gives."""
    for p in problems(3 * len(KINDS), seed=7, kinds=KINDS):
        r = sandbox.run_program(p.code, timeout_s=5.0)
        assert agent.same_number(agent.final_number(r.stdout), p.answer), (p.kind, p.code, r)


def test_the_tool_output_is_read_never_trained_on(tmp_path):
    """Only the assistant turns carry loss; a model trained to predict the tool's output
    would learn to guess instead of calling."""
    from aksharallm.tokenizer.tokenizer import Tokenizer
    tok = Tokenizer("data/blend/tokenizer.json")
    p = problems(1, seed=0)[0]
    msgs = conversation(p)
    ids, mask = tok.render_chat(msgs)
    out_ids = tok.encode(msgs[2]["content"])
    # find the tool output inside the rendered conversation, and require mask 0 over it
    n = len(out_ids)
    at = next(i for i in range(len(ids) - n) if ids[i:i + n] == out_ids
              and ids[i - 2:i] != tok.encode("is "))
    assert sum(mask[at:at + n]) == 0
    call = tok.encode(msgs[1]["content"])
    j = next(i for i in range(len(ids)) if ids[i:i + len(call)] == call)
    assert all(mask[j:j + len(call)])


def test_one_kind_is_never_in_the_training_data():
    assert HELD_OUT_KIND in KINDS and HELD_OUT_KIND not in TRAIN_KINDS
    assert {p.kind for p in problems(60, seed=1)} == set(TRAIN_KINDS)


def test_the_final_number_is_the_last_one_with_commas_removed():
    assert agent.final_number("48213 x 9071: the answer is 437,340,123.") == "437340123"
    assert agent.same_number("1234.50", "1234.5")
    assert not agent.same_number(None, "3")


@needs_sandbox
def test_the_loop_runs_the_call_and_hands_back_its_output(monkeypatch):
    """Scripted replies in place of a model: a call, then an answer. With the tool, the
    sandbox's stdout must reach the second turn; without it, the call must not be run."""
    seen = []

    def fake_turn(model, tok, messages, device, max_new=160):
        seen.append([m["content"] for m in messages])
        if messages[-1]["role"] == "user":
            return "<tool>\nprint(6 * 7)\n</tool>"
        return "The answer is 42."

    monkeypatch.setattr(agent, "_turn", fake_turn)
    r = agent.solve(None, None, "What is 6 times 7?", use_tool=True)
    assert r["calls"] == 1 and seen[1][-1] == "42" and r["final"] == "The answer is 42."
    seen.clear()
    r = agent.solve(None, None, "What is 6 times 7?", use_tool=False)
    assert r["calls"] == 0 and "42" not in seen[1][-1]
