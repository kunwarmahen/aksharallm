"""GRPO with the tool in the loop (tools/rollout.py, train/grpo.py --reward tools).

The rollout splices the sandbox's output between two sampled turns. Three things decide
whether that trains the right thing, and each would be silent if wrong: the sequence must be
token for token the conversation SFT trained on (else the policy is scored on a format it
never saw), only sampled tokens may carry loss (else it is trained to predict the tool), and
a call is run only when the turn actually finished it.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch

from aksharallm.tokenizer.tokenizer import Tokenizer, train_bpe
from aksharallm.tools.rollout import ToolReward, rollouts, tool_segment
from aksharallm.train.grpo import build_batch

CALL = "<tool>\nprint(6 * 7)\n</tool>"
ANSWER = "The answer is 42."


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tok.json"
    train_bpe(iter([CALL, ANSWER, "What is 6 times 7?", "tool\n42", "assistant\n"] * 50),
              vocab_size=300, out_path=path)
    return Tokenizer(str(path))


def _fake_sample(tok, first_turn):
    """A sampler that writes `first_turn` as turn 1 and ANSWER as turn 2 (a prompt holding
    a tool turn is a turn-2 prompt). `(full, gen)` groups, the trainer's contract."""
    def sample(prompts, k):
        out = []
        for pids in prompts:
            turn2 = pids.count(tok.im_start_id) > 2
            gen = tok.encode(ANSWER) + [tok.im_end_id] if turn2 else first_turn
            out.append([(list(pids) + gen, list(gen)) for _ in range(k)])
        return out
    return sample


def _prompt(tok):
    ids, _ = tok.render_chat([{"role": "user", "content": "What is 6 times 7?"}],
                             add_generation_prompt=True)
    return ids


def test_a_tool_rollout_is_token_for_token_the_conversation_sft_trained_on(tok):
    ran = []
    first = tok.encode(CALL) + [tok.im_end_id]
    groups = rollouts(_fake_sample(tok, first), tok, [_prompt(tok)], 2,
                      run=lambda code: ran.append(code) or "42")
    full, final, trainable = groups[0][0]
    want, _ = tok.render_chat([{"role": "user", "content": "What is 6 times 7?"},
                               {"role": "assistant", "content": CALL},
                               {"role": "tool", "content": "42"},
                               {"role": "assistant", "content": ANSWER}])
    newline = tok.encode("\n")
    assert full == want[:-len(newline)]          # the rollout ends at the last <|im_end|>
    assert ran == ["print(6 * 7)"] * 2
    assert tok.decode(final[:-1]) == ANSWER
    # loss on exactly the two sampled turns: never the prompt, never the tool's output
    p = len(_prompt(tok))
    seg = len(tool_segment(tok, "42"))
    assert trainable == [0] * p + [1] * len(first) + [0] * seg + [1] * len(final)


def test_no_call_is_one_turn_and_an_unfinished_call_is_not_run(tok):
    ran = []
    plain = tok.encode(ANSWER) + [tok.im_end_id]
    g = rollouts(_fake_sample(tok, plain), tok, [_prompt(tok)], 1, run=ran.append)
    full, final, trainable = g[0][0]
    assert final == plain and trainable == [0] * len(_prompt(tok)) + [1] * len(plain)
    cut = tok.encode(CALL)                       # max tokens hit before <|im_end|>
    g = rollouts(_fake_sample(tok, cut), tok, [_prompt(tok)], 1, run=ran.append)
    assert ran == [] and g[0][0][1] == cut


def test_the_loss_mask_follows_the_trainable_flags_not_the_prompt_length():
    full = [5, 6, 7, 8, 9, 10]
    trainable = [0, 0, 1, 0, 0, 1]               # a tool segment in the middle
    seq, mask = build_batch([[(full, full[2:], trainable)]], pad_id=0, device="cpu")
    assert mask[0].tolist() == [0, 1, 0, 0, 1]   # target t is seq[t+1]
    # with no flags it is still "everything after the prompt"
    _, plain = build_batch([[(full, full[2:])]], pad_id=0, device="cpu")
    assert plain[0].tolist() == [0, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        build_batch([[(full, full[2:], trainable[:-1])]], pad_id=0, device="cpu")


def test_the_reward_is_the_final_number_and_nothing_else():
    r = ToolReward("42")
    assert r("q", "The answer is 42.<|im_end|>") == 1.0
    assert r("q", "6 times 7 is 42") == 1.0      # no tool needed if the number is right
    assert r("q", "The answer is 41.") == 0.0 and r("q", "no idea") == 0.0


def test_grpo_runs_with_the_tool_in_the_loop(tmp_path, monkeypatch, tok):
    """Through `main()`, serial sampler, a model that always calls: the call reaches the
    sandbox, the reward reads the turn after it, and the log carries the held-out check."""
    from aksharallm.config import ModelConfig
    from aksharallm.model.transformer import Transformer
    from aksharallm.train import grpo
    cfg = dict(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
               max_seq_len=256, tie_embeddings=True)
    torch.manual_seed(0)
    torch.save({"model": Transformer(ModelConfig(**cfg)).state_dict(), "model_config": cfg,
                "config": {}, "step": 0}, tmp_path / "init.pt")
    calls = []
    first = tok.encode(CALL) + [tok.im_end_id]

    def fake_group(model, pids, k, *a, **kw):
        turn2 = pids.count(tok.im_start_id) > 2
        gen = tok.encode(ANSWER) + [tok.im_end_id] if turn2 else first
        return [(list(pids) + gen, list(gen)) for _ in range(k)]
    monkeypatch.setattr(grpo, "sample_group", fake_group)
    monkeypatch.setattr("aksharallm.tools.rollout.run_call", lambda code: calls.append(code) or "42")
    monkeypatch.setattr(sys, "argv", [
        "grpo", "--init", str(tmp_path / "init.pt"), "--tokenizer", tok.path,
        "--out-dir", str(tmp_path / "out"), "--reward", "tools", "--tool-problems", "4",
        "--holdout", "2", "--eval-every", "1", "--steps", "1", "--group-size", "2",
        "--prompts-per-step", "2", "--max-new-tokens", "8", "--device", "cpu",
        "--sampler", "serial"])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32",
                        torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    grpo.main()
    rows = [json.loads(l) for l in (tmp_path / "out" / "grpo_log.jsonl").read_text().splitlines()]
    step = [r for r in rows if "reward" in r and "step" in r]
    assert len(step) == 1 and calls                      # the calls went to the sandbox
    assert step[0]["called"] == 1.0         # every rollout ran its call
    assert [r["val_n"] for r in rows if "val_n" in r] == [2]
    start = [r for r in rows if r.get("event") == "session_start"][0]
    assert start["reward"] == "tools"
