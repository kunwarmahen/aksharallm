"""The Playground and everything built after it: what each checkpoint IS, and tool calls.

Since the Playground was written the repo has grown codecs, recognisers, picture models,
world models and recall models, and every one of them is a `.pt` under `checkpoints/`. The
picker lists them all. These tests pin the two things that keep that honest:

  * a checkpoint's `kind` is read from what is inside the file (a `stage` key, a trainer's
    config section, the tokenizer's own meta.json), the Playground's default is always a
    text model, and asking a non-text model for text is refused *before* a stream starts,
    with a sentence naming the tab that uses it;
  * in chat, a `<tool>…</tool>` call is run in the sandbox and its output handed back as a
    `tool` turn — the format the tool-use models were trained on (tools/problems.py).
"""

from __future__ import annotations

import json

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer import sandbox
from aksharallm.infer.checkpoints import CheckpointStore, InferError, kind_of
from aksharallm.infer.playground import Playground
from aksharallm.model.transformer import Transformer
from aksharallm.tokenizer.tokenizer import train_bpe

MODEL_CFG = dict(vocab_size=300, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                 max_seq_len=64, tie_embeddings=True)


def _lm_payload(tok="data/tok.json", **mcfg):
    cfg = {**MODEL_CFG, **mcfg}
    return {"model": Transformer(ModelConfig(**cfg)).state_dict(), "model_config": cfg,
            "config": {"name": "x", "data": {"tokenizer": tok}, "model": {},
                       "diffusion": {}, "train": {"max_steps": 10}},
            "step": 5, "best_val": 3.0}


@pytest.fixture
def repo(tmp_path):
    tok = tmp_path / "data" / "tok.json"
    tok.parent.mkdir(parents=True)
    train_bpe(iter(["hello world, a small corpus for a small tokenizer. 6 * 7\n"] * 40),
              vocab_size=300, out_path=tok)
    # A world corpus describes itself in a meta.json beside its tokenizer.
    world = tmp_path / "data" / "world"
    world.mkdir()
    (world / "tokenizer.json").write_text(tok.read_text())
    (world / "meta.json").write_text(json.dumps({"vocab": ["#", "."], "frame_tokens": 64}))

    ck = tmp_path / "checkpoints"

    def save(run, payload, name="ckpt_best.pt"):
        (ck / run).mkdir(parents=True, exist_ok=True)
        torch.save(payload, ck / run / name)

    save("lm", _lm_payload())
    save("lm", {**_lm_payload(), "step": 6}, "sft_best.pt")
    save("hyb", _lm_payload(full_attn_every=2))
    save("world", _lm_payload(tok="data/world/tokenizer.json"))
    # Not transformers at all: a codec (records `stage`) and a flow model (config section).
    save("codec", {"codec": {}, "stage": "codec", "config": {"codec": {}}, "step": 1})
    save("flow", {"flow": {}, "config": {"flow": {}, "data": {}}, "step": 1})
    return tmp_path


@pytest.fixture
def pg(repo):
    p = Playground(repo, busy_cb=lambda: [])
    yield p
    p.close()


def test_every_kind_is_read_from_the_file_not_the_name(repo):
    kinds = {c.run: c.kind for c in CheckpointStore(repo).list()}
    assert kinds == {"lm": "text", "hyb": "text", "world": "world", "codec": "codec",
                     "flow": "flow"}
    # And the two model-level kinds, from the model config alone.
    assert kind_of({"model_config": {**MODEL_CFG, "causal": False,
                                     "mask_token_id": 299}}) == "diffusion"
    assert kind_of({"model_config": {**MODEL_CFG, "tag_classes": 12}}) == "tagger"


def test_a_non_text_checkpoint_says_what_it_is_and_which_tab_uses_it(repo):
    info = CheckpointStore(repo).get("codec/ckpt_best.pt").as_dict()
    assert info["playable"] is False and info["modes"] == []
    assert info["elsewhere"]["view"] == "audio" and info["elsewhere"]["tab"] == "Audio"
    assert "not a text language model" in info["stage_note"]


def test_the_default_is_a_text_model_even_when_something_else_is_newer(repo):
    # The newest file on disk is a codec; opening the Playground on it would show nothing
    # it can do.
    import os
    import time
    # Without the SFT file every candidate is stage "base", so only the kind can decide —
    # an SFT checkpoint would outrank the codec by stage and hide a missing filter.
    (repo / "checkpoints" / "lm" / "sft_best.pt").unlink()
    later = time.time() + 100
    os.utime(repo / "checkpoints" / "codec" / "ckpt_best.pt", (later, later))
    assert CheckpointStore(repo).default().kind == "text"


def test_architecture_notes_explain_what_changes_when_sampled(repo):
    hyb = CheckpointStore(repo).get("hyb/ckpt_best.pt")
    assert any("DeltaNet" in n and "draft" in n for n in hyb.notes)
    assert CheckpointStore(repo).get("lm/ckpt_best.pt").notes == ()


@pytest.mark.parametrize("ckpt,tab", [("codec/ckpt_best.pt", "Audio"),
                                      ("flow/ckpt_best.pt", "Pictures"),
                                      ("world/ckpt_best.pt", "World")])
def test_text_is_refused_before_the_stream_with_the_tab_to_use(pg, ckpt, tab):
    # Raised by `stream` itself (so the portal answers 4xx), not from inside the generator.
    with pytest.raises(InferError, match=f"{tab} tab"):
        pg.stream(ckpt_id=ckpt, mode="complete", prompt="hello", device="cpu")


# ---------------------------------------------------------------- tool calls -------------

class Scripted:
    """Stands in for the engine: each `stream` call answers the next scripted reply, and
    remembers the messages it was handed so the test can see what the model saw."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def stream(self, ckpt_id, mode, *, prompt="", messages=None, **kw):
        turns = list(messages or [])
        if prompt:
            turns.append({"role": "user", "content": prompt})
        self.seen.append(turns)
        reply = self.replies.pop(0)

        def gen():
            yield "start", {"checkpoint": {}}
            yield "delta", reply
            yield "done", {"text": reply, "tokens": 3, "elapsed_s": 0.5}
        return gen()

    def close(self):
        pass


needs_sandbox = pytest.mark.skipif(not sandbox.available()[0], reason="no sandbox here")


@needs_sandbox
def test_a_tool_call_is_run_and_its_output_handed_back(pg):
    fake = Scripted(["<tool>\nprint(6 * 7)\n</tool>", "The answer is 42."])
    pg.engine = fake
    events = list(pg.stream(ckpt_id="lm/sft_best.pt", mode="chat", prompt="6 times 7?",
                            tools=True, record=False))
    tool = [p for k, p in events if k == "tool"]
    done = [p for k, p in events if k == "done"][0]
    assert tool and tool[0]["output"] == "42" and tool[0]["status"] == "pass"
    # The second turn saw the call and its output, in the training format.
    assert fake.seen[1][-2] == {"role": "assistant",
                                "content": "<tool>\nprint(6 * 7)\n</tool>"}
    assert fake.seen[1][-1] == {"role": "tool", "content": "42"}
    assert done["text"] == "The answer is 42." and done["tool_calls"] == 1
    assert [t["role"] for t in done["turns"]] == ["user", "assistant", "tool", "assistant"]
    assert done["tokens"] == 6


def test_without_tools_a_call_is_only_text(pg):
    fake = Scripted(["<tool>\nprint(6 * 7)\n</tool>"])
    pg.engine = fake
    events = list(pg.stream(ckpt_id="lm/sft_best.pt", mode="chat", prompt="6 times 7?",
                            tools=False, record=False))
    assert not [k for k, _ in events if k == "tool"] and len(fake.seen) == 1


@needs_sandbox
def test_tool_calls_stop_at_the_cap(pg):
    from aksharallm.infer.playground import MAX_TOOL_CALLS
    fake = Scripted(["<tool>\nprint(1)\n</tool>"] * (MAX_TOOL_CALLS + 1))
    pg.engine = fake
    events = list(pg.stream(ckpt_id="lm/sft_best.pt", mode="chat", prompt="loop",
                            tools=True, record=False))
    assert sum(k == "tool" for k, _ in events) == MAX_TOOL_CALLS
    assert len(fake.seen) == MAX_TOOL_CALLS + 1


def test_every_trainer_stage_is_a_known_kind():
    """A trainer that saves `{"stage": "<x>", ...}` must name a kind in KINDS, or its
    checkpoints land in the Playground as `other` at best. Reads the trainers' source, so a
    new one fails here the day it is written rather than the day someone picks its file."""
    import re
    from pathlib import Path

    from aksharallm.infer.checkpoints import KINDS
    root = Path(__file__).resolve().parents[1] / "aksharallm"
    found = {}
    for path in root.rglob("*.py"):
        if "portal" in path.parts or path.name == "runlog.py":
            continue
        for m in re.finditer(r'torch\.save\(\{[^}]*?"stage":\s*"([a-z]+)"|'
                             r'"stage":\s*"([a-z]+)",\s*"data_rng"|'
                             r'"best_val":\s*\w+,\s*"stage":\s*"([a-z]+)"|'
                             r'^\s+"stage":\s*"([a-z]+)",?\s*$',
                             path.read_text(), re.M):
            stage = next(g for g in m.groups() if g)
            found.setdefault(stage, path.relative_to(root.parent))
    # The post-training stages are text; they are named by filename prefix, not KINDS.
    unknown = {s: str(p) for s, p in found.items()
               if s not in KINDS and s not in ("sft", "dpo", "grpo", "base")}
    assert len(found) >= 8, found          # the scan still sees the trainers
    assert not unknown, f"add these to infer/checkpoints.KINDS: {unknown}"
