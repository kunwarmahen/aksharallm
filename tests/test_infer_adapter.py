"""Tests for running a checkpoint with an adapter attached.

Two behaviours here are load-bearing and neither is obvious:

  * **A base checkpoint plus an SFT adapter is a chat model.** The playground refuses chat
    on a base model, and it decides that from the *filename* prefix. An adapter changes
    what the combination can do without changing the filename, so the gate has to read the
    pair rather than the file. Getting this wrong refuses exactly the thing adapters were
    built to produce.
  * **The wrong adapter must be refused, loudly.** An adapter is a delta. Applied to a
    different base it produces a model that still runs and still emits fluent text and is
    simply worse — the worst possible failure mode, because nothing tells you.
"""

from pathlib import Path

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer.checkpoints import InferError
from aksharallm.infer.engine import Engine
from aksharallm.lora.adapter import base_identity, save_adapter
from aksharallm.lora.inject import LoRAConfig, apply_lora
from aksharallm.lora.layer import LoRALinear
from aksharallm.model.transformer import Transformer
from aksharallm.tokenizer.tokenizer import Tokenizer

CFG = ModelConfig(vocab_size=None, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)


@pytest.fixture
def repo(tmp_path):
    """A miniature checkout: one tokenizer, one base checkpoint, one SFT adapter."""
    (tmp_path / "checkpoints" / "tiny").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "configs").mkdir()

    tok_path = _tokenizer(tmp_path)
    tok = Tokenizer(tok_path)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=32)

    torch.manual_seed(0)
    model = Transformer(cfg)
    ckpt = {
        "model": model.state_dict(),
        "model_config": dict(vars(cfg)),
        "config": {"data": {"tokenizer": "tokenizer.json"}, "train": {"seq_len": 32}},
        "step": 100, "best_val": 1.5,
    }
    torch.save(ckpt, tmp_path / "checkpoints" / "tiny" / "ckpt_best.pt")

    adapted = Transformer(cfg)
    lcfg = LoRAConfig(r=4, targets="all-linear")
    apply_lora(adapted, lcfg)
    with torch.no_grad():
        for mod in adapted.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_B.normal_(0, 0.2)
    save_adapter(tmp_path / "checkpoints" / "tiny" / "sft_best.lora.pt", adapted, lcfg,
                 base_identity(ckpt, "checkpoints/tiny/ckpt_best.pt"),
                 extra={"training": {"stage": "sft", "val_loss": 1.1}})
    return tmp_path


def _tokenizer(root: Path) -> Path:
    """A real (tiny) BPE tokenizer, built the same way `tests/test_infer.py` builds one."""
    from aksharallm.tokenizer.tokenizer import train_bpe

    path = root / "tokenizer.json"
    train_bpe(iter(["hello world this is a small corpus for a small tokenizer. "
                    "def add(a, b): return a + b\n"] * 40),
              vocab_size=300, out_path=path)
    return path


def _engine(repo) -> Engine:
    return Engine(repo, busy_cb=lambda: [])


def test_the_adapter_is_listed_and_the_base_still_loads_without_it(repo):
    eng = _engine(repo)
    assert [a.rel for a in eng.adapters.list()] == ["tiny/sft_best.lora.pt"]
    loaded = eng.load("tiny/ckpt_best.pt", device="cpu")
    assert loaded.adapter is None
    assert loaded.stage == "base"


def test_attaching_an_adapter_changes_the_model(repo):
    eng = _engine(repo)
    x = torch.randint(0, 200, (1, 4))
    plain = eng.load("tiny/ckpt_best.pt", device="cpu")
    with torch.no_grad():
        before, _ = plain.model(x)
    with_ad = eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/sft_best.lora.pt")
    with torch.no_grad():
        after, _ = with_ad.model(x)
    assert with_ad.adapter is not None
    assert not torch.allclose(before, after, atol=1e-4)


def test_an_sft_adapter_makes_a_base_checkpoint_a_chat_model(repo):
    """The whole point of the feature: the checkpoint is still `ckpt_`, and chat works."""
    eng = _engine(repo)
    base = eng.load("tiny/ckpt_best.pt", device="cpu")
    assert base.stage == "base"
    with pytest.raises(InferError, match="base model"):
        eng.build_prompt(base, "chat", prompt="hello")

    adapted = eng.load("tiny/ckpt_best.pt", device="cpu",
                       adapter="tiny/sft_best.lora.pt")
    assert adapted.stage == "sft"
    ids, stop_id, _ = eng.build_prompt(adapted, "chat", prompt="hello")
    assert ids and stop_id is not None


def test_swapping_the_adapter_reloads_rather_than_returning_the_cached_model(repo):
    eng = _engine(repo)
    a = eng.load("tiny/ckpt_best.pt", device="cpu")
    b = eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/sft_best.lora.pt")
    c = eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/sft_best.lora.pt")
    assert a is not b, "attaching an adapter must not reuse the plain model"
    assert b is c, "reselecting the same pair should be a no-op"
    d = eng.load("tiny/ckpt_best.pt", device="cpu")
    assert d is not b, "detaching the adapter must not reuse the adapted model"


def test_status_reports_the_attached_adapter(repo):
    eng = _engine(repo)
    eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/sft_best.lora.pt")
    st = eng.status()
    assert st["adapter"]["rel"] == "tiny/sft_best.lora.pt"
    assert st["stage"] == "sft"


def test_an_adapter_for_a_different_architecture_is_refused(repo):
    """The silent-degradation case. It must raise rather than load."""
    other = ModelConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = Transformer(other)
    lcfg = LoRAConfig(r=4, targets="attn")
    apply_lora(model, lcfg)
    save_adapter(repo / "checkpoints" / "tiny" / "wrong.lora.pt", model, lcfg,
                 base_identity({"model_config": dict(vars(other)),
                                "config": {"data": {"tokenizer": "tokenizer.json"}}},
                               "other.pt"),
                 extra={"training": {"stage": "sft"}})
    eng = _engine(repo)
    with pytest.raises(InferError, match="not trained on this checkpoint"):
        eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/wrong.lora.pt")


def test_an_unknown_adapter_is_refused(repo):
    eng = _engine(repo)
    with pytest.raises(InferError):
        eng.load("tiny/ckpt_best.pt", device="cpu", adapter="tiny/nope.lora.pt")


def test_generation_records_which_adapter_produced_it(repo):
    """`logs/playground.jsonl` is how output is compared across steps. With adapters in
    play, "which model said this" now includes which adapter was on."""
    eng = _engine(repo)
    stats = eng.generate("tiny/ckpt_best.pt", "complete", prompt="hello",
                         device="cpu", adapter="tiny/sft_best.lora.pt")
    assert stats["adapter"]["rel"] == "tiny/sft_best.lora.pt"
    assert stats["provenance"]["adapter"] == "tiny/sft_best.lora.pt"
    assert stats["provenance"]["stage"] == "sft"
