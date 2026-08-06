"""The Diffusion tab's back end, and the flag that keeps the two paradigms apart.

Nothing here loads a real tokenizer: the tab's own logic is bounds, refusals and the mapping
from token ids to display cells, and those are what break. The engine is stubbed so a test
can hand it whichever kind of model the case is about.

The refusals are the point. An autoregressive checkpoint denoised, or a diffusion checkpoint
sampled left-to-right, produces fluent nonsense and no error — so both directions are
asserted, here and in `test_diffusion.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer.checkpoints import CheckpointStore, InferError
from aksharallm.model.transformer import Transformer
from aksharallm.portal.diffusion import MAX_LENGTH, MAX_STEPS, Diffusion

VOCAB = 64
MASK = VOCAB - 1

DIFF_CFG = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2,
                       max_seq_len=32, causal=False, mask_token_id=MASK)
AR_CFG = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2,
                     max_seq_len=32)


class _Tok:
    """Enough tokenizer for a display cell: one letter per id."""

    vocab_size = VOCAB - 1
    bos_id = 0

    def encode(self, text, bos=False, eos=False):
        ids = [1 + (ord(c) % 30) for c in (text or "")]
        return ([self.bos_id] + ids) if bos else ids

    def decode(self, ids, skip_special=True):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


class _Info:
    rel = "tiny-diffusion/ckpt_best.pt"
    val_bin = None


class _Loaded:
    def __init__(self, model):
        self.model = model
        self.tokenizer = _Tok()
        self.device = "cpu"
        self.info = _Info()


class _Engine:
    def __init__(self, model):
        self._loaded = _Loaded(model)

    def load(self, ckpt_id, **kw):
        return self._loaded


class _Playground:
    def __init__(self, model, checkpoints=()):
        self.engine = _Engine(model)
        self._cks = list(checkpoints)

    def overview(self):
        return {"checkpoints": self._cks, "loaded": None, "device": "cpu", "plan": {}}


@pytest.fixture
def tab(tmp_path):
    torch.manual_seed(0)
    return Diffusion(_Playground(Transformer(DIFF_CFG).eval()), tmp_path)


# ---- refusals ---------------------------------------------------------------------------

def test_an_autoregressive_checkpoint_is_refused_with_a_reason(tmp_path):
    tab = Diffusion(_Playground(Transformer(AR_CFG).eval()), tmp_path)
    with pytest.raises(InferError, match="autoregressive"):
        tab.generate("tiny/ckpt_best.pt")


def test_empty_text_is_refused_rather_than_corrupted(tab):
    with pytest.raises(InferError, match="type something"):
        tab.corrupt_preview("x", "", 0.4)


def test_infilling_needs_at_least_one_end(tab):
    with pytest.raises(InferError, match="at least one end"):
        tab.infill("x", "  ", "")


def test_measuring_needs_a_validation_split(tab):
    with pytest.raises(InferError, match="validation split"):
        tab.measure("x")


# ---- the forward-process preview ----------------------------------------------------------

def test_the_preview_reports_the_weight_the_loss_would_apply(tab):
    out = tab.corrupt_preview("x", "hello there", 0.25)
    assert out["weight"] == 4.0                      # 1/t, the number the panel exists to show
    assert len(out["tokens"]) == len(out["masked"]) == out["n_tokens"]
    assert out["n_masked"] == sum(out["masked"])


def test_the_preview_is_repeatable_under_its_seed(tab):
    a = tab.corrupt_preview("x", "hello there world", 0.5, seed=7)
    b = tab.corrupt_preview("x", "hello there world", 0.5, seed=7)
    assert a["masked"] == b["masked"]


def test_a_mask_rate_of_zero_does_not_divide_by_zero(tab):
    out = tab.corrupt_preview("x", "hello", 0.0)
    assert out["n_masked"] == 0
    assert out["weight"] == 1000.0                   # clamped at t_min, not infinite


# ---- generation ---------------------------------------------------------------------------

def test_generate_returns_every_intermediate_state(tab):
    out = tab.generate("x", prompt="hi", length=12, steps=4)
    assert out["passes"] == len(out["steps"]) - 1
    assert out["steps"][0]["remaining"] == 12        # step 0 is the all-masked start
    assert out["steps"][-1]["remaining"] == 0
    # A cell is None exactly where the position is still masked — that None is what the
    # page renders as a blank, so it must not be an empty string.
    assert out["steps"][0]["cells"][-1] is None
    assert all(c is not None for c in out["steps"][-1]["cells"])


def test_the_prompt_is_reported_so_the_page_can_mark_it_as_given(tab):
    out = tab.generate("x", prompt="hello", length=8, steps=2)
    assert out["prefix_len"] == len(_Tok().encode("hello", bos=True))


def test_length_and_steps_are_clamped_not_rejected(tab):
    out = tab.generate("x", length=10_000, steps=10_000)
    # The model's context is 32, so the length ceiling bites after MAX_LENGTH does.
    assert len(out["steps"][0]["cells"]) <= DIFF_CFG.max_seq_len
    assert out["passes"] <= MAX_STEPS
    assert MAX_LENGTH == 256


def test_tokens_per_pass_is_the_ratio_the_paradigm_claims(tab):
    out = tab.generate("x", length=16, steps=4)
    assert out["tokens_per_pass"] == pytest.approx(4.0)


def test_infill_returns_the_middle_and_both_ends(tab):
    out = tab.infill("x", "before", "after", length=6, steps=3)
    assert out["prefix"] == "before" and out["suffix"] == "after"
    assert isinstance(out["middle"], str)
    assert out["steps"][-1]["remaining"] == 0


# ---- the flag on a real checkpoint ---------------------------------------------------------

def _save(root: Path, cfg: ModelConfig, run: str) -> Path:
    folder = root / "checkpoints" / run
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ckpt_best.pt"
    torch.save({"model": Transformer(cfg).state_dict(),
                "model_config": dict(vars(cfg)),
                "config": {"data": {"tokenizer": "t.json"}, "train": {"seq_len": 16}},
                "step": 10, "best_val": 1.5}, path)
    return path


def test_a_checkpoint_says_which_paradigm_it_is_without_loading_weights(tmp_path):
    """`mmap`-read from the header, so a picker can grey out what it cannot use."""
    _save(tmp_path, DIFF_CFG, "tiny-diffusion")
    _save(tmp_path, AR_CFG, "tiny")
    store = CheckpointStore(tmp_path)
    by_run = {c.run: c for c in store.list()}
    assert by_run["tiny-diffusion"].is_diffusion
    assert by_run["tiny-diffusion"].as_dict()["diffusion"] is True
    assert not by_run["tiny"].is_diffusion
    assert "bidirectional" in by_run["tiny-diffusion"].arch
    assert "bidirectional" not in by_run["tiny"].arch


def test_a_checkpoint_written_before_any_of_this_is_autoregressive(tmp_path):
    """Every `.pt` in this repo predates `causal`, and none of them carry the key."""
    folder = tmp_path / "checkpoints" / "old"
    folder.mkdir(parents=True)
    mcfg = dict(vars(AR_CFG))
    del mcfg["causal"], mcfg["mask_token_id"]
    torch.save({"model": {}, "model_config": mcfg,
                "config": {"data": {}}, "step": 1}, folder / "ckpt_last.pt")
    ck = CheckpointStore(tmp_path).get("old/ckpt_last.pt")
    assert ck.causal is True and not ck.is_diffusion


def test_the_overview_flags_when_there_is_nothing_to_denoise(tmp_path):
    """A picker that hides every checkpoint on a machine with no diffusion run looks
    broken; the tab says so instead."""
    pg = _Playground(Transformer(DIFF_CFG).eval(),
                     checkpoints=[{"rel": "tiny/ckpt_best.pt", "diffusion": False}])
    out = Diffusion(pg, tmp_path).overview()
    assert out["any"] is False
    assert out["checkpoints"][0]["diffusion"] is False
