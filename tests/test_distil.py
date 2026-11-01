"""Knowledge distillation (`train/distil.py`, docs/24).

Each test pins one way a distillation run can look healthy and be wrong: a chunked loss that
differs from the unchunked one, a temperature that silently turns the soft term down, a
teacher with another tokenizer, a quantized teacher, a teacher that gets trained.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aksharallm.config import Config, DataConfig, ModelConfig, TrainConfig, DistilConfig
from aksharallm.model.transformer import Transformer
from aksharallm.train.distil import (DistilError, DistilObjective, check_teacher, distil_loss,
                                     load_teacher)

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data" / "tinystories" / "tokenizer.json"
needs_tok = pytest.mark.skipif(not TOK.exists(), reason="needs the TinyStories tokenizer")


def logits(seed, shape=(2, 12, 50)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 3


# ---------------------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------------------


def test_a_student_equal_to_its_teacher_has_no_soft_loss():
    t = logits(0)
    y = torch.randint(0, 50, (2, 12))
    _, soft, hard = distil_loss(t.clone(), t, y, temperature=2.0, alpha=0.5)
    assert soft.abs() < 1e-5
    assert torch.allclose(hard, F.cross_entropy(t.reshape(-1, 50), y.reshape(-1)), atol=1e-5)


def test_alpha_zero_is_ordinary_pretraining():
    """The from-scratch control is the same config with alpha 0; it must be plain CE."""
    s, t = logits(1), logits(2)
    y = torch.randint(0, 50, (2, 12))
    loss, _, _ = distil_loss(s, t, y, alpha=0.0)
    assert torch.allclose(loss, F.cross_entropy(s.reshape(-1, 50), y.reshape(-1)), atol=1e-6)


def test_the_soft_term_is_the_kl_written_out_by_hand():
    s, t = logits(3), logits(4)
    y = torch.randint(0, 50, (2, 12))
    T = 3.0
    p = F.softmax(t / T, -1)
    want = (p * (p.log() - F.log_softmax(s / T, -1))).sum(-1).mean() * T * T
    _, soft, _ = distil_loss(s, t, y, temperature=T, alpha=1.0)
    assert torch.allclose(soft, want, atol=1e-5)


def test_chunking_changes_memory_and_nothing_else():
    """Chunks run under checkpoint and are recomputed in backward. The loss and every
    gradient must match the one-piece computation, or the memory saving costs correctness."""
    t = logits(5, (2, 37, 50))
    y = torch.randint(0, 50, (2, 37))
    y[0, :5] = -100                       # ignored positions must stay ignored per chunk
    grads, losses = [], []
    for chunk in (None, 8):               # 37 is not a multiple of 8: a ragged last chunk
        s = logits(6, (2, 37, 50)).requires_grad_()
        loss, _, _ = distil_loss(s, t, y, temperature=2.0, alpha=0.7, chunk=chunk)
        loss.backward()
        grads.append(s.grad)
        losses.append(loss.detach())
    assert torch.allclose(losses[0], losses[1], atol=1e-6)
    assert torch.allclose(grads[0], grads[1], atol=1e-7)


def test_an_ignored_position_contributes_nothing_to_either_term():
    """A -100 target (SFT padding, a masked prompt) must drop out of the soft term too, not
    only the hard one. Found by mutation: the chunking test compares two paths that share
    the same masking, so it cannot tell whether either one masks."""
    s, t = logits(11), logits(12)
    y = torch.randint(0, 50, (2, 12))
    y[:, :4] = -100
    t2 = t.clone()
    t2[:, :4] = logits(13)[:, :4]          # rewrite the teacher only where nothing counts
    a = distil_loss(s, t, y, alpha=1.0)[0]
    b = distil_loss(s, t2, y, alpha=1.0)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_the_t_squared_factor_keeps_the_soft_gradient_on_one_scale():
    """Without T^2, the soft gradient shrinks as 1/T^2 and raising the temperature would
    quietly hand the loss to the hard term. With it, T = 1 and T = 4 pull with comparable
    force: the ratio stays near 1 instead of near 1/16."""
    t = logits(7, (4, 64, 200))
    norms = {}
    for T in (1.0, 4.0):
        s = logits(8, (4, 64, 200)).requires_grad_()
        loss, _, _ = distil_loss(s, t, torch.zeros(4, 64, dtype=torch.long), T, alpha=1.0)
        loss.backward()
        norms[T] = s.grad.norm().item()
    assert 0.25 < norms[4.0] / norms[1.0] < 4.0, norms


def test_no_gradient_reaches_the_teacher_logits():
    s = logits(9).requires_grad_()
    t = logits(10).requires_grad_()
    loss, _, _ = distil_loss(s, t, torch.randint(0, 50, (2, 12)))
    loss.backward()
    assert t.grad is None


# ---------------------------------------------------------------------------------------
# the teacher
# ---------------------------------------------------------------------------------------


def tiny_cfg(**model):
    m = dict(vocab_size=8192, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    m.update(model)
    return Config(model=ModelConfig(**m), data=DataConfig(tokenizer=str(TOK)),
                  train=TrainConfig(seq_len=32))


def teacher_ckpt(tokenizer=TOK, **model):
    cfg = tiny_cfg(**model)
    torch.manual_seed(0)
    from dataclasses import asdict
    return {"model": Transformer(cfg.model).state_dict(), "model_config": asdict(cfg.model),
            "config": {"data": {"tokenizer": str(tokenizer)}}, "step": 99}


@needs_tok
def test_the_same_tokenizer_passes():
    check_teacher(teacher_ckpt(), tiny_cfg())


@needs_tok
def test_a_different_tokenizer_of_the_same_size_is_refused(tmp_path):
    """The case a vocab-size check alone misses: two 8,192-token vocabularies agree on the
    number and disagree on what every id means."""
    other = tmp_path / "other.json"
    other.write_text(TOK.read_text().replace('"Once"', '"Twice"', 1))
    with pytest.raises(DistilError, match="tokenizer mismatch"):
        check_teacher(teacher_ckpt(tokenizer=other), tiny_cfg())


@needs_tok
def test_the_same_tokenizer_at_another_path_passes(tmp_path):
    copy = tmp_path / "copy.json"
    copy.write_bytes(TOK.read_bytes())
    check_teacher(teacher_ckpt(tokenizer=copy), tiny_cfg())


@needs_tok
def test_a_teacher_that_does_not_record_its_tokenizer_is_refused():
    ck = teacher_ckpt()
    ck["config"] = {}
    with pytest.raises(DistilError, match="does not record its tokenizer"):
        check_teacher(ck, tiny_cfg())


@needs_tok
def test_a_vocabulary_mismatch_is_refused():
    with pytest.raises(DistilError, match="vocabulary mismatch"):
        check_teacher(teacher_ckpt(), tiny_cfg(vocab_size=8193))


@needs_tok
def test_a_diffusion_teacher_is_refused():
    ck = teacher_ckpt()
    ck["model_config"].update(causal=False, mask_token_id=8191)
    with pytest.raises(DistilError, match="not a next-token model"):
        check_teacher(ck, tiny_cfg())


@needs_tok
def test_a_teacher_with_a_shorter_window_is_refused():
    cfg = tiny_cfg(max_seq_len=64)
    cfg.train.seq_len = 64
    with pytest.raises(DistilError, match="window"):
        check_teacher(teacher_ckpt(max_seq_len=32), cfg)


@needs_tok
def test_an_moe_student_is_refused(tmp_path):
    path = tmp_path / "t.pt"
    torch.save(teacher_ckpt(), path)
    cfg = tiny_cfg(n_experts=4, moe_top_k=1)
    cfg.distil = DistilConfig(teacher=str(path))
    with pytest.raises(DistilError, match="MoE student"):
        DistilObjective(cfg)


@pytest.mark.parametrize("quant", [None, "int8", "nf4"])
@needs_tok
def test_the_teacher_is_frozen_and_quantizes_with_our_own_code(tmp_path, quant):
    path = tmp_path / "t.pt"
    ck = teacher_ckpt(d_model=64)
    torch.save(ck, path)
    teacher = load_teacher(path, quant, "cpu")
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    ref = Transformer(ModelConfig(**ck["model_config"]))
    ref.load_state_dict(ck["model"])
    x = torch.randint(0, 8192, (1, 16))
    want = ref(x, full_logits=True)[0]
    got = teacher(x, full_logits=True)[0]
    # Float is exact; a quantized teacher is close, not equal -- that difference is what the
    # "does a quantized teacher hurt the student?" experiment measures.
    if quant is None:
        assert torch.allclose(got, want, atol=1e-5)
    else:
        from aksharallm.quant import QuantLinear
        assert any(isinstance(m, QuantLinear) for m in teacher.modules())
        assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.5


# ---------------------------------------------------------------------------------------
# end to end, through pretrain.main
# ---------------------------------------------------------------------------------------


@needs_tok
def test_a_distillation_run_trains_through_the_ordinary_loop(tmp_path):
    """The objective is picked from `distil.teacher` alone, the student learns, the teacher's
    weights are untouched, and the step log carries both halves of the loss."""
    rng = np.random.default_rng(0)
    for name in ("train.bin", "val.bin"):
        rng.integers(0, 8192, size=20_000, dtype=np.uint16).tofile(tmp_path / name)
    tpath = tmp_path / "teacher.pt"
    ck = teacher_ckpt(d_model=64)
    torch.save(ck, tpath)
    out = tmp_path / "out"
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"""
name: distil-test
model: {{vocab_size: 8192, d_model: 32, n_layers: 2, n_heads: 4, max_seq_len: 32}}
data: {{train_bin: {tmp_path / 'train.bin'}, val_bin: {tmp_path / 'val.bin'}, tokenizer: {TOK}}}
optim: {{lr: 0.001, warmup_steps: 2}}
train: {{out_dir: {out}, batch_size: 2, grad_accum: 2, seq_len: 32, max_steps: 6,
        eval_every: 0, sample_every: 0, ckpt_every: 0, log_every: 1, eval_batches: 2,
        compile: false, seed: 1, resume: auto}}
distil: {{teacher: {tpath}, temperature: 2.0, alpha: 0.5, chunk: 8}}
""")
    proc = subprocess.run([sys.executable, "-m", "aksharallm.train.pretrain", str(cfg)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "objective    distillation" in proc.stdout
    import json
    recs = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in recs if "loss" in r and "step" in r]
    assert steps and all({"kd", "ce"} <= set(r.get("objective", {})) for r in steps)
    after = torch.load(tpath, map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(after[k], ck["model"][k]) for k in after)
