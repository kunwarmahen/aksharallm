"""Correctness tests for the from-scratch transformer.

The KV-cache test is the important one: a cache bug produces a model that trains fine
and generates garbage, which is a miserable thing to debug later.
"""

import math

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import RMSNorm, Transformer, apply_rope, build_rope_cache


def tiny_cfg(**kw):
    base = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


def test_init_loss_is_uniform():
    """A freshly initialised model should predict ~uniformly => loss ~ ln(vocab_size)."""
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=8192, d_model=256, n_layers=4, n_heads=4, max_seq_len=128)
    m = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (4, 65))
    x, y = idx[:, :-1], idx[:, 1:]  # the shift that makes this next-token prediction
    with torch.no_grad():
        _, loss = m(x, targets=y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.25


def test_rmsnorm_matches_reference():
    x = torch.randn(3, 7, 16, dtype=torch.float32)
    n = RMSNorm(16)
    with torch.no_grad():
        n.weight.normal_()
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + n.eps) * n.weight
    assert torch.allclose(n(x), ref, atol=1e-5)


def test_rope_preserves_norm_and_relative_position():
    """RoPE is a rotation: it must not change vector norms, and the dot product between
    two rotated vectors must depend only on their *relative* offset."""
    cos, sin = build_rope_cache(head_dim=32, max_seq_len=64, theta=10000.0)
    q = torch.randn(1, 1, 64, 32)
    qr = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)

    v = torch.randn(32)
    # same vector placed at (5, 10) vs (20, 25) -> identical offset -> identical dot product
    a = apply_rope(v.expand(1, 1, 64, 32).clone(), cos, sin)
    d1 = (a[0, 0, 5] * a[0, 0, 10]).sum()
    d2 = (a[0, 0, 20] * a[0, 0, 25]).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


def test_causality():
    """Changing a future token must not change the logits at earlier positions."""
    torch.manual_seed(0)
    m = Transformer(tiny_cfg()).eval()
    idx = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        a, _ = m(idx, targets=idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % 512
        b, _ = m(idx2, targets=idx2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_kv_cache_matches_full_forward(n_kv_heads):
    """Token-by-token decoding with the cache must reproduce the full-sequence forward pass.
    Run in fp32 so we can use a tight tolerance."""
    torch.manual_seed(0)
    cfg = tiny_cfg(n_kv_heads=n_kv_heads)
    m = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        full, _ = m(idx, targets=idx)  # (B, T, V) -- all positions

        caches = m.init_caches(2, cfg.max_seq_len, dtype=torch.float32, device="cpu")
        # prefill on everything but the last token, then decode one at a time
        m(idx[:, :8], caches=caches)
        incremental = []
        for t in range(8, 16):
            logits, _ = m(idx[:, t : t + 1], caches=caches)
            incremental.append(logits[:, -1])
        inc = torch.stack(incremental, dim=1)  # (B, 8, V)

    # Feeding token t with a warm cache yields the logits *at* position t, which is
    # full[:, t] -- so the decode steps t=8..15 line up with full[:, 8:16].
    assert torch.allclose(full[:, 8:16], inc, atol=1e-4), (full[:, 8:16] - inc).abs().max()


def test_gqa_reduces_kv_cache():
    cfg = tiny_cfg(n_heads=8, n_kv_heads=2, d_model=64)
    m = Transformer(cfg)
    caches = m.init_caches(1, 32, dtype=torch.float32, device="cpu")
    assert caches[0].k.shape[1] == 2  # not 8


def test_weight_tying():
    m = Transformer(tiny_cfg(tie_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    m2 = Transformer(tiny_cfg(tie_embeddings=False))
    assert m2.lm_head.weight.data_ptr() != m2.tok_emb.weight.data_ptr()


def test_optimizer_param_grouping():
    m = Transformer(tiny_cfg())
    opt, (n_decay, n_nodecay) = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device_type="cpu")
    # every RMSNorm gain is 1-D and must land in the no-decay group
    expected_nodecay = sum(p.numel() for p in m.parameters() if p.dim() < 2)
    assert n_nodecay == expected_nodecay
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0
