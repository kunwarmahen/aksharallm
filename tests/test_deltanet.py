"""Gated DeltaNet and the hybrid model (docs/04 § "Linear attention", model/deltanet.py).

The chunked form is only worth having if it is the SAME function as the three-line
recurrence. Everything here holds the two to each other, in float64, on lengths that are not
a multiple of the chunk, with values and gradients."""

from __future__ import annotations

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.model.deltanet import (
    GatedDeltaNet,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from aksharallm.model.transformer import Transformer


def _inputs(B=2, H=3, T=37, Dk=8, Dv=5, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, Dv, generator=g, dtype=torch.float64)
    # log decays from "keep almost all" to "forget a lot", and write strengths in (0, 1)
    lg = -torch.rand(B, H, T, generator=g, dtype=torch.float64) * 2.0
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    xs = [q, k, v, lg, beta]
    if requires_grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
def test_the_chunked_form_is_the_recurrence(chunk):
    """37 tokens: every chunk size leaves a ragged last chunk except 1."""
    xs = _inputs()
    ref = recurrent_gated_delta_rule(*xs)
    got = chunk_gated_delta_rule(*xs, chunk=chunk)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-10, rtol=0)


def test_the_gradients_agree_too():
    a = _inputs(requires_grad=True)
    b = [x.detach().clone().requires_grad_(True) for x in a]
    w = torch.randn(a[2].shape[:3] + (a[2].shape[-1],), dtype=torch.float64)
    (recurrent_gated_delta_rule(*a) * w).sum().backward()
    (chunk_gated_delta_rule(*b, chunk=16) * w).sum().backward()
    for x, y in zip(a, b):
        assert torch.allclose(x.grad, y.grad, atol=1e-9, rtol=0)


def test_the_delta_rule_recalls_an_overwritten_value_exactly():
    """What makes it a DELTA rule: write v1 at key k, then v2 at the same k with beta 1 and no
    decay, and reading k returns v2 alone. An additive linear attention returns v1 + v2."""
    k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 6, dtype=torch.float64), dim=-1)
    k = k.expand(1, 1, 2, 6)
    v = torch.randn(1, 1, 2, 4, dtype=torch.float64)
    out = chunk_gated_delta_rule(k, k, v, torch.zeros(1, 1, 2, dtype=torch.float64),
                                 torch.ones(1, 1, 2, dtype=torch.float64), chunk=2)
    assert torch.allclose(out[0, 0, 1], v[0, 0, 1], atol=1e-12)


def _layer_cfg(**kw):
    return ModelConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4, max_seq_len=128, **kw)


def test_the_layer_is_causal():
    """Changing token t must leave every output before t untouched, through the short
    convolution as well as the recurrence."""
    torch.manual_seed(0)
    layer = GatedDeltaNet(_layer_cfg(), chunk=8).double().eval()
    x = torch.randn(1, 30, 32, dtype=torch.float64)
    y = x.clone()
    y[:, 17] += 1.0
    a, b = layer(x), layer(y)
    assert torch.equal(a[:, :17], b[:, :17])
    assert not torch.allclose(a[:, 17:], b[:, 17:])


def test_the_hybrid_puts_full_attention_in_one_block_of_n():
    m = Transformer(_layer_cfg(full_attn_every=2))
    assert [type(b.attn).__name__ for b in m.blocks] == [
        "GatedDeltaNet", "Attention", "GatedDeltaNet", "Attention"]
    plain = Transformer(_layer_cfg())
    assert all(type(b.attn).__name__ == "Attention" for b in plain.blocks)


def test_a_hybrid_model_trains_and_refuses_a_kv_cache():
    torch.manual_seed(0)
    m = Transformer(_layer_cfg(full_attn_every=2))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 64, (4, 33))
    losses = []
    for _ in range(30):
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 1.0
    with pytest.raises(ValueError, match="recurrent state"):
        m(x[:, :4], caches=m.init_caches(4, 32, dtype=torch.float32, device="cpu"))


def test_the_config_refuses_what_a_recurrence_cannot_be():
    with pytest.raises(ValueError, match="causal"):
        _layer_cfg(full_attn_every=2, causal=False, mask_token_id=63)
    with pytest.raises(ValueError, match="attn_window"):
        _layer_cfg(full_attn_every=2, attn_window=16)


def test_a_hybrid_model_generates_by_rereading_its_prefix():
    """No KV cache for the recurrent blocks, so `generate` recomputes the prefix. Trained on
    one sequence where token 1 is followed by 2, 4, 6 and 8 in turn, so only a model reading
    its whole prefix can continue it (an untrained model barely depends on context, and a
    decode that saw only the last token would pass against one)."""
    from aksharallm.infer.generate import generate
    torch.manual_seed(0)
    seq = [1, 2, 3, 1, 4, 5, 1, 6, 7, 1, 8, 9, 1, 10, 11]
    m = Transformer(_layer_cfg(full_attn_every=2))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.tensor([seq])
    for _ in range(150):
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    got = generate(m, seq[:4], max_new_tokens=len(seq) - 4, temperature=0.0, device="cpu")
    assert got == seq


def test_the_hybrid_arm_is_its_control_plus_one_key():
    """tiny-adamw (2026-11-19) is the control; the hybrid differs in full_attn_every alone."""
    import yaml
    a = yaml.safe_load(open("configs/tiny-adamw.yaml"))
    b = yaml.safe_load(open("configs/tiny-hybrid.yaml"))
    assert "full_attn_every" not in a["model"] and b["model"].pop("full_attn_every") == 3
    assert b.pop("name") == "tiny-hybrid" and a.pop("name") == "tiny-adamw"
    assert b["train"].pop("out_dir") == "checkpoints/tiny-hybrid"
    a["train"].pop("out_dir")
    assert a == b
