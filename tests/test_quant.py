"""Tests for quantization.

The failure modes worth guarding against here are all *silent*. A packing bug does not
raise, it returns a model that generates plausible-looking rubbish. A scale computed over
the wrong axis does not raise either -- it just makes the model slightly worse in a way
that looks like "well, that's 4-bit for you". So these tests pin the arithmetic first and
the plumbing second.
"""

import math

import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.quant.convert import (
    apply_quant_metadata,
    build_from_checkpoint,
    linear_layers,
    model_nbytes,
    quantize_model,
    save_quantized,
)
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import (
    QuantScheme,
    dequantize,
    fake_quantize,
    pack,
    pack4,
    quantize_group,
    resolve_group_size,
    unpack,
    unpack4,
)

ALL_SCHEMES = [
    QuantScheme(bits=8, group_size=64, sym=True),
    QuantScheme(bits=8, group_size=64, sym=False),
    QuantScheme(bits=8, group_size=-1, sym=True),
    QuantScheme(bits=4, group_size=64, sym=True),
    QuantScheme(bits=4, group_size=64, sym=False),
    QuantScheme(bits=4, group_size=-1, sym=False),
]


def tiny_cfg(**kw):
    base = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=32,
                d_ff=128, tie_embeddings=True)
    base.update(kw)
    return ModelConfig(**base)


# ---- the representation -------------------------------------------------------------

@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=lambda s: s.label())
def test_pack_unpack_is_lossless(scheme):
    """Packing must be a pure re-encoding. Any loss here is a bug, not quantization
    error -- the error happens in quantize_group, before this point."""
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    codes, _, _ = quantize_group(w, scheme)
    restored = unpack(pack(codes, scheme), scheme)
    assert torch.equal(restored.to(torch.int32), codes.to(torch.int32))


def test_symmetric_all_positive_group_still_round_trips():
    """The regression this file exists for. pack4 shifts symmetric codes by +8 to make
    the nibble unsigned; if that shift is decided by looking at the data ("does it contain
    a negative?") instead of the scheme, a group that happens to be all-positive is packed
    unshifted and unpacked shifted, and comes back 8 levels wrong."""
    scheme = QuantScheme(bits=4, group_size=-1, sym=True)
    w = torch.linspace(0.1, 1.0, 64).reshape(1, 64)  # strictly positive
    codes, scales, zeros = quantize_group(w, scheme)
    assert codes.min() >= 0
    restored = unpack4(pack4(codes, sym=True), sym=True)
    assert torch.equal(restored.to(torch.int32), codes.to(torch.int32))


def test_packing_actually_halves_the_bytes():
    scheme = QuantScheme(bits=4, group_size=64)
    codes, _, _ = quantize_group(torch.randn(16, 128), scheme)
    packed = pack(codes, scheme)
    assert packed.dtype == torch.uint8
    assert packed.numel() == codes.numel() // 2


def test_odd_in_features_cannot_be_packed():
    with pytest.raises(ValueError, match="odd"):
        pack4(torch.zeros(4, 7, dtype=torch.int8))


@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=lambda s: s.label())
def test_zero_survives_exactly(scheme):
    """A zero weight must dequantize to exactly zero. With an asymmetric scheme that is
    only true if the zero-point is an integer code -- a fractional one leaves a small DC
    offset on every zero in the model, which is precisely the kind of error that is
    invisible per-layer and fatal in aggregate."""
    w = torch.zeros(4, 128)
    assert fake_quantize(w, scheme).abs().max().item() == 0.0


@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=lambda s: s.label())
def test_dequantize_inverts_quantize_within_one_step(scheme):
    """Every weight must land within half a quantization step of its original value.
    This is the definition of correct rounding, and it catches an off-by-one in the
    clamp range or a scale that is too small to reach the extremes."""
    torch.manual_seed(1)
    w = torch.randn(8, 128)
    g = 128 if scheme.group_size == -1 else scheme.group_size
    codes, scales, zeros = quantize_group(w, scheme)
    deq = dequantize(codes, scales, zeros, g)
    step = scales.repeat_interleave(g, dim=1)
    assert ((w - deq).abs() <= step * 0.5 + 1e-6).all()


def test_more_bits_is_never_worse():
    torch.manual_seed(2)
    w = torch.randn(16, 256)
    err = lambda s: (w - fake_quantize(w, s)).pow(2).mean().item()  # noqa: E731
    assert err(QuantScheme(bits=8, group_size=64)) < err(QuantScheme(bits=4, group_size=64))


def test_smaller_groups_are_never_worse():
    """The whole justification for storing a scale per 64 weights instead of per row."""
    torch.manual_seed(3)
    w = torch.randn(16, 256)
    err = lambda g: (w - fake_quantize(w, QuantScheme(bits=4, group_size=g))).pow(2).mean()  # noqa: E731
    assert err(32) < err(64) < err(128) < err(-1)


def test_asymmetric_beats_symmetric_on_a_skewed_group():
    """A group that never goes negative wastes half a symmetric grid. This is why
    asymmetric is the default at 4 bits, where there are only 16 levels to waste."""
    w = torch.rand(8, 128) + 1.0  # all in [1, 2]
    sym = (w - fake_quantize(w, QuantScheme(bits=4, group_size=64, sym=True))).pow(2).mean()
    asym = (w - fake_quantize(w, QuantScheme(bits=4, group_size=64, sym=False))).pow(2).mean()
    assert asym < sym


def test_constant_group_round_trips():
    """A group of identical values has zero range; the scale must not become zero or the
    dequantized weight is NaN. Real models have such groups -- a dead channel is one."""
    w = torch.full((4, 64), 0.37)
    out = fake_quantize(w, QuantScheme(bits=4, group_size=64))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, w, atol=1e-3)


# ---- group size resolution ---------------------------------------------------------

def test_group_size_falls_back_to_a_divisor():
    """Our own 300M config has d_ff=2752, which 128 does not divide. The SwiGLU down
    projection reduces over d_ff, so a blanket group_size=128 would break exactly one
    layer per block; it degrades to 64 instead of failing."""
    assert resolve_group_size(128, 2752) == 64
    assert resolve_group_size(128, 1024) == 128
    assert resolve_group_size(-1, 2752) == -1


def test_quantize_group_rejects_an_indivisible_group():
    with pytest.raises(ValueError, match="does not divide"):
        quantize_group(torch.randn(4, 100), QuantScheme(bits=4, group_size=64), group_size=64)


# ---- QuantLinear -------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=lambda s: s.label())
def test_qlinear_matches_its_own_dequantized_weight(scheme):
    """QuantLinear.forward must equal F.linear with the weight it claims to hold. If
    these disagree, the packing and the forward path have drifted apart."""
    torch.manual_seed(4)
    lin = nn.Linear(128, 64, bias=False)
    q = QuantLinear.from_linear(lin, scheme)
    x = torch.randn(3, 5, 128)
    expected = torch.nn.functional.linear(x, q.dequantize_weight(x.dtype))
    assert torch.allclose(q(x), expected, atol=1e-5)


def test_qlinear_is_close_to_the_float_layer_at_int8():
    torch.manual_seed(5)
    lin = nn.Linear(256, 128, bias=False)
    q = QuantLinear.from_linear(lin, QuantScheme(bits=8, group_size=64, sym=True))
    x = torch.randn(4, 256)
    rel = (q(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01, f"int8 should be near-free, got {rel:.4f} relative error"


def test_qlinear_holds_no_float_weight():
    """The point of the exercise. If a `weight` attribute survives, nothing was saved."""
    q = QuantLinear.from_linear(nn.Linear(128, 64, bias=False),
                                QuantScheme(bits=4, group_size=64))
    assert not hasattr(q, "weight")
    assert q.nbytes() < q.float_nbytes() / 3


def test_qlinear_refuses_a_bias():
    lin = nn.Linear(64, 32, bias=True)
    with pytest.raises(ValueError, match="bias"):
        QuantLinear.from_linear(lin, QuantScheme())


def test_qlinear_buffers_are_not_parameters():
    """Quantized weights carry no gradient and must never reach the optimiser or weight
    decay -- registering them as Parameters would silently make a fine-tune update bytes
    that are then reinterpreted as packed nibbles."""
    q = QuantLinear.from_linear(nn.Linear(64, 32, bias=False), QuantScheme())
    assert list(q.parameters()) == []


# ---- whole-model conversion ---------------------------------------------------------

def test_quantize_model_replaces_every_linear_but_the_tied_head():
    model = Transformer(tiny_cfg())
    n_linear = len(linear_layers(model))
    report = quantize_model(model, QuantScheme(bits=4, group_size=64))
    remaining = linear_layers(model)
    assert len(remaining) == 1 and "lm_head" in next(iter(remaining))
    assert len(report.quantized) == n_linear - 1
    assert "tied" in report.skipped[0].skipped


def test_tied_head_is_quantized_on_request():
    model = Transformer(tiny_cfg())
    report = quantize_model(model, QuantScheme(bits=4, group_size=64), quantize_head=True)
    assert linear_layers(model) == {}
    assert report.skipped == []


def test_untied_head_is_quantized_by_default():
    """The skip exists because of the *tie*, not because lm_head is special."""
    model = Transformer(tiny_cfg(tie_embeddings=False))
    quantize_model(model, QuantScheme(bits=4, group_size=64))
    assert linear_layers(model) == {}


def test_quantizing_shrinks_the_model():
    model = Transformer(tiny_cfg())
    before = model_nbytes(model)
    quantize_model(model, QuantScheme(bits=4, group_size=64))
    assert model_nbytes(model) < before


def test_quantized_model_still_runs_and_stays_finite():
    torch.manual_seed(6)
    model = Transformer(tiny_cfg())
    model.eval()
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref, _ = model(x, targets=x)
        quantize_model(model, QuantScheme(bits=8, group_size=64, sym=True))
        out, _ = model(x, targets=x)
    assert torch.isfinite(out).all()
    # int8 should barely move the logits.
    assert (out - ref).abs().max() < 0.5 * ref.abs().max()


def test_int8_costs_less_quality_than_int4():
    """The headline ordering. If this ever inverts, a scale axis is wrong."""
    torch.manual_seed(7)
    x = torch.randint(0, 128, (2, 16))
    base = Transformer(tiny_cfg())
    base.eval()
    with torch.no_grad():
        ref, _ = base(x, targets=x)

    def err(scheme):
        m = Transformer(tiny_cfg())
        m.load_state_dict(base.state_dict())
        m.eval()
        quantize_model(m, scheme)
        with torch.no_grad():
            out, _ = m(x, targets=x)
        return (out - ref).pow(2).mean().item()

    assert err(QuantScheme(bits=8, group_size=64)) < err(QuantScheme(bits=4, group_size=64))


def test_report_totals_do_not_double_count_the_tied_embedding():
    """With tie_embeddings the skipped lm_head and tok_emb are one allocation. Counting
    it in both the layer table and 'everything else' would overstate the float baseline
    and flatter the compression ratio."""
    model = Transformer(tiny_cfg())
    report = quantize_model(model, QuantScheme(bits=4, group_size=64))
    cfg = tiny_cfg()
    emb_bytes = cfg.vocab_size * cfg.d_model * 2
    assert report.other_bytes < emb_bytes  # the embedding is charged to lm_head, once


# ---- save / load --------------------------------------------------------------------

def _fake_checkpoint(model, cfg):
    return {
        "model": model.state_dict(),
        "model_config": {k: getattr(cfg, k) for k in
                         ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                          "d_ff", "max_seq_len", "tie_embeddings")},
        "config": {"data": {"tokenizer": "data/tinystories/tokenizer.json"}},
        "step": 42, "best_val": 1.5,
    }


@pytest.mark.parametrize("scheme", [QuantScheme(bits=4, group_size=64),
                                    QuantScheme(bits=8, group_size=64, sym=True)],
                         ids=lambda s: s.label())
def test_quantized_checkpoint_round_trips_through_disk(tmp_path, scheme):
    torch.manual_seed(8)
    cfg = tiny_cfg()
    model = Transformer(cfg)
    model.eval()
    src = _fake_checkpoint(model, cfg)

    quantize_model(model, scheme)
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        before, _ = model(x, targets=x)

    path = tmp_path / "q.pt"
    save_quantized(path, model, scheme, _empty_report(scheme), src, source_path="orig.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    reloaded = build_from_checkpoint(ckpt, device="cpu", dtype=torch.float32)
    reloaded.eval()
    with torch.no_grad():
        after, _ = reloaded(x, targets=x)
    assert torch.allclose(before, after, atol=1e-5)


def _empty_report(scheme):
    from aksharallm.quant.convert import QuantReport

    return QuantReport(scheme=scheme)


def test_saved_checkpoint_keeps_the_tokenizer_path(tmp_path):
    """The BPE vocabulary *is* the embedding index. A quantized checkpoint that loses
    which tokenizer it belongs to decodes to fluent nonsense, and the inference engine
    refuses to load it -- so the metadata must survive quantization."""
    cfg = tiny_cfg()
    model = Transformer(cfg)
    src = _fake_checkpoint(model, cfg)
    scheme = QuantScheme(bits=4, group_size=64)
    report = quantize_model(model, scheme)
    path = tmp_path / "q.pt"
    save_quantized(path, model, scheme, report, src, source_path="orig.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["config"]["data"]["tokenizer"] == "data/tinystories/tokenizer.json"
    assert ckpt["step"] == 42
    assert ckpt["quant"]["scheme"]["bits"] == 4
    assert ckpt["quant"]["source"] == "orig.pt"


def test_loading_quantized_weights_into_a_float_model_fails_loudly(tmp_path):
    """The good failure. If apply_quant_metadata is skipped, load_state_dict must raise
    rather than quietly leaving the float weights in place -- a model that loaded
    'successfully' but was never quantized is the worst outcome here."""
    cfg = tiny_cfg()
    model = Transformer(cfg)
    src = _fake_checkpoint(model, cfg)
    scheme = QuantScheme(bits=4, group_size=64)
    report = quantize_model(model, scheme)
    path = tmp_path / "q.pt"
    save_quantized(path, model, scheme, report, src, source_path=None)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    plain = Transformer(cfg)
    with pytest.raises(RuntimeError):
        plain.load_state_dict(ckpt["model"])


def test_apply_quant_metadata_rebuilds_the_right_shapes(tmp_path):
    cfg = tiny_cfg()
    model = Transformer(cfg)
    scheme = QuantScheme(bits=4, group_size=64)
    report = quantize_model(model, scheme)
    src = _fake_checkpoint(Transformer(cfg), cfg)
    path = tmp_path / "q.pt"
    save_quantized(path, model, scheme, report, src)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    fresh = Transformer(cfg)
    apply_quant_metadata(fresh, ckpt["quant"])
    qlayers = [m for m in fresh.modules() if isinstance(m, QuantLinear)]
    assert len(qlayers) == len(ckpt["quant"]["layers"])
    fresh.load_state_dict(ckpt["model"])  # must not raise


# ---- scheme bookkeeping -------------------------------------------------------------

def test_bits_per_weight_includes_the_scales():
    """4-bit is never 4 bits per weight -- the scales are real bytes. Quoting the nominal
    number is how a '4x smaller' claim turns into 3.7x on the scale."""
    s = QuantScheme(bits=4, group_size=64, sym=False)
    assert 4.3 < s.bits_per_weight(1024) < 4.5
    coarse = QuantScheme(bits=4, group_size=-1, sym=False)
    assert coarse.bits_per_weight(1024) < s.bits_per_weight(1024)


def test_scheme_round_trips_through_a_dict():
    s = QuantScheme(bits=4, group_size=128, sym=True, method="gptq")
    assert QuantScheme.from_dict(s.as_dict()) == s


@pytest.mark.parametrize("bits", [1, 3, 16])
def test_unsupported_bit_widths_are_rejected(bits):
    with pytest.raises(ValueError, match="bits must be"):
        QuantScheme(bits=bits)
