"""Tests for the NF4 datatype and double quantization.

Both are additions to a storage format that already worked, which makes the risks here
almost entirely *silent* ones:

  * NF4 codes are indices into a table, not points on an even grid. Dequantizing one with
    the uniform formula produces plausible small numbers rather than an error.
  * The levels are asymmetric (7 negative, 8 positive). An off-by-one in the derivation
    still gives a monotonic table that quantizes and dequantizes fine — and is slightly
    wrong everywhere.
  * Double quantization compresses the scales. Getting the block padding or the mean
    wrong degrades every group in the last block, which no shape check would catch.

So these pin the numbers against the published table, and round-trip everything.
"""

import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.quant.convert import build_from_checkpoint, quantize_model, save_quantized
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import (
    DQ_BLOCK,
    NF4_LEVELS,
    QuantScheme,
    compress_scales,
    decompress_scales,
    fake_quantize,
    quantize_group,
)

#: The NF4 table as published with the QLoRA paper. `NF4_LEVELS` is *derived* from the
#: normal distribution rather than pasted in, so this is the test that the derivation is
#: the real thing and not merely something plausible with the right shape.
PUBLISHED_NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def test_derived_levels_match_the_published_table():
    got = NF4_LEVELS
    want = torch.tensor(PUBLISHED_NF4)
    assert got.shape == (16,)
    assert torch.allclose(got, want, atol=1e-6), f"derived {got.tolist()}"


def test_levels_are_sorted_span_minus_one_to_one_and_contain_exact_zero():
    assert torch.all(NF4_LEVELS[1:] > NF4_LEVELS[:-1]), "levels must be strictly increasing"
    assert NF4_LEVELS[0].item() == pytest.approx(-1.0)
    assert NF4_LEVELS[-1].item() == pytest.approx(1.0)
    # Exactly representable zero is the property that makes padded and dead weights
    # quantize to 0.0 rather than to something merely close.
    assert (NF4_LEVELS == 0.0).sum().item() == 1


def test_the_grid_is_denser_near_zero_than_in_the_tails():
    """The entire point of NF4: spend levels where a normal distribution has its mass."""
    gaps = (NF4_LEVELS[1:] - NF4_LEVELS[:-1]).tolist()
    middle = gaps[len(gaps) // 2]
    assert middle < gaps[0] and middle < gaps[-1]


def test_nf4_rejects_settings_that_have_no_meaning():
    with pytest.raises(ValueError, match="4-bit"):
        QuantScheme(bits=8, dtype="nf4")
    with pytest.raises(ValueError, match="sym"):
        QuantScheme(bits=4, dtype="nf4", sym=True)
    with pytest.raises(ValueError, match="dtype"):
        QuantScheme(bits=4, dtype="fp4")


def test_nf4_codes_are_indices_and_round_trip_exactly():
    torch.manual_seed(0)
    scheme = QuantScheme(bits=4, group_size=32, dtype="nf4")
    w = torch.randn(8, 64) * 0.02
    codes, scales, zeros = quantize_group(w, scheme, group_size=32)
    assert zeros is None, "NF4 has no zero-point"
    assert codes.min() >= 0 and codes.max() <= 15
    # The largest-magnitude weight in each group must land on an extreme level, because the
    # scale is exactly that group's absmax.
    g = w.reshape(8, 2, 32)
    hottest = g.abs().argmax(dim=-1)
    picked = codes.reshape(8, 2, 32).gather(-1, hottest.unsqueeze(-1)).squeeze(-1)
    assert set(picked.unique().tolist()) <= {0, 15}


@pytest.mark.parametrize("scheme", [
    QuantScheme(bits=4, group_size=32, dtype="nf4"),
    QuantScheme(bits=4, group_size=32, dtype="nf4", double_quant=True),
    QuantScheme(bits=4, group_size=-1, dtype="nf4"),
])
def test_quantlinear_storage_matches_fake_quantize(scheme):
    """The packed path and the arithmetic path must agree — this is what catches a packing
    or scale-storage bug, which otherwise shows up only as a slightly worse model.

    `scale_dtype=float16` is not a fudge factor: QuantLinear *stores* its scales in fp16,
    so fp32 scales are numerics the shipped layer does not have. Comparing against them
    would be comparing the storage path to something nothing ever runs.
    """
    torch.manual_seed(0)
    lin = nn.Linear(64, 16, bias=False)
    lin.weight.data.normal_(0, 0.02)
    q = QuantLinear.from_linear(lin, scheme)
    got = q.dequantize_weight(torch.float32)
    want = fake_quantize(lin.weight.data, scheme, group_size=q.group_size,
                         scale_dtype=torch.float16)
    assert torch.allclose(got, want, atol=1e-6)


def test_nf4_beats_int4_per_bit_on_gaussian_weights():
    """NF4 is fitted to a normal distribution, so on normal weights it should match int4's
    error while storing fewer bits (no zero-point). Both halves are asserted, because a
    scheme that is merely *smaller* is not the claim."""
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    nf4 = QuantScheme(bits=4, group_size=64, dtype="nf4")
    int4 = QuantScheme(bits=4, group_size=64, sym=False)
    e_nf4 = (w - fake_quantize(w, nf4)).norm() / w.norm()
    e_int4 = (w - fake_quantize(w, int4)).norm() / w.norm()
    assert e_nf4 < 1.15 * e_int4, f"nf4 {e_nf4:.4f} vs int4 {e_int4:.4f}"
    assert nf4.bits_per_weight(256) < int4.bits_per_weight(256)


# ---- double quantization ---------------------------------------------------------------


def _dq_tolerance(scales: torch.Tensor) -> float:
    """What double quantization actually promises.

    The error is int8 over a block re-centred on its mean, so it is bounded by
    `spread / 254` in *absolute* terms — it is not a relative bound per value. A scale that
    happens to sit near the block mean has tiny absolute error and could have arbitrarily
    large relative error, which is fine and is why this is the right thing to assert.

    In practice the scales inside one layer are far more tightly clustered than the uniform
    random ones used here, so this is a pessimistic bound.
    """
    return float((scales.max() - scales.min()).item()) / 254 + 1e-9


@pytest.mark.parametrize("n", [1, DQ_BLOCK - 1, DQ_BLOCK, DQ_BLOCK + 1, 3 * DQ_BLOCK])
def test_scale_compression_round_trips_at_every_padding_boundary(n):
    torch.manual_seed(0)
    scales = torch.rand(n).abs() * 0.01 + 1e-4
    codes, absmax, mean = compress_scales(scales)
    back = decompress_scales(codes, absmax, mean, scales.shape)
    assert back.shape == scales.shape
    assert torch.allclose(back, scales, rtol=0, atol=_dq_tolerance(scales))


def test_scale_compression_preserves_shape_for_a_matrix():
    torch.manual_seed(0)
    scales = torch.rand(7, 40) * 0.01
    back = decompress_scales(*compress_scales(scales), scales.shape)
    assert back.shape == (7, 40)
    assert torch.allclose(back, scales, rtol=0, atol=_dq_tolerance(scales))


def test_padding_a_short_block_does_not_drag_the_real_values():
    """The last block is padded to 256. Padding with zeros would pull that block's mean and
    absmax toward zero and cost precision on every real scale in it — so it pads with the
    last value instead. A run whose scale count is just over a block boundary is the case
    that would otherwise silently degrade."""
    scales = torch.full((DQ_BLOCK + 3,), 0.5)
    scales[:DQ_BLOCK] = torch.linspace(0.4, 0.6, DQ_BLOCK)
    back = decompress_scales(*compress_scales(scales), scales.shape)
    tail = back[DQ_BLOCK:]
    assert torch.allclose(tail, scales[DQ_BLOCK:], atol=1e-6), tail


def test_double_quantization_actually_stores_fewer_bytes():
    torch.manual_seed(0)
    lin = nn.Linear(256, 64, bias=False)
    plain = QuantLinear.from_linear(lin, QuantScheme(bits=4, group_size=64, dtype="nf4"))
    dq = QuantLinear.from_linear(
        lin, QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=True))
    assert dq.nbytes() < plain.nbytes()
    # ...and the reported bits/weight has to agree with the bytes actually held.
    for layer in (plain, dq):
        expected = layer.scheme.bits_per_weight(256) * 256 * 64 / 8
        assert layer.nbytes() == pytest.approx(expected, rel=0.02)


def test_double_quantization_barely_changes_the_weights():
    """If compressing the scales cost real accuracy it would not be worth the complexity;
    this pins that it does not."""
    torch.manual_seed(0)
    w = torch.randn(32, 256) * 0.02
    plain = QuantScheme(bits=4, group_size=64, dtype="nf4")
    dq = QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=True)
    e_plain = (w - fake_quantize(w, plain)).norm() / w.norm()
    e_dq = (w - fake_quantize(w, dq)).norm() / w.norm()
    assert e_dq < e_plain * 1.05


def test_scales_property_returns_fp16_either_way():
    torch.manual_seed(0)
    lin = nn.Linear(128, 32, bias=False)
    for double in (False, True):
        q = QuantLinear.from_linear(
            lin, QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=double))
        assert q.scales.dtype == torch.float16
        assert q.scales.shape == (32, 2)


# ---- whole models ----------------------------------------------------------------------


@pytest.mark.parametrize("double", [False, True])
def test_a_model_quantized_to_nf4_saves_and_reloads(tmp_path, double):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    model = Transformer(cfg).eval()
    scheme = QuantScheme(bits=4, group_size=32, dtype="nf4", double_quant=double)
    report = quantize_model(model, scheme)
    assert report.quantized, "nothing was quantized"

    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        before, _ = model(x)

    src = {"model_config": vars(cfg), "config": {"data": {"tokenizer": "t.json"}},
           "step": 1, "best_val": 1.0}
    path = save_quantized(tmp_path / "q.pt", model, scheme, report, src)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # The scheme must survive the round trip, or the reloaded model dequantizes with the
    # wrong grid and is silently wrong.
    assert ckpt["quant"]["scheme"]["dtype"] == "nf4"
    assert ckpt["quant"]["scheme"]["double_quant"] is double

    rebuilt = build_from_checkpoint(ckpt, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        after, _ = rebuilt(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_old_checkpoints_without_a_dtype_key_still_load():
    """Checkpoints written before NF4 existed have no `dtype`/`double_quant` in their
    scheme. They are int4/int8, and must keep loading as such."""
    scheme = QuantScheme.from_dict({"bits": 4, "group_size": 64, "sym": False,
                                    "method": "gptq"})
    assert scheme.dtype == "int"
    assert scheme.double_quant is False
    assert scheme.label() == "gptq-int4-g64-asym"


def test_labels_distinguish_every_variant():
    labels = {
        QuantScheme(bits=4, group_size=64, sym=False).label(),
        QuantScheme(bits=4, group_size=64, dtype="nf4").label(),
        QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=True).label(),
        QuantScheme(bits=8, group_size=64, sym=True).label(),
    }
    assert len(labels) == 4, labels
    assert "rtn-nf4-g64" in labels
    assert "rtn-nf4-g64-dq" in labels
