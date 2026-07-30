"""Tests for the calibration-based methods: GPTQ and AWQ.

Both are easy to implement in a way that runs, produces a model, and is *worse than RTN*
while looking fine. So the tests here are mostly comparative: each method must actually
beat the baseline it claims to beat, on data with the structure it claims to exploit.
"""

import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.quant.awq import (
    ALPHA_GRID,
    _gqa_share,
    apply_awq,
    channel_importance,
    search_scale,
)
from aksharallm.quant.calib import Calibration, LayerStats, collect, damped_hessian
from aksharallm.quant.convert import quantize_model
from aksharallm.quant.gptq import gptq_quantize_weight, make_gptq_quantizer
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import QuantScheme

INT4 = QuantScheme(bits=4, group_size=64, sym=False)


def correlated_inputs(n: int, in_f: int, seed: int = 0) -> torch.Tensor:
    """Inputs with real channel correlation and uneven per-channel energy -- the only
    conditions under which GPTQ and AWQ have anything to work with. On white noise with
    uniform variance, both correctly reduce to something very close to RTN."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(in_f, in_f, generator=g) * 0.3 + torch.eye(in_f)
    x = torch.randn(n, in_f, generator=g) @ mix
    scale = torch.logspace(-1, 1, in_f)  # channel j is far more energetic than channel 0
    return x * scale


def output_mse(x, w_ref, w_hat):
    return (x @ w_hat.T.float() - x @ w_ref.T.float()).pow(2).mean().item()


def tiny_cfg(**kw):
    base = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2,
                max_seq_len=32, d_ff=128, tie_embeddings=True)
    base.update(kw)
    return ModelConfig(**base)


def fake_calibration(model, seed=0, hessian=False):
    """Plausible per-layer statistics without running any data through the model."""
    g = torch.Generator().manual_seed(seed)
    calib = Calibration()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            st = LayerStats(in_features=mod.in_features, n_samples=512)
            st.abs_mean = torch.rand(mod.in_features, generator=g) * 3 + 0.1
            st.abs_max = st.abs_mean * 4
            if hessian:
                a = torch.randn(256, mod.in_features, generator=g) * st.abs_mean
                st.hessian = (a.T @ a) / 256
            calib.stats[name] = st
    return calib


# ---- Hessian handling ---------------------------------------------------------------

def test_damping_makes_a_singular_hessian_invertible():
    """A channel that is dead over the calibration set gives an all-zero row and column.
    Cholesky then fails outright -- or, worse, a slightly-negative eigenvalue sneaks
    through and the inverse comes back as noise."""
    h = torch.randn(32, 8)
    h = h @ h.T  # rank 8 in 32 dimensions: very singular
    with pytest.raises(RuntimeError):
        torch.linalg.cholesky(h)
    torch.linalg.cholesky(damped_hessian(h))  # must not raise


def test_damping_zeroes_are_replaced_not_just_padded():
    h = torch.eye(8)
    h[3, 3] = 0.0
    d = damped_hessian(h)
    assert d[3, 3] > 0


def test_collect_records_every_linear(tmp_path):
    """Calibration must hook the real modules, and the shapes must match in_features --
    a transposed hook is a class of bug that only shows up as a bad model much later."""
    import numpy as np

    cfg = tiny_cfg()
    model = Transformer(cfg).eval()
    bin_path = tmp_path / "val.bin"
    np.arange(4096, dtype=np.uint16).__mod__(cfg.vocab_size).astype(np.uint16).tofile(bin_path)

    calib = collect(model, str(bin_path), seq_len=16, n_sequences=8, batch_size=4,
                    device="cpu", want_hessian=True)
    assert calib.n_sequences == 8
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            st = calib.get(name)
            assert st is not None, f"{name} was not calibrated"
            assert st.hessian.shape == (mod.in_features, mod.in_features)
            assert st.abs_mean.shape == (mod.in_features,)


def test_free_releases_the_hessian():
    """A 2752x2752 fp32 Hessian is 30 MB. Holding all of a 24-block model's at once is
    over a gigabyte, so GPTQ frees each one as it finishes the layer."""
    model = Transformer(tiny_cfg())
    calib = fake_calibration(model, hessian=True)
    name = next(iter(calib.stats))
    assert calib.get(name).hessian is not None
    calib.free(name)
    assert calib.get(name).hessian is None


# ---- GPTQ ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", [
    QuantScheme(bits=4, group_size=64, sym=False),
    QuantScheme(bits=4, group_size=-1, sym=False),
    QuantScheme(bits=8, group_size=64, sym=True),
], ids=lambda s: s.label())
def test_gptq_beats_rtn_on_correlated_inputs(scheme):
    """The point of the algorithm. If this fails, the error compensation is either not
    happening or is being applied with the wrong sign."""
    torch.manual_seed(0)
    in_f, out_f = 256, 128
    x = correlated_inputs(2048, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    h = (x.T @ x) / x.shape[0]
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight.data = w.clone()

    rtn = QuantLinear.from_linear(lin, scheme).dequantize_weight(torch.float32)
    packed, scales, zeros = gptq_quantize_weight(w, h, scheme)
    gptq = QuantLinear.from_linear(
        lin, scheme, qweight=packed, scales=scales, zeros=zeros
    ).dequantize_weight(torch.float32)

    assert output_mse(x, w, gptq) < output_mse(x, w, rtn)


def test_gptq_produces_the_same_storage_format_as_rtn():
    """GPTQ is a smarter search, not a different representation -- everything downstream
    (packing, the kernel, the checkpoint format) must be shared."""
    torch.manual_seed(1)
    w = torch.randn(64, 128) * 0.05
    h = torch.eye(128)
    packed, scales, zeros = gptq_quantize_weight(w, h, INT4)
    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 64)          # 4-bit: two per byte
    assert scales.shape == (64, 2)           # 128 / group 64
    assert zeros.shape == (64, 2)


def test_gptq_with_an_identity_hessian_is_close_to_rtn():
    """With uncorrelated, equal-energy inputs there is nothing to compensate *between*
    columns, so GPTQ should land near RTN. A large divergence here means the update is
    firing on curvature that isn't there."""
    torch.manual_seed(2)
    in_f = 128
    w = torch.randn(32, in_f) * 0.05
    lin = nn.Linear(in_f, 32, bias=False)
    lin.weight.data = w.clone()
    rtn = QuantLinear.from_linear(lin, INT4).dequantize_weight(torch.float32)
    packed, scales, zeros = gptq_quantize_weight(w, torch.eye(in_f), INT4)
    gptq = QuantLinear.from_linear(lin, INT4, qweight=packed, scales=scales,
                                   zeros=zeros).dequantize_weight(torch.float32)
    x = torch.randn(512, in_f)
    assert output_mse(x, w, gptq) < 2 * output_mse(x, w, rtn)


def test_gptq_handles_a_dead_channel():
    """A channel with no activation over the calibration set contributes a zero row and
    column. It must not produce NaNs in the rest of the layer."""
    torch.manual_seed(3)
    in_f = 128
    x = correlated_inputs(512, in_f)
    x[:, 5] = 0.0
    w = torch.randn(32, in_f) * 0.05
    h = (x.T @ x) / x.shape[0]
    packed, scales, zeros = gptq_quantize_weight(w, h, INT4)
    assert torch.isfinite(scales.float()).all()
    assert torch.isfinite(zeros.float()).all()


def test_gptq_quantizer_falls_back_loudly_without_stats():
    """A layer with no Hessian gets RTN — but the caller must be able to find out, or a
    'GPTQ model' could be mostly RTN and nobody would know."""
    model = Transformer(tiny_cfg())
    calib = Calibration()  # deliberately empty
    q = make_gptq_quantizer(calib)
    quantize_model(model, INT4, quantizer=q)
    assert len(q.fell_back) > 0


def test_gptq_runs_over_a_whole_model():
    torch.manual_seed(4)
    model = Transformer(tiny_cfg()).eval()
    calib = fake_calibration(model, hessian=True)
    x = torch.randint(0, 128, (2, 16))
    report = quantize_model(model, INT4, quantizer=make_gptq_quantizer(calib))
    with torch.no_grad():
        out, _ = model(x, targets=x)
    assert torch.isfinite(out).all()
    assert len(report.quantized) > 0


# ---- AWQ ----------------------------------------------------------------------------

def test_awq_fold_leaves_the_model_function_unchanged():
    """The whole premise: `x W^T == (x/s)(W diag(s))^T`. AWQ rewrites the model but must
    not change what it computes -- before quantization, outputs must be bit-for-bit
    equivalent within float noise. A fold applied to the wrong axis still runs, still
    produces plausible logits, and is completely wrong."""
    torch.manual_seed(5)
    model = Transformer(tiny_cfg()).eval()
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref, _ = model(x, targets=x)
    apply_awq(model, fake_calibration(model), INT4)
    with torch.no_grad():
        out, _ = model(x, targets=x)
    assert torch.allclose(ref, out, atol=1e-3), (ref - out).abs().max()


def test_awq_fold_is_exact_under_plain_mha_too():
    """The GQA sharing constraint must not break the n_kv_heads == n_heads case."""
    torch.manual_seed(6)
    model = Transformer(tiny_cfg(n_kv_heads=8)).eval()
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref, _ = model(x, targets=x)
    apply_awq(model, fake_calibration(model), INT4)
    with torch.no_grad():
        out, _ = model(x, targets=x)
    assert torch.allclose(ref, out, atol=1e-3)


def test_gqa_share_makes_grouped_heads_agree():
    """Under GQA several query heads read one value head, so the fold into wv can only
    apply one scale to all of them. If their scales were allowed to differ, folding would
    silently change the layer's function."""
    n_heads, n_kv, hd = 8, 2, 4
    imp = torch.arange(n_heads * hd, dtype=torch.float32)
    shared = _gqa_share(imp, n_heads, n_kv, hd)
    per_head = shared.reshape(n_heads, hd)
    n_rep = n_heads // n_kv
    for k in range(n_kv):
        group = per_head[k * n_rep:(k + 1) * n_rep]
        assert torch.allclose(group, group[0].expand_as(group))


def test_gqa_share_is_identity_without_gqa():
    imp = torch.rand(32)
    assert torch.equal(_gqa_share(imp, 8, 8, 4), imp)


def test_search_scale_can_choose_to_do_nothing():
    """alpha=0 must be reachable. On a layer where scaling does not help, AWQ returning
    a distortion anyway would make it strictly worse than RTN."""
    torch.manual_seed(7)
    w = [torch.randn(32, 128) * 0.05]
    imp = torch.ones(128)  # perfectly uniform: nothing to exploit
    s, alpha, gain = search_scale(w, imp, INT4)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3) or gain >= 1.0


def test_search_scale_never_returns_worse_than_baseline():
    """alpha=0 is in the grid and is the baseline, so the winner can never be worse."""
    torch.manual_seed(8)
    w = [torch.randn(64, 256) * 0.05]
    imp = torch.logspace(-2, 2, 256)
    _s, _alpha, gain = search_scale(w, imp, INT4)
    assert gain >= 1.0


def test_awq_helps_when_channel_importance_is_uneven():
    """The condition AWQ is designed for: a few channels carry most of the energy."""
    torch.manual_seed(9)
    in_f, out_f = 256, 128
    x = correlated_inputs(2048, in_f, seed=9)
    w = torch.randn(out_f, in_f) * 0.05
    imp = (x * x).mean(dim=0)

    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight.data = w.clone()
    rtn = QuantLinear.from_linear(lin, INT4).dequantize_weight(torch.float32)

    s, _alpha, _gain = search_scale([w], imp, INT4)
    lin.weight.data = w * s.unsqueeze(0)
    scaled = QuantLinear.from_linear(lin, INT4).dequantize_weight(torch.float32)
    awq = scaled / s.unsqueeze(0)

    assert output_mse(x, w, awq) < output_mse(x, w, rtn)


def test_alpha_grid_includes_both_ends():
    assert ALPHA_GRID[0] == 0.0 and ALPHA_GRID[-1] == 1.0


def test_channel_importance_prefers_the_hessian_diagonal():
    st = LayerStats(in_features=4, n_samples=10)
    st.abs_mean = torch.ones(4)
    st.hessian = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(channel_importance(st), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    st.hessian = None
    assert torch.allclose(channel_importance(st), torch.ones(4))


def test_awq_reports_every_site_it_touched():
    model = Transformer(tiny_cfg())
    rep = apply_awq(model, fake_calibration(model), INT4)
    # 4 fold sites per block: attn_norm->qkv, wv->wo, ffn_norm->w1w3, w3->w2
    assert rep["n_sites"] == 4 * len(model.blocks)
    assert rep["mean_gain"] >= 1.0


def test_awq_can_skip_the_attention_output_site():
    model = Transformer(tiny_cfg())
    rep = apply_awq(model, fake_calibration(model), INT4, include_attn_out=False)
    assert rep["n_sites"] == 3 * len(model.blocks)
    assert len(rep["skipped"]) == len(model.blocks)


def test_awq_then_quantize_gives_a_working_model():
    torch.manual_seed(10)
    model = Transformer(tiny_cfg()).eval()
    apply_awq(model, fake_calibration(model), INT4)
    quantize_model(model, INT4)
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out, _ = model(x, targets=x)
    assert torch.isfinite(out).all()
