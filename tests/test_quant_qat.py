"""Tests for quantization-aware training.

The straight-through estimator is the thing to pin. It is three tokens of code and every
part of it is load-bearing: get the `.detach()` wrong and either the forward pass is not
quantized (so QAT does nothing and you cannot tell) or the gradient is zero (so nothing
trains, and you also cannot tell, because the loss simply stops moving and that looks like
convergence).
"""

import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.quant.convert import linear_layers, model_nbytes
from aksharallm.quant.qat import QATLinear, convert_qat, prepare_qat, train_qat
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import QuantScheme, fake_quantize

INT4 = QuantScheme(bits=4, group_size=64, sym=False)


def tiny_cfg(**kw):
    base = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=32,
                d_ff=128, tie_embeddings=True)
    base.update(kw)
    return ModelConfig(**base)


# ---- the straight-through estimator --------------------------------------------------

def test_forward_value_is_actually_quantized():
    """The whole point: training must see the *quantized* weight, not the float one. If
    this fails, QAT is an expensive no-op that reports plausible losses throughout.

    The comparison uses fp16 scales because that is what QuantLinear stores; training
    against fp32 scales would be simulating a slightly better model than the one that
    ships (see `test_convert_preserves_what_training_saw`, which is what caught it)."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=False)
    q = QATLinear(lin, INT4)
    expected = fake_quantize(lin.weight.data.float(), INT4, group_size=q.group_size,
                             scale_dtype=torch.float16)
    assert torch.allclose(q.quantized_weight(), expected, atol=1e-6)
    assert not torch.allclose(q.quantized_weight(), lin.weight.data, atol=1e-6)


def test_qat_simulates_the_stored_scale_precision():
    """Scales live in fp16 on disk and in QuantLinear. Fake-quantizing with fp32 scales
    gives a *different* weight, so QAT has to round them too or the converted model is
    not the model that was trained."""
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    fp32 = fake_quantize(w, INT4)
    fp16 = fake_quantize(w, INT4, scale_dtype=torch.float16)
    assert not torch.equal(fp32, fp16)
    # Small -- fp16 has ~3 decimal digits, so scales shift by ~1e-4 relative -- but a
    # systematic shift applied to every weight in the model, not random noise.
    rel = (fp32 - fp16).abs().max() / fp32.abs().max()
    assert rel < 1e-2, rel


def test_gradient_passes_straight_through():
    """d(w_q)/dw must be exactly 1, not the true derivative of rounding (which is 0
    almost everywhere and would stop training dead)."""
    torch.manual_seed(1)
    lin = nn.Linear(64, 32, bias=False)
    q = QATLinear(lin, INT4)
    x = torch.randn(4, 64)
    q(x).sum().backward()
    assert q.weight.grad is not None
    assert q.weight.grad.abs().sum() > 0, "no gradient reached the weight"

    # The gradient must equal what a plain Linear would have produced.
    plain = nn.Linear(64, 32, bias=False)
    plain.weight.data.copy_(lin.weight.data)
    plain(x).sum().backward()
    assert torch.allclose(q.weight.grad, plain.weight.grad, atol=1e-5)


def test_the_weight_parameter_is_shared_not_copied():
    """QATLinear must train the *same* Parameter, or an optimiser built before the swap
    updates a tensor nobody reads."""
    lin = nn.Linear(64, 32, bias=False)
    q = QATLinear(lin, INT4)
    assert q.weight is lin.weight


def test_qat_layer_still_has_a_trainable_parameter():
    q = QATLinear(nn.Linear(64, 32, bias=False), INT4)
    params = list(q.parameters())
    assert len(params) == 1 and params[0].requires_grad


# ---- prepare / convert ---------------------------------------------------------------

def test_prepare_wraps_every_linear_but_the_tied_head():
    model = Transformer(tiny_cfg())
    n_linear = len(linear_layers(model))
    wrapped = prepare_qat(model, INT4)
    assert len(wrapped) == n_linear - 1
    assert not any("lm_head" in w for w in wrapped)


def test_prepare_does_not_shrink_the_model():
    """During QAT the weights are still float -- the model is not smaller yet, and
    claiming otherwise would make the eventual saving look like it appeared twice."""
    model = Transformer(tiny_cfg())
    before = model_nbytes(model)
    prepare_qat(model, INT4)
    assert model_nbytes(model) >= before


def test_convert_produces_real_quantized_layers():
    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    before = model_nbytes(model)
    n = convert_qat(model, INT4)
    assert n > 0
    assert not any(isinstance(m, QATLinear) for m in model.modules())
    assert any(isinstance(m, QuantLinear) for m in model.modules())
    assert model_nbytes(model) < before


def test_convert_preserves_what_training_saw():
    """Because training and conversion use identical arithmetic, the converted model must
    compute what the last training forward pass computed. A mismatch here means the model
    silently gets worse the moment you save it."""
    torch.manual_seed(2)
    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    model.eval()
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        during, _ = model(x, targets=x)
    convert_qat(model, INT4)
    with torch.no_grad():
        after, _ = model(x, targets=x)
    assert torch.allclose(during, after, atol=1e-4)


def test_round_trip_through_prepare_and_convert_runs():
    torch.manual_seed(3)
    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    convert_qat(model, INT4)
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.randint(0, 128, (1, 8)))
    assert torch.isfinite(out).all()


# ---- the training loop ---------------------------------------------------------------

def test_qat_training_reduces_loss(tmp_path):
    """A smoke test with teeth: on data the model can actually fit, a QAT fine-tune must
    move the loss down. If the straight-through estimator is broken the loss sits flat."""
    torch.manual_seed(4)
    cfg = tiny_cfg()
    bin_path = tmp_path / "train.bin"
    # A short repeating pattern: learnable in a handful of steps on a tiny model.
    pattern = np.tile(np.arange(16, dtype=np.uint16), 4000)
    pattern.astype(np.uint16).tofile(bin_path)

    model = Transformer(cfg)
    prepare_qat(model, INT4)
    res = train_qat(model, str(bin_path), seq_len=16, scheme=INT4, steps=40,
                    batch_size=4, lr=1e-3, warmup=5, device="cpu", log=None)
    assert res.steps == 40
    assert res.loss_end < res.loss_start, (res.loss_start, res.loss_end)


def test_a_stop_file_ends_qat_early_and_reports_the_steps_it_did(tmp_path):
    """QAT is the one quantization method with a loop to interrupt. Stopping it must leave
    the model converted and the count honest — a QATResult still claiming 40 steps would
    put a fiction in the results table that nothing downstream could contradict."""
    torch.manual_seed(4)
    bin_path = tmp_path / "train.bin"
    np.tile(np.arange(16, dtype=np.uint16), 4000).astype(np.uint16).tofile(bin_path)
    stop = tmp_path / "STOP"
    stop.write_text("5")

    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    res = train_qat(model, str(bin_path), seq_len=16, scheme=INT4, steps=40, batch_size=2,
                    device="cpu", log=None, stop_file=stop)
    assert res.steps == 5
    assert not model.training       # still handed back ready to measure


def test_a_qat_deadline_that_has_passed_stops_at_the_first_step(tmp_path):
    torch.manual_seed(4)
    bin_path = tmp_path / "train.bin"
    np.tile(np.arange(16, dtype=np.uint16), 4000).astype(np.uint16).tofile(bin_path)
    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    res = train_qat(model, str(bin_path), seq_len=16, scheme=INT4, steps=40, batch_size=2,
                    device="cpu", log=None, stop_by=time.time() - 1)
    assert res.steps == 1


def test_qat_leaves_the_model_in_eval_mode(tmp_path):
    """`train_qat` puts the model in train mode; leaving it there would silently enable
    dropout for whatever measurement runs next."""
    torch.manual_seed(5)
    bin_path = tmp_path / "train.bin"
    np.tile(np.arange(16, dtype=np.uint16), 1000).astype(np.uint16).tofile(bin_path)
    model = Transformer(tiny_cfg())
    prepare_qat(model, INT4)
    train_qat(model, str(bin_path), seq_len=16, scheme=INT4, steps=3, batch_size=2,
              device="cpu", log=None)
    assert not model.training


@pytest.mark.parametrize("scheme", [
    QuantScheme(bits=4, group_size=64, sym=False),
    QuantScheme(bits=4, group_size=-1, sym=False),
    QuantScheme(bits=8, group_size=64, sym=True),
], ids=lambda s: s.label())
def test_every_scheme_survives_prepare_and_convert(scheme):
    torch.manual_seed(6)
    model = Transformer(tiny_cfg())
    prepare_qat(model, scheme)
    model.eval()
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        a, _ = model(x, targets=x)
    convert_qat(model, scheme)
    with torch.no_grad():
        b, _ = model(x, targets=x)
    assert torch.allclose(a, b, atol=1e-4)


# ---- stage-aware data guards ---------------------------------------------------------
# Quantization runs at the *end* of the pipeline, so the checkpoint is usually a chat
# model. These pin the guards that keep QAT from quietly undoing SFT.

class _Args:
    train_bin = None
    calib_bin = None
    val_bin = None
    calib_seqs = 8
    calib_batch = 2
    device = "cpu"


def test_qat_refuses_an_sft_checkpoint_without_explicit_data():
    """The dangerous default. `sft.py` carries the *base run's* data config forward, so an
    SFT checkpoint's train_bin is raw pretraining text. Running QAT on that fine-tunes a
    chat model back towards next-token prediction — and the perplexity number, measured on
    that same pretraining split, would *improve* while the model forgot how to converse."""
    from pathlib import Path

    from aksharallm.quant.cli import _train_bin

    ckpt = {"config": {"data": {"train_bin": "data/blend/fineweb.bin"}}}
    with pytest.raises(SystemExit, match="undo the SFT"):
        _train_bin(ckpt, Path("checkpoints/x/sft_best.pt"), _Args())


@pytest.mark.parametrize("name", ["sft_best.pt", "dpo_best.pt", "code_best.pt"])
def test_every_post_base_stage_is_guarded(name):
    from pathlib import Path

    from aksharallm.quant.cli import _train_bin

    with pytest.raises(SystemExit):
        _train_bin({"config": {"data": {"train_bin": "x.bin"}}},
                   Path(f"checkpoints/x/{name}"), _Args())


def test_explicit_train_bin_satisfies_the_guard():
    """Naming the data is the intended escape hatch — it is the *silent* default that is
    dangerous, not QAT on a chat model per se."""
    from pathlib import Path

    from aksharallm.quant.cli import _train_bin

    args = _Args()
    args.train_bin = "data/sft/train.bin"
    assert _train_bin({}, Path("checkpoints/x/sft_best.pt"), args) == "data/sft/train.bin"


def test_a_base_checkpoint_still_resolves_its_own_data(tmp_path):
    from pathlib import Path

    from aksharallm.quant.cli import _train_bin

    bin_path = tmp_path / "train.bin"
    bin_path.write_bytes(b"\x00\x00")
    got = _train_bin({"config": {"data": {"train_bin": str(bin_path)}}},
                     Path("checkpoints/x/ckpt_best.pt"), _Args())
    assert got == str(bin_path)


def test_train_sources_are_used_without_shadowing_the_checkpoint_path(tmp_path):
    """Regression: the blended-source loop used to rebind `src`, the checkpoint path."""
    from pathlib import Path

    from aksharallm.quant.cli import _train_bin

    bin_path = tmp_path / "fineweb.bin"
    bin_path.write_bytes(b"\x00\x00")
    got = _train_bin({"config": {"data": {"train_sources": [{"bin": str(bin_path)}]}}},
                     Path("checkpoints/x/ckpt_best.pt"), _Args())
    assert got == str(bin_path)
