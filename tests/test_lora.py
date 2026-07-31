"""Tests for LoRA and QLoRA.

The failure modes here are the quiet ones, and there are four that matter:

  1. **Forgetting to freeze.** Adapters get added, the base is still trainable, the loss
     goes down, and the ~4 MB adapter file you save does not contain most of what you
     trained. Nothing errors; the adapter simply does very little when reloaded.
  2. **A non-identity init.** If `lora_B` is not zero, step 0 is not the base model, and
     the first hundred steps are spent undoing a random perturbation of a pretrained model.
  3. **Gradients not reaching through a quantized base.** QLoRA backprops *through* the
     dequantization. If that path is detached — which the Triton kernel would do, having no
     backward — the adapters below it train against a constant.
  4. **An adapter applied to the wrong base.** It is a delta. On the wrong model it does
     not raise, it silently degrades. Hence the identity checks.

Each of those gets a test that would fail loudly if it regressed.
"""

import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.lora.adapter import (
    AdapterError,
    attach_adapter,
    base_identity,
    describe,
    load_adapter_file,
    save_adapter,
)
from aksharallm.lora.inject import (
    PRESETS,
    LoRAConfig,
    apply_lora,
    has_lora,
    lora_layers,
    prepare_for_training,
    resolve_targets,
)
from aksharallm.lora.layer import LoRALinear, disable_adapters
from aksharallm.lora.merge import merge_lora
from aksharallm.lora.setup import describe_memory
from aksharallm.model.transformer import Transformer
from aksharallm.quant.convert import quantize_model
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import QuantScheme

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
NF4 = QuantScheme(bits=4, group_size=32, dtype="nf4", double_quant=True, method="rtn")


def _model(quantize=False):
    torch.manual_seed(0)
    m = Transformer(CFG)
    if quantize:
        quantize_model(m, NF4)
    return m


def _batch(n=2, t=8):
    return torch.randint(0, CFG.vocab_size, (n, t))


# ---- the layer -------------------------------------------------------------------------


def test_the_adapter_is_the_identity_at_initialisation():
    """B starts at zero, so `Wx + BAx == Wx` exactly. Training therefore starts from the
    pretrained model rather than from a randomly perturbed one."""
    torch.manual_seed(0)
    lin = nn.Linear(16, 24, bias=False)
    x = torch.randn(3, 16)
    layer = LoRALinear(lin, r=4)
    assert torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B))
    assert torch.allclose(layer(x), lin(x), atol=1e-6)


def test_lora_a_is_not_zero_or_nothing_would_ever_move():
    layer = LoRALinear(nn.Linear(16, 24, bias=False), r=4)
    assert layer.lora_A.abs().sum() > 0


def test_scaling_is_alpha_over_r():
    layer = LoRALinear(nn.Linear(8, 8, bias=False), r=4, alpha=16)
    assert layer.scaling == pytest.approx(4.0)
    # The default is alpha = 2r, i.e. a scaling of 2 at any rank — which is the point:
    # sweeping the rank should not also change the size of the update.
    for r in (4, 8, 16, 32):
        assert LoRALinear(nn.Linear(8, 8, bias=False), r=r).scaling == pytest.approx(2.0)


def test_delta_weight_matches_the_forward_pass():
    """`delta_weight` is used by merging; if it disagreed with forward, merging would
    silently change the model."""
    torch.manual_seed(0)
    lin = nn.Linear(16, 24, bias=False)
    layer = LoRALinear(lin, r=4)
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.5)
    x = torch.randn(3, 16)
    want = lin(x) + x @ layer.delta_weight().T
    assert torch.allclose(layer(x), want, atol=1e-5)


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(8, 8, bias=False), r=0)


# ---- injection -------------------------------------------------------------------------


def test_apply_lora_freezes_everything_that_is_not_an_adapter():
    m = _model()
    report = apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable), trainable
    assert report.trainable == sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_only_adapters_receive_gradients():
    m = _model()
    apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    x = _batch()
    _, loss = m(x, targets=x)
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} is trainable but got no gradient"
        else:
            assert p.grad is None, f"{name} is frozen but received a gradient"


def test_lm_head_is_never_adapted_because_it_is_tied():
    m = _model()
    report = apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    assert not any(n.endswith("lm_head") for n in report.adapted)
    assert any("tied" in reason for name, reason in report.skipped if "lm_head" in name)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_adapts_something(preset):
    m = _model()
    report = apply_lora(m, LoRAConfig(r=4, targets=preset))
    assert report.adapted, f"{preset} adapted nothing"
    suffixes = {n.split(".")[-1] for n in report.adapted}
    assert suffixes == set(PRESETS[preset])


def test_targets_accept_an_explicit_list():
    assert resolve_targets("wq,wv") == ("wq", "wv")
    m = _model()
    report = apply_lora(m, LoRAConfig(r=4, targets="wq,wv"))
    assert {n.split(".")[-1] for n in report.adapted} == {"wq", "wv"}


def test_a_bigger_rank_trains_more_parameters():
    counts = []
    for r in (4, 8, 16):
        m = _model()
        counts.append(apply_lora(m, LoRAConfig(r=r, targets="all-linear")).trainable)
    assert counts[0] < counts[1] < counts[2]
    # Trainable parameters are linear in the rank: doubling r doubles them exactly.
    assert counts[1] == 2 * counts[0]


def test_trainable_fraction_is_small():
    m = _model()
    report = apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    assert 0 < report.fraction < 0.5
    assert "trainable" in report.summary()


# ---- disable_adapters ------------------------------------------------------------------


def test_disabling_adapters_reproduces_the_base_model_exactly():
    """This is what makes DPO's reference model free — so it has to be exact, not close."""
    m = _model().eval()
    x = _batch()
    with torch.no_grad():
        before, _ = m(x)
    apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_B.normal_(0, 0.3)
        adapted, _ = m(x)
        with disable_adapters(m):
            base, _ = m(x)
        after, _ = m(x)

    assert not torch.allclose(adapted, base, atol=1e-4), "the adapter changed nothing"
    assert torch.allclose(base, before, atol=1e-6)
    # The context manager must restore, not just clear.
    assert torch.allclose(after, adapted, atol=1e-6)


def test_disable_adapters_restores_state_even_on_an_exception():
    m = _model()
    apply_lora(m, LoRAConfig(r=4, targets="attn"))
    with pytest.raises(RuntimeError):
        with disable_adapters(m):
            raise RuntimeError("boom")
    assert all(mod.adapter_enabled for mod in lora_layers(m).values())


# ---- QLoRA ------------------------------------------------------------------------------


def test_qlora_wraps_quantized_layers_and_says_so():
    m = _model(quantize=True)
    report = apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    assert report.quantized_base
    bases = [mod.base for mod in lora_layers(m).values()]
    assert bases and all(isinstance(b, QuantLinear) for b in bases)


def test_gradients_reach_the_adapters_through_a_four_bit_base():
    """The load-bearing QLoRA property: backprop passes *through* the dequantization.

    At step 0 only `lora_B` has a non-zero gradient, and that is arithmetic rather than a
    bug: the update is `B @ A`, so `dL/dA` carries a factor of `B` — which is exactly zero
    at initialisation. `A` starts moving on the second step, once `B` has left zero. A test
    that expected both to move immediately would be asserting something false.
    """
    m = _model(quantize=True)
    apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    n_layers = len(lora_layers(m))
    x = _batch()

    _, loss = m(x, targets=x)
    loss.backward()
    moved = {n for n, p in m.named_parameters()
             if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0}
    assert len(moved) == n_layers, moved
    assert all(n.endswith("lora_B") for n in moved), moved

    # One step, then the A matrices join in — which is what proves the gradient really is
    # flowing back through the quantized base rather than stopping at B.
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    opt.step()
    opt.zero_grad(set_to_none=True)
    _, loss = m(x, targets=x)
    loss.backward()
    moved = {n for n, p in m.named_parameters()
             if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0}
    assert len(moved) == 2 * n_layers, moved


def test_the_quantized_base_holds_no_trainable_parameters():
    """QuantLinear stores buffers, not Parameters. If that ever changed, the optimiser
    would start trying to descend on packed uint8 weights."""
    m = _model(quantize=True)
    apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    for mod in m.modules():
        if isinstance(mod, QuantLinear):
            assert not list(mod.parameters(recurse=False))


def test_prepare_for_training_pins_the_torch_backend():
    """The fused Triton kernel has no autograd backward. Training through it would either
    error or silently detach the base — so the backend is pinned before any step runs."""
    saved = QuantLinear.backend
    try:
        QuantLinear.backend = "auto"
        m = _model(quantize=True)
        apply_lora(m, LoRAConfig(r=4, targets="attn"))
        note = prepare_for_training(m)
        assert QuantLinear.backend == "torch"
        assert note and "backward" in note
    finally:
        QuantLinear.backend = saved


def test_prepare_for_training_is_a_no_op_without_a_quantized_base():
    m = _model()
    apply_lora(m, LoRAConfig(r=4, targets="attn"))
    assert prepare_for_training(m) is None


def test_qlora_holds_far_less_than_full_fine_tuning():
    """Measured at a realistic width, not on the toy config.

    At d_model=32 the adapters are a *sixth* of the model, because rank 4 is not small
    relative to 32. The ~1% figure everyone quotes is a property of real widths, so the
    claim has to be tested at one — otherwise the test enshrines a number that says more
    about the fixture than about LoRA.
    """
    big = ModelConfig(vocab_size=1024, d_model=512, n_layers=8, n_heads=8, max_seq_len=128)
    n = sum(p.numel() for p in Transformer(big).parameters())

    m = Transformer(big)
    quantize_model(m, NF4)
    apply_lora(m, LoRAConfig(r=8, targets="all-linear"))
    mem = describe_memory(m)

    assert mem["trainable_params"] / n < 0.05
    # Optimiser state is the term LoRA is really attacking: full fine-tuning needs 8 bytes
    # per parameter for Adam's two fp32 moments, and this needs 8 bytes per *adapter*
    # parameter.
    assert mem["optimizer_bytes"] < n * 8 / 20
    # ...and the whole QLoRA footprint stays well under what full fine-tuning would need
    # for its optimiser state alone.
    assert mem["total_bytes"] < n * 8


# ---- merging ----------------------------------------------------------------------------


@pytest.mark.parametrize("quantize", [False, True], ids=["float-base", "nf4-base"])
def test_merging_preserves_the_output(quantize):
    m = _model(quantize=quantize).eval()
    apply_lora(m, LoRAConfig(r=4, targets="all-linear"))
    x = _batch()
    with torch.no_grad():
        for mod in lora_layers(m).values():
            mod.lora_B.normal_(0, 0.2)
        unmerged, _ = m(x)
        info = merge_lora(m)
        merged, _ = m(x)
    assert torch.allclose(unmerged, merged, atol=1e-4)
    assert not has_lora(m), "adapters should be gone after merging"
    if quantize:
        # Merging a 4-bit base must dequantize, and must say so rather than pretending the
        # result is still a 4-bit model.
        assert info["dequantized"]
        assert "not a 4-bit tensor" in info["note"]
        assert not any(isinstance(mod, QuantLinear) for mod in m.modules())


# ---- adapter files -----------------------------------------------------------------------


def _save(tmp_path, model, config, ckpt, name="a.lora.pt"):
    return save_adapter(tmp_path / name, model, config, base_identity(ckpt, "base.pt"))


def _ckpt(**over):
    d = {"model_config": dict(vars(CFG)), "step": 42, "best_val": 1.5,
         "config": {"data": {"tokenizer": "tok.json"}}}
    d.update(over)
    return d


def test_an_adapter_round_trips(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=4, targets="all-linear")
    apply_lora(m, cfg)
    with torch.no_grad():
        for mod in lora_layers(m).values():
            mod.lora_B.normal_(0, 0.2)
    x = _batch()
    m.eval()
    with torch.no_grad():
        want, _ = m(x)

    path = _save(tmp_path, m, cfg, _ckpt())
    fresh = _model().eval()
    attach_adapter(fresh, load_adapter_file(path), ckpt=_ckpt())
    with torch.no_grad():
        got, _ = fresh(x)
    assert torch.allclose(want, got, atol=1e-5)


def test_an_adapter_file_is_far_smaller_than_a_checkpoint(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=4, targets="all-linear")
    apply_lora(m, cfg)
    path = _save(tmp_path, m, cfg, _ckpt())
    full = tmp_path / "full.pt"
    torch.save({"model": _model().state_dict()}, full)
    assert path.stat().st_size < full.stat().st_size / 2


def test_the_adapter_carries_its_config_so_loading_needs_no_flags(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=16, alpha=8, dropout=0.1, targets="qv")
    apply_lora(m, cfg)
    d = describe(load_adapter_file(_save(tmp_path, m, cfg, _ckpt())))
    assert (d["r"], d["alpha"], d["targets"]) == (16, 8.0, "qv")
    assert d["layers"] == len(lora_layers(m))


def test_loading_onto_a_different_architecture_is_refused(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=4, targets="attn")
    apply_lora(m, cfg)
    path = _save(tmp_path, m, cfg, _ckpt())

    other = dict(vars(CFG))
    other["n_layers"] = 4
    with pytest.raises(AdapterError, match="not trained on this checkpoint"):
        attach_adapter(_model(), load_adapter_file(path),
                       ckpt=_ckpt(model_config=other))


def test_loading_with_a_different_tokenizer_is_refused(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=4, targets="attn")
    apply_lora(m, cfg)
    path = _save(tmp_path, m, cfg, _ckpt())
    bad = _ckpt(config={"data": {"tokenizer": "other.json"}})
    with pytest.raises(AdapterError, match="tokenizer"):
        attach_adapter(_model(), load_adapter_file(path), ckpt=bad)


def test_a_dropout_difference_alone_is_not_a_mismatch(tmp_path):
    """SFT raises dropout and inference sets it to zero, so it differs on almost every
    legitimate load. Treating it as an architecture change would refuse everything."""
    m = _model()
    cfg = LoRAConfig(r=4, targets="attn")
    apply_lora(m, cfg)
    path = _save(tmp_path, m, cfg, _ckpt())
    other = dict(vars(CFG))
    other["dropout"] = 0.05
    attach_adapter(_model(), load_adapter_file(path), ckpt=_ckpt(model_config=other))


def test_a_mismatch_can_be_forced(tmp_path):
    m = _model()
    cfg = LoRAConfig(r=4, targets="attn")
    apply_lora(m, cfg)
    path = _save(tmp_path, m, cfg, _ckpt())
    bad = _ckpt(config={"data": {"tokenizer": "other.json"}})
    attach_adapter(_model(), load_adapter_file(path), ckpt=bad, strict=False)


def test_a_checkpoint_is_not_an_adapter(tmp_path):
    p = tmp_path / "not-an-adapter.pt"
    torch.save({"model": {}, "model_config": dict(vars(CFG))}, p)
    with pytest.raises(AdapterError, match="not a LoRA adapter"):
        load_adapter_file(p)


def test_an_adapter_from_a_newer_format_is_refused(tmp_path):
    p = tmp_path / "future.lora.pt"
    torch.save({"kind": "lora-adapter", "format": 99, "lora": {},
                "lora_config": {"r": 4, "alpha": 8}}, p)
    with pytest.raises(AdapterError, match="newer format"):
        load_adapter_file(p)
