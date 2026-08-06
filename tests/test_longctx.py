"""Tests for long context: RoPE scaling, sliding windows, and the two measurements.

Most of this runs on a CPU. The parts that need a GPU are the ones checking that the two
implementations of a sliding window — the bool mask SDPA gets, and the two integers the
Triton kernel gets — describe the same rule, which is exactly the kind of thing that drifts.

Several tests here exist because the failure they catch is invisible:

  * `test_none_is_bit_for_bit_the_old_cache` — every model in the repo gained a
    `rope_scaling` field. If the default changed a single angle, every existing checkpoint
    would decode slightly differently and nothing else would say so.
  * `test_dynamic_scaling_is_stateful` — pinning a surprise rather than a bug. After one
    long sequence, a short one is served by the long factor.
  * `test_a_window_wider_than_the_sequence_changes_nothing` — the identity case. A window
    that is supposed to be inert has to actually be inert.
  * `test_extending_twice_compounds` — the bookkeeping trap. After one extension
    `max_seq_len` no longer says what the weights were trained on, so a second extension
    computed from it would silently rebase and report the wrong factor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.longctx import curve as curvemod
from aksharallm.longctx import extend as extmod  # the module, not the function it holds
from aksharallm.longctx import haystack as hay
from aksharallm.model import flash, rope
from aksharallm.model.transformer import Transformer, build_rope_cache, sliding_window_mask

triton_only = pytest.mark.skipif(
    not flash.available(), reason="needs triton on a CUDA device")

HEAD_DIM, THETA = 64, 10000.0


def cfg(**kw):
    base = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, max_seq_len=128)
    base.update(kw)
    return ModelConfig(**base)


# ---- the scaling maths -------------------------------------------------------------------

def test_none_is_bit_for_bit_the_old_cache():
    """The default must not move a single angle — every checkpoint in the repo depends on
    it. The reference here is the formula as it was written before scaling existed."""
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    pos = torch.arange(256).float()
    emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    cos, sin = build_rope_cache(HEAD_DIM, 256, THETA)
    assert torch.equal(cos, emb.cos()) and torch.equal(sin, emb.sin())


def test_linear_divides_every_frequency_equally():
    base = rope.base_inv_freq(HEAD_DIM, THETA)
    got, mscale = rope.plan(HEAD_DIM, THETA, rope.RopeScaling("linear", 4.0), 4096)
    assert torch.allclose(got, base / 4.0)
    assert mscale == 1.0


def test_ntk_barely_touches_the_fast_pairs_and_fully_slows_the_slow_ones():
    """The whole claim of NTK-aware scaling, as an assertion rather than a paragraph."""
    base = rope.base_inv_freq(HEAD_DIM, THETA)
    got, _ = rope.plan(HEAD_DIM, THETA, rope.RopeScaling("ntk", 4.0), 4096)
    fastest, slowest = (base[0] / got[0]).item(), (base[-1] / got[-1]).item()
    assert fastest == pytest.approx(1.0, abs=0.02), "the fast pair should be almost untouched"
    assert slowest == pytest.approx(4.0, rel=0.02), "the slow pair should take the full factor"


def test_yarn_keeps_fast_pairs_interpolates_slow_ones_and_ramps_between():
    base = rope.base_inv_freq(HEAD_DIM, THETA)
    sc = rope.RopeScaling("yarn", 4.0, original_max_seq_len=1024)
    got, mscale = rope.plan(HEAD_DIM, THETA, sc, 4096)
    ratio = base / got
    assert ratio[0] == pytest.approx(1.0, abs=1e-4), "fastest pair untouched"
    assert ratio[-1] == pytest.approx(4.0, rel=1e-3), "slowest pair fully interpolated"
    assert torch.all(ratio[1:] >= ratio[:-1] - 1e-6), "the ramp must be monotone"
    assert mscale == pytest.approx(0.1 * math.log(4.0) + 1.0)


def test_yarn_attention_temperature_scales_both_cos_and_sin():
    """The temperature is applied through the cache, so q and k each pick it up once and
    the logits get its square. Applying it to only one of them is a silent half-fix."""
    sc = rope.RopeScaling("yarn", 8.0, original_max_seq_len=512)
    cos, sin = rope.build_cache(HEAD_DIM, 4096, THETA, sc)
    plain_cos, plain_sin = rope.build_cache(HEAD_DIM, 4096, THETA, None)
    m = 0.1 * math.log(8.0) + 1.0
    # cos(0) == 1 everywhere, so row 0 reads the multiplier straight off.
    assert cos[0].max().item() == pytest.approx(m, rel=1e-5)
    assert (cos.abs().max() / plain_cos.abs().max()).item() == pytest.approx(m, rel=1e-5)
    assert sin.abs().max().item() > plain_sin.abs().max().item()


@pytest.mark.parametrize("method", ["linear", "ntk", "yarn"])
def test_factor_one_is_a_no_op(method):
    """`enabled` is false at factor 1, so a config that names a method but does not scale
    must produce the unscaled cache rather than something almost-but-not-quite it."""
    got, mscale = rope.plan(HEAD_DIM, THETA, rope.RopeScaling(method, 1.0), 1024)
    assert torch.equal(got, rope.base_inv_freq(HEAD_DIM, THETA)) and mscale == 1.0


def test_dynamic_is_unscaled_inside_the_original_window():
    sc = rope.RopeScaling("dynamic", 8.0, original_max_seq_len=1024)
    got, _ = rope.plan(HEAD_DIM, THETA, sc, 8192, seq_len=900)
    assert torch.equal(got, rope.base_inv_freq(HEAD_DIM, THETA))


def test_dynamic_scales_with_the_length_it_is_given():
    sc = rope.RopeScaling("dynamic", 8.0, original_max_seq_len=1024)
    short, _ = rope.plan(HEAD_DIM, THETA, sc, 8192, seq_len=2048)
    long, _ = rope.plan(HEAD_DIM, THETA, sc, 8192, seq_len=8192)
    assert long[-1] < short[-1], "a longer input must interpolate the slow pairs further"


def test_scaling_rejects_nonsense():
    with pytest.raises(ValueError, match="type"):
        rope.RopeScaling(type="magic")
    with pytest.raises(ValueError, match="factor"):
        rope.RopeScaling(type="yarn", factor=0.5)
    with pytest.raises(ValueError, match="beta_fast"):
        rope.RopeScaling(type="yarn", factor=2.0, beta_fast=1.0, beta_slow=8.0)


# ---- config plumbing ---------------------------------------------------------------------

def test_a_checkpoints_nested_dict_becomes_a_dataclass():
    """Ten call sites do `ModelConfig(**ckpt["model_config"])` with a plain dict inside.
    Without the coercion in __post_init__ this silently yields a dict, and every `.type`
    lookup downstream fails somewhere far away from the cause."""
    c = ModelConfig(rope_scaling={"type": "yarn", "factor": 4.0,
                                  "original_max_seq_len": 1024})
    assert isinstance(c.rope_scaling, rope.RopeScaling)
    assert c.rope_scaling.enabled and c.rope_scaling.original_len(4096) == 1024


def test_a_serialised_config_is_plain_data():
    """`rope_scaling` is the repo's first *nested* model config, and a checkpoint has to
    carry it as a dict. Written with `vars()` instead of `asdict()` it becomes a pickled
    live object, and `torch.load`'s weights_only default then refuses the whole file —
    which is how this was found. The trainer uses `asdict`; this keeps it that way."""
    from dataclasses import asdict

    blob = asdict(ModelConfig(rope_scaling=rope.RopeScaling("yarn", 4.0)))
    assert isinstance(blob["rope_scaling"], dict)
    assert not any(hasattr(v, "__dataclass_fields__") for v in blob.values())
    assert ModelConfig(**blob).rope_scaling.type == "yarn"


@pytest.mark.parametrize("form", ["dict", "dataclass"])
def test_reading_a_checkpoint_tolerates_either_config_form(form, tmp_path):
    """`checkpoints.py` reads files it did not write. Most carry `rope_scaling` as a dict
    (`asdict`), some as a live dataclass (`vars`), and being strict about it raised
    `AttributeError` from inside a checkpoint *listing* — which took the whole Finetune tab
    down, several tabs away from the cause. Both forms must read."""
    from aksharallm.infer.checkpoints import CheckpointStore

    scaling = rope.RopeScaling("yarn", 4.0, original_max_seq_len=1024)
    mcfg = dict(vars(cfg(max_seq_len=4096)))
    mcfg["rope_scaling"] = scaling.__dict__ if form == "dict" else scaling

    run = tmp_path / "checkpoints" / "toy"
    run.mkdir(parents=True)
    torch.save({"model": {}, "model_config": mcfg, "step": 1}, run / "ckpt_best.pt")
    info = CheckpointStore(tmp_path).get("toy/ckpt_best.pt")
    assert info.error is None, info.error
    assert info.trained_window == 1024 and info.max_seq_len == 4096
    assert "yarn x4 from 1024" in info.arch
    assert isinstance(info.as_dict()["rope_scaling"], dict)


def test_default_config_has_scaling_off():
    c = ModelConfig()
    assert c.rope_scaling.type == "none" and not c.rope_scaling.enabled
    assert c.attn_window is None and c.attn_sinks == 0


def test_sinks_without_a_window_is_refused():
    """Sinks are defined relative to a window. Silently ignoring them would mean a config
    that reads as StreamingLLM and behaves as plain full attention."""
    with pytest.raises(ValueError, match="attn_sinks only means something"):
        ModelConfig(attn_sinks=4)


# ---- the sliding window ------------------------------------------------------------------

def test_the_window_mask_is_the_rule_written_out():
    T = S = 32
    m = sliding_window_mask(T, S, window=8, sinks=2, device="cpu")
    q = torch.arange(T)[:, None]
    k = torch.arange(S)[None, :]
    assert torch.equal(m, (k <= q) & ((k > q - 8) | (k < 2)))


def test_the_window_mask_is_bottom_right_aligned_like_everything_else():
    """T < S is a warm cache. Query 0 of 4 against 100 keys sits at position 96, so with a
    window of 8 it sees keys 89..96 — not 0..7."""
    m = sliding_window_mask(T=4, S=100, window=8, sinks=0, device="cpu")
    assert m[0].nonzero().flatten().tolist() == list(range(89, 97))
    assert m[3].nonzero().flatten().tolist() == list(range(92, 100))


def test_sinks_stay_visible_however_far_the_window_has_slid():
    m = sliding_window_mask(T=64, S=64, window=4, sinks=3, device="cpu")
    assert m[-1, :3].all(), "the first three keys must survive to the last query"
    assert not m[-1, 3:60].any(), "everything between the sinks and the window is gone"


def test_a_window_wider_than_the_sequence_changes_nothing():
    """The identity case, end to end through the model. A window that cannot bite must
    produce the same logits as no window at all."""
    torch.manual_seed(0)
    idx = torch.randint(0, 256, (2, 48))
    base = Transformer(cfg()).eval()
    wide = Transformer(cfg(attn_window=4096)).eval()
    wide.load_state_dict(base.state_dict())
    with torch.no_grad():
        a, _ = base(idx, full_logits=True)
        b, _ = wide(idx, full_logits=True)
    assert torch.allclose(a, b, atol=1e-6)


def test_a_window_actually_blinds_the_model_to_the_far_past():
    """The other half: a narrow window must *change* the answer, or nothing is being
    masked and every other test here is passing for the wrong reason."""
    torch.manual_seed(0)
    idx = torch.randint(0, 256, (1, 64))
    m = Transformer(cfg(attn_window=8, attn_sinks=0)).eval()
    with torch.no_grad():
        narrow, _ = m(idx, full_logits=True)
        for block in m.blocks:
            block.attn.window = None
        full, _ = m(idx, full_logits=True)
    assert not torch.allclose(narrow[:, -1], full[:, -1], atol=1e-3)


def test_changing_a_token_outside_the_window_cannot_reach_the_last_position():
    """The property a sliding window promises, stated directly."""
    torch.manual_seed(0)
    m = Transformer(cfg(attn_window=8, attn_sinks=0)).eval()
    a = torch.randint(0, 256, (1, 64))
    b = a.clone()
    b[0, 5] = (b[0, 5] + 7) % 256          # far outside the last query's 8-token window
    with torch.no_grad():
        assert torch.allclose(m(a, full_logits=True)[0][:, -1],
                              m(b, full_logits=True)[0][:, -1], atol=1e-5)


# ---- the two implementations of the window agree ------------------------------------------

@triton_only
@pytest.mark.parametrize("window,sinks", [(32, 0), (32, 4), (16, 1), (256, 4)])
def test_the_kernel_and_the_reference_window_identically(window, sinks):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 128, 32, device="cuda", requires_grad=True)
    k = torch.randn(2, 2, 128, 32, device="cuda", requires_grad=True)
    v = torch.randn(2, 2, 128, 32, device="cuda", requires_grad=True)
    out = flash.flash_attention(q, k, v, window=window, sinks=sinks)
    ref = flash.reference_attention(q, k, v, window=window, sinks=sinks)
    assert torch.allclose(out, ref, atol=1e-5)

    g = torch.randn_like(out)
    ours = torch.autograd.grad(out, (q, k, v), g)
    ins = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    theirs = torch.autograd.grad(
        flash.reference_attention(*ins, window=window, sinks=sinks), ins, g)
    for name, a, b in zip("qkv", ours, theirs):
        assert torch.allclose(a, b, atol=1e-4), f"d{name}"


@triton_only
def test_a_windowed_model_agrees_whichever_kernel_runs_it():
    """The integration that matters: SDPA gets a bool mask, Triton gets two integers, and
    they have to be the same rule. This is where a drift between them would show up."""
    torch.manual_seed(0)
    c = dict(vocab_size=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
             max_seq_len=256, attn_window=48, attn_sinks=4)
    m = Transformer(ModelConfig(**c)).cuda().eval()
    idx = torch.randint(0, 128, (2, 192), device="cuda")
    with torch.no_grad():
        sdpa, _ = m(idx, full_logits=True)
        for block in m.blocks:
            block.attn.attn_impl = "flash"
        ours, _ = m(idx, full_logits=True)
    assert torch.allclose(sdpa, ours, atol=2e-4), (sdpa - ours).abs().max().item()


# ---- dynamic scaling's statefulness --------------------------------------------------------

def test_dynamic_scaling_is_stateful():
    """Documented surprise, pinned. After a long sequence, a short one is served by the
    long factor — every published implementation behaves this way and it still catches
    people out, which is why `pin_rope` exists."""
    m = Transformer(cfg(max_seq_len=512,
                        rope_scaling={"type": "dynamic", "factor": 4.0,
                                      "original_max_seq_len": 128})).eval()
    before = m.rope_cos.clone()
    with torch.no_grad():
        m(torch.randint(0, 256, (1, 100)))          # inside the original window
    assert torch.equal(m.rope_cos, before), "a short input must not rescale anything"
    with torch.no_grad():
        m(torch.randint(0, 256, (1, 400)))          # past it -> rebuild
    assert not torch.equal(m.rope_cos, before)
    grown = m.rope_cos.clone()
    with torch.no_grad():
        m(torch.randint(0, 256, (1, 100)))          # short again -> stays grown
    assert torch.equal(m.rope_cos, grown)


def test_pin_rope_fixes_the_factor_up_front():
    m = Transformer(cfg(max_seq_len=512,
                        rope_scaling={"type": "dynamic", "factor": 4.0,
                                      "original_max_seq_len": 128})).eval()
    m.pin_rope(512)
    pinned = m.rope_cos.clone()
    with torch.no_grad():
        m(torch.randint(0, 256, (1, 300)))
    assert torch.equal(m.rope_cos, pinned), "a pinned cache must not move mid-generation"


def test_pin_rope_is_a_no_op_for_every_other_method():
    m = Transformer(cfg(rope_scaling={"type": "yarn", "factor": 4.0})).eval()
    before = m.rope_cos.clone()
    m.pin_rope(4096)
    assert torch.equal(m.rope_cos, before)


# ---- extending a checkpoint ---------------------------------------------------------------

def base_cfg_dict() -> dict:
    from aksharallm.config import config_to_dict
    return config_to_dict(cfg(max_seq_len=1024))


def test_extend_records_the_window_the_weights_were_trained_on():
    after = extmod.plan_extension(base_cfg_dict(), "yarn", 4.0)
    assert after["max_seq_len"] == 4096
    assert after["rope_scaling"]["original_max_seq_len"] == 1024
    assert after["rope_scaling"]["type"] == "yarn"


def test_extending_twice_compounds():
    """After one extension `max_seq_len` is 4096, so a second extension that trusted it
    would report 4096 as the trained window and compute a factor against the wrong base."""
    once = extmod.plan_extension(base_cfg_dict(), "yarn", 4.0)
    twice = extmod.plan_extension(once, "yarn", 8.0)
    assert twice["rope_scaling"]["original_max_seq_len"] == 1024
    assert twice["max_seq_len"] == 8192


def test_extend_back_to_none_restores_the_trained_window():
    once = extmod.plan_extension(base_cfg_dict(), "linear", 4.0)
    back = extmod.plan_extension(once, "none", 1.0)
    assert back["max_seq_len"] == 1024
    assert back["rope_scaling"]["type"] == "none"


def test_extend_refuses_to_overwrite_the_source(tmp_path):
    p = tmp_path / "ckpt.pt"
    torch.save({"model": {}, "model_config": base_cfg_dict()}, p)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        extmod.extend(p, p, "yarn", 4.0)


def test_extend_writes_the_same_weights(tmp_path):
    """The headline claim of the whole module, as a test: not one tensor changes."""
    torch.manual_seed(0)
    model = Transformer(cfg(max_seq_len=1024))
    src, out = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"model": model.state_dict(), "model_config": base_cfg_dict()}, src)
    extmod.extend(src, out, "yarn", 4.0)
    a = torch.load(src, map_location="cpu", weights_only=False)["model"]
    b = torch.load(out, map_location="cpu", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_an_extended_checkpoint_reloads_extended(tmp_path):
    """End to end through the ten-call-sites path: save, extend, `ModelConfig(**dict)`,
    and the model must come back with a 4x cache without anyone passing a flag."""
    model = Transformer(cfg(max_seq_len=1024))
    src, out = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"model": model.state_dict(), "model_config": base_cfg_dict()}, src)
    extmod.extend(src, out, "yarn", 4.0)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    reloaded = Transformer(ModelConfig(**ck["model_config"]))
    reloaded.load_state_dict(ck["model"], strict=True)
    assert reloaded.rope_cos.shape[0] == 4096
    assert reloaded.cfg.rope_scaling.type == "yarn"


# ---- the measurements ----------------------------------------------------------------------

def test_the_curve_covers_every_position_exactly_once(tmp_path):
    bin_path = tmp_path / "val.bin"
    np.arange(20000, dtype=np.uint16).tofile(bin_path)
    m = Transformer(cfg(max_seq_len=256, vocab_size=20000)).eval()
    out = curvemod.position_curve(m, str(bin_path), seq_len=256, bucket=64, n_windows=2)
    assert [b["start"] for b in out["buckets"]] == [0, 64, 128, 192]
    assert sum(b["tokens"] for b in out["buckets"]) == 256 * 2
    assert out["buckets"][-1]["end"] == 255


def test_the_cliff_is_measured_against_the_in_window_baseline_not_position_zero():
    """Position 0 has no context and is always the worst point on a healthy curve.
    Anchoring to it would report a cliff on every model ever measured."""
    flat = {"buckets": [{"start": i * 100, "end": i * 100 + 99, "loss": loss}
                        for i, loss in enumerate([9.0, 2.0, 2.0, 2.1, 2.0])]}
    assert curvemod.cliff(flat, trained_len=300) is None


def test_the_cliff_is_found_when_there_is_one():
    broken = {"buckets": [{"start": i * 100, "end": i * 100 + 99, "loss": loss}
                          for i, loss in enumerate([3.0, 2.0, 2.0, 8.0, 11.0])]}
    got = curvemod.cliff(broken, trained_len=300)
    assert got["position"] == 300 and got["excess"] > 5


def test_a_collapsed_bucket_cannot_blow_up_the_chart(tmp_path):
    """exp(15) is 3.2 million. One capped cell keeps every other point readable, and the
    `loss` column stays the unclipped truth."""
    bin_path = tmp_path / "val.bin"
    np.arange(20000, dtype=np.uint16).tofile(bin_path)
    m = Transformer(cfg(max_seq_len=128, vocab_size=20000)).eval()
    out = curvemod.position_curve(m, str(bin_path), seq_len=128, bucket=64, n_windows=1)
    assert all(b["perplexity"] <= 1e6 for b in out["buckets"])


class FakeTok:
    """Whitespace tokenizer with ids well clear of the filler's range."""

    def encode(self, text, bos=False):
        return [10_000 + (abs(hash(w)) % 1000) for w in text.split()]


def test_the_needle_lands_at_the_depth_it_was_asked_for():
    tok = FakeTok()
    filler = np.arange(4000, dtype=np.uint16)
    for depth in (0.0, 0.5, 1.0):
        ids, at = hay.build_context(tok, filler, 512, depth, "Bengaluru", "7431")
        assert len(ids) == 512, "the context must be exactly the length requested"
        room = 512 - len(tok.encode(hay.NEEDLE.format(city="Bengaluru", code="7431"))) \
            - len(tok.encode(hay.PROBE.format(city="Bengaluru")))
        assert at == round(depth * room)


def test_a_context_too_short_for_its_own_needle_is_refused():
    with pytest.raises(ValueError, match="no room"):
        hay.build_context(FakeTok(), np.arange(100, dtype=np.uint16), 8, 0.5, "X", "1")


def test_the_grid_reports_chance_beside_every_number():
    """A grid of 25%s from a four-way choice is a grid of zeros. The chance line is not
    decoration — without it the table reads as partial success."""
    trials = [hay.Trial(512, 0.5, "X", "1", True, 0.4),
              hay.Trial(512, 0.5, "X", "2", False, -0.2)]
    out = hay.summarise(trials, [512], [0.5], n_candidates=4)
    assert out["chance"] == 0.25
    assert out["grid"][0][0]["accuracy"] == 0.5
    assert out["by_length"][0]["n"] == 2
