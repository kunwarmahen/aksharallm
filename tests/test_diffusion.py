"""Masked diffusion: the objective, the bidirectional attention it needs, and unmasking.

The tests worth reading here are the ones that pin something which would otherwise fail
*silently*:

* `test_attention_really_is_bidirectional` — the whole paradigm rests on one boolean, and a
  causal mask left on by accident trains fine and generates fluent nonsense. So it is
  asserted from the outside: change a LATER token, and position 0's prediction must move.
* `test_a_kv_cache_is_refused` — a cache holds keys for tokens that a diffusion model may
  still rewrite. Wrong answers, no error, unless something raises.
* `test_no_mask_survives_generation` — the model assigns probability to `[MASK]` because it
  has seen the id on every training step; a committed one is a hole nothing can fill.
* `test_the_weight_makes_mask_rates_comparable` — the `1/t` term is the part of the formula
  that is easy to drop and impossible to notice, because the loss still goes down.
"""

from __future__ import annotations

import math

import pytest
import torch

from aksharallm.config import Config, DiffusionConfig, ModelConfig, load_config
from aksharallm.diffusion.corrupt import corrupt, diffusion_loss, sample_t
from aksharallm.diffusion.evaluate import elbo, loss_by_t
from aksharallm.diffusion.generate import (
    DiffusionError,
    decode_with_masks,
    diffusion_generate,
    infill,
)
from aksharallm.diffusion.objective import DiffusionObjective
from aksharallm.model.transformer import Transformer
from aksharallm.train.pretrain import ARObjective, objective_for

VOCAB = 64
MASK = VOCAB - 1


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2,
                max_seq_len=32, causal=False, mask_token_id=MASK)
    base.update(kw)
    return ModelConfig(**base)


def tiny_model(**kw) -> Transformer:
    torch.manual_seed(0)
    return Transformer(tiny_cfg(**kw)).eval()


# ---- configuration ---------------------------------------------------------------------

def test_a_diffusion_config_describes_itself():
    cfg = tiny_cfg()
    assert cfg.is_diffusion
    assert not ModelConfig(vocab_size=VOCAB).is_diffusion


def test_bidirectional_with_no_mask_token_is_refused():
    """Otherwise there is no objective at all: predicting the next token from a sequence
    that contains it is solved by reading it, and the run trains to zero loss."""
    with pytest.raises(ValueError, match="mask_token_id"):
        ModelConfig(vocab_size=VOCAB, causal=False)


def test_a_sliding_window_is_refused_bidirectionally():
    with pytest.raises(ValueError, match="causal idea"):
        ModelConfig(vocab_size=VOCAB, causal=False, mask_token_id=MASK, attn_window=8)


def test_a_mask_id_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        ModelConfig(vocab_size=VOCAB, causal=False, mask_token_id=VOCAB + 5)


def test_the_shipped_config_loads_and_is_a_diffusion_run():
    cfg = load_config("configs/tiny-diffusion.yaml")
    assert cfg.model.is_diffusion
    # The vocabulary rule: one more id than the tokenizer has, and the mask sits on it.
    assert cfg.model.mask_token_id == cfg.model.vocab_size - 1
    assert isinstance(cfg.diffusion, DiffusionConfig)


# ---- the model change ------------------------------------------------------------------

def test_attention_really_is_bidirectional():
    """One boolean is the whole architectural difference, so assert it from the outside.

    In a causal model nothing after position 0 can reach it. Here everything can.
    """
    model = tiny_model()
    a = torch.randint(0, VOCAB - 1, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % (VOCAB - 1)
    la, _ = model(a, full_logits=True)
    lb, _ = model(b, full_logits=True)
    assert not torch.allclose(la[0, 0], lb[0, 0], atol=1e-6)


def test_a_causal_model_is_unchanged_by_the_new_flag():
    """The other direction: the default must still hide the future, or every existing run
    in this repo has quietly changed meaning."""
    model = Transformer(ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                                    n_kv_heads=2, max_seq_len=32)).eval()
    a = torch.randint(0, VOCAB, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % VOCAB
    la, _ = model(a, full_logits=True)
    lb, _ = model(b, full_logits=True)
    assert torch.allclose(la[0, 0], lb[0, 0], atol=1e-6)


def test_a_kv_cache_is_refused():
    model = tiny_model()
    caches = model.init_caches(1, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="KV cache"):
        model(torch.randint(0, VOCAB - 1, (1, 4)), caches=caches)


# ---- the forward process ---------------------------------------------------------------

def test_corrupt_masks_about_t_of_the_tokens():
    x = torch.randint(0, VOCAB - 1, (64, 64))
    c = corrupt(x, MASK, torch.full((64,), 0.3))
    assert 0.25 < c.rate < 0.35
    # Only masked positions changed, and they all became the mask id.
    assert (c.x_t[c.masked] == MASK).all()
    assert (c.x_t[~c.masked] == x[~c.masked]).all()


def test_sample_t_respects_its_floor():
    t = sample_t(4096, 0.1, "cpu")
    assert t.min() >= 0.1 and t.max() < 1.0


def test_keep_positions_are_never_masked():
    """What makes scoring an infill possible: the given ends carry no loss."""
    x = torch.randint(0, VOCAB - 1, (8, 16))
    keep = torch.zeros_like(x, dtype=torch.bool)
    keep[:, :4] = True
    c = corrupt(x, MASK, torch.full((8,), 0.9), keep=keep)
    assert not c.masked[:, :4].any()


def test_loss_is_computed_only_on_masked_positions():
    """Change an unmasked token's *target* and the loss must not move — if it does, the
    mask is not being applied and the model is being scored on tokens it can see."""
    torch.manual_seed(1)
    logits = torch.randn(2, 8, VOCAB)
    x = torch.randint(0, VOCAB - 1, (2, 8))
    masked = torch.zeros(2, 8, dtype=torch.bool)
    masked[:, :3] = True
    from aksharallm.diffusion.corrupt import Corruption
    c = Corruption(x_t=x, masked=masked, t=torch.full((2,), 0.5))
    before, _ = diffusion_loss(logits, x, c)
    x2 = x.clone()
    x2[:, 5] = (x2[:, 5] + 1) % (VOCAB - 1)
    after, _ = diffusion_loss(logits, x2, c)
    assert torch.allclose(before, after)


def test_the_weight_makes_mask_rates_comparable():
    """The point of `1/t`. With uniform logits every masked token costs log(V) nats, so the
    weighted per-position loss is log(V) whatever fraction was masked. Without the weight it
    would scale with `t` and the model would train almost entirely on near-blank sequences.
    """
    from aksharallm.diffusion.corrupt import Corruption
    torch.manual_seed(2)
    B, T = 16, 64
    x = torch.randint(0, VOCAB - 1, (B, T))
    logits = torch.zeros(B, T, VOCAB)          # uniform: ce == log(V) everywhere
    seen = []
    for rate in (0.1, 0.5, 0.9):
        t = torch.full((B,), rate)
        c = corrupt(x, MASK, t)
        loss, stats = diffusion_loss(logits, x, Corruption(c.x_t, c.masked, t))
        seen.append(loss.item())
        assert stats["ce_masked"].item() == pytest.approx(math.log(VOCAB), abs=1e-4)
    for value in seen:
        assert value == pytest.approx(math.log(VOCAB), rel=0.15)


def test_an_empty_draw_contributes_zero_rather_than_being_forced():
    """Deliberately not "at least one mask per sequence": zero is the correct value of an
    empty sum, and forcing one would bias the estimator."""
    from aksharallm.diffusion.corrupt import Corruption
    x = torch.randint(0, VOCAB - 1, (1, 8))
    c = Corruption(x_t=x, masked=torch.zeros(1, 8, dtype=torch.bool),
                   t=torch.full((1,), 0.5))
    loss, _ = diffusion_loss(torch.randn(1, 8, VOCAB), x, c)
    assert loss.item() == 0.0


# ---- generation -------------------------------------------------------------------------

def test_no_mask_survives_generation():
    model = tiny_model()
    ids, _ = diffusion_generate(model, length=12, steps=4, prefix=[1, 2], device="cpu")
    assert MASK not in ids
    assert len(ids) == 14


def test_given_positions_are_never_overwritten():
    model = tiny_model()
    ids, _ = diffusion_generate(model, length=8, steps=3, prefix=[3, 4, 5],
                                suffix=[7, 8], device="cpu")
    assert ids[:3] == [3, 4, 5]
    assert ids[-2:] == [7, 8]


def test_every_step_commits_something_and_the_trace_is_complete():
    model = tiny_model()
    _, trace = diffusion_generate(model, length=12, steps=4, device="cpu", trace=True)
    assert trace[0].step == 0 and trace[0].remaining == 12
    assert [s.remaining for s in trace] == sorted((s.remaining for s in trace), reverse=True)
    assert trace[-1].remaining == 0
    assert sum(len(s.committed) for s in trace) == 12


def test_more_steps_than_positions_is_clamped():
    model = tiny_model()
    _, trace = diffusion_generate(model, length=4, steps=50, device="cpu", trace=True)
    assert len(trace) - 1 <= 4


def test_generation_is_repeatable_under_a_seed():
    model = tiny_model()
    kw = dict(length=10, steps=3, device="cpu", temperature=0.9, seed=11)
    assert diffusion_generate(model, **kw)[0] == diffusion_generate(model, **kw)[0]


def test_greedy_commits_the_most_confident_first():
    """At temperature 0 the confidence is a real maximum probability, so the first step's
    commits must all be at least as confident as anything left masked."""
    model = tiny_model()
    _, trace = diffusion_generate(model, length=16, steps=4, temperature=0.0,
                                  device="cpu", trace=True)
    first = trace[1]
    assert first.confidence == sorted(first.confidence, reverse=True)


def test_a_long_request_gives_up_generated_positions_not_the_prompt():
    model = tiny_model()                                   # max_seq_len 32
    prefix = list(range(20))
    ids, _ = diffusion_generate(model, length=100, steps=2, prefix=prefix, device="cpu")
    assert len(ids) == 32
    assert ids[:20] == prefix


def test_no_room_at_all_is_an_error_rather_than_a_truncated_prompt():
    model = tiny_model()
    with pytest.raises(DiffusionError, match="no room"):
        diffusion_generate(model, length=4, prefix=list(range(32)), device="cpu")


def test_an_autoregressive_model_is_refused():
    model = Transformer(ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                                    n_kv_heads=2, max_seq_len=32)).eval()
    with pytest.raises(DiffusionError, match="autoregressive"):
        diffusion_generate(model, length=4, device="cpu")


def test_an_unknown_remasking_strategy_is_refused():
    with pytest.raises(DiffusionError, match="remasking"):
        diffusion_generate(tiny_model(), length=4, steps=2, remask="vibes", device="cpu")


def test_infill_returns_only_the_middle():
    model = tiny_model()
    middle, _ = infill(model, [1, 2, 3], [9, 9], length=5, steps=2, device="cpu")
    assert len(middle) == 5
    assert MASK not in middle


class _FakeTok:
    vocab_size = VOCAB - 1

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_masks_decode_to_a_placeholder_rather_than_a_wrong_word():
    """The tokenizer has never heard of the mask id — it is a row the model was given."""
    text = decode_with_masks(_FakeTok(), [1, MASK, 2], MASK, placeholder="_")
    assert text == "b_c"


# ---- measurement -------------------------------------------------------------------------

class _Dataset:
    """Two fixed batches, so an evaluation is deterministic given its own seed."""

    def __init__(self, n=4, t=16):
        torch.manual_seed(3)
        self.x = torch.randint(0, VOCAB - 1, (n, t))

    def iter_eval_batches(self, batch_size, n_batches, seed=0):
        for _ in range(n_batches):
            yield self.x[:batch_size], self.x[:batch_size]


def test_the_elbo_is_repeatable_and_labelled_a_bound():
    model = tiny_model()
    ds = _Dataset()
    a = elbo(model, ds, 4, 2)
    b = elbo(model, ds, 4, 2)
    assert a["nelbo"] == pytest.approx(b["nelbo"])
    # The key name is load-bearing: nothing downstream may mistake it for a perplexity.
    assert "ppl_upper_bound" in a and "ppl" not in a
    assert "upper bound" in a["note"]


def test_a_different_seed_gives_a_different_draw():
    """Which is exactly why the trainer fixes one: otherwise 'best val' records the kindest
    corruption rather than the best model."""
    model = tiny_model()
    ds = _Dataset()
    assert elbo(model, ds, 4, 2, seed=1)["nelbo"] != elbo(model, ds, 4, 2, seed=2)["nelbo"]


def test_loss_by_t_covers_the_whole_range_at_fixed_rates():
    rows = loss_by_t(tiny_model(), _Dataset(), 4, 1, buckets=4)
    assert [r["t"] for r in rows] == [0.125, 0.375, 0.625, 0.875]
    assert all(r["ce_masked"] > 0 for r in rows)


def test_evaluation_leaves_the_model_in_training_mode_if_it_found_it_there():
    model = tiny_model()
    model.train()
    elbo(model, _Dataset(), 4, 1)
    assert model.training


# ---- the objective seam --------------------------------------------------------------------

def _cfg() -> Config:
    return Config(name="t", model=tiny_cfg(), diffusion=DiffusionConfig())


def test_the_objective_is_chosen_by_one_config_key():
    assert isinstance(objective_for(_cfg()), DiffusionObjective)
    assert isinstance(objective_for(Config(model=ModelConfig(vocab_size=VOCAB))), ARObjective)


def test_the_diffusion_objective_produces_a_gradient_on_the_masked_positions():
    cfg = _cfg()
    model = Transformer(cfg.model)
    obj = objective_for(cfg)
    loss = obj.loss(model, (torch.randint(0, VOCAB - 1, (4, 16)),))
    loss.backward()
    assert loss.item() > 0
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum() > 0


def test_the_objective_reports_the_unweighted_cross_entropy():
    cfg = _cfg()
    obj = objective_for(cfg)
    obj.loss(Transformer(cfg.model), (torch.randint(0, VOCAB - 1, (4, 16)),))
    stats = obj.stats()
    # ~log(V) at initialisation, and NOT the 1/t-weighted loss, which has no such scale.
    assert stats["ce"] == pytest.approx(math.log(VOCAB), rel=0.25)
    assert 0.2 < stats["mask"] < 0.8


class _Tok:
    vocab_size = VOCAB - 1
    bos_id = 0


def test_a_vocabulary_with_no_room_for_mask_is_refused():
    cfg = _cfg()
    obj = DiffusionObjective(cfg)
    obj.check(_Tok())                       # 63 tokenizer ids, 64 model ids: fine
    too_big = type("T", (), {"vocab_size": VOCAB})()
    with pytest.raises(ValueError, match="room for"):
        obj.check(too_big)


def test_a_mask_id_that_collides_with_a_real_token_is_refused():
    cfg = Config(name="t", model=tiny_cfg(mask_token_id=10), diffusion=DiffusionConfig())
    with pytest.raises(ValueError, match="collides"):
        DiffusionObjective(cfg).check(_Tok())
