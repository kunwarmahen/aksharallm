"""Tests for GRPO's math. The training loop itself is smoke-tested separately (it needs a
real model and sampling); these pin the two pure functions the same way test_pipeline pins
dpo_loss -- an off-by-one or a sign error here silently produces an RL run that optimises
nothing, or optimises backwards.
"""

import math

import pytest
import torch

from aksharallm.train.grpo import (
    SubstringReward,
    build_batch,
    grpo_loss,
    group_advantages,
)


# ---- group_advantages --------------------------------------------------------------

def test_advantages_are_zero_mean_within_group():
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    a = group_advantages(r)
    assert torch.allclose(a.mean(dim=1), torch.zeros(2), atol=1e-5)


def test_uniform_group_gives_no_signal():
    """Every completion equally good (or bad) -> nothing to learn -> all-zero advantage.
    This is the property that makes GRPO spend gradient only on boundary prompts."""
    for val in (0.0, 0.1, 1.0):
        a = group_advantages(torch.full((1, 6), val))
        assert torch.allclose(a, torch.zeros(1, 6), atol=1e-6)


def test_higher_reward_higher_advantage():
    a = group_advantages(torch.tensor([[0.0, 0.3, 0.7, 1.0]]))
    assert a[0, 0] < a[0, 1] < a[0, 2] < a[0, 3]
    assert a[0, 0] < 0 < a[0, 3]  # below-average negative, above-average positive


def test_advantage_normalised_by_spread():
    # same shape of rewards, different scale -> same advantages (std-normalised)
    a1 = group_advantages(torch.tensor([[0.0, 1.0]]))
    a2 = group_advantages(torch.tensor([[0.0, 10.0]]))
    assert torch.allclose(a1, a2, atol=1e-3)


# ---- grpo_loss ---------------------------------------------------------------------

def _flat(*vals):
    return torch.tensor([vals], dtype=torch.float32)


def test_loss_zero_when_no_advantage_and_on_reference():
    lp = _flat(-1.0, -2.0, -1.5)
    mask = torch.ones_like(lp)
    loss, m = grpo_loss(lp, lp.clone(), lp.clone(), adv=torch.zeros(1), mask=mask, beta=0.04)
    assert abs(loss.item()) < 1e-6
    assert m["kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["ratio"] == pytest.approx(1.0, abs=1e-6)


def test_positive_advantage_pushes_logprob_up():
    """The core of RL: an above-average completion must get a gradient that *raises* its
    token logprobs. So d(loss)/d(new_lp) must be negative there."""
    old = _flat(-1.0, -1.0, -1.0)
    new = old.clone().requires_grad_(True)
    ref = old.clone()
    mask = torch.ones_like(old)
    loss, _ = grpo_loss(new, old, ref, adv=torch.tensor([1.0]), mask=mask, beta=0.0)
    loss.backward()
    assert (new.grad < 0).all()  # increasing logprob lowers loss -> gradient descent raises it


def test_negative_advantage_pushes_logprob_down():
    old = _flat(-1.0, -1.0)
    new = old.clone().requires_grad_(True)
    loss, _ = grpo_loss(new, old, old.clone(), adv=torch.tensor([-1.0]),
                        mask=torch.ones_like(old), beta=0.0)
    loss.backward()
    assert (new.grad > 0).all()


def test_kl_penalty_is_nonnegative_and_grows_with_divergence():
    old = _flat(-1.0, -1.0)
    ref = _flat(-1.0, -1.0)
    mask = torch.ones_like(old)
    zero_adv = torch.zeros(1)
    # with zero advantage, the whole loss is the KL term
    near = grpo_loss(_flat(-1.1, -1.1), old, ref, zero_adv, mask, beta=1.0)[0].item()
    far = grpo_loss(_flat(-3.0, -3.0), old, ref, zero_adv, mask, beta=1.0)[0].item()
    assert near >= -1e-6 and far >= -1e-6
    assert far > near  # further from the reference -> larger KL penalty


def test_mask_excludes_prompt_tokens():
    # token 0 has a huge advantage-carrying logprob change but is masked out -> ignored
    new = _flat(5.0, -1.0)
    old = _flat(0.0, -1.0)
    ref = _flat(0.0, -1.0)
    mask = torch.tensor([[0.0, 1.0]])  # only the second (completion) token counts
    loss, _ = grpo_loss(new, old, ref, adv=torch.tensor([1.0]), mask=mask, beta=0.0)
    # second token: ratio=1, adv=1 -> surr=1 -> per_tok=-1; masked mean over 1 token = -1
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_clip_caps_the_ratio():
    """A completion whose policy prob has run far above its sampling prob (ratio >> 1) with
    a positive advantage must be clipped, so one step can't chase it arbitrarily far."""
    old = _flat(-5.0)
    new = _flat(0.0)          # exp(0 - -5) = e^5 ratio, way above 1+eps
    ref = _flat(-5.0)
    mask = torch.ones_like(old)
    loss, _ = grpo_loss(new, old, ref, adv=torch.tensor([1.0]), mask=mask,
                        beta=0.0, clip_eps=0.2)
    # min(ratio*A, clip*A) with A>0 picks the clipped (smaller) surrogate = 1.2
    assert loss.item() == pytest.approx(-1.2, abs=1e-4)


# ---- build_batch -------------------------------------------------------------------

def test_build_batch_masks_only_completion_tokens():
    # one prompt, group of 2: prompt = [1,2,3], completions [4,5] and [6,7,8]
    groups = [[([1, 2, 3, 4, 5], [4, 5]), ([1, 2, 3, 6, 7, 8], [6, 7, 8])]]
    seq, mask = build_batch(groups, pad_id=0, device="cpu")
    assert seq.shape[0] == 2
    # row 0: targets are seq[1:] = [2,3,4,5,pad]; completion tokens 4,5 sit at target idx 2,3
    assert mask[0].tolist()[:5] == [0, 0, 1, 1, 0]
    # row 1: targets [2,3,6,7,8]; completion 6,7,8 at idx 2,3,4
    assert mask[1].tolist() == [0, 0, 1, 1, 1]


# ---- reward ------------------------------------------------------------------------

def test_substring_reward():
    r = SubstringReward(" dragon")
    assert r("prompt", "there was a dragon here") == 1.0
    assert r("prompt", "there was a cat here") == 0.0
