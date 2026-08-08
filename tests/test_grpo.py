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


# --- splitting the group across backward passes must not change the step -------------------
# The whole group is one optimizer step: advantages are normalised *within* it, so splitting
# the group would change the algorithm. What is split is only when the activations exist.
# Scoring all P*G completions at once materialises `(B, L-1, vocab)` logits — 1.15 GiB at
# 32 x 294 x 32768 fp32 — and `log_softmax` allocates another, three times over (old,
# reference, new). That OOMed a 24 GB card at 300M; the weights were never the problem.

def test_a_chunked_update_is_gradient_identical_to_the_undivided_one():
    """The property the whole micro-batching change rests on.

    `grpo_loss` is a masked *sum* over a denominator. Hold the denominator at the group's
    total and each chunk contributes its own share of the same fraction, so the chunk losses
    add to the undivided loss and their gradients accumulate into the same step.
    """
    torch.manual_seed(0)
    B, T = 12, 9
    mask = torch.zeros(B, T)
    for i in range(B):            # deliberately uneven: completions stop at different lengths
        mask[i, : 2 + (i * 3) % (T - 1)] = 1.0
    old, ref, adv = torch.randn(B, T), torch.randn(B, T), torch.randn(B)
    p = torch.randn(B, T, requires_grad=True)

    loss_full, m_full = grpo_loss(p, old, ref, adv, mask)
    loss_full.backward()
    g_full, p.grad = p.grad.clone(), None

    denom = mask.sum().clamp(min=1)
    total, kl_sum, n_sum = 0.0, 0.0, 0.0
    for i in range(0, B, 5):      # 5 does not divide 12, so the last chunk is short
        sl = slice(i, i + 5)
        loss_c, m_c = grpo_loss(p[sl], old[sl], ref[sl], adv[sl], mask[sl], denom=denom)
        loss_c.backward()
        total += loss_c.item()
        kl_sum += m_c["kl_sum"]
        n_sum += m_c["n_tokens"]

    assert torch.allclose(g_full, p.grad, atol=1e-6), "chunking changed the gradient"
    assert abs(total - loss_full.item()) < 1e-6, "chunk losses do not sum to the whole"
    assert abs(kl_sum / n_sum - m_full["kl"]) < 1e-6, "KL metric is not recovered exactly"


def test_the_denominator_is_what_makes_it_exact():
    """Without a shared denominator the chunks are means of means, and that is not the mean
    when they hold different numbers of completion tokens — which they always do."""
    torch.manual_seed(1)
    B, T = 12, 9
    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, : 2 + (i * 3) % (T - 1)] = 1.0
    old, ref, adv, p = torch.randn(B, T), torch.randn(B, T), torch.randn(B), torch.randn(B, T)

    whole = grpo_loss(p, old, ref, adv, mask)[0].item()
    naive = sum(grpo_loss(p[i:i + 5], old[i:i + 5], ref[i:i + 5], adv[i:i + 5],
                          mask[i:i + 5])[0].item() for i in range(0, B, 5)) / 3
    assert abs(naive - whole) > 0.01, (
        "the uneven-chunk trap has stopped biting, so this test no longer guards anything")


# --- batched sampling must be the serial sampler, only faster ------------------------------
# Sampling is ~90% of a GRPO step: 32 completions generated one at a time at ~50 tok/s where
# a batch of 32 runs at 236 (docs/17). But the sampled completions ARE the training data —
# they feed the reward and the advantage — so a batched sampler that produced subtly
# different sequences would train on different data while every curve looked healthy.
#
# The check is greedy equivalence against the serial path, and the prompts are deliberately
# *different lengths*: a ragged prefill is where position and mask bugs live, and a batch of
# equal-length prompts would not exercise them at all.

def _trained_tiny(seed: int = 0, vocab: int = 64, max_seq_len: int = 128):
    """A briefly trained model. An untrained transformer predicts nearly the same token
    whatever it is shown, so it agrees with any implementation and proves nothing — the same
    trap recorded in tests/test_serve.py."""
    from aksharallm.config import ModelConfig
    from aksharallm.model.transformer import Transformer

    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=max_seq_len, dropout=0.0)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(300):
        pattern = torch.randint(0, vocab, (8, 7), generator=gen)
        x = pattern.repeat(1, 16)[:, : min(64, max_seq_len)]
        _, loss = model(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def _cpu_engine(model, max_batch: int):
    from aksharallm.serve.batch import BatchEngine
    from aksharallm.serve.paged import BLOCK_SIZE, BlockPool

    cfg = model.cfg
    per_seq = cfg.max_seq_len // BLOCK_SIZE + 2
    pool = BlockPool(n_layers=cfg.n_layers, n_blocks=per_seq * max_batch,
                     n_kv_heads=cfg.n_kv_heads, head_dim=cfg.d_model // cfg.n_heads,
                     dtype=torch.float32, device="cpu")
    return BatchEngine(model, pool, max_batch=max_batch, device="cpu")


def test_batched_sampling_matches_the_serial_sampler_token_for_token():
    """Greedy, so the comparison is exact rather than distributional."""
    from aksharallm.train.grpo import sample_group, sample_groups_batched

    model = _trained_tiny()
    torch.manual_seed(0)
    # deliberately ragged: 5, 9 and 3 tokens of prompt
    prompts = [[3, 9, 14, 2, 7], [11, 4, 6, 1, 19, 22, 8, 5, 13], [2, 30, 17]]
    greedy = dict(temperature=1e-6, top_k=1, top_p=None, eos_id=None)
    G, MAX_NEW = 2, 12

    serial = [sample_group(model, p, G, MAX_NEW, greedy["temperature"], greedy["top_k"],
                           greedy["top_p"], greedy["eos_id"], "cpu") for p in prompts]
    batched = sample_groups_batched(_cpu_engine(model, G * len(prompts)), prompts, G,
                                    MAX_NEW, greedy["temperature"], greedy["top_k"],
                                    greedy["top_p"], greedy["eos_id"])

    assert len(batched) == len(serial)
    for p_idx, (s_grp, b_grp) in enumerate(zip(serial, batched)):
        assert len(b_grp) == G, f"prompt {p_idx}: expected {G} completions, got {len(b_grp)}"
        for k, ((s_full, s_gen), (b_full, b_gen)) in enumerate(zip(s_grp, b_grp)):
            assert b_gen == s_gen, (
                f"prompt {p_idx} completion {k}: batched sampling diverged from serial.\n"
                f"  serial : {s_gen}\n  batched: {b_gen}")
            assert b_full == s_full, "the prompt was not carried through intact"


def test_the_batched_sampler_keeps_each_completion_with_its_own_prompt():
    """The advantage is normalised *within a prompt's group*, so a completion filed under
    the wrong prompt would be compared against the wrong baseline — and nothing downstream
    could notice, because every shape still lines up."""
    from aksharallm.train.grpo import sample_groups_batched

    model = _trained_tiny()
    prompts = [[3, 9, 14, 2, 7], [11, 4, 6, 1, 19, 22, 8, 5, 13], [2, 30, 17]]
    groups = sample_groups_batched(_cpu_engine(model, 9), prompts, 3, 8,
                                   1e-6, 1, None, None)
    for pids, grp in zip(prompts, groups):
        for full, gen in grp:
            assert full[:len(pids)] == list(pids), "a completion is under the wrong prompt"
            assert full[len(pids):] == gen
