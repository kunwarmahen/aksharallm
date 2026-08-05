"""Tests for speculative decoding.

One claim is being defended here, and everything else is bookkeeping around it: **the output
is the target model's output**. A faster decoder that changes what the model says is not a
faster decoder, it is a different model — and the difference would be invisible, because the
text stays fluent. So the two tests that matter are the exact ones:

* greedy speculative decoding must equal greedy target-only decoding, token for token;
* the acceptance rule must emit exactly `p`, proved on random distributions rather than
  observed on samples.

Everything runs on the CPU with tiny models, and the draft's disagreement is *constructed*
rather than hoped for — see `Disagrees` for why a second random model proved useless.

One thing is deliberately not tested here: that a block of tokens against a warm cache is
masked correctly. These models are too small to notice wrong-but-plausible logits, so it is
pinned where it can be checked exactly, in
`tests/test_model.py::test_several_tokens_against_a_warm_cache_match_one_at_a_time`.
"""

from __future__ import annotations

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer.generate import generate
from aksharallm.infer.speculative import (
    NgramDrafter,
    SpecStats,
    accept_or_correct,
    SpeculativeError,
    residual_distribution,
    speculative_collect,
    speculative_generate,
)
from aksharallm.model.transformer import Transformer


def tiny(seed: int, vocab: int = 64, d_model: int = 32, n_layers: int = 2) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=4,
                      n_kv_heads=2, max_seq_len=64, dropout=0.0)
    return Transformer(cfg).eval()


PROMPT = [3, 9, 27, 5]


# ---- the claim -------------------------------------------------------------------------

class Disagrees(torch.nn.Module):
    """A draft model that disagrees with the target on **every `every`-th position**.

    Two untrained models are useless as a pair here, and finding that out was worth the
    detour: an untrained transformer's prediction barely depends on its input, so a random
    draft either agrees on everything (tied embeddings) or on nothing (untied), and a round
    is then uniformly all-accepted or all-rejected. Neither shape exercises the rewind — the
    place where a rejected draft's keys and values have to stop existing — so both would pass
    while the interesting bug lived.

    So the disagreement is *constructed*: wrap the real target, swap its top two logits at
    chosen positions, and the draft proposes the second-best token exactly there. It is the
    real model with real caches, so a broken rewind still shows up as wrong text.
    """

    def __init__(self, base: Transformer, every: int = 3):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.every = every

    def init_caches(self, *a, **kw):
        return self.base.init_caches(*a, **kw)

    def forward(self, idx, caches=None, **kw):
        logits, loss = self.base(idx, caches=caches, **kw)
        end = caches[0].pos if caches is not None else idx.shape[1]
        start = end - logits.shape[1]
        for j in range(logits.shape[1]):
            if (start + j) % self.every == 0:
                top2 = torch.topk(logits[0, j], 2).indices
                a, b = int(top2[0]), int(top2[1])
                logits[0, j, a], logits[0, j, b] = logits[0, j, b].clone(), logits[0, j, a].clone()
        return logits, loss


@pytest.mark.parametrize("gamma", [1, 2, 4, 7])
def test_greedy_speculative_decoding_is_token_for_token_the_target_alone(gamma):
    """The point of the whole module: a draft that is sometimes wrong must not change one
    token of the output — it can only make it slower."""
    target = tiny(0)
    draft = Disagrees(tiny(0), every=3)
    alone = generate(target, PROMPT, max_new_tokens=24, temperature=0.0, device="cpu")
    with_draft, stats = speculative_collect(target, draft, PROMPT, max_new_tokens=24,
                                            gamma=gamma, temperature=0.0, device="cpu")
    assert with_draft == alone
    assert stats.emitted == len(alone) - len(PROMPT)
    # The scenario has to be the interesting one. A draft that agreed with everything (or
    # with nothing) would satisfy the assertion above while leaving the rewind untested, and
    # this line is the only thing that would notice.
    if gamma > 1:
        assert 0.0 < stats.accept_rate < 1.0, "degenerate draft: acceptance was all or nothing"


def test_a_draft_that_is_wrong_about_everything_still_produces_the_targets_text():
    """The other end of the range: every single guess rejected. The text is still exactly
    the target's, and the stats say the draft earned nothing."""
    target = tiny(0)
    draft = Disagrees(tiny(0), every=1)
    alone = generate(target, PROMPT, max_new_tokens=16, temperature=0.0, device="cpu")
    out, stats = speculative_collect(target, draft, PROMPT, max_new_tokens=16, gamma=4,
                                     temperature=0.0, device="cpu")
    assert out == alone
    assert stats.accept_rate == 0.0 and stats.corrections == stats.rounds


def test_the_acceptance_rule_emits_exactly_the_targets_distribution():
    """P(emit x) = min(q,p)(x) + P(reject)·residual(x) = p(x), for every x.

    Asserted analytically on random distributions rather than sampled, because a statistical
    test of this would be both slow and flaky — and this identity is the reason the speedup
    is free rather than a quality trade.
    """
    torch.manual_seed(0)
    for _ in range(50):
        p = torch.softmax(torch.randn(32) * 2, dim=-1)
        q = torch.softmax(torch.randn(32) * 2, dim=-1)
        accept_prob = torch.minimum(p / q, torch.ones_like(q)) * q      # = min(p, q)
        p_reject = 1.0 - float(accept_prob.sum())
        emitted = accept_prob + p_reject * residual_distribution(p, q)
        assert torch.allclose(emitted, p, atol=1e-6), (emitted - p).abs().max()


def test_a_rejected_token_is_replaced_from_the_residual_and_not_from_the_target(monkeypatch):
    """The correction must be drawn from `norm(max(p - q, 0))`, not from `p`.

    With a greedy draft the two are the same token, so an end-to-end test cannot tell them
    apart — which is exactly how the wrong one would survive. Here the draft is certain of A
    while the target is split evenly between A and B, so the residual is B *with certainty*
    and sampling from `p` would return A about half the time.
    """
    p = torch.tensor([0.5, 0.5, 0.0])          # A or B
    q = torch.tensor([1.0, 0.0, 0.0])          # the draft is sure it is A
    monkeypatch.setattr("aksharallm.infer.speculative.torch.rand",
                        lambda *a, **k: torch.tensor(0.99))     # force the rejection
    for _ in range(20):
        ok, replacement = accept_or_correct(p, q, token=0)
        assert ok is False and replacement == 1


def test_an_agreed_token_is_accepted_without_consuming_the_draw():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.5, 0.5])               # p/q = 1.8 > 1, so acceptance is certain
    assert accept_or_correct(p, q, token=0) == (True, None)


def test_the_residual_is_never_a_zero_vector():
    """Identical distributions make rejection impossible, so this branch is unreachable —
    but an unreachable crash is still a crash, and `multinomial` on zeros is one."""
    p = torch.tensor([0.25, 0.25, 0.5])
    assert torch.allclose(residual_distribution(p, p), p)


# ---- the mechanism ----------------------------------------------------------------------

def test_a_perfect_draft_is_accepted_every_time_and_earns_a_bonus_token():
    """Drafting with the target itself (through a copy, since the same object is refused)
    means p == q at every position: nothing can be rejected, and each round emits gamma+1
    tokens — which is why the speedup is not capped at gamma."""
    target = tiny(0)
    twin = tiny(0)                       # same seed: identical weights, different object
    _, stats = speculative_collect(target, twin, PROMPT, max_new_tokens=12, gamma=3,
                                   temperature=0.0, device="cpu")
    assert stats.accept_rate == 1.0
    assert stats.corrections == 0 and stats.bonus == stats.rounds
    assert stats.emitted == 12 and stats.rounds == 3        # 4 tokens per round


def test_a_useless_draft_costs_target_forwards_but_still_emits_one_token_each(monkeypatch):
    """The worst case, stated as a number: every guess rejected still emits the correction,
    so the loop cannot stall — it degenerates to plain decoding plus wasted draft passes."""
    target, draft = tiny(0), tiny(9)
    stats = SpecStats()
    monkeypatch.setattr("aksharallm.infer.speculative.torch.rand",
                        lambda *a, **k: torch.tensor(1.0))   # reject everything
    out = list(speculative_generate(target, draft, PROMPT, max_new_tokens=6, gamma=4,
                                    temperature=0.8, device="cpu", stats=stats))
    assert len(out) == 6
    assert stats.accepted == 0 and stats.corrections == stats.rounds == 6


def test_stats_add_up_and_report_tokens_per_forward():
    target, draft = tiny(0), tiny(1)
    _, stats = speculative_collect(target, draft, PROMPT, max_new_tokens=20, gamma=4,
                                   temperature=0.0, device="cpu")
    assert stats.emitted == stats.accepted + stats.corrections + stats.bonus
    assert sum(stats.per_round) == stats.accepted
    assert stats.drafted == sum(min(4, x + 1) for x in stats.per_round)
    # One forward for the prefill, one per round; plain decoding would need one per token.
    assert stats.target_forwards == stats.rounds + 1
    assert stats.tokens_per_forward > 1.0


def test_eos_stops_the_round_it_lands_in():
    """A round can accept several tokens at once, so EOS has to end generation *inside* the
    batch it arrived in — not after the rest of the accepted tokens have been emitted."""
    target, draft = tiny(0), tiny(1)
    alone = generate(target, PROMPT, max_new_tokens=24, temperature=0.0, device="cpu")
    eos = alone[len(PROMPT) + 3]          # whatever the 4th generated token happens to be
    out, _ = speculative_collect(target, draft, PROMPT, max_new_tokens=24, gamma=4,
                                 temperature=0.0, eos_id=eos, device="cpu")
    assert out[-1] == eos
    assert out == alone[: len(out)]


def test_the_context_window_is_never_exceeded():
    target, draft = tiny(0), tiny(1)
    prompt = list(range(60))              # max_seq_len is 64
    out, _ = speculative_collect(target, draft, prompt, max_new_tokens=50, gamma=8,
                                 temperature=0.0, device="cpu")
    assert len(out) <= 64


# ---- drafting with no model at all -------------------------------------------------------

def test_the_ngram_drafter_copies_the_most_recent_continuation():
    """It is a lookup, and the *most recent* match is the one to trust: the same three
    tokens can have been followed by different things earlier in the text."""
    d = NgramDrafter(vocab_size=64, n=2)
    tokens = [5, 6, 7, 1, 2, 3, 5, 6, 9, 9, 5, 6]
    guess, probs = d.propose(tokens, 2, dist=None)
    assert guess == [9, 9]                       # from the later "5, 6", not the first
    assert probs[0].argmax() == 9 and float(probs[0].sum()) == 1.0


def test_the_ngram_drafter_falls_back_to_a_shorter_match_then_gives_up():
    d = NgramDrafter(vocab_size=64, n=3)
    # No 3-token repeat, but "8" has occurred before and was followed by 4.
    assert d.propose([8, 4, 1, 2, 8], 2, dist=None)[0] == [4, 1]
    # Nothing repeats at all: propose nothing rather than something made up.
    assert d.propose([1, 2, 3], 2, dist=None) == ([], [])


def test_a_round_where_nothing_was_proposed_still_emits_a_token():
    """The no-match case must degenerate to plain decoding, not stall. This is the whole
    safety argument for drafting by lookup: where the text is novel it costs nothing."""
    target = tiny(0)
    never = NgramDrafter(vocab_size=target.cfg.vocab_size, n=8, min_n=8)
    alone = generate(target, PROMPT, max_new_tokens=8, temperature=0.0, device="cpu")
    out, stats = speculative_collect(target, never, PROMPT, max_new_tokens=8, gamma=4,
                                     temperature=0.0, device="cpu")
    assert out == alone
    assert stats.drafted == 0 and stats.emitted == 8 and stats.rounds == 8


def test_ngram_drafting_returns_the_targets_text_on_repetitive_input():
    """The case it exists for: a prompt that repeats itself. The copy is accepted, several
    tokens land per target forward, and the text is still exactly the target's."""
    target = tiny(0)
    prompt = [2, 3, 4, 5] * 5
    drafter = NgramDrafter(vocab_size=target.cfg.vocab_size, n=3)
    alone = generate(target, prompt, max_new_tokens=16, temperature=0.0, device="cpu")
    out, stats = speculative_collect(target, drafter, prompt, max_new_tokens=16, gamma=4,
                                     temperature=0.0, device="cpu")
    assert out == alone
    assert stats.drafted > 0


# ---- refusals ---------------------------------------------------------------------------

def test_a_vocabulary_mismatch_is_refused_rather_than_warned_about():
    """A shared tokenizer is not a nicety: the acceptance rule only ever compares the
    probabilities of a token *id*, so two vocabularies make the comparison meaningless while
    everything continues to run."""
    with pytest.raises(SpeculativeError, match="vocabulary mismatch"):
        list(speculative_generate(tiny(0, vocab=64), tiny(1, vocab=32), PROMPT, device="cpu"))


def test_drafting_with_the_target_itself_is_refused():
    target = tiny(0)
    with pytest.raises(SpeculativeError, match="same model"):
        list(speculative_generate(target, target, PROMPT, device="cpu"))
