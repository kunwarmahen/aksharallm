"""Tests for sampling and generation."""

import torch
import torch.nn.functional as F

from aksharallm.config import ModelConfig
from aksharallm.infer.generate import _filter_logits, generate
from aksharallm.model.transformer import Transformer


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=48)
    return Transformer(cfg).eval()


# ---- logit filtering ----------------------------------------------------------------

def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    out = _filter_logits(logits.clone(), top_k=2, top_p=None)
    assert torch.isinf(out).sum() == 3
    assert not torch.isinf(out[0, 0]) and not torch.isinf(out[0, 1])


def test_top_k_larger_than_vocab_is_a_noop():
    logits = torch.randn(1, 5)
    out = _filter_logits(logits.clone(), top_k=100, top_p=None)
    assert torch.equal(out, logits)


def test_top_p_keeps_the_nucleus():
    # The nucleus is the smallest set whose probabilities sum to >= p.
    # cumulative: 0.5, 0.8, 0.95, 1.0 -- so p=0.9 needs the first three.
    # (Values chosen to avoid landing exactly on the threshold, where float rounding
    # decides the answer.)
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    out = _filter_logits(logits.clone(), top_k=None, top_p=0.9)
    kept = (~torch.isinf(out)).sum().item()
    assert kept == 3


def test_top_p_always_keeps_the_top_token():
    """Even when one token holds more mass than p, we must not filter everything away."""
    logits = torch.tensor([[10.0, 0.0, 0.0]])  # top token has ~1.0 probability
    out = _filter_logits(logits.clone(), top_k=None, top_p=0.5)
    assert not torch.isinf(out[0, 0])
    assert (~torch.isinf(out)).sum() >= 1


def test_top_p_preserves_token_positions():
    """Filtering sorts internally; the result must be scattered back to the original
    indices or sampling would return the wrong token ids."""
    logits = torch.tensor([[1.0, 9.0, 2.0, 8.0]])
    out = _filter_logits(logits.clone(), top_k=2, top_p=None)
    assert not torch.isinf(out[0, 1]) and not torch.isinf(out[0, 3])  # the two largest
    assert torch.isinf(out[0, 0]) and torch.isinf(out[0, 2])


def test_filtering_does_not_change_relative_order_of_kept_logits():
    logits = torch.tensor([[3.0, 1.0, 2.0, 0.5]])
    out = _filter_logits(logits.clone(), top_k=3, top_p=None)
    keep = ~torch.isinf(out)
    assert torch.equal(out[keep], logits[keep])


# ---- generation ---------------------------------------------------------------------

def test_greedy_is_deterministic():
    m = tiny_model()
    a = generate(m, [1, 2, 3], max_new_tokens=12, temperature=0.0, device="cpu")
    b = generate(m, [1, 2, 3], max_new_tokens=12, temperature=0.0, device="cpu")
    assert a == b


def test_generate_returns_prompt_plus_new_tokens():
    m = tiny_model()
    prompt = [5, 6, 7]
    out = generate(m, prompt, max_new_tokens=10, temperature=0.0, device="cpu")
    assert out[: len(prompt)] == prompt
    assert len(out) == len(prompt) + 10


def test_generate_stops_at_eos():
    m = tiny_model()
    # Force token 0 to always win, then ask generation to stop on it.
    with torch.no_grad():
        m.lm_head.weight[0] += 100.0
    out = generate(m, [1, 2], max_new_tokens=30, temperature=0.0, eos_id=0, device="cpu")
    assert out[-1] == 0
    assert out.count(0) == 1, "should stop at the first EOS, not keep going"


def test_generate_respects_context_limit():
    m = tiny_model()  # max_seq_len = 48
    out = generate(m, list(range(10)), max_new_tokens=200, temperature=0.0, device="cpu")
    assert len(out) <= m.cfg.max_seq_len


def test_long_prompt_is_truncated_not_crashed():
    m = tiny_model()
    long_prompt = list(range(1, 60))  # longer than max_seq_len=48
    out = generate(m, long_prompt, max_new_tokens=3, temperature=0.0, device="cpu")
    assert len(out) <= m.cfg.max_seq_len


def test_stream_callback_fires_once_per_token():
    m = tiny_model()
    seen = []
    out = generate(m, [1, 2], max_new_tokens=8, temperature=0.0,
                   device="cpu", stream_cb=seen.append)
    assert len(seen) == 8
    assert seen == out[2:]


def test_temperature_zero_matches_argmax_of_the_model():
    m = tiny_model()
    prompt = [3, 4, 5]
    with torch.no_grad():
        logits, _ = m(torch.tensor([prompt]))
    expected = int(logits[0, -1].argmax())
    out = generate(m, prompt, max_new_tokens=1, temperature=0.0, device="cpu")
    assert out[-1] == expected


def test_repetition_penalty_handles_negative_logits():
    """Dividing a negative logit would *raise* it. The sign branch must multiply instead,
    or the penalty makes repetition worse."""
    m = tiny_model()
    with torch.no_grad():
        m.lm_head.weight.mul_(-1)  # push logits negative
    out = generate(m, [1, 2, 3], max_new_tokens=10, temperature=0.0,
                   repetition_penalty=1.5, device="cpu")
    assert len(out) == 13  # ran without inverting the penalty into a bonus
