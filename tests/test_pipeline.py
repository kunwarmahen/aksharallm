"""Tests for the data, tokenizer, schedule and post-training pieces.

The mask-alignment tests matter most. An off-by-one in the SFT loss mask trains the model
on the user's half of the conversation, which produces a model that interviews you instead
of answering -- and nothing in the loss curve reveals it.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aksharallm.data.loader import MixedTokenDataset, TokenDataset
from aksharallm.tokenizer.tokenizer import Tokenizer, train_bpe
from aksharallm.train.dpo import dpo_loss
from aksharallm.train.schedule import get_lr

CORPUS = [
    "Once upon a time there was a little girl who loved to read books.",
    "The quick brown fox jumps over the lazy dog again and again.",
    "She opened the door and saw a garden full of bright red flowers.",
    "He said hello to his friend and they walked to the park together.",
] * 60


@pytest.fixture(scope="module")
def tok(tmp_path_factory) -> Tokenizer:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_bpe(iter(CORPUS), vocab_size=512, out_path=path, min_frequency=1)
    return Tokenizer(path)


# ---- tokenizer ---------------------------------------------------------------------

def test_roundtrip_including_unicode(tok):
    for s in ["Hello world", "café", "emoji 🎈 here", "tabs\tand\nnewlines", "1234567890"]:
        assert tok.decode(tok.encode(s)) == s, s


def test_no_unknown_tokens(tok):
    """Byte-level BPE must encode arbitrary bytes -- there is no <UNK>."""
    weird = "".join(chr(i) for i in range(32, 127)) + "日本語 Ѐ Ω"
    assert tok.decode(tok.encode(weird)) == weird


def test_special_token_ids_are_stable(tok):
    assert (tok.bos_id, tok.pad_id, tok.im_start_id, tok.im_end_id) == (0, 1, 2, 3)
    assert tok.eos_id == tok.bos_id


def test_bos_eos_flags(tok):
    ids = tok.encode("hello", bos=True, eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


# ---- chat template ------------------------------------------------------------------

def test_chat_mask_covers_only_assistant_content(tok):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
    ]
    ids, mask = tok.render_chat(messages)
    assert len(ids) == len(mask)
    assert any(mask), "nothing marked trainable"

    # Decoding only the masked tokens must recover the assistant's text and nothing else.
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "Four." in trained
    assert "2+2" not in trained, "user content leaked into the trainable region"


def test_chat_mask_excludes_system_and_all_user_turns(tok):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question here."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question here."},
        {"role": "assistant", "content": "Second answer."},
    ]
    ids, mask = tok.render_chat(messages)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "First answer." in trained and "Second answer." in trained
    for leaked in ("You are helpful", "First question", "Second question"):
        assert leaked not in trained


def test_generation_prompt_is_not_trainable(tok):
    ids, mask = tok.render_chat(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True
    )
    assert ids[-3:-1] != [], "expected an assistant header"
    assert not any(mask), "a prompt-only render must have nothing to train on"
    assert ids[-1] != tok.im_end_id, "generation prompt must be left open for the model"


def test_sft_mask_alignment_matches_targets(tok):
    """Reproduces SFTDataset's shift: mask[1:] must line up with the *targets*."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
    ]
    ids, mask = tok.render_chat(messages)
    arr, msk = np.array(ids), np.array(mask)

    y = arr[1:].copy()
    m = msk[1:]
    y_masked = y.copy()
    y_masked[m == 0] = -100

    kept = y_masked[y_masked != -100]
    # Every surviving target must be a token the assistant actually produced.
    assert len(kept) > 0
    assert "Four." in tok.decode(kept.tolist())


# ---- loader ------------------------------------------------------------------------

def test_loader_shift_and_bounds(tmp_path):
    data = np.arange(1000, dtype=np.uint16)
    p = tmp_path / "toy.bin"
    data.tofile(p)

    ds = TokenDataset(p, seq_len=16, device="cpu")
    assert ds.n_tokens == 1000
    x, y = ds.get_batch(8)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.int64
    # y is x shifted left by one, everywhere.
    assert torch.equal(x[:, 1:], y[:, :-1])
    # and since our toy data is a ramp, y == x + 1 exactly
    assert torch.equal(y, x + 1)


def test_eval_batches_are_deterministic(tmp_path):
    np.arange(5000, dtype=np.uint16).tofile(tmp_path / "toy.bin")
    ds = TokenDataset(tmp_path / "toy.bin", seq_len=32, device="cpu")
    a = [x for x, _ in ds.iter_eval_batches(4, 3, seed=7)]
    b = [x for x, _ in ds.iter_eval_batches(4, 3, seed=7)]
    for u, v in zip(a, b):
        assert torch.equal(u, v), "val batches must be reproducible across evals"


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenDataset(tmp_path / "nope.bin", seq_len=8, device="cpu")


# ---- mixed (blended) loader --------------------------------------------------------

def _make_bin(path, value, n=4000):
    """A bin filled with a single token value, so we can tell sources apart by content."""
    np.full(n, value, dtype=np.uint16).tofile(path)


def test_mixed_respects_weights_every_batch(tmp_path):
    """The mix must be exact per batch, not just on average. Source A is all 10s, source B
    all 20s, so we can count where each row came from."""
    _make_bin(tmp_path / "a.bin", 10)
    _make_bin(tmp_path / "b.bin", 20)
    ds = MixedTokenDataset(
        [{"bin": tmp_path / "a.bin", "weight": 0.75},
         {"bin": tmp_path / "b.bin", "weight": 0.25}],
        seq_len=8, device="cpu",
    )
    x, y = ds.get_batch(8)
    assert x.shape == (8, 8)
    from_a = int((x[:, 0] == 10).sum())
    from_b = int((x[:, 0] == 20).sum())
    assert from_a == 6 and from_b == 2  # exactly 0.75 / 0.25 of 8


def test_mixed_weights_are_normalised(tmp_path):
    _make_bin(tmp_path / "a.bin", 10)
    _make_bin(tmp_path / "b.bin", 20)
    # unnormalised 3:1 should behave the same as 0.75:0.25
    ds = MixedTokenDataset(
        [{"bin": tmp_path / "a.bin", "weight": 30},
         {"bin": tmp_path / "b.bin", "weight": 10}],
        seq_len=8, device="cpu",
    )
    assert ds.weights[0] == pytest.approx(0.75)
    x, _ = ds.get_batch(8)
    assert int((x[:, 0] == 10).sum()) == 6


def test_mixed_largest_remainder_sums_to_batch_size(tmp_path):
    _make_bin(tmp_path / "a.bin", 10)
    _make_bin(tmp_path / "b.bin", 20)
    _make_bin(tmp_path / "c.bin", 30)
    ds = MixedTokenDataset(
        [{"bin": tmp_path / "a.bin", "weight": 0.5},
         {"bin": tmp_path / "b.bin", "weight": 0.3},
         {"bin": tmp_path / "c.bin", "weight": 0.2}],
        seq_len=4, device="cpu",
    )
    for bs in (1, 3, 7, 16, 100):
        assert int(ds._counts(bs).sum()) == bs

    # n_tokens is the sum across sources
    assert ds.n_tokens == 3 * 4000


def test_mixed_batch_is_shifted_and_typed(tmp_path):
    _make_bin(tmp_path / "a.bin", 10)
    _make_bin(tmp_path / "b.bin", 20)
    ds = MixedTokenDataset(
        [{"bin": tmp_path / "a.bin", "weight": 1},
         {"bin": tmp_path / "b.bin", "weight": 1}],
        seq_len=8, device="cpu",
    )
    x, y = ds.get_batch(6)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])  # the next-token shift holds through the mix


def test_mixed_rejects_bad_input(tmp_path):
    _make_bin(tmp_path / "a.bin", 10)
    with pytest.raises(ValueError):
        MixedTokenDataset([], seq_len=8, device="cpu")
    with pytest.raises(ValueError):
        MixedTokenDataset([{"bin": tmp_path / "a.bin", "weight": 0}], seq_len=8, device="cpu")


# ---- LR schedule --------------------------------------------------------------------

def test_warmup_is_linear_and_peaks_at_base_lr():
    kw = dict(base_lr=1e-3, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert get_lr(0, **kw) == pytest.approx(1e-5)
    assert get_lr(49, **kw) == pytest.approx(5e-4)
    assert get_lr(99, **kw) == pytest.approx(1e-3)


def test_cosine_decays_to_the_floor_and_never_below():
    kw = dict(base_lr=1e-3, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert get_lr(1000, **kw) == pytest.approx(1e-4)
    lrs = [get_lr(s, **kw) for s in range(100, 1001)]
    assert all(l >= 1e-4 - 1e-12 for l in lrs)
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:])), "must be monotonically decreasing"


def test_wsd_holds_flat_then_decays():
    kw = dict(base_lr=1e-3, warmup_steps=10, max_steps=1000,
              min_lr_ratio=0.1, schedule="wsd")
    assert get_lr(500, **kw) == pytest.approx(1e-3)   # stable phase
    assert get_lr(1000, **kw) == pytest.approx(1e-4)  # fully decayed
    assert get_lr(900, **kw) < 1e-3                   # decaying


# ---- DPO loss -----------------------------------------------------------------------

def _t(v):
    return torch.tensor([v], dtype=torch.float32)


def test_dpo_loss_is_ln2_when_policy_equals_reference():
    """At initialisation the policy *is* the reference, so the margin is 0 and the loss
    is exactly -log(sigmoid(0)) = ln 2. If your DPO run doesn't start here, it's wrong."""
    loss, acc, margin = dpo_loss(_t(-5.0), _t(-8.0), _t(-5.0), _t(-8.0), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert margin.item() == pytest.approx(0.0, abs=1e-6)


def test_dpo_loss_falls_when_chosen_gains_relative_to_reference():
    ref_c, ref_r = _t(-5.0), _t(-5.0)
    better = dpo_loss(_t(-4.0), _t(-6.0), ref_c, ref_r, beta=0.1)[0]
    worse = dpo_loss(_t(-6.0), _t(-4.0), ref_c, ref_r, beta=0.1)[0]
    assert better.item() < math.log(2) < worse.item()


def test_dpo_reference_term_actually_matters():
    """Same policy logprobs, different reference => different loss. If these were equal,
    the reference model would be doing nothing and the KL anchor would be absent."""
    a = dpo_loss(_t(-4.0), _t(-6.0), _t(-5.0), _t(-5.0), beta=0.1)[0]
    b = dpo_loss(_t(-4.0), _t(-6.0), _t(-3.0), _t(-7.0), beta=0.1)[0]
    assert not math.isclose(a.item(), b.item(), abs_tol=1e-4)


def test_dpo_beta_scales_the_margin():
    args = (_t(-4.0), _t(-6.0), _t(-5.0), _t(-5.0))
    small = dpo_loss(*args, beta=0.01)[0].item()
    large = dpo_loss(*args, beta=1.0)[0].item()
    assert large < small < math.log(2)


def test_dpo_accuracy_counts_pairs_above_the_reference():
    pi_c = torch.tensor([-4.0, -8.0])   # first pair improved, second regressed
    pi_r = torch.tensor([-6.0, -4.0])
    ref = torch.tensor([-5.0, -5.0])
    _, acc, _ = dpo_loss(pi_c, pi_r, ref, ref.clone(), beta=0.1)
    assert acc.item() == pytest.approx(0.5)
