"""Tests for the data, tokenizer, schedule and post-training pieces.

The mask-alignment tests matter most. An off-by-one in the SFT loss mask trains the model
on the user's half of the conversation, which produces a model that interviews you instead
of answering -- and nothing in the loss curve reveals it.
"""

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aksharallm.config import load_config
from aksharallm.data.loader import MixedTokenDataset, TokenDataset
from aksharallm.tokenizer.tokenizer import Tokenizer, train_bpe
from aksharallm.train.dpo import dpo_loss
from aksharallm.train import stopfile
from aksharallm.train.pretrain import fmt_dur, resolve_stop_step, stop_file_target
from aksharallm.train.schedule import get_lr

CORPUS = [
    "Once upon a time there was a little girl who loved to read books.",
    "The quick brown fox jumps over the lazy dog again and again.",
    "She opened the door and saw a garden full of bright red flowers.",
    "He said hello to his friend and they walked to the park together.",
] * 60


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory) -> Path:
    """The tokenizer on disk. `sft.py` takes a path, not an object, so the CLI test needs
    this separately from the `tok` fixture below."""
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_bpe(iter(CORPUS), vocab_size=512, out_path=path, min_frequency=1)
    return path


@pytest.fixture(scope="module")
def tok(tok_path) -> Tokenizer:
    return Tokenizer(tok_path)


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


# ---- run timing and bounded stops ---------------------------------------------------

def test_fmt_dur_uses_the_right_unit_at_each_scale():
    assert fmt_dur(0.4) == "0.4s"
    assert fmt_dur(45.25) == "45.2s"
    assert fmt_dur(90) == "1m30s"
    assert fmt_dur(3600) == "1h00m"
    assert fmt_dur(3 * 86400 + 4 * 3600) == "3d04h"


def test_fmt_dur_handles_a_negative_interval():
    """Clock jumps (NTP, suspend/resume) can make a delta negative mid-run. Formatting it
    must not crash a run that's days in."""
    assert fmt_dur(-90) == "-1m30s"


def test_empty_stop_file_means_stop_now(tmp_path):
    p = tmp_path / "STOP"
    p.write_text("")
    assert stop_file_target(p) is None
    assert stopfile.read(p).now is True


def test_stop_file_with_a_step_number_is_a_deferred_stop(tmp_path):
    p = tmp_path / "STOP"
    p.write_text("20000\n")
    assert stop_file_target(p) == 20000
    assert stopfile.read(p) == stopfile.StopRequest(step=20000)


def test_garbage_stop_file_is_treated_as_stop_now(tmp_path):
    """An ambiguous STOP must fail towards stopping, never towards ignoring the request."""
    p = tmp_path / "STOP"
    p.write_text("soon-ish")
    assert stop_file_target(p) is None
    assert stopfile.read(p).now is True
    # …including a deadline that is not a number, which is the shape a truncated write takes.
    p.write_text("@later")
    assert stopfile.read(p).now is True


def test_a_missing_stop_file_is_not_a_stop(tmp_path):
    """The distinction the trainer polls on: no file at all means carry on."""
    assert stopfile.read(tmp_path / "STOP") is None
    assert stopfile.reached(None, 500) is None


def test_a_deadline_stop_file_fires_on_time_not_on_a_step(tmp_path):
    """The contract that makes "stop in 20 minutes" survive a portal restart: the deadline
    lives in the file, and the trainer -- not a timer somewhere -- decides when it lands."""
    p = tmp_path / "STOP"
    stopfile.write(p, stopfile.StopRequest(deadline=1_000_000.0))
    assert p.read_text() == "@1000000"
    req = stopfile.read(p)
    assert (req.step, req.now) == (None, False)
    assert stopfile.reached(req, step=999_999, now=999_999.0) is None   # not yet
    assert "reached stop time" in stopfile.reached(req, step=3, now=1_000_000.0)


def test_a_step_stop_fires_at_or_past_its_step():
    req = stopfile.StopRequest(step=700)
    assert stopfile.reached(req, 699) is None
    assert "700" in stopfile.reached(req, 700)
    assert stopfile.reached(req, 5000) is not None  # a target already behind us still fires


@pytest.mark.parametrize("text,seconds", [
    ("30m", 1800), ("90s", 90), ("2h", 7200), ("1h30m", 5400), ("45", 2700), ("1h30m15s", 5415),
])
def test_durations_parse_the_way_people_say_them(text, seconds):
    """A bare number is minutes: "give it another 30" is never half a minute. The same
    grammar is implemented in scripts/stop.sh, so both doors take the same words."""
    assert stopfile.parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["", "soon", "-5", "0s", "30x"])
def test_an_unreadable_duration_is_refused_rather_than_guessed(bad):
    with pytest.raises(ValueError):
        stopfile.parse_duration(bad)


def test_a_clock_time_already_past_means_tomorrow(tmp_path):
    """"--by 06:30" typed at midnight is the whole night, not a deadline in the past."""
    from datetime import datetime
    noon = datetime(2026, 7, 31, 12, 0, 0).timestamp()
    assert stopfile.deadline_from_clock("18:00", now=noon) == noon + 6 * 3600
    assert stopfile.deadline_from_clock("06:30", now=noon) == noon + 18.5 * 3600


@pytest.mark.parametrize("start,stop_after,stop_at,last_step", [
    (0, 5, None, 4),        # a fresh run doing 5 steps ends on step 4
    (620, 80, None, 699),   # 80 more steps from a resume at 620
    (620, None, 700, 700),  # "stop at 700" trains step 700 -- inclusive, not max_steps-like
    (620, 80, 650, 650),    # whichever bound comes first wins
    (620, None, None, None),  # no bound -> run to max_steps
])
def test_bounded_stop_resolves_to_an_inclusive_last_step(start, stop_after, stop_at, last_step):
    """`--at 700` leaving the checkpoint at 699, with no step-700 line in the log, is the
    off-by-one this pins down."""
    assert resolve_stop_step(start, stop_after, stop_at) == last_step


def test_stop_after_gives_exactly_that_many_steps():
    last = resolve_stop_step(620, 80, None)
    assert last - 620 + 1 == 80


def test_a_bound_already_behind_the_resume_point_is_an_error():
    """Silently training one step (or zero) would look like a hang. Say so instead."""
    with pytest.raises(ValueError, match="nothing to do"):
        resolve_stop_step(620, None, 500)


def test_stop_after_and_stop_at_default_to_off_and_parse_from_overrides(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("name: t\n")
    assert load_config(cfg_path).train.stop_after is None
    assert load_config(cfg_path).train.stop_at is None
    assert load_config(cfg_path).train.stop_after_s is None
    cfg = load_config(cfg_path, ["train.stop_after=500", "train.stop_at=20000",
                                 "train.stop_after_s=1800"])
    assert (cfg.train.stop_after, cfg.train.stop_at, cfg.train.stop_after_s) \
        == (500, 20000, 1800)


# ---- SFT stop and resume -------------------------------------------------------------
#
# The property that matters is not "it restarts" but "it restarts *in the right place*".
# SFT iterates a shuffled epoch rather than sampling a stream, so a resume that re-shuffles
# shows the model some conversations twice in one epoch and others not at all — the exact
# overfitting SFT is most exposed to, and invisible in the loss curve.

def _sft_dataset(tmp_path, n=64, seq_len=8):
    from aksharallm.train.sft import SFTDataset
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 100, size=(n, seq_len), dtype=np.uint16)
    mask = np.ones((n, seq_len), dtype=np.uint8)
    np.save(tmp_path / "train_tokens.npy", tokens)
    np.save(tmp_path / "train_mask.npy", mask)
    return SFTDataset(tmp_path / "train_tokens.npy", tmp_path / "train_mask.npy", "cpu")


def test_a_resumed_epoch_draws_the_batch_the_stopped_run_would_have_drawn_next(tmp_path):
    """Restore the epoch's rng state, skip the batches already seen, land on the next one."""
    ds = _sft_dataset(tmp_path)
    bs, stop_after = 4, 5

    # The uninterrupted run: one epoch, in order.
    rng = np.random.default_rng(1234)
    state = rng.bit_generator.state          # what `_progress` records
    uninterrupted = [x for x, _ in ds.epoch_batches(bs, rng)]

    # The resumed run: same state, skip what was done, continue.
    rng2 = np.random.default_rng(1234)
    rng2.bit_generator.state = state
    it = ds.epoch_batches(bs, rng2)
    for _ in range(stop_after):
        next(it)
    continued = [x for x, _ in it]

    assert len(continued) == len(uninterrupted) - stop_after
    for a, b in zip(continued, uninterrupted[stop_after:]):
        assert torch.equal(a, b), "a resume must not re-shuffle the epoch"


def test_a_reshuffled_resume_would_repeat_data(tmp_path):
    """Why the rng state is saved at all: without it the epoch order changes."""
    ds = _sft_dataset(tmp_path)
    first = [x for x, _ in ds.epoch_batches(4, np.random.default_rng(1234))]
    # A resume that just made a fresh generator from the *next* epoch's state:
    rng = np.random.default_rng(1234)
    ds.epoch_batches(4, rng).__next__()      # advance the generator as one epoch would
    second = [x for x, _ in ds.epoch_batches(4, rng)]
    assert not all(torch.equal(a, b) for a, b in zip(first, second)), (
        "if a re-shuffle produced the same order this test could not detect the bug")


def test_progress_survives_a_checkpoint_round_trip(tmp_path):
    """`_progress` has to come back out of a .pt file intact — it is a numpy rng state
    dict, not a scalar, and that is the part most likely to break silently."""
    from aksharallm.train import resume
    rng = np.random.default_rng(1234)
    p = resume.epoch_progress(epoch=1, batches_done=17,
                              epoch_rng_state=rng.bit_generator.state, best=1.25)
    torch.save({"sft_progress": p}, tmp_path / "ckpt.pt")
    back = torch.load(tmp_path / "ckpt.pt", weights_only=False)["sft_progress"]
    assert back["epoch"] == 1 and back["batches_done"] == 17 and back["best"] == 1.25
    # and it must still drive a generator to the same sequence
    r2 = np.random.default_rng(0)
    r2.bit_generator.state = back["epoch_rng_state"]
    assert np.array_equal(r2.permutation(50), np.random.default_rng(1234).permutation(50))


def _tiny_sft_run(tmp_path, tok_path):
    """A real (tiny) SFT setup: a base checkpoint, packed blocks, and a mask."""
    from aksharallm.config import ModelConfig
    from aksharallm.model.transformer import Transformer

    cfg = ModelConfig(vocab_size=512, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
                      max_seq_len=16, tie_embeddings=True, dropout=0.0)
    base = tmp_path / "base.pt"
    torch.save({"model": Transformer(cfg).state_dict(), "model_config": dict(vars(cfg)),
                "config": {"data": {"tokenizer": str(tok_path)}}, "step": 7,
                "best_val": 2.0}, base)

    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 32), ("val", 8)):
        np.save(data / f"{split}_tokens.npy",
                rng.integers(0, 512, size=(n, 16), dtype=np.uint16))
        np.save(data / f"{split}_mask.npy", np.ones((n, 16), dtype=np.uint8))
    return base, data


def test_a_stopped_sft_resumes_where_it_stopped(tmp_path, tok_path):
    """End-to-end, through the real CLI: stop at a step, resume, continue from step+1.

    The unit tests above pin the shuffle mechanics; this one pins that `main()` actually
    uses them. A resume that silently restarted at step 0 would pass every test above.
    """
    base, data = _tiny_sft_run(tmp_path, tok_path)
    out = tmp_path / "out"
    stop = tmp_path / "STOP"
    common = [sys.executable, "-m", "aksharallm.train.sft",
              "--base", str(base), "--data-dir", str(data),
              "--tokenizer", str(tok_path), "--out-dir", str(out),
              "--epochs", "2", "--batch-size", "2", "--grad-accum", "2",
              "--device", "cpu", "--eval-every", "10000", "--log-every", "1",
              "--stop-file", str(stop)]

    stop.write_text("2")                      # stop on reaching step 2
    first = subprocess.run(common, capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1], timeout=600)
    assert first.returncode == 0, first.stderr[-3000:]
    assert "STOP file asked for step 2" in first.stdout, first.stdout[-2000:]
    assert (out / "sft_last.pt").exists()
    saved = torch.load(out / "sft_last.pt", weights_only=False)
    assert saved["sft_progress"]["batches_done"] > 0

    # The honoured request is cleared, so the resume is not stopped again at step 0.
    assert not stop.exists(), "sft.py must clear the STOP file it acted on"

    second = subprocess.run(common + ["--resume", "auto"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=600)
    assert second.returncode == 0, second.stderr[-3000:]
    assert "resumed from" in second.stdout, second.stdout[-2000:]
    # It continues rather than restarting: the first step it logs is past where it stopped.
    steps = [int(m) for m in re.findall(r"step\s+(\d+)/", second.stdout)]
    assert steps and steps[0] > 2, f"resumed run restarted at {steps[:3]}"


@pytest.mark.parametrize("value", ["none", "None", "off", "no", "  "])
def test_resume_can_be_switched_off_by_word(tmp_path, tok_path, value):
    """`RESUME=none scripts/stage.sh sft ...` must start over, not hunt for a file called
    "none". The value comes from the environment, where a word is the natural way to say
    "don't", and the failure mode is a FileNotFoundError at launch."""
    base, data = _tiny_sft_run(tmp_path, tok_path)
    out = tmp_path / "out"
    (out).mkdir()
    # a checkpoint that `auto` WOULD pick up, to prove the word is what turned it off
    stop = tmp_path / "STOP"
    stop.write_text("1")
    argv = [sys.executable, "-m", "aksharallm.train.sft",
            "--base", str(base), "--data-dir", str(data), "--tokenizer", str(tok_path),
            "--out-dir", str(out), "--epochs", "1", "--batch-size", "2", "--grad-accum", "2",
            "--device", "cpu", "--eval-every", "10000", "--log-every", "1",
            "--stop-file", str(stop)]
    subprocess.run(argv, capture_output=True, text=True,
                   cwd=Path(__file__).resolve().parents[1], timeout=600)
    assert (out / "sft_last.pt").exists()

    stop.write_text("1")
    r = subprocess.run(argv + ["--resume", value], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1], timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "resumed from" not in r.stdout, "the word should have switched resuming off"


# ---- the resume contract, shared by SFT / DPO / GRPO -------------------------------------

def test_resume_resolve_reads_auto_and_the_off_words(tmp_path):
    from aksharallm.train import resume
    last = tmp_path / "sft_last.pt"
    assert resume.resolve(None, last) is None
    assert resume.resolve("auto", last) is None          # nothing to continue yet
    last.write_bytes(b"x")
    assert resume.resolve("auto", last) == last
    for word in ("none", "None", " off ", "no", "false", ""):
        assert resume.resolve(word, last) is None, word
    assert resume.resolve(str(tmp_path / "other.pt"), last) == tmp_path / "other.pt"


def test_restore_rng_survives_a_missing_state(capsys):
    from aksharallm.train import resume
    rng = np.random.default_rng(0)
    assert resume.restore_rng(rng, None, "the sampler") is False   # not an error
    assert resume.restore_rng(rng, {"nonsense": 1}, "the sampler") is False
    assert "could not restore the sampler" in capsys.readouterr().out


def test_a_resume_loads_the_policy_and_never_the_reference(tmp_path):
    """The rule that is unique to post-training, and silent when broken.

    DPO and GRPO hold a trained policy and a frozen reference; the KL term measures drift
    away from that reference. If a resume reloads the reference from the same checkpoint as
    the policy, the anchor moves with the policy: KL collapses toward zero and the run is
    free to wander arbitrarily far from the SFT model *while reporting a small KL*.
    """
    from aksharallm.config import ModelConfig
    from aksharallm.model.transformer import Transformer
    from aksharallm.train import resume

    cfg = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                      max_seq_len=8, tie_embeddings=True, dropout=0.0)
    reference = Transformer(cfg)                       # what --init/--sft produced
    anchor = {k: v.clone() for k, v in reference.state_dict().items()}

    drifted = Transformer(cfg)                         # a policy that has trained a while
    with torch.no_grad():
        for p in drifted.parameters():
            p.add_(torch.randn_like(p))
    torch.save({"model": drifted.state_dict()}, tmp_path / "last.pt")

    policy = Transformer(cfg)
    resume.load(tmp_path / "last.pt", policy, optimizer=None, device="cpu")

    # the policy moved to the checkpoint...
    for k, v in policy.state_dict().items():
        assert torch.allclose(v, drifted.state_dict()[k]), k
    # ...and the reference did not move at all
    for k, v in reference.state_dict().items():
        assert torch.allclose(v, anchor[k]), f"the KL reference drifted at {k}"


def test_a_stage_log_carries_what_the_dashboard_reads(tmp_path, tok_path):
    """The complaint that prompted this: a finished SFT showed a loss curve and four empty
    tiles. Throughput, MFU, ETA, progress and the Sessions table are all read by key name
    from the step log, and SFT was writing none of them.
    """
    from aksharallm.train import runlog

    base, data = _tiny_sft_run(tmp_path, tok_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "aksharallm.train.sft", "--base", str(base),
         "--data-dir", str(data), "--tokenizer", str(tok_path), "--out-dir", str(out),
         "--epochs", "1", "--batch-size", "2", "--grad-accum", "2", "--device", "cpu",
         "--eval-every", "4", "--log-every", "2"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]

    records = runlog.load_records(out / "sft_log.jsonl")
    steps = [x for x in records if "step" in x and "loss" in x]
    assert steps, "no step records"
    for key in ("tok_per_sec", "mfu", "grad_norm", "eta_s", "s_per_step"):
        assert steps[-1].get(key) is not None, f"the dashboard reads {key} and it is missing"

    # session brackets: the Sessions table, and where max_steps/progress come from
    sessions = runlog.summarise_sessions(runlog.split_sessions(records))
    assert len(sessions) == 1, sessions
    start = next(x for x in records if x.get("event") == "session_start")
    end = next(x for x in records if x.get("event") == "session_end")
    assert start["max_steps"] > 0 and start["tokens_per_step"] == 2 * 2 * 16
    # a session record must NOT look like an eval: it has no step to place one at
    assert "val_loss" not in end and end["final_val_loss"] is not None
    assert (out / "report.md").exists(), "the run report must still write"
