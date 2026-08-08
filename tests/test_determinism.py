"""Stopping and resuming must be the same as never stopping.

`docs/10` says a run is trained "over evenings using stop/resume", and every controlled
experiment in this repo — the MoE comparison, the diffusion comparison, the LoRA-vs-full
table — rests on two runs seeing *the same data in the same order*. Neither claim was true
before these tests existed:

    self.rng = np.random.default_rng()        # loader.py, unseeded

`torch.manual_seed(cfg.train.seed)` made the weights and dropout reproducible, so a run
looked seeded. The **data order** came from OS entropy, so two runs of one config saw
different batches, and a resume re-drew from a fresh stream rather than continuing the old
one — which means a stopped-and-resumed run trains on some of the same data twice and skips
some entirely. Nothing about the loss curve would say so.

These are slow-ish (they train a real model for a few steps) and deliberately tiny.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aksharallm.data.loader import MixedTokenDataset, TokenDataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A tiny token stream and a tokenizer, so the trainer has something real to read."""
    d = tmp_path_factory.mktemp("determinism")
    rng = np.random.default_rng(0)
    for name in ("train.bin", "val.bin"):
        # Ids must be inside the real tokenizer's vocabulary: the trainer refuses a
        # config whose vocab_size disagrees with the tokenizer on disk, which is the
        # check that stops a checkpoint being trained against the wrong embedding table.
        rng.integers(0, 8192, size=40_000, dtype=np.uint16).tofile(d / name)
    return d


# ---------------------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------------------


def test_one_seed_gives_one_data_order(corpus):
    a = TokenDataset(corpus / "train.bin", 32, "cpu", seed=11)
    b = TokenDataset(corpus / "train.bin", 32, "cpu", seed=11)
    for _ in range(5):
        assert torch.equal(a.get_batch(4)[0], b.get_batch(4)[0])


def test_different_seeds_give_different_data_orders(corpus):
    """Otherwise the seed is decorative, which is the more embarrassing failure."""
    a = TokenDataset(corpus / "train.bin", 32, "cpu", seed=1)
    b = TokenDataset(corpus / "train.bin", 32, "cpu", seed=2)
    assert not torch.equal(a.get_batch(8)[0], b.get_batch(8)[0])


def test_no_seed_still_means_no_seed(corpus):
    """`seed=None` keeps the old behaviour on purpose: a quick interactive load should not
    silently be pinned to one arbitrary stream just because the default changed."""
    a = TokenDataset(corpus / "train.bin", 32, "cpu")
    b = TokenDataset(corpus / "train.bin", 32, "cpu")
    assert not torch.equal(a.get_batch(8)[0], b.get_batch(8)[0])


def test_restoring_the_rng_state_replays_the_same_batches(corpus):
    """Saving the *seed* is not enough — a resume would restart the stream and re-show the
    batches the run already trained on. The state is what makes a resume a continuation."""
    ds = TokenDataset(corpus / "train.bin", 32, "cpu", seed=3)
    ds.get_batch(4)  # get somewhere into the stream
    state = ds.rng_state
    expected = [ds.get_batch(4)[0] for _ in range(3)]
    ds.rng_state = state
    for want in expected:
        assert torch.equal(ds.get_batch(4)[0], want)


def test_the_blended_loader_is_seeded_all_the_way_down(corpus):
    """`get_batch` passes the parent generator down, but anything reaching for
    `mixed.datasets[i]` directly should be reproducible too."""
    sources = [{"bin": str(corpus / "train.bin"), "weight": 0.7},
               {"bin": str(corpus / "val.bin"), "weight": 0.3}]
    a = MixedTokenDataset(sources, 32, "cpu", seed=5)
    b = MixedTokenDataset(sources, 32, "cpu", seed=5)
    assert torch.equal(a.get_batch(8)[0], b.get_batch(8)[0])
    assert torch.equal(a.datasets[0].get_batch(2)[0], b.datasets[0].get_batch(2)[0])


def test_the_blended_loader_round_trips_its_whole_state(corpus):
    sources = [{"bin": str(corpus / "train.bin"), "weight": 0.5},
               {"bin": str(corpus / "val.bin"), "weight": 0.5}]
    ds = MixedTokenDataset(sources, 32, "cpu", seed=5)
    ds.get_batch(8)
    state = ds.rng_state
    want = ds.get_batch(8)[0]
    ds.rng_state = state
    assert torch.equal(ds.get_batch(8)[0], want)


# ---------------------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------------------


def write_config(path: Path, out_dir: Path, corpus: Path, max_steps: int) -> Path:
    path.write_text(f"""
name: determinism
model: {{vocab_size: 8192, d_model: 32, n_layers: 2, n_heads: 4, max_seq_len: 32}}
data:
  train_bin: {corpus / 'train.bin'}
  val_bin: {corpus / 'val.bin'}
  tokenizer: {ROOT / 'data' / 'tinystories' / 'tokenizer.json'}
optim: {{lr: 0.001, warmup_steps: 2}}
train:
  out_dir: {out_dir}
  batch_size: 2
  grad_accum: 1
  seq_len: 32
  max_steps: {max_steps}
  eval_every: 0
  sample_every: 0
  ckpt_every: 0
  log_every: 0
  compile: false
  seed: 1234
  resume: auto
""")
    return path


def train(config: Path, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", "aksharallm.train.pretrain", str(config), *(extra or [])]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, corpus):
    """**The test this file exists for.** Train 8 steps. Then train 4, stop, and train 4
    more. The two final checkpoints must be bit-for-bit identical.

    It fails if the data stream is not restored on resume, if the optimizer state is not,
    or if the LR schedule is computed from the session's step rather than the run's — three
    independent bugs that all present as "the resumed run is slightly worse", which is
    indistinguishable from noise and is therefore never investigated.
    """
    if not (ROOT / "data" / "tinystories" / "tokenizer.json").exists():
        pytest.skip("needs a tokenizer on disk")

    straight = tmp_path / "straight"
    train(write_config(tmp_path / "a.yaml", straight, corpus, 8))

    stopped = tmp_path / "stopped"
    cfg = write_config(tmp_path / "b.yaml", stopped, corpus, 8)
    train(cfg, ["-o", "train.stop_after=4"])
    train(cfg)  # resume:auto picks up ckpt_last.pt

    a, b = weights(straight / "ckpt_last.pt"), weights(stopped / "ckpt_last.pt")
    assert a.keys() == b.keys()
    drift = {k: float((a[k] - b[k]).abs().max()) for k in a if a[k].is_floating_point()}
    worst = max(drift.values())
    assert worst == 0.0, (
        f"a resumed run diverged from an uninterrupted one by {worst:.3e}; "
        f"worst tensor: {max(drift, key=drift.get)}"
    )
