"""Batch loader over the flat uint16 token files.

There is no PyTorch Dataset/DataLoader here on purpose. Our data is a single contiguous
array of tokens on disk; the "dataset" is just `np.memmap`. Sampling a batch is B random
offsets and a slice. No workers, no collate, no queue -- and it's faster than a DataLoader
because the OS page cache is already doing the readahead for us.

Read with: docs/01-data.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, bin_path: str | Path, seq_len: int, device: str = "cuda",
                 seed: int | None = None):
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run `python -m aksharallm.data.prepare` first"
            )
        self.seq_len = seq_len
        self.device = device
        # RNG for training batches. `np.random` (the module) has no .integers -- only
        # Generator does -- so we always hold a real Generator.
        #
        # `seed=None` draws from OS entropy, which is what this used to do unconditionally.
        # That made `train.seed` a lie: it seeded torch, so the initial weights and dropout
        # were reproducible, while the DATA ORDER was not -- and the data order is most of
        # what a training run is. Every "same seed and same data" comparison in this repo
        # (the MoE experiment, the diffusion experiment, the LoRA-vs-full table) was
        # therefore comparing runs that saw different batches. Callers pass the seed now;
        # `rng_state` below is the other half, for resume.
        self.rng = np.random.default_rng(seed)
        self.n_tokens = self.path.stat().st_size // 2  # uint16
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{self.path} has only {self.n_tokens} tokens, need > {seq_len}")

    @property
    def rng_state(self) -> dict:
        """The generator's internal state, for a checkpoint.

        Saving the *seed* is not enough: a resume would restart the stream from the
        beginning and re-show the batches the run already trained on. Saving the state means
        a stopped-and-resumed run draws exactly the batches the uninterrupted one would,
        which is what makes `tests/test_determinism.py` able to assert bitwise equality.
        """
        return self.rng.bit_generator.state

    @rng_state.setter
    def rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

    def _data(self) -> np.memmap:
        # Re-opened per call. np.memmap leaks virtual address space if kept alive across
        # a long training run with many reads; reopening is cheap and avoids the leak.
        return np.memmap(self.path, dtype=np.uint16, mode="r")

    def get_batch(self, batch_size: int, generator: np.random.Generator | None = None):
        """Returns (x, y) int64 tensors of shape (batch_size, seq_len).

        y is x shifted left by one -- the model at position t sees x[t] and must predict
        y[t] == x[t+1]. That single-token shift *is* the pretraining objective.
        """
        rng = generator if generator is not None else self.rng
        data = self._data()
        ix = rng.integers(0, self.n_tokens - self.seq_len - 1, size=batch_size)

        # Build in one contiguous host buffer, then a single H2D copy.
        xs = np.stack([data[i : i + self.seq_len] for i in ix]).astype(np.int64)
        ys = np.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if self.device.startswith("cuda"):
            # pin + non_blocking lets the copy overlap with the previous step's compute
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def iter_eval_batches(self, batch_size: int, n_batches: int, seed: int = 0):
        """Deterministic batches for validation, so val loss is comparable across steps."""
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            yield self.get_batch(batch_size, generator=rng)

    def __repr__(self):
        return f"TokenDataset({self.path.name}, {self.n_tokens:,} tokens, seq_len={self.seq_len})"


class MixedTokenDataset:
    """Sample each batch from several token files by fixed weight.

    Used to blend corpora during pretraining -- e.g. 85% general web text + 15% Python.
    The mix is exact *every step* (largest-remainder apportionment of the batch) rather
    than only correct on average, which keeps the gradient's data composition stable.

    Why do this in the loader instead of interleaving the tokens on disk: the ratio
    becomes a config knob you can change without re-tokenizing 20 GB of data. The same
    loader then powers the code-heavy continued-pretraining phase just by flipping the
    weights.
    """

    def __init__(self, sources: list[dict], seq_len: int, device: str = "cuda",
                 seed: int | None = None):
        # sources: [{"bin": path, "weight": float}, ...]
        if not sources:
            raise ValueError("MixedTokenDataset needs at least one source")
        # Sub-datasets get a derived seed as well. `get_batch` passes this object's generator
        # down, so theirs is normally unused -- but anything that reaches for
        # `mixed.datasets[i]` directly (a measurement, a notebook) should be reproducible too.
        self.datasets = [
            TokenDataset(s["bin"], seq_len, device,
                         seed=None if seed is None else seed + 1 + i)
            for i, s in enumerate(sources)
        ]
        w = np.array([float(s.get("weight", 1.0)) for s in sources], dtype=np.float64)
        if not (w > 0).all():
            raise ValueError("all source weights must be positive")
        self.weights = w / w.sum()
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)

    @property
    def rng_state(self) -> dict:
        """This generator's state plus every source's -- see `TokenDataset.rng_state`."""
        return {"top": self.rng.bit_generator.state,
                "sources": [d.rng_state for d in self.datasets]}

    @rng_state.setter
    def rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state["top"]
        for d, s in zip(self.datasets, state.get("sources", [])):
            d.rng_state = s

    @property
    def n_tokens(self) -> int:
        return sum(d.n_tokens for d in self.datasets)

    def _counts(self, batch_size: int) -> np.ndarray:
        """How many rows to draw from each source. Sums to batch_size exactly."""
        exact = self.weights * batch_size
        base = np.floor(exact).astype(int)
        leftover = batch_size - int(base.sum())
        if leftover:
            # hand the remaining slots to the largest fractional parts
            order = np.argsort(-(exact - base))
            base[order[:leftover]] += 1
        return base

    def get_batch(self, batch_size: int, generator: np.random.Generator | None = None):
        rng = generator if generator is not None else self.rng
        counts = self._counts(batch_size)
        xs, ys = [], []
        for ds, c in zip(self.datasets, counts):
            if c == 0:
                continue
            x, y = ds.get_batch(int(c), generator=rng)
            xs.append(x)
            ys.append(y)
        # Row order within a batch doesn't affect the loss (it's a mean over all tokens,
        # with no cross-example interaction), so we simply concatenate the per-source rows.
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def iter_eval_batches(self, batch_size: int, n_batches: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            yield self.get_batch(batch_size, generator=rng)

    def __repr__(self):
        parts = ", ".join(f"{d.path.name}:{w:.2f}" for d, w in zip(self.datasets, self.weights))
        return f"MixedTokenDataset([{parts}], {self.n_tokens:,} tokens)"
