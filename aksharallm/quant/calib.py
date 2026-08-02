"""Calibration: showing the model some real text and watching what its layers see.

RTN needs nothing but the weights. Every method that beats it needs to know something
about the *inputs* those weights get multiplied by, because that is the only way to tell
an important weight from an unimportant one. A weight that multiplies a channel which is
always near zero can be rounded carelessly; a weight on a channel that regularly hits 50
cannot.

So calibration is: run a few hundred sequences of real text through the model, and record
per-layer statistics of the input `x` to each Linear.

Two statistics, for two methods:

  Hessian H = sum over samples of x x^T   (in_features x in_features)
      GPTQ needs this. It is the curvature of the layer's output error with respect to
      its weights -- if you perturb weight column i, how much does the output move, and
      how does that interact with a perturbation of column j? Storing it costs
      in_features^2 floats per layer, which for d_ff=2752 is 30 MB in fp32. That is why
      GPTQ processes one layer at a time and frees as it goes.

  channel scale s_j = mean |x_j|         (in_features,)
      AWQ needs only this. Far cheaper, and it turns out to be almost as good.

How much data?
--------------
Very little. 128 sequences is the number the literature settled on and it holds up here:
the statistics being estimated are second moments of a distribution the model sees
millions of times, and they converge fast. More calibration data mostly buys noise
reduction in the tail, and costs linear time.

A warning worth stating plainly: calibrate on data that resembles what the model will
*do*. Calibrating a code model purely on prose measures the wrong activations, and the
damage shows up exactly where you care. Our blended run has both in `val.bin`, so the
default draws from there.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..data.loader import TokenDataset


@dataclass
class LayerStats:
    """Accumulated input statistics for one Linear layer."""

    in_features: int
    n_samples: int = 0
    hessian: torch.Tensor | None = None      # (in, in), fp32
    abs_mean: torch.Tensor | None = None     # (in,), fp32
    abs_max: torch.Tensor | None = None      # (in,), fp32

    def add(self, x: torch.Tensor, want_hessian: bool):
        """x: (..., in_features) -- one layer's input for a batch of tokens."""
        flat = x.reshape(-1, x.shape[-1]).float()
        n = flat.shape[0]
        if want_hessian:
            if self.hessian is None:
                self.hessian = torch.zeros(self.in_features, self.in_features,
                                           dtype=torch.float32, device=flat.device)
            # Running mean of x x^T. Keeping it as a mean rather than a sum stops the
            # magnitude drifting with calibration size, so the damping below means the
            # same thing regardless of how many sequences were used.
            self.hessian *= self.n_samples / (self.n_samples + n)
            self.hessian += (flat.T @ flat) / (self.n_samples + n)
        a = flat.abs()
        s = a.sum(dim=0)
        m = a.amax(dim=0)
        if self.abs_mean is None:
            self.abs_mean = torch.zeros(self.in_features, dtype=torch.float32,
                                        device=flat.device)
            self.abs_max = torch.zeros_like(self.abs_mean)
        self.abs_mean = (self.abs_mean * self.n_samples + s) / (self.n_samples + n)
        self.abs_max = torch.maximum(self.abs_max, m)
        self.n_samples += n


@dataclass
class Calibration:
    """Statistics for every Linear in the model, by name."""

    stats: dict[str, LayerStats] = field(default_factory=dict)
    n_sequences: int = 0
    source: str | None = None

    def get(self, name: str) -> LayerStats | None:
        return self.stats.get(name)

    def free(self, name: str):
        """Drop one layer's Hessian. GPTQ calls this as it finishes each layer; without
        it a 24-layer model holds every Hessian at once and runs out of memory."""
        if name in self.stats:
            self.stats[name].hessian = None


@torch.no_grad()
def collect(
    model: nn.Module,
    val_bin: str,
    seq_len: int,
    n_sequences: int = 128,
    batch_size: int = 4,
    device: str = "cuda",
    want_hessian: bool = True,
    layer_filter=None,
    progress=None,
) -> Calibration:
    """Run text through the model, recording each Linear's input statistics.

    Implemented with forward hooks so the model itself needs no modification -- and so
    this works identically on a float model (what GPTQ/AWQ quantize *from*) and on a
    partially quantized one.
    """
    from .qlinear import QuantLinear

    targets = {
        n: m for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, QuantLinear)) and (layer_filter is None or layer_filter(n))
    }
    calib = Calibration(source=val_bin)
    handles = []

    def make_hook(name: str, in_features: int):
        calib.stats[name] = LayerStats(in_features=in_features)

        def hook(_mod, inputs, _out):
            calib.stats[name].add(inputs[0].detach(), want_hessian)

        return hook

    for name, mod in targets.items():
        in_f = mod.in_features
        handles.append(mod.register_forward_hook(make_hook(name, in_f)))

    try:
        ds = TokenDataset(val_bin, seq_len, device)
        n_batches = max(1, n_sequences // batch_size)
        for i, (x, _y) in enumerate(ds.iter_eval_batches(batch_size, n_batches, seed=7)):
            model(x)
            calib.n_sequences += x.shape[0]
            if progress:
                progress(i + 1, n_batches)
    finally:
        for h in handles:
            h.remove()
    return calib


def damped_hessian(h: torch.Tensor, percent: float = 0.01) -> torch.Tensor:
    """Add a ridge to the diagonal before inverting.

    H is a sum of outer products of activations. It is positive *semi*-definite, and in
    practice frequently singular: any input channel that is dead over the calibration set
    contributes an all-zero row and column. Cholesky then fails, or worse, succeeds with
    garbage.

    Adding `percent` of the mean diagonal fixes it, and the choice is not arbitrary --
    it is a prior saying "assume every channel has at least this much energy". Too small
    and the inverse amplifies noise in the rarely-used channels; too large and the error
    compensation is throttled towards plain RTN. 1% is the standard value and behaves
    well here.
    """
    h = h.clone()
    diag = torch.arange(h.shape[0], device=h.device)
    mean_diag = torch.diagonal(h).mean().clamp(min=1e-8)
    dead = torch.diagonal(h) == 0
    if dead.any():
        h[diag[dead], diag[dead]] = mean_diag
    h[diag, diag] += percent * mean_diag
    return h
