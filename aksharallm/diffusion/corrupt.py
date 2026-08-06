"""The forward process and the loss — all of masked diffusion training, in one file.

The objective is short enough to state completely::

    t ~ U(t_min, 1)                       one mask rate per sequence
    x_t[i] = [MASK]  with probability t   independently per position
    loss    = (1/t) * (1/T) * sum over MASKED positions of  -log p(x[i] | x_t)

and that weighted sum is a variational bound (an ELBO) on the log-likelihood of `x`. There
is no noise schedule to tune, no separate denoiser network and no continuous latent: for
discrete tokens, "noise" means "replaced by `[MASK]`", and the amount of it is one number
drawn uniformly.

Three details in that formula are the whole of the implementation, and each one is a place
to get it quietly wrong.

**Why `1/t`.** Without it, a sequence masked at 90% would contribute nine times the loss of
one masked at 10% simply for having more terms in the sum, and the model would spend its
capacity on the hardest, least informative corruptions. Dividing by `t` — the *expected*
fraction of masked positions — makes every mask rate contribute the same expected weight.
It is also exactly what the ELBO derivation produces, which is the better reason.

**Why `t` is per sequence, not per batch.** Drawing one `t` for the whole batch is legal and
gives an unbiased estimate; it also means every sequence in a micro-batch sees the same
difficulty, and the gradient noise across steps goes up sharply. One draw per row costs
nothing and is what every published implementation does.

**Why `t_min`.** `1/t` is unbounded as `t → 0`. In the limit the numerator goes to zero too
(nothing is masked, so the sum is empty) and the estimator is fine in expectation — but a
single sequence that draws `t = 1e-9` *and* happens to mask one token multiplies that
token's cross-entropy by a billion, which reaches the optimiser as one enormous gradient
that the clip then swallows for the entire step. Clamping the draw to `[t_min, 1)` is the
standard fix; it makes this an exact ELBO for `t ~ U(t_min, 1)` rather than an approximate
one for `U(0, 1)`, and at `t_min = 1e-3` the slice given up is a tenth of a percent.

Nothing is forced: a sequence that happens to mask zero tokens contributes exactly zero,
which is the correct value of an empty sum. Forcing "at least one mask" would look tidier
and bias the estimator.

Read with: docs/19-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Corruption:
    """One draw of the forward process: what the model sees, and what it has to recover."""

    #: (B, T) int64 — the sequence with `[MASK]` written into the corrupted positions.
    x_t: torch.Tensor
    #: (B, T) bool — True where a token was replaced. The loss lives exactly here.
    masked: torch.Tensor
    #: (B,) float — the mask rate each row was drawn with. Needed by the `1/t` weight.
    t: torch.Tensor

    @property
    def rate(self) -> float:
        """The fraction of positions actually masked, over the whole batch.

        Worth watching next to `t`: they should agree to within sampling noise, and if they
        do not, the Bernoulli draw is not using the `t` the weight is dividing by.
        """
        return self.masked.float().mean().item()


def sample_t(batch: int, t_min: float, device, generator: torch.Generator | None = None
             ) -> torch.Tensor:
    """`batch` mask rates drawn uniformly from `[t_min, 1)`."""
    u = torch.rand(batch, device=device, generator=generator)
    return t_min + (1.0 - t_min) * u


def corrupt(x: torch.Tensor, mask_id: int, t: torch.Tensor,
            generator: torch.Generator | None = None,
            keep: torch.Tensor | None = None) -> Corruption:
    """Mask each token of `x` independently with its row's probability `t`.

    `keep` is an optional (B, T) bool of positions that must survive — used at *evaluation*
    time to score an infill (the prefix and suffix are given, so they are never corrupted
    and never contribute loss). Training passes None: every position is fair game, which is
    what makes one trained model able to fill any hole.
    """
    if x.dim() != 2:
        raise ValueError(f"expected (B, T) token ids, got {tuple(x.shape)}")
    probs = t[:, None].expand_as(x)
    masked = torch.rand(x.shape, device=x.device, generator=generator) < probs
    if keep is not None:
        masked = masked & ~keep
    x_t = torch.where(masked, torch.full_like(x, mask_id), x)
    return Corruption(x_t=x_t, masked=masked, t=t)


def diffusion_loss(logits: torch.Tensor, x: torch.Tensor, c: Corruption) -> tuple:
    """The ELBO term for one batch. Returns `(loss, stats)`.

    `logits` are the model's output over the *corrupted* sequence, in full — every position,
    not just the masked ones. Predictions at unmasked positions are thrown away here rather
    than never computed: the transformer is bidirectional, so producing them costs one
    matmul over positions it had to attend to anyway.

    `stats` carries the two numbers worth logging beside the loss:

    * `ce_masked` — the plain mean cross-entropy over masked positions, *unweighted*. This
      is the number with an intuitive meaning ("how surprised was it by a token it could not
      see"), and it is **not** the loss: the loss is its `1/t`-weighted, per-position form.
    * `mask_rate` — the fraction actually masked, as a check on the draw.
    """
    B, T, V = logits.shape
    per_tok = F.cross_entropy(
        logits.reshape(-1, V).float(), x.reshape(-1), reduction="none"
    ).view(B, T)
    m = c.masked.to(per_tok.dtype)
    # Per sequence: sum the masked positions, spread it over the whole length, undo the
    # thinning by 1/t. Averaging over the batch last keeps every sequence equally weighted
    # regardless of how many of its tokens the draw happened to hit.
    per_seq = (per_tok * m).sum(dim=1) / (c.t * T)
    loss = per_seq.mean()
    n_masked = m.sum()
    stats = {
        "ce_masked": ((per_tok * m).sum() / n_masked.clamp(min=1)).detach(),
        "mask_rate": (n_masked / (B * T)).detach(),
        "t_mean": c.t.mean().detach(),
    }
    return loss, stats
