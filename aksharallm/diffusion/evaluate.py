"""Measuring a diffusion model — and the one number you must not compare.

**The validation loss of a diffusion model is not the validation loss of an autoregressive
model.** An AR run's val loss is the exact per-token cross-entropy: the negative log
likelihood the model assigns to the held-out text, and `exp()` of it is a real perplexity.
A diffusion model's is a *variational bound* — a Monte-Carlo estimate of an upper bound on
the same quantity. Two consequences follow, and both have to be said out loud every time a
number is printed:

1. It is an **upper bound**, so a diffusion model's 1.9 and an AR model's 1.9 do not mean
   the same thing; the diffusion model is at least that good and might be better. Comparing
   the two directly favours the AR model by an unknown margin.
2. It is **noisy**, because it averages over random draws of `t` and of the mask. The fix is
   not more batches but a *fixed* set of draws: `elbo()` seeds its own generator, so the
   same checkpoint measured twice returns the same number and two checkpoints measured a
   thousand steps apart differ because the model changed.

`loss_by_t` is the diagnostic that has no AR equivalent and is worth reading before
anything else: cross-entropy bucketed by how much of the sequence was masked. At `t = 0.1`
the model is doing a cloze test with nine tenths of the context available; at `t = 0.9` it
is nearly writing from nothing. A model that is only good at low `t` will produce good
infills and poor unconditional samples, and the loss curve alone will not say so.

Read with: docs/20-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from .corrupt import corrupt, diffusion_loss, sample_t


@torch.no_grad()
def elbo(model, dataset, batch_size: int, n_batches: int, *, t_min: float = 1e-3,
         seed: int = 20260806, ctx=None, repeats: int = 1) -> dict:
    """Per-token NELBO on `dataset`, in nats. Lower is better; it bounds the true NLL above.

    `repeats` re-corrupts the *same* batches with fresh draws of `t`. It costs a forward pass
    each and buys variance reduction, which matters when the number is being used to pick a
    best checkpoint. The trainer leaves it at 1 and relies on the fixed seed instead — the
    two together are what make the curve readable.
    """
    mask_id = int(model.cfg.mask_token_id)
    ctx = ctx or nullcontext()
    was_training = model.training
    model.eval()

    # One generator for the whole evaluation, seeded here: the sequence of draws is then a
    # deterministic function of (seed, batch order), so this is repeatable to the bit.
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))

    total, ce_sum, n_batches_done, rates = 0.0, 0.0, 0, 0.0
    for _ in range(max(1, repeats)):
        for x, _y in dataset.iter_eval_batches(batch_size, n_batches, seed=1234):
            x = x.to(device)
            t = sample_t(x.shape[0], t_min, device, gen)
            c = corrupt(x, mask_id, t, generator=gen)
            with ctx:
                logits, _ = model(c.x_t, full_logits=True)
            loss, stats = diffusion_loss(logits, x, c)
            total += loss.item()
            ce_sum += stats["ce_masked"].item()
            rates += stats["mask_rate"].item()
            n_batches_done += 1

    if was_training:
        model.train()
    n = max(1, n_batches_done)
    nelbo = total / n
    return {
        "nelbo": nelbo,
        # Labelled a bound everywhere it is shown. `exp(nelbo)` is an upper bound on
        # perplexity, not a perplexity, and the name is the only thing stopping it being
        # put in a table beside an AR model's real one.
        "ppl_upper_bound": math.exp(min(nelbo, 20)),
        "ce_masked": ce_sum / n,
        "mask_rate": rates / n,
        "batches": n_batches_done,
        "seed": seed,
        "note": "NELBO is an upper bound on the true NLL and is not comparable with an "
                "autoregressive run's cross-entropy.",
    }


@torch.no_grad()
def loss_by_t(model, dataset, batch_size: int, n_batches: int, *, buckets: int = 10,
              seed: int = 20260806, ctx=None) -> list[dict]:
    """Cross-entropy on masked positions, bucketed by mask rate.

    Each bucket is evaluated at a *fixed* `t` — the centre of the bucket — rather than by
    binning random draws. That makes the curve smooth with far fewer batches, and it is the
    honest thing to plot: the x axis is then the mask rate the model was actually given.
    """
    mask_id = int(model.cfg.mask_token_id)
    ctx = ctx or nullcontext()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))

    rows = []
    for b in range(buckets):
        lo, hi = b / buckets, (b + 1) / buckets
        centre = (lo + hi) / 2
        ce_sum, n = 0.0, 0
        for x, _y in dataset.iter_eval_batches(batch_size, n_batches, seed=1234):
            x = x.to(device)
            t = torch.full((x.shape[0],), centre, device=device)
            c = corrupt(x, mask_id, t, generator=gen)
            with ctx:
                logits, _ = model(c.x_t, full_logits=True)
            _, stats = diffusion_loss(logits, x, c)
            ce_sum += stats["ce_masked"].item()
            n += 1
        rows.append({"t": centre, "lo": lo, "hi": hi,
                     "ce_masked": ce_sum / max(1, n), "batches": n})

    if was_training:
        model.train()
    return rows
