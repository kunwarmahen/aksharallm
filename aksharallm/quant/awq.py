"""AWQ: make the important channels easier to quantize, by scaling them up first.

The observation
---------------
Quantization error in a group is set by the group's *range*: one large weight stretches
the scale and coarsens every other weight in the group. But not all weights deserve equal
protection -- a weight multiplying an input channel that is always tiny contributes almost
nothing to the output, no matter how badly it is rounded.

AWQ exploits that with a change of variables that leaves the layer's function untouched:

    x W^T  ==  (x / s) (W diag(s))^T          for any positive per-input-channel s

Quantize `W diag(s)` instead of `W`. Channels with big activations get scaled *up*, so
they occupy more of their group's range and are represented more finely; channels that
barely matter get scaled down and absorb the rounding instead. Nothing is added to the
model -- it is the same linear map, written differently.

Where does the 1/s go?
----------------------
Into whatever produced `x`, so it costs nothing at runtime. Our architecture gives four
sites where that is possible, and this is the part that is genuinely architecture-specific:

    attn_norm  ->  wq, wk, wv     one RMSNorm feeds all three: fold 1/s into its gain
    ffn_norm   ->  w1, w3         same story
    w3         ->  w2             w2's input is silu(w1 x) * (w3 x) elementwise, so
                                  dividing row j of w3 divides input channel j of w2
    wv         ->  wo             wo's input is the attention output, whose channel j
                                  comes from value head j // head_dim

The last one is constrained by GQA. With n_kv_heads < n_heads, several query heads read
the *same* value head, so their scales are not independent -- channel p of every query
head sharing a kv head must get the same scale. We enforce that by averaging the
importance across those heads before searching. Miss this and the fold silently changes
the function of the layer, which is the worst kind of bug: the model still runs.

Choosing s
----------
AWQ does not derive s, it searches it, over a one-parameter family:

    s = importance ^ alpha,   alpha in [0, 1]

alpha=0 is "do nothing" (plain RTN); alpha=1 scales in direct proportion to activation
magnitude, which over-corrects. The best value is somewhere in between and differs per
layer, so we try a grid and keep whichever minimises the weighted error. `s` is then
normalised so its geometric mean is 1, which keeps the fold from drifting the norm gains
up or down over many layers.

The error we minimise is `sum over channels of E[x_j^2] * (W - Wq)_ij^2` -- the squared
weight error, weighted by how much energy that input channel actually carries. That is
the diagonal of the same Hessian GPTQ uses, which is why AWQ is so much cheaper: it needs
one number per channel, not an in_features x in_features matrix.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .calib import Calibration
from .qtensor import QuantScheme, fake_quantize, resolve_group_size

#: alpha grid for the scale search. 0.0 is included deliberately: if no scaling helps,
#: AWQ should be able to return "do nothing" rather than being forced to pick a distortion.
ALPHA_GRID = tuple(i / 20 for i in range(21))


def channel_importance(stats) -> torch.Tensor:
    """Per-input-channel activation energy, E[x_j^2], from calibration.

    Uses the Hessian diagonal when it was collected (exact), otherwise falls back to
    mean|x| squared, which is the cheap estimate AWQ can run on.
    """
    if stats.hessian is not None:
        return torch.diagonal(stats.hessian).clamp(min=1e-8).float()
    return stats.abs_mean.clamp(min=1e-8).float() ** 2


def _weighted_error(w: torch.Tensor, w_hat: torch.Tensor, imp: torch.Tensor) -> torch.Tensor:
    """sum_ij  imp_j * (w_ij - w_hat_ij)^2 -- the diagonal-Hessian proxy for output error."""
    return ((w - w_hat).float().pow(2) * imp.unsqueeze(0)).sum()


def search_scale(
    weights: list[torch.Tensor],
    imp: torch.Tensor,
    scheme: QuantScheme,
    grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[torch.Tensor, float, float]:
    """Find the per-channel scale `s` shared by a group of layers with the same input.

    Returns (s, best_alpha, improvement) where improvement is the error ratio against
    alpha=0, i.e. against plain RTN. A value of 1.0 means AWQ found nothing to gain here,
    which is a legitimate and informative outcome.
    """
    imp = imp.float()
    best_s, best_alpha, best_err, base_err = None, 0.0, None, None

    for alpha in grid:
        if alpha == 0.0:
            s = torch.ones_like(imp)
        else:
            s = imp.pow(alpha)
            # Normalise to geometric mean 1 so repeated folds do not drift the norm gains.
            s = s / s.log().mean().exp()
            s = s.clamp(min=1e-4, max=1e4)

        err = torch.zeros((), device=imp.device, dtype=torch.float32)
        for w in weights:
            g = resolve_group_size(scheme.group_size, w.shape[1])
            scaled = w.float() * s.unsqueeze(0)
            # Quantize the scaled weight, then undo the scaling: this is exactly what the
            # deployed model computes, so the error measured here is the real one.
            w_hat = fake_quantize(scaled, scheme, group_size=g) / s.unsqueeze(0)
            err = err + _weighted_error(w.float(), w_hat, imp)

        if alpha == 0.0:
            base_err = err
        if best_err is None or err < best_err:
            best_err, best_s, best_alpha = err, s, alpha

    improvement = float(base_err / best_err) if best_err and best_err > 0 else 1.0
    return best_s, best_alpha, improvement


# ---- folding the inverse scale into the producer -----------------------------------


def _fold_into_norm(norm: nn.Module, s: torch.Tensor):
    """RMSNorm output is `normalised * weight`, so dividing the gain divides the output."""
    norm.weight.data = (norm.weight.data.float() / s).to(norm.weight.dtype)


def _fold_into_linear_rows(lin: nn.Linear, s: torch.Tensor):
    """Dividing row j of a Linear divides output channel j."""
    lin.weight.data = (lin.weight.data.float() / s.unsqueeze(1)).to(lin.weight.dtype)


def _scale_columns(lin: nn.Linear, s: torch.Tensor):
    """Multiply input channel j of a Linear -- i.e. `W diag(s)`."""
    lin.weight.data = (lin.weight.data.float() * s.unsqueeze(0)).to(lin.weight.dtype)


def _gqa_share(imp: torch.Tensor, n_heads: int, n_kv_heads: int, head_dim: int
               ) -> torch.Tensor:
    """Average importance across query heads that read the same value head.

    Under GQA the scale for `wo`'s input channels is not free per channel: query heads
    h and h' sharing a kv head must agree, because the fold happens once, in `wv`.
    """
    if n_kv_heads == n_heads:
        return imp
    n_rep = n_heads // n_kv_heads
    x = imp.reshape(n_kv_heads, n_rep, head_dim).mean(dim=1)   # (n_kv, head_dim)
    return x.repeat_interleave(n_rep, dim=0).reshape(-1)       # back to (n_heads*head_dim)


# ---- the pre-pass -------------------------------------------------------------------


def apply_awq(
    model,
    calib: Calibration,
    scheme: QuantScheme,
    include_attn_out: bool = True,
    progress=None,
) -> dict:
    """Rewrite the model in place so a later RTN/GPTQ pass quantizes better.

    This is a *pre-pass*, not a quantizer -- when it returns, the model is still float,
    still computes the same function, and is simply easier to quantize. That is why AWQ
    composes with anything: run it, then quantize however you like.

    Returns a report of which sites were scaled, the alpha chosen, and the predicted
    error improvement over doing nothing.
    """
    cfg = model.cfg
    sites, skipped = [], []

    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}"

        # --- attn_norm -> wq, wk, wv ---------------------------------------------
        st = calib.get(f"{p}.attn.wq")
        if st is not None:
            imp = channel_importance(st)
            ws = [block.attn.wq.weight.data, block.attn.wk.weight.data,
                  block.attn.wv.weight.data]
            s, alpha, gain = search_scale(ws, imp, scheme)
            _fold_into_norm(block.attn_norm, s)
            for lin in (block.attn.wq, block.attn.wk, block.attn.wv):
                _scale_columns(lin, s)
            sites.append({"site": f"{p}.attn_norm->qkv", "alpha": alpha, "gain": gain})

        # --- wv -> wo (GQA-constrained) ------------------------------------------
        st = calib.get(f"{p}.attn.wo")
        if st is not None and include_attn_out:
            imp = _gqa_share(channel_importance(st), cfg.n_heads,
                             cfg.n_kv_heads, cfg.head_dim)
            s, alpha, gain = search_scale([block.attn.wo.weight.data], imp, scheme)
            # wv has n_kv_heads*head_dim outputs; take one representative per kv head.
            n_rep = cfg.n_heads // cfg.n_kv_heads
            s_v = s.reshape(cfg.n_heads, cfg.head_dim)[::n_rep].reshape(-1)
            _fold_into_linear_rows(block.attn.wv, s_v)
            _scale_columns(block.attn.wo, s)
            sites.append({"site": f"{p}.wv->wo", "alpha": alpha, "gain": gain})
        elif st is not None:
            skipped.append(f"{p}.attn.wo")

        # --- ffn_norm -> w1, w3 ---------------------------------------------------
        st = calib.get(f"{p}.ffn.w1")
        if st is not None:
            imp = channel_importance(st)
            ws = [block.ffn.w1.weight.data, block.ffn.w3.weight.data]
            s, alpha, gain = search_scale(ws, imp, scheme)
            _fold_into_norm(block.ffn_norm, s)
            _scale_columns(block.ffn.w1, s)
            _scale_columns(block.ffn.w3, s)
            sites.append({"site": f"{p}.ffn_norm->w1,w3", "alpha": alpha, "gain": gain})

        # --- w3 -> w2 -------------------------------------------------------------
        st = calib.get(f"{p}.ffn.w2")
        if st is not None:
            imp = channel_importance(st)
            s, alpha, gain = search_scale([block.ffn.w2.weight.data], imp, scheme)
            _fold_into_linear_rows(block.ffn.w3, s)
            _scale_columns(block.ffn.w2, s)
            sites.append({"site": f"{p}.w3->w2", "alpha": alpha, "gain": gain})

        if progress:
            progress(i + 1, len(model.blocks))

    gains = [x["gain"] for x in sites] or [1.0]
    return {
        "sites": sites,
        "skipped": skipped,
        "n_sites": len(sites),
        "mean_gain": sum(gains) / len(gains),
        "mean_alpha": sum(x["alpha"] for x in sites) / max(1, len(sites)),
    }


def rescale_hessians(calib: Calibration, model, scheme: QuantScheme):
    """After AWQ, a layer's inputs are `x / s`, so its calibrated Hessian is stale.

    H' = diag(1/s) H diag(1/s). Cheap to apply and necessary if GPTQ is going to run on
    top of AWQ -- otherwise GPTQ compensates against curvature the layer no longer has.
    Requires the scales, so it is only correct when called immediately after `apply_awq`
    with the same model.
    """
    raise NotImplementedError(
        "AWQ+GPTQ composition needs the per-site scales threaded through; run them "
        "separately for now")
