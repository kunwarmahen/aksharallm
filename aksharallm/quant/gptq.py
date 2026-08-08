"""GPTQ: quantize one column at a time, and make the rest of the layer absorb the error.

The idea
--------
RTN quantizes every weight independently and accepts whatever error results. But a layer
is not a bag of independent weights -- it is a linear map, and what we actually care
about is that `x @ W.T` barely changes, not that each individual weight barely changes.
Those are very different objectives.

So: quantize column j, measure the error you just made, and *adjust the columns you have
not quantized yet* to cancel that error's effect on the output. The later columns are
still full-precision at that moment, so they are free to move. By the time you reach the
last column it has absorbed the accumulated debt of every column before it.

The weighting comes from the Hessian
------------------------------------
How much should column k move to compensate for an error in column j? That depends on how
correlated their inputs are, which is exactly what `H = E[x x^T]` records. The optimal
update falls out of minimising ||x W^T - x Wq^T||^2, and is

    W[:, j+1:]  -=  (err_j / [H^-1]_jj) * [H^-1]_j,j+1:

with `err_j` the quantization error of column j. Everything else in this file is the
numerics needed to get `H^-1` safely and to apply that update in cache-friendly blocks.

Why a Cholesky factor rather than H^-1 itself
---------------------------------------------
The update only ever needs the *upper triangle* of H^-1 -- column j only talks to columns
after it. Taking the Cholesky factor of H^-1 once, up front, gives exactly that in a
numerically stable form, and turns what would be a fresh linear solve per column into a
row lookup.

What this implementation does not do
------------------------------------
The published method runs *sequentially*: each block is calibrated on activations that
have already passed through the quantized blocks before it, so the error compensation
accounts for upstream damage too. Here the statistics are collected once, from the float
model. That is simpler, roughly one pass instead of one per block, and gives up a little
quality at 4 bits. Recorded here rather than glossed over, because it is the honest gap
between this file and the paper.

Read with: docs/11-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .calib import Calibration, damped_hessian
from .qlinear import QuantLinear
from .qtensor import (
    NF4_BOUNDARIES,
    NF4_LEVELS,
    QuantScheme,
    pack,
    quantize_group,
    resolve_group_size,
)

#: Columns processed per block. The inner loop is sequential and column-at-a-time; the
#: block exists so the *trailing* update (the expensive one) is a single big matmul over
#: everything after the block rather than one skinny matmul per column.
BLOCK = 128


def _group_params(w_block: torch.Tensor, scheme: QuantScheme):
    """Scale and zero-point for one group, from the weights as they stand *now*.

    Deliberately recomputed mid-algorithm rather than fixed up front: by the time GPTQ
    reaches a group, the earlier columns' error has already been pushed into it, so the
    group's range is not what it was in the original weight matrix. Using stale scales
    here is a subtle way to lose most of GPTQ's benefit.
    """
    _, scales, zeros = quantize_group(w_block, scheme, group_size=w_block.shape[1])
    return scales, zeros  # each (out, 1)


def _quantize_column(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None,
                     scheme: QuantScheme):
    """One column -> integer codes and the dequantized value. w, scale: (out,).

    Both grids are handled here, and this is the only place GPTQ needs to know which one
    it is on: everything above works in terms of "quantize this column, tell me the
    error", which is grid-agnostic.
    """
    if scheme.is_nf4:
        bounds = NF4_BOUNDARIES.to(device=w.device, dtype=w.dtype)
        q = torch.bucketize((w / scale).clamp(-1.0, 1.0), bounds)
        return q.to(w.dtype), NF4_LEVELS.to(w.device)[q.long()] * scale
    if zero is None:
        q = torch.round(w / scale)
    else:
        q = torch.round(w / scale) + zero
    q = q.clamp(scheme.qmin, scheme.qmax)
    deq = (q - zero) * scale if zero is not None else q * scale
    return q, deq


@torch.no_grad()
def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    scheme: QuantScheme,
    damp: float = 0.01,
    block: int = BLOCK,
):
    """Quantize one weight matrix with Hessian-guided error compensation.

    weight:  (out_features, in_features), float
    hessian: (in_features, in_features), the calibrated E[x x^T]

    Returns (packed_qweight, scales, zeros) in exactly the layout RTN produces, so the
    storage format and everything downstream is shared.
    """
    dev = weight.device
    W = weight.detach().clone().float()
    out_f, in_f = W.shape
    g = resolve_group_size(scheme.group_size, in_f)
    gsize = in_f if g == -1 else g
    n_groups = in_f // gsize

    H = damped_hessian(hessian.to(dev, torch.float32), damp)
    # Columns whose input is always zero carry no information; zero them so they cannot
    # inject noise through the inverse, and let them quantize to whatever RTN says.
    dead = torch.diagonal(H) == 0
    if dead.any():
        W[:, dead] = 0

    # H^-1, then its upper Cholesky factor. `cholesky_inverse` is the stable route:
    # invert the factor, not the matrix.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    scales = torch.zeros(out_f, n_groups, device=dev, dtype=torch.float32)
    zeros = (torch.zeros(out_f, n_groups, device=dev, dtype=torch.float32)
             if scheme.has_zeros else None)
    # One block == one group. Aligning them is what keeps the scale fitting honest: at
    # the top of a block, `W[:, i1:i2]` has already absorbed the error of every preceding
    # block, so the scale is fitted to the weights this group will actually be storing.
    # Letting a group straddle a block boundary would fit half of it to stale values.
    block = min(gsize, in_f)

    for i1 in range(0, in_f, block):
        i2 = min(i1 + block, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        s, z = _group_params(W1, scheme)
        cur_scale = s[:, 0]
        cur_zero = None if z is None else z[:, 0]
        scales[:, i1 // gsize] = cur_scale
        if zeros is not None:
            zeros[:, i1 // gsize] = cur_zero

        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q, deq = _quantize_column(w, cur_scale, cur_zero, scheme)
            Q1[:, i] = q
            err = (w - deq) / d
            # Push the error into the columns of this block that are still untouched.
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err

        Q[:, i1:i2] = Q1
        # ...and into every column after this block, in one matmul.
        if i2 < in_f:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    packed = pack(Q.to(torch.int32), scheme)
    return packed, scales.to(torch.float16), (None if zeros is None else zeros)


def make_gptq_quantizer(calib: Calibration, damp: float = 0.01,
                        progress=None, free: bool = True):
    """A `Quantizer` for `convert.quantize_model` that uses GPTQ where it can.

    Falls back to RTN for any layer with no calibration statistics, and says so, rather
    than silently producing a mixed-method model that claims to be GPTQ.
    """
    fell_back: list[str] = []

    def quantizer(name: str, lin: nn.Linear, scheme: QuantScheme) -> QuantLinear:
        stats = calib.get(name)
        if stats is None or stats.hessian is None:
            fell_back.append(name)
            return QuantLinear.from_linear(lin, scheme)
        packed, scales, zeros = gptq_quantize_weight(
            lin.weight.data, stats.hessian, scheme, damp=damp)
        if free:
            calib.free(name)  # a 2752x2752 fp32 Hessian is 30 MB; do not hold 24 of them
        q = QuantLinear.from_linear(lin, scheme, qweight=packed, scales=scales, zeros=zeros)
        if progress:
            progress(name)
        return q

    quantizer.fell_back = fell_back  # type: ignore[attr-defined]
    return quantizer
