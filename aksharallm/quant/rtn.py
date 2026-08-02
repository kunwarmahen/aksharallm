"""RTN -- round-to-nearest. The baseline every other method is measured against.

The entire algorithm: for each group, find the scale that spans it, divide, round, clamp.
No calibration data, no search, no forward passes. Quantizing our 300M model this way
takes seconds and is embarrassingly parallel.

It is also, at 8 bits, essentially free in quality -- 256 levels per group of 64 weights
is more resolution than a trained weight distribution needs. The reason the field did not
stop here is 4 bits: 16 levels is coarse enough that *which* value you round to starts to
matter, and RTN answers that question with "the nearest one", which is only the right
answer if you assume every weight matters equally. GPTQ and AWQ are two different ways of
saying that it doesn't.

So the RTN-vs-GPTQ gap at int4 is the single most informative number this package
produces, and it is why RTN is built first rather than skipped.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .qlinear import QuantLinear
from .qtensor import QuantScheme, pack, quantize_group, resolve_group_size


def quantize_linear_rtn(lin: nn.Linear, scheme: QuantScheme) -> QuantLinear:
    """Quantize one Linear layer, round-to-nearest."""
    return QuantLinear.from_linear(lin, scheme)


def rtn_weight(
    w: torch.Tensor, scheme: QuantScheme
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize a raw weight matrix. Returns (packed, scales, zeros).

    Exposed separately from `quantize_linear_rtn` because GPTQ and AWQ both end by
    calling exactly this on a modified weight -- AWQ on a rescaled one, GPTQ on one
    column block at a time.
    """
    g = resolve_group_size(scheme.group_size, w.shape[1])
    codes, scales, zeros = quantize_group(w, scheme, group_size=g)
    return pack(codes, scheme), scales, zeros
