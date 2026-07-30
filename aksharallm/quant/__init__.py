"""Quantization, from scratch.

Making a trained model smaller by storing its weights in 8 or 4 bits instead of 16.

The whole field rests on one observation: a weight matrix is not uniformly important.
Within any small group of weights the values are tightly clustered, so if you store one
scale factor per group you can represent the group with very few bits per weight and
lose almost nothing. Everything else -- RTN, GPTQ, AWQ, QAT -- is a different answer to
the question *how do you choose the rounding so the model notices least?*

Modules:
    qtensor   the representation: group-wise scales, packing 4-bit values two-per-byte
    qlinear   QuantLinear, a drop-in for nn.Linear that holds packed weights
    rtn       round-to-nearest, the baseline
    convert   swap a model's Linear layers for QuantLinear, save/load quantized checkpoints
    bench     measure what it actually bought: bytes, perplexity, tokens/sec
"""

from .qtensor import (
    QuantScheme,
    dequantize,
    pack4,
    quantize_group,
    unpack4,
)
from .qlinear import QuantLinear
from .rtn import quantize_linear_rtn

__all__ = [
    "QuantScheme",
    "QuantLinear",
    "dequantize",
    "pack4",
    "quantize_group",
    "quantize_linear_rtn",
    "unpack4",
]
