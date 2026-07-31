"""Quantization, from scratch.

Making a trained model smaller by storing its weights in 8 or 4 bits instead of 16.

The whole field rests on one observation: a weight matrix is not uniformly important.
Within any small group of weights the values are tightly clustered, so if you store one
scale factor per group you can represent the group with very few bits per weight and
lose almost nothing. Everything else -- RTN, GPTQ, AWQ, QAT -- is a different answer to
the question *how do you choose the rounding so the model notices least?*

There are two grids to choose from, not one: evenly spaced integer levels (`dtype='int'`)
or the normal-quantile levels of NF4 (`dtype='nf4'`), which is what QLoRA fine-tunes on
top of. Both are 4 bits and both go through the same storage, so every method here works
with either.

Modules:
    qtensor   the representation: group-wise scales, NF4 levels, double quantization,
              packing 4-bit values two-per-byte
    qlinear   QuantLinear, a drop-in for nn.Linear that holds packed weights
    rtn       round-to-nearest, the baseline
    convert   swap a model's Linear layers for QuantLinear, save/load quantized checkpoints
    bench     measure what it actually bought: bytes, perplexity, tokens/sec
"""

from .qtensor import (
    DQ_BLOCK,
    NF4_LEVELS,
    QuantScheme,
    compress_scales,
    decompress_scales,
    dequantize,
    pack4,
    quantize_group,
    unpack4,
)
from .qlinear import QuantLinear
from .rtn import quantize_linear_rtn

__all__ = [
    "DQ_BLOCK",
    "NF4_LEVELS",
    "QuantScheme",
    "QuantLinear",
    "compress_scales",
    "decompress_scales",
    "dequantize",
    "pack4",
    "quantize_group",
    "quantize_linear_rtn",
    "unpack4",
]
