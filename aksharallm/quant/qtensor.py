"""The representation: how a bf16 weight matrix becomes bytes.

Read this file first. Everything else in `quant/` is a strategy for *choosing* the
numbers; this is where they are actually stored.

The idea
--------
A weight `w` is stored as a small integer `q` plus a per-group scale `s` (and, if the
group isn't symmetric about zero, a zero-point `z`):

    w  ~=  s * (q - z)

`q` needs only `bits` bits. `s` and `z` are stored once per *group* of consecutive
weights, so their cost is amortised: at group_size=64 in 4 bits, the true cost per weight
is 4 + 16/64 + 4/64 ~= 4.3 bits, not 4. That overhead is the price of not having one
scale for the entire matrix, and it is worth every bit -- a single scale per matrix has
to stretch to cover the largest outlier in 2 million weights, which crushes everything
else towards zero.

Which axis do we group along?
-----------------------------
`nn.Linear` holds weight as (out_features, in_features) and computes `x @ W.T`, so each
output element is a dot product along `in_features`. We group along **in_features** --
i.e. along the reduction axis, within one output row. That is the axis the matmul sums
over, so each group's scale can be factored straight out of a partial dot product, which
is exactly what makes the fused kernel in `kernels.py` possible later.

    W  (out_features, in_features)
        row 0:  [ g0 | g1 | g2 | ... ]   each g = `group_size` weights, one scale each
        row 1:  [ g0 | g1 | g2 | ... ]
        ...

Symmetric vs asymmetric
-----------------------
Symmetric assumes the group is centred on zero and stores only a scale; the integer range
is [-2^(b-1)+1, 2^(b-1)-1] (we give up the extra negative value so that the range is
actually symmetric and `q = 0` means `w = 0` exactly).

Asymmetric fits the group's true [min, max] using a zero-point, giving the full 2^b
levels. It costs one more small tensor and is meaningfully better at 4 bits, where you
only have 16 levels to spend and wasting half of them on a range the weights never visit
is expensive. Defaults: asymmetric at 4 bits, symmetric at 8.

Packing
-------
At 8 bits a value is a byte and there is nothing to do. At 4 bits, two values share a
byte: the even index goes in the low nibble, the odd index in the high nibble. Everything
in this file works on the *reduction* axis last, so "even/odd index" means adjacent
weights in a dot product -- which is also the order the kernel wants them in.

Two grids, not one: int4 and NF4
--------------------------------
Everything above assumes the 16 codes are *evenly spaced* -- that is what "s * (q - z)"
means. That is the right assumption if you know nothing about the weights, and the wrong
one if you know they are roughly Gaussian, which trained weights are.

NF4 ("normal float") spends its 16 levels where a normal distribution actually puts its
mass: closely spaced near zero, widely spaced out in the tails. The levels are fixed
constants, so a group stores only an absmax scale and no zero-point at all. See
`NF4_LEVELS` below, which is derived rather than pasted in.

    int4 asym:  levels evenly spaced across the group's [min, max]
    NF4:        levels at the quantiles of N(0,1), scaled by the group's absmax

`QuantScheme.dtype` selects between them, and every method in this package (RTN, GPTQ,
AWQ, QAT) works with either, because they all go through `quantize_group`.

Double quantization
-------------------
At group_size=64 the scales are one fp16 per 64 weights = 0.25 bits/weight, and with a
zero-point 0.375. Small, but not nothing: on a 4-bit model that is ~9% of the total.
`double_quant` quantizes the *scales* themselves -- int8 codes with one fp32 scale and
one fp32 mean per block of 256 scales -- taking that 0.25 down to 0.129. See
`compress_scales`. It is a real saving and a small one, and the docs say so.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Q_MIN/Q_MAX for the symmetric case give up one negative level (e.g. -127..127 rather
# than -128..127) so the grid is genuinely symmetric and zero is representable.
_SYM_RANGE = {8: (-127, 127), 4: (-7, 7)}
_ASYM_RANGE = {8: (0, 255), 4: (0, 15)}

#: Scales per block for double quantization. 256 is what the QLoRA paper uses; the block
#: only has to be big enough that its own fp32 scale+mean amortise away (2 values per 256
#: is 1/16 bit per weight at group_size=64).
DQ_BLOCK = 256


def _norm_ppf(p: torch.Tensor) -> torch.Tensor:
    """Inverse CDF of the standard normal, via erfinv. Avoids a scipy dependency."""
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def _derive_nf4_levels() -> torch.Tensor:
    """Build the 16 NF4 levels from the normal distribution they are meant to fit.

    The recipe, which is worth understanding rather than copying:

      * Take evenly spaced *probabilities* and map them through the normal inverse CDF.
        Even spacing in probability means even spacing in *mass*, so every level ends up
        representing about the same number of weights. That is the whole idea.
      * Stop at `offset` = 0.9677 rather than 1.0, because the inverse CDF goes to
        infinity at 1 and the top level has to be a finite number.
      * Use 7 negative levels and 8 positive ones. 16 levels cannot be split evenly
        around a zero that must itself be representable -- one side gets the spare. The
        asymmetry is why the negative levels are slightly coarser than the positive ones.
      * Normalise so the extremes are exactly -1 and +1, because the stored scale is the
        group's absmax and has to map onto them.

    Zero being an exact level matters more than it looks: padding, masked positions and
    genuinely dead weights all quantize to exactly 0.0 instead of to some nearby level.
    """
    offset = 0.9677083
    neg = -_norm_ppf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    pos = _norm_ppf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    v = torch.cat([neg, torch.zeros(1, dtype=torch.float64), pos]).sort().values
    return (v / v.abs().max()).to(torch.float32)


#: The 16 NF4 levels, ascending. Index into this with a 4-bit code to dequantize.
NF4_LEVELS = _derive_nf4_levels()

#: Midpoints between consecutive levels: quantizing is `bucketize` against these, which is
#: exactly "round to the nearest level" for a non-uniform grid.
NF4_BOUNDARIES = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2


@dataclass(frozen=True)
class QuantScheme:
    """How to quantize. One of these describes an entire checkpoint.

    bits:        4 or 8.
    group_size:  weights per scale along in_features. -1 means one scale per output row
                 ("per-channel"). Must divide in_features -- see `resolve_group_size`.
    sym:         symmetric (scale only) vs asymmetric (scale + zero-point). Ignored when
                 dtype='nf4', whose grid is fixed and neither.
    dtype:       'int' (evenly spaced levels) or 'nf4' (levels at the normal quantiles).
    double_quant: also quantize the scales. See `compress_scales`.
    method:      which algorithm picked the values -- 'rtn', 'gptq', 'awq', 'qat'.
                 Recorded so a loaded checkpoint can say how it was made.
    """

    bits: int = 4
    group_size: int = 64
    sym: bool = False
    method: str = "rtn"
    dtype: str = "int"
    double_quant: bool = False

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"group_size must be -1 or positive, got {self.group_size}")
        if self.dtype not in ("int", "nf4"):
            raise ValueError(f"dtype must be 'int' or 'nf4', got {self.dtype!r}")
        if self.dtype == "nf4":
            if self.bits != 4:
                raise ValueError("nf4 is a 4-bit type; bits must be 4")
            if self.sym:
                # Not a stylistic objection: the NF4 grid is neither symmetric nor
                # zero-point-based, so `sym` has no meaning here and silently accepting it
                # would put a flag in the checkpoint metadata that describes nothing.
                raise ValueError("nf4 has a fixed grid; sym does not apply")

    @property
    def is_nf4(self) -> bool:
        return self.dtype == "nf4"

    @property
    def qmin(self) -> int:
        if self.is_nf4:
            return 0  # codes are indices into NF4_LEVELS
        return (_SYM_RANGE if self.sym else _ASYM_RANGE)[self.bits][0]

    @property
    def qmax(self) -> int:
        if self.is_nf4:
            return len(NF4_LEVELS) - 1
        return (_SYM_RANGE if self.sym else _ASYM_RANGE)[self.bits][1]

    @property
    def packed(self) -> bool:
        """True when two values share a byte."""
        return self.bits == 4

    @property
    def has_zeros(self) -> bool:
        """True when the scheme stores a zero-point tensor alongside the scales."""
        return not self.sym and not self.is_nf4

    def bits_per_weight(self, in_features: int) -> float:
        """The honest number, including the scales. Quote this, not `bits`."""
        g = in_features if self.group_size == -1 else self.group_size
        # fp16 scale, or int8 + (fp32 scale + fp32 mean per DQ_BLOCK scales)
        scale_bits = 8.0 + 64.0 / DQ_BLOCK if self.double_quant else 16.0
        per_group = scale_bits + (8.0 if self.has_zeros else 0.0)
        return self.bits + per_group / g

    def label(self) -> str:
        g = "chan" if self.group_size == -1 else f"g{self.group_size}"
        kind = "nf4" if self.is_nf4 else f"int{self.bits}"
        parts = [self.method, kind, g]
        if not self.is_nf4:
            parts.append("sym" if self.sym else "asym")
        if self.double_quant:
            parts.append("dq")
        return "-".join(parts)

    def as_dict(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "method": self.method,
            "dtype": self.dtype,
            "double_quant": self.double_quant,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantScheme":
        return cls(
            bits=int(d["bits"]),
            group_size=int(d["group_size"]),
            sym=bool(d["sym"]),
            method=str(d.get("method", "rtn")),
            # Absent in checkpoints written before NF4 existed, and their default is the
            # only thing they could have been.
            dtype=str(d.get("dtype", "int")),
            double_quant=bool(d.get("double_quant", False)),
        )


def resolve_group_size(group_size: int, in_features: int) -> int:
    """Pick a group size that actually divides this layer's reduction axis.

    This is not a hypothetical. Our own 300M config has `d_ff = 2752`, because d_ff is
    `8/3 * d_model` rounded up to a multiple of 64 -- and 2752 is *not* divisible by 128,
    the group size most quantization papers use. The SwiGLU down-projection `w2` reduces
    over d_ff, so a blanket group_size=128 would fail on exactly one layer per block.

    Rather than refuse, fall back to the largest power-of-two divisor that is <= the
    requested size. The caller is expected to report when this happened, so a layer
    quietly getting coarser groups than you asked for is visible rather than silent.
    """
    if group_size == -1 or in_features % group_size == 0:
        return group_size
    g = group_size
    while g > 1 and in_features % g != 0:
        g //= 2
    return g if g > 1 else -1


def quantize_group(
    w: torch.Tensor, scheme: QuantScheme, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize a weight matrix group-wise along its last axis.

    w: (out_features, in_features) float. Returns integer codes `q` of the same shape
    (not yet packed), plus `scales` and `zeros` of shape (out_features, n_groups).

    Kept separate from packing so that fake-quant (QAT, and the error term GPTQ needs)
    can use the exact same arithmetic without ever touching a nibble.
    """
    out_f, in_f = w.shape
    g = scheme.group_size if group_size is None else group_size
    g = in_f if g == -1 else g
    if in_f % g != 0:
        raise ValueError(
            f"group_size {g} does not divide in_features {in_f}; "
            f"call resolve_group_size() first"
        )
    n_groups = in_f // g

    wf = w.float().reshape(out_f, n_groups, g)

    if scheme.is_nf4:
        # The whole scheme in three lines: normalise the group into [-1, 1] by its absmax,
        # then snap each value to the nearest of the 16 fixed levels. No zero-point --
        # the grid already straddles zero, and 0.0 is one of its levels exactly.
        scales = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = (wf / scales).clamp(-1.0, 1.0)
        bounds = NF4_BOUNDARIES.to(device=wf.device, dtype=wf.dtype)
        q = torch.bucketize(normed, bounds)
        zeros = None
    elif scheme.sym:
        # One scale, grid centred on zero. The largest magnitude in the group defines it.
        amax = wf.abs().amax(dim=-1, keepdim=True)
        scales = (amax / scheme.qmax).clamp(min=1e-8)
        zeros = None
        q = torch.round(wf / scales)
    else:
        # Fit the group's [min, max] -- but *always stretched to include zero*.
        #
        # That last part is not an optimisation, it is what makes the scheme valid. The
        # zero-point `z` is the integer code standing for 0.0, and it has to be a code
        # that exists, i.e. inside [qmin, qmax]. Extending the range so wmin <= 0 <= wmax
        # guarantees it: -wmin/s then lands in [0, qmax-qmin] by construction.
        #
        # Fit the raw [min, max] instead and a group that never crosses zero -- weights in
        # [1, 2], say -- wants z = -15, which has to be clamped back to 0, and every value
        # in the group then quantizes to the same saturated code. The group collapses to a
        # constant. It does not raise, it does not produce NaN, it just quietly destroys
        # that group; only a test comparing against the symmetric scheme catches it.
        wmin = wf.amin(dim=-1, keepdim=True).clamp(max=0.0)
        wmax = wf.amax(dim=-1, keepdim=True).clamp(min=0.0)
        # An all-zero group has zero range even after stretching; keep the scale finite.
        span = (wmax - wmin).clamp(min=1e-8)
        scales = span / (scheme.qmax - scheme.qmin)
        zeros = torch.round(-wmin / scales) + scheme.qmin
        q = torch.round(wf / scales) + zeros

    q = q.clamp(scheme.qmin, scheme.qmax).reshape(out_f, in_f)
    scales = scales.reshape(out_f, n_groups)
    if zeros is not None:
        zeros = zeros.reshape(out_f, n_groups)
    return q, scales, zeros


def dequantize(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    group_size: int,
    dtype: torch.dtype = torch.float32,
    nf4: bool = False,
) -> torch.Tensor:
    """The inverse of `quantize_group`: integer codes back to floats.

    q: (out_features, in_features) integer codes, already unpacked.
    nf4: the codes are indices into NF4_LEVELS rather than points on an even grid.
    """
    out_f, in_f = q.shape
    g = in_f if group_size == -1 else group_size
    n_groups = in_f // g
    s = scales.reshape(out_f, n_groups, 1).float()
    if nf4:
        levels = NF4_LEVELS.to(q.device)
        qf = levels[q.long().reshape(out_f, n_groups, g)]
        return (qf * s).reshape(out_f, in_f).to(dtype)
    qf = q.float().reshape(out_f, n_groups, g)
    if zeros is not None:
        qf = qf - zeros.reshape(out_f, n_groups, 1).float()
    return (qf * s).reshape(out_f, in_f).to(dtype)


def fake_quantize(w: torch.Tensor, scheme: QuantScheme, group_size: int | None = None,
                  scale_dtype: torch.dtype | None = None):
    """Quantize and immediately dequantize: the *error* without the storage win.

    This is the workhorse of the whole package. It is what QAT puts in the forward pass,
    what GPTQ uses to compute the error it then compensates for, and what the tests use
    to check that packing round-trips. `w - fake_quantize(w)` is precisely the damage
    quantization does to a layer.

    `scale_dtype` rounds the scales to the precision they will actually be *stored* in
    before using them. QuantLinear keeps scales in fp16, so a QAT run that computes them
    in fp32 is training against numerics a shade better than the ones it will ship with,
    and the model shifts slightly the moment it is converted. Passing torch.float16 here
    closes that gap.
    """
    g = scheme.group_size if group_size is None else group_size
    q, scales, zeros = quantize_group(w, scheme, group_size=g)
    # Order matters, and it is the order QuantLinear stores in: the int8 scale codes are
    # computed from the *fp32* scales, and only the decompressed result is narrowed to
    # fp16. Rounding to fp16 first and compressing that would be a third set of numerics
    # that nothing ever runs -- and would make this function disagree with the layer it
    # exists to simulate.
    if scheme.double_quant:
        # Same reasoning as `scale_dtype`: a QAT run whose forward uses exact scales is
        # training against numerics the shipped model will not have.
        scales = decompress_scales(*compress_scales(scales), scales.shape).to(scales.dtype)
    if scale_dtype is not None:
        scales = scales.to(scale_dtype).to(scales.dtype)
    gg = w.shape[1] if g == -1 else g
    return dequantize(q, scales, zeros, gg, dtype=w.dtype, nf4=scheme.is_nf4)


# ---- double quantization ----------------------------------------------------------------


def compress_scales(scales: torch.Tensor, block: int = DQ_BLOCK):
    """Quantize the scales themselves to int8: the "double" in double quantization.

    scales: (out_features, n_groups) float. Returns (codes, absmax, mean), where `codes`
    is a *flat* int8 tensor padded up to a multiple of `block`, and `absmax`/`mean` are
    one fp32 each per block.

    Why subtract a mean first
    -------------------------
    Scales are all positive -- they are absmax or span values. A symmetric int8 grid
    centred on zero would therefore waste its entire negative half, throwing away one of
    the eight bits before it starts. Subtracting the block mean re-centres the values on
    zero so both halves are used. That single subtraction is worth about a bit of
    precision, which is the difference between double quantization being harmless and
    being visible.

    The block is flat across the whole matrix rather than per row: a block boundary that
    followed rows would need padding per row, and there is nothing special about a row
    boundary as far as scale magnitudes are concerned.
    """
    flat = scales.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        # Pad with the last value rather than zero: a block of real scales followed by a
        # run of zeros has a mean and an absmax pulled towards the padding, which costs
        # precision on the real values for no reason.
        flat = torch.cat([flat, flat[-1:].expand(pad)])
    blocks = flat.reshape(-1, block)
    mean = blocks.mean(dim=1, keepdim=True)
    centred = blocks - mean
    absmax = centred.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    codes = torch.round(centred / absmax * 127.0).clamp(-127, 127).to(torch.int8)
    return codes.reshape(-1), (absmax / 127.0).reshape(-1), mean.reshape(-1)


def decompress_scales(codes: torch.Tensor, absmax: torch.Tensor, mean: torch.Tensor,
                      shape, block: int = DQ_BLOCK) -> torch.Tensor:
    """Inverse of `compress_scales`, trimming the padding back off."""
    blocks = codes.reshape(-1, block).float()
    out = blocks * absmax.reshape(-1, 1) + mean.reshape(-1, 1)
    n = int(torch.Size(shape).numel())
    return out.reshape(-1)[:n].reshape(shape)


# ---- packing --------------------------------------------------------------------------


def pack4(q: torch.Tensor, sym: bool = False) -> torch.Tensor:
    """Pack 4-bit codes two per byte along the last axis.

    q holds values in [0, 15] (asymmetric) or [-7, 7] (symmetric). Symmetric codes are
    shifted by +8 into [1, 15] before packing so the nibble is unsigned; `unpack4`
    undoes it. The last axis must be even.

    The shift is decided by `sym`, never by inspecting the data -- a symmetric group that
    happens to contain no negative value must still be shifted, or it will not survive
    the round trip through `unpack4(..., sym=True)`.
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"cannot pack an odd number of 4-bit values: {q.shape[-1]}")
    qi = q.to(torch.int16)
    if sym:
        qi = qi + 8  # symmetric range [-7,7] -> [1,15]
    if qi.min() < 0 or qi.max() > 15:
        raise ValueError(f"4-bit codes out of range: [{int(qi.min())}, {int(qi.max())}]")
    lo = qi[..., 0::2]
    hi = qi[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack4(packed: torch.Tensor, sym: bool) -> torch.Tensor:
    """Inverse of `pack4`. Returns int8 codes with the last axis doubled."""
    p = packed.to(torch.int16)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    if sym:
        out = out - 8
    return out.to(torch.int8)


def pack(q: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """Store integer codes as bytes according to the scheme.

    NF4 codes are already unsigned indices in [0, 15], so they pack like the asymmetric
    case with no shift -- `scheme.sym` is False for NF4 by construction (see
    `QuantScheme.__post_init__`), which is what makes this fall out correctly.
    """
    if scheme.packed:
        return pack4(q, sym=scheme.sym)
    if scheme.sym:
        return q.to(torch.int8)
    return q.to(torch.uint8)


def unpack(packed: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """Bytes back to integer codes."""
    if scheme.packed:
        return unpack4(packed, scheme.sym)
    return packed.to(torch.int16)


def packed_shape(out_features: int, in_features: int, scheme: QuantScheme) -> tuple[int, int]:
    return (out_features, in_features // 2 if scheme.packed else in_features)
