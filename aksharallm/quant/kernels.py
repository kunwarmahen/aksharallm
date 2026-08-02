"""A fused dequantize-and-matmul kernel, in Triton.

Why this file exists
--------------------
The torch backend stores 4-bit weights and then, on every forward pass, rebuilds the full
bf16 weight matrix and hands it to cuBLAS. That is correct and it saves memory at rest,
but it is *slower* than not quantizing at all -- measured at 0.35x on our 300M model --
because it does everything the bf16 matmul does, plus an unpack, plus writing and
re-reading a temporary the size of the original weight.

Fusing fixes that specific waste: load the packed bytes into registers, unpack and scale
them *there*, and accumulate straight into the output, so the 4-bit form is the only
thing that ever crosses the memory bus. On one 1024x1024 layer at batch 1 that takes the
matmul from 206 us (dequantize + cuBLAS) to 25 us. End to end it takes decode from 19.3
to 31.7 tok/s -- a real 1.6x.

And then it stops, well short of bf16's 55 tok/s. Read on, because the reason is the most
useful thing in this file.

What actually limits decode here (measured, and not what the textbook says)
--------------------------------------------------------------------------
The standard story is that single-token decoding is memory-bandwidth bound: the GPU reads
every weight and does two flops with each, so time is set by bytes read and 4x fewer bytes
should be ~4x faster. That story is *false at this scale*, and it is worth knowing why.

Our 300M model in bf16 holds ~525 MB of Linear weights. A 3090 has ~936 GB/s. If decode
were bandwidth bound a step would take **0.56 ms**. Measured: **17.5 ms**. Thirty-one
times slower than the bandwidth limit.

The giveaway is what happens when the batch grows. Bandwidth-bound and overhead-bound both
predict flat ms/step as batch grows, so that alone proves nothing -- but the *absolute
level* does:

    bf16   B=1  17.5 ms/step     B=8  18.3 ms/step     B=32  17.9 ms/step
    int4   B=1  30.7 ms/step     B=8  31.2 ms/step     B=32  42.8 ms/step

Thirty-two tokens for the price of one means the GPU is not remotely saturated at B=1. The
time is going into per-operation dispatch: a step is ~170 Linear calls plus norms,
attention and sampling, each an individual eager-mode kernel launch, and 17.5 ms spread
over several hundred launches is about 25 us apiece. That is launch and Python overhead,
not memory traffic.

So at this size the honest summary is:

  * quantization's win is **memory footprint**, which is real and immediate (2.8x);
  * the fused kernel's win is **real but capped** (1.6x) because it optimises the part
    that was not the bottleneck;
  * beating bf16 at batch 1 needs the launch overhead removed first -- CUDA graphs or
    `torch.compile(mode="reduce-overhead")` -- and only *then* does the bandwidth
    argument start to bite;
  * the bandwidth story becomes true as the model grows. At 7B the weights are 20x
    larger while the launch count is only ~3x, so the fixed overhead stops dominating.

None of which is a reason not to have written the kernel. It is a reason to measure before
believing a performance story, including this one.

Shape strategy
--------------
This is a GEMV-shaped problem, not a GEMM one. During decode M (the number of rows of
activations) is 1, and a tensor-core `tl.dot` wants at least 16 -- padding a 1-row matmul
to 16 wastes fifteen sixteenths of the tensor core. So the kernel is written as an
explicit broadcast-multiply-and-reduce over K, which for small M is both simpler and
faster than pretending this is a matrix multiply.

That is also why `AUTO_MAX_ROWS` exists. Prefill (hundreds of rows at once) is genuinely
compute bound and cuBLAS on a dequantized bf16 matrix wins there. The "auto" backend
sends decode to this kernel and prefill to torch, which is the right answer for both.

The unpacking trick
-------------------
Weights are packed two per byte, low nibble first, so byte i holds k = 2i and k = 2i+1.
Recovering that ordering inside the kernel would normally need an interleave; `tl.join`
does it for free by stacking lo and hi along a new trailing axis and reshaping, which
lands element (n, 2i) = lo and (n, 2i+1) = hi -- exactly the packing order.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import torch

#: Above this many rows, `backend="auto"` prefers dequantize + cuBLAS. Decode is 1 row;
#: prefill is the whole prompt at once and belongs on the other path.
AUTO_MAX_ROWS = 16

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    _HAVE_TRITON = False


def available(device: torch.device | None = None) -> bool:
    """True when the fused path can run here."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True


if _HAVE_TRITON:

    @triton.jit
    def _qgemv_kernel(
        X, QW, SC, ZP, LUT, Y,
        M, N, K,
        stride_xm, stride_xk,
        stride_qn, stride_qk,
        stride_sn, stride_sg,
        stride_ym, stride_yn,
        GROUP_SIZE: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        GROUP_TILE: tl.constexpr,
        HAS_ZERO: tl.constexpr,
        NF4: tl.constexpr,
        BITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """y[m, n] = sum_k x[m, k] * dequant(qw[n, k])

        One program computes a BLOCK_M x BLOCK_N tile of the output, streaming over K.
        The weight tile is unpacked and scaled in registers and never written to memory.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        mask_m = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Per-channel means one scale for the whole row, so it is loop-invariant: hoist
        # it out entirely rather than re-reading it once per K tile.
        if PER_CHANNEL:
            sc_row = tl.load(SC + offs_n * stride_sn, mask=mask_n, other=0.0).to(tl.float32)
            if HAS_ZERO:
                zp_row = tl.load(ZP + offs_n * stride_sn, mask=mask_n, other=0).to(tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)

            if BITS == 4:
                # Two weights per byte: load half as many bytes as we need values.
                offs_b = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
                mask_b = offs_b < (K // 2)
                b = tl.load(
                    QW + offs_n[:, None] * stride_qn + offs_b[None, :] * stride_qk,
                    mask=mask_n[:, None] & mask_b[None, :], other=0,
                )
                lo = (b & 0x0F).to(tl.float32)
                hi = ((b >> 4) & 0x0F).to(tl.float32)
                # join+reshape puts them back in packing order: lo0, hi0, lo1, hi1, ...
                codes = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
            else:
                codes = tl.load(
                    QW + offs_n[:, None] * stride_qn + offs_k[None, :] * stride_qk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0,
                ).to(tl.float32)

            # One scale per group of GROUP_SIZE consecutive k.
            #
            # PER_CHANNEL is a separate branch rather than letting "GROUP_SIZE == K" fall
            # out of the arithmetic: with a single group per row the index is identically
            # zero, but written as `offs_k // GROUP_SIZE` it becomes a real division by a
            # large non-power-of-two constant, executed every iteration for nothing.
            # GROUP_TILE is the case worth optimising: when the K tile is exactly one
            # group, every element in the tile shares a scale, so the load is (BLOCK_N, 1)
            # instead of (BLOCK_N, BLOCK_K). Loading it per element re-reads the same
            # fp16 value BLOCK_K times -- at BLOCK_K=128 that is *more* traffic in scales
            # than in the 4-bit weights the kernel exists to avoid reading.
            if PER_CHANNEL:
                sc = sc_row[:, None]
            elif GROUP_TILE:
                sc = tl.load(SC + offs_n * stride_sn + (k0 // GROUP_SIZE) * stride_sg,
                             mask=mask_n, other=0.0).to(tl.float32)[:, None]
            else:
                offs_g = offs_k // GROUP_SIZE
                sc = tl.load(
                    SC + offs_n[:, None] * stride_sn + offs_g[None, :] * stride_sg,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0,
                ).to(tl.float32)

            if HAS_ZERO:
                if PER_CHANNEL:
                    zp = zp_row[:, None]
                elif GROUP_TILE:
                    zp = tl.load(ZP + offs_n * stride_sn + (k0 // GROUP_SIZE) * stride_sg,
                                 mask=mask_n, other=0).to(tl.float32)[:, None]
                else:
                    zp = tl.load(
                        ZP + offs_n[:, None] * stride_sn + offs_g[None, :] * stride_sg,
                        mask=mask_n[:, None] & mask_k[None, :], other=0,
                    ).to(tl.float32)
                w = (codes - zp) * sc
            elif NF4:
                # The non-uniform grid costs one extra load: the code is an *index* into
                # the 16-entry level table rather than a point on an even grid. The table
                # is 64 bytes and every program reads all of it, so it lands in L1 after
                # the first access and the gather is effectively free.
                w = tl.load(LUT + codes.to(tl.int32)) * sc
            else:
                # Symmetric 4-bit codes were shifted by +8 at pack time to keep the
                # nibble unsigned; undo that here rather than in a separate pass.
                w = (codes - 8.0) * sc if BITS == 4 else codes * sc

            w = tl.where(mask_k[None, :], w, 0.0)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc, mask=mask_m[:, None] & mask_n[None, :],
        )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


#: One fp32 copy of the NF4 levels per device, built once. Rebuilding a 16-element tensor
#: on every decode step would be a host-side allocation in the hottest loop we have.
_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _nf4_lut(device: torch.device) -> torch.Tensor:
    key = torch.device(device)
    if key not in _LUT_CACHE:
        from .qtensor import NF4_LEVELS

        _LUT_CACHE[key] = NF4_LEVELS.to(device=key, dtype=torch.float32).contiguous()
    return _LUT_CACHE[key]


def qlinear_forward(x: torch.Tensor, layer) -> torch.Tensor:
    """Fused forward for a QuantLinear. Falls back to the torch path if unsupported."""
    scheme = layer.scheme
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = layer.out_features
    g = layer.in_features if layer.group_size == -1 else layer.group_size

    # BLOCK_K has to be a power of two -- `tl.arange` requires it, and so does the
    # BLOCK_K//2 byte tile. It does *not* have to divide K (the tail is masked) and it
    # does not have to line up with the group size: the group index is computed per
    # element, so a tile may span several groups or sit inside one.
    # Tuned on a 3090 for decode; see docs/10-quantization.md for the sweep. The two
    # things that mattered were occupancy (BLOCK_N=32 gives 32 programs on 82 SMs;
    # BLOCK_N=128 gives 8 and is 5x slower) and matching BLOCK_K to the group size so the
    # scales are read once per tile instead of once per element.
    block_n, num_warps = 32, 2
    block_k = g if (g < K and 32 <= g <= 256 and (g & (g - 1)) == 0) else (
        128 if K >= 128 else max(32, _next_pow2(K)))
    # A group size that is neither per-channel nor a power of two would need that same
    # expensive division; resolve_group_size only ever halves, so this should not happen,
    # but fall back rather than compile something pathological.
    if g < K and (g & (g - 1)) != 0:
        import torch.nn.functional as F

        return F.linear(x, layer.dequantize_weight(x.dtype))
    if not _HAVE_TRITON:
        import torch.nn.functional as F

        return F.linear(x, layer.dequantize_weight(x.dtype))

    y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    # `.scales` is a property: with double quantization it rebuilds fp16 scales from int8
    # codes here, once per call, rather than inside the kernel. The scales are 1/64th the
    # size of the weights, so this costs little -- but it does mean double quantization is
    # a size-on-disk and size-at-rest win, not a bandwidth win. The docs say so.
    scales = layer.scales
    zp = layer.qzeros if layer.qzeros is not None else scales  # unused when sym/nf4
    lut = _nf4_lut(x.device) if scheme.is_nf4 else scales  # unused when not nf4
    # Rows beyond BLOCK_M get their own program: capping BLOCK_M without gridding over M
    # would silently leave the rest of the output uninitialised.
    #
    # The cap is 4, and it is a *compile time* limit rather than a runtime one. The inner
    # loop reduces a (BLOCK_M, BLOCK_N, BLOCK_K) broadcast; at BLOCK_M=16 that is 131k
    # elements and Triton's optimiser takes ~100 seconds on it (measured, on several
    # unrelated schemes -- it is the broadcast size, not the scheme). At BLOCK_M=4 the
    # same kernel compiles in under four seconds and decode, which is one row, is
    # unaffected either way.
    block_m = min(4, max(1, _next_pow2(M)))
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m))

    _qgemv_kernel[grid](
        x2, layer.qweight, scales, zp, lut, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        layer.qweight.stride(0), layer.qweight.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=g,
        PER_CHANNEL=(g >= K),
        GROUP_TILE=(block_k == g),
        HAS_ZERO=layer.qzeros is not None,
        NF4=scheme.is_nf4,
        BITS=scheme.bits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N).to(x.dtype)
