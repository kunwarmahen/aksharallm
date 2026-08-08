"""FlashAttention, written from scratch in Triton -- forward and backward.

Why this file exists
--------------------
`Attention.forward` calls `F.scaled_dot_product_attention`, which dispatches to somebody
else's FlashAttention kernel. That is the right thing for a training run and the wrong
thing for a repo whose point is that every core piece is hand-written. This is the same
algorithm, ours, in ~400 lines: tiled, never materialising the (T, T) score matrix, with a
backward pass that recomputes the scores instead of storing them.

The one idea
------------
Ordinary attention is three passes over an N-by-N matrix: compute `S = QK^T`, softmax it,
multiply by `V`. The middle matrix is the problem -- at T=4096 with 16 heads it is 4 GB in
fp32, and every byte of it is written to HBM and read back twice.

FlashAttention never builds it. It walks the keys in blocks and keeps three small running
values per query row:

    m  the largest score seen so far
    l  the sum of exp(score - m) so far
    o  the weighted sum of values so far, expressed relative to that same m

When a new block arrives with a larger maximum, the two accumulators are *rescaled* by
`exp(m_old - m_new)` and the new block is folded in. That is the **online softmax**, and it
is the whole trick: it makes softmax associative, so the sum can be computed in one pass
over blocks that never leave on-chip SRAM.

    exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)

The output is exact -- not an approximation, not a windowed attention. Same numbers,
different memory schedule.

The backward pass, and the thing that makes it possible
-------------------------------------------------------
The gradient needs `P = softmax(S)`, which is exactly the matrix the forward pass refused
to keep. Two options: store it (and lose everything) or recompute it. Recomputing needs the
softmax denominator, so the forward saves **one float per query row** -- the log-sum-exp
`L = m + log(l)` -- and the backward rebuilds `P = exp(S - L)` block by block from Q and K.
`(B, H, T)` instead of `(B, H, T, T)`: at our 300M's shape that is 786 KB in place of 3 GB.

The rest is the standard chain rule through a softmax, with one non-obvious term:

    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - rowsum(dO * O))      <- the softmax Jacobian, collapsed
    dQ = dS K * scale ,  dK = dS^T Q * scale

`rowsum(dO * O)` (called `delta` below) is what is left of the softmax Jacobian
`diag(p) - p p^T` once you notice that `p^T dP` equals `dO . O` for that row. It is a
single number per row, computed in one cheap elementwise pass before the kernels run.

Causal masking is bottom-right aligned
--------------------------------------
Query `m` of `T` may see key `n` of `S` iff `n <= m + (S - T)`. The offset matters and it
is the same trap documented in `Attention.forward`: when Q and K have different lengths --
several tokens verified against a warm KV cache, which is what speculative decoding does --
a top-left aligned triangle hides most of the prompt from every query, trains fine, and
generates fluent nonsense. One integer, `DIAG = S - T`, and the three cases (training
prefill, one decode step, a draft block) all fall out of the same expression.

Two blocks per program-pair are skipped entirely because of it: for a query block, key
blocks past the diagonal are never loaded, which is where causal attention gets its ~2x
over the full version rather than only masking it away.

What it is worth, measured
--------------------------
See `docs/04-model.md` for the table. The short version: against PyTorch's SDPA (which is
FlashAttention-2 with hand-written PTX and years of tuning) this lands in the same
neighbourhood rather than ahead of it, and the interesting number is not the speedup, it is
the *memory*: the naive implementation cannot run the shapes this does at all.

Run `python -m aksharallm.model.flash` to reproduce all of it on your own card.

Read with: docs/04-model.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math

import torch

#: `tl.dot` needs at least 16 rows, and tiling a 1-row problem is pointless anyway: a
#: single decode step is a matrix-vector product where SDPA's own kernel wins easily. Below
#: this many query rows the wrapper hands the work back to SDPA. Same idea as
#: `AUTO_MAX_ROWS` in `quant/kernels.py`, pointing the other way.
MIN_QUERY_ROWS = 16

#: Head dimensions the kernel compiles for. Powers of two only (`tl.arange`), and 128 is
#: where a block of scores plus a block of values stops fitting in SRAM on an Ampere card.
SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)

#: log2(e). The kernels work in base 2 because `exp2` is a single hardware instruction on
#: the SFU while `exp` is a multiply plus that instruction. Folding the constant into the
#: score scale makes it free.
LOG2E = 1.4426950408889634

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    _HAVE_TRITON = False


def available(device: torch.device | None = None) -> bool:
    """True when the Triton path can run here at all."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True


def usable(q: torch.Tensor, k: torch.Tensor, dropout_p: float = 0.0,
           attn_mask: torch.Tensor | None = None) -> bool:
    """True when *this particular call* is one the kernel handles.

    Everything it says no to has a correct SDPA path behind it, so this is a routing
    decision and never a failure. The refusals are: an arbitrary additive/boolean mask
    (only the causal shape is built in), dropout inside attention (we train with none, and
    a dropout kernel needs its own RNG state to be reproducible across the backward),
    a head dimension the kernel is not compiled for, and a query block too short to fill
    one tensor-core tile.
    """
    if not available(q.device):
        return False
    if attn_mask is not None or dropout_p:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS or q.shape[-1] != k.shape[-1]:
        return False
    if q.shape[2] < MIN_QUERY_ROWS:
        return False
    return q.shape[2] <= k.shape[2]  # queries sit at the END of the keys; see DIAG


if _HAVE_TRITON:

    @triton.jit
    def _fwd_kernel(
        Q, K, V, O, L,
        qk_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        H, T, S, DIAG, N_REP, WINDOW, SINKS,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        WINDOWED: tl.constexpr,
    ):
        """One program owns BLOCK_M query rows of one (batch, head) and streams the keys.

        Writes `O` (the attention output) and `L` (the natural log-sum-exp per row, which
        is the only thing the backward pass needs in order to rebuild the softmax).
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        hk = h // N_REP  # GQA: several query heads read one key/value head

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < T

        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None], other=0.0,
        )

        # The three running values. `m_i` starts at a large finite negative rather than
        # -inf so that a row whose every key is masked yields exp2(-inf - m) = 0 instead of
        # the nan that (-inf) - (-inf) would give. Costs nothing and removes a whole class
        # of "the loss went nan at step 3" mysteries.
        m_i = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        # Causal: no query in this block can see a key past `(pid_m+1)*BLOCK_M - 1 + DIAG`,
        # so those blocks are never loaded. This is where the ~2x comes from -- masking
        # them after the fact would still pay for the loads and the matmul.
        if CAUSAL:
            hi = tl.minimum(S, (pid_m + 1) * BLOCK_M + DIAG)
        else:
            hi = S

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            k = tl.load(
                K + b * stride_kb + hk * stride_kh
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0,
            )
            v = tl.load(
                V + b * stride_vb + hk * stride_vh
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0,
            )
            # `qk_scale` already carries the 1/sqrt(D) *and* log2(e), so `s` is in base-2
            # units and the softmax below can use the single-instruction exp2.
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale

            keep = mask_n[None, :]
            if CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + DIAG)
            if WINDOWED:
                keep = keep & ((offs_n[None, :] > offs_m[:, None] + DIAG - WINDOW)
                               | (offs_n[None, :] < SINKS))
            s = tl.where(keep, s, float("-inf"))

            # --- the online softmax rescale, in four lines --------------------------
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)   # how much the old accumulators shrink
            p = tl.exp2(s - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]

        # Saved for the backward: the natural log-sum-exp. Converting out of base 2 here
        # keeps every other reader of this tensor (tests, the backward, a debugger) in the
        # units the maths is written in.
        lse = (m_i + tl.log2(l_safe)) / 1.4426950408889634
        tl.store(L + b * stride_lb + h * stride_lh + offs_m * stride_lm, lse, mask=mask_m)
        tl.store(
            O + b * stride_ob + h * stride_oh
            + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc.to(O.dtype.element_ty), mask=mask_m[:, None],
        )

    @triton.jit
    def _bwd_kv_kernel(
        Q, K, V, DO, DK, DV, L, D,
        qk_scale, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_db, stride_dh, stride_dm, stride_dd,
        stride_gb, stride_gh, stride_gn, stride_gd,
        stride_lb, stride_lh, stride_lm,
        H, T, S, DIAG, N_REP, WINDOW, SINKS,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        WINDOWED: tl.constexpr,
    ):
        """dK and dV for one block of keys, accumulated over the queries that see it.

        Fixing the *key* block and looping over queries is what keeps this atomic-free:
        each program owns its slice of the output and writes it once. The dQ kernel below
        fixes the query block instead, for the same reason. Two passes over the scores
        rather than one, and no contention.
        """
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        hk = h // N_REP

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_n = offs_n < S

        k = tl.load(
            K + b * stride_kb + hk * stride_kh
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None], other=0.0,
        )
        v = tl.load(
            V + b * stride_vb + hk * stride_vh
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None], other=0.0,
        )

        dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

        # The mirror of the forward's `hi`: key n is invisible to every query before
        # `n - DIAG`, so those query blocks contribute nothing and are skipped.
        if CAUSAL:
            lo = tl.maximum(0, pid_n * BLOCK_N - DIAG)
            lo = (lo // BLOCK_M) * BLOCK_M
        else:
            lo = 0

        for start_m in range(lo, T, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T

            q = tl.load(
                Q + b * stride_qb + h * stride_qh
                + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=mask_m[:, None], other=0.0,
            )
            do = tl.load(
                DO + b * stride_db + h * stride_dh
                + offs_m[:, None] * stride_dm + offs_d[None, :] * stride_dd,
                mask=mask_m[:, None], other=0.0,
            )
            lse = tl.load(L + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                          mask=mask_m, other=0.0)
            delta = tl.load(D + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                            mask=mask_m, other=0.0)

            # Everything here is transposed relative to the forward -- (key, query)
            # instead of (query, key) -- so that both matmuls land with the key block on
            # the rows, which is the axis dK and dV are accumulated over.
            s_t = tl.dot(k, tl.trans(q), input_precision="ieee") * qk_scale
            keep = mask_m[None, :] & mask_n[:, None]
            if CAUSAL:
                keep = keep & (offs_n[:, None] <= offs_m[None, :] + DIAG)
            if WINDOWED:
                keep = keep & ((offs_n[:, None] > offs_m[None, :] + DIAG - WINDOW)
                               | (offs_n[:, None] < SINKS))
            s_t = tl.where(keep, s_t, float("-inf"))
            p_t = tl.exp2(s_t - (lse * 1.4426950408889634)[None, :])

            dv += tl.dot(p_t.to(do.dtype), do, input_precision="ieee")
            dp_t = tl.dot(v, tl.trans(do), input_precision="ieee")
            ds_t = p_t * (dp_t - delta[None, :])
            dk += tl.dot(ds_t.to(q.dtype), q, input_precision="ieee")

        # Written per QUERY head, not per key/value head: with GQA several query heads
        # share one KV head and their gradients must be summed. Doing that sum in torch
        # afterwards costs one reduction and keeps this kernel free of atomics.
        base = b * stride_gb + h * stride_gh + offs_n[:, None] * stride_gn \
            + offs_d[None, :] * stride_gd
        tl.store(DK + base, dk * sm_scale, mask=mask_n[:, None])
        tl.store(DV + base, dv, mask=mask_n[:, None])

    @triton.jit
    def _bwd_q_kernel(
        Q, K, V, DO, DQ, L, D,
        qk_scale, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_db, stride_dh, stride_dm, stride_dd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        H, T, S, DIAG, N_REP, WINDOW, SINKS,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        WINDOWED: tl.constexpr,
    ):
        """dQ for one block of queries, accumulated over the keys it sees.

        Same shape as the forward pass, walking the same blocks in the same order, which is
        why the recomputed scores are bit-identical to the ones the forward used.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        hk = h // N_REP

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < T

        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None], other=0.0,
        )
        do = tl.load(
            DO + b * stride_db + h * stride_dh
            + offs_m[:, None] * stride_dm + offs_d[None, :] * stride_dd,
            mask=mask_m[:, None], other=0.0,
        )
        lse = tl.load(L + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                      mask=mask_m, other=0.0)
        delta = tl.load(D + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                        mask=mask_m, other=0.0)

        dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        if CAUSAL:
            hi = tl.minimum(S, (pid_m + 1) * BLOCK_M + DIAG)
        else:
            hi = S

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            k = tl.load(
                K + b * stride_kb + hk * stride_kh
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0,
            )
            v = tl.load(
                V + b * stride_vb + hk * stride_vh
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0,
            )

            s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
            keep = mask_n[None, :]
            if CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + DIAG)
            if WINDOWED:
                keep = keep & ((offs_n[None, :] > offs_m[:, None] + DIAG - WINDOW)
                               | (offs_n[None, :] < SINKS))
            s = tl.where(keep, s, float("-inf"))
            p = tl.exp2(s - (lse * 1.4426950408889634)[:, None])

            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")

        tl.store(
            DQ + b * stride_ob + h * stride_oh
            + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            (dq * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None],
        )


def _blocks(head_dim: int, itemsize: int, forward: bool) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps, num_stages).

    Hand-picked on a 3090 rather than autotuned, because autotuning costs a multi-second
    stall the first time each new shape appears and one training step meets several. The
    sweep is in `docs/04-model.md`.

    The budget being spent is **SRAM**, not registers: an Ampere SM will give one program
    99 KB, and it has to hold the Q tile, the K and V tiles, and `num_stages` copies of
    whatever the pipeliner is prefetching. So every axis that makes a tile bigger has to
    take a block size back out --

      head_dim doubling      halves the block, obviously;
      the backward           holds five tiles rather than three (Q, K, V, dO, and the
                             transposed score block);
      fp32                   is twice the bytes of bf16 for the same block, which is why
                             this takes `itemsize` at all. Getting it wrong is not a slow
                             kernel, it is a hard `OutOfResources` at launch -- the one
                             good thing about this failure mode.

    The result of the sweep was flatter than expected and had one sharp edge in it. Block
    sizes barely moved the forward (0.61-0.68 ms over the whole 64-128 grid at T=2048), but
    **`num_warps` moved the backward by 1.7x** -- 8 warps on a 64x64 tile gave 4.85 ms
    against 4 warps' 2.90 ms. More warps than the tile has work for is not free
    parallelism; it splits a small tile into slivers and spends the difference on
    cross-warp reduction. Every default here is 4 warps for that reason.
    """
    wide = head_dim > 64 or itemsize > 2
    if forward:
        return (64, 32, 4, 2) if wide else (64, 64, 4, 2)
    return (32, 32, 4, 2) if wide else (64, 64, 4, 2)


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, window, sinks):
        B, H, T, D = q.shape
        Hk, S = k.shape[1], k.shape[2]
        n_rep = H // Hk

        q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
        o = torch.empty_like(q)
        lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

        block_m, block_n, warps, stages = _blocks(D, q.element_size(), forward=True)
        grid = (triton.cdiv(T, block_m), B * H)
        _fwd_kernel[grid](
            q, k, v, o, lse,
            sm_scale * LOG2E,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(), *lse.stride(),
            H, T, S, S - T, n_rep, window or 0, sinks,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
            CAUSAL=causal, WINDOWED=window is not None,
            num_warps=warps, num_stages=stages,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.sm_scale = causal, sm_scale
        ctx.window, ctx.sinks = window, sinks
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        Hk, S = k.shape[1], k.shape[2]
        n_rep = H // Hk
        sm_scale = ctx.sm_scale

        do = do if do.stride(-1) == 1 else do.contiguous()
        # `delta = rowsum(dO * O)` -- the collapsed softmax Jacobian. One elementwise pass
        # in torch: it reads the same two tensors a dedicated kernel would, so writing one
        # buys nothing but a file to keep correct.
        delta = (do.float() * o.float()).sum(-1)

        dq = torch.empty_like(q)
        # dK/dV are accumulated per QUERY head and reduced afterwards -- see the store at
        # the end of `_bwd_kv_kernel`. On a dense model (n_rep == 1) the reduction is a
        # no-op sum over a length-1 axis; on GQA it is the whole point.
        dk_buf = torch.empty((B, H, S, D), dtype=torch.float32, device=q.device)
        dv_buf = torch.empty((B, H, S, D), dtype=torch.float32, device=q.device)

        block_m, block_n, warps, stages = _blocks(D, q.element_size(), forward=False)
        common = (sm_scale * LOG2E, sm_scale)
        # Every tensor carries its own strides. They usually all agree -- attention hands
        # this function five dense (B, H, *, D) transposes -- but "usually" is how a kernel
        # that reads the wrong addresses for one caller gets written.
        _bwd_kv_kernel[(triton.cdiv(S, block_n), B * H)](
            q, k, v, do, dk_buf, dv_buf, lse, delta,
            *common,
            *q.stride(), *k.stride(), *v.stride(), *do.stride(),
            *dk_buf.stride(), *lse.stride(),
            H, T, S, S - T, n_rep, ctx.window or 0, ctx.sinks,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
            CAUSAL=ctx.causal, WINDOWED=ctx.window is not None,
            num_warps=warps, num_stages=stages,
        )
        _bwd_q_kernel[(triton.cdiv(T, block_m), B * H)](
            q, k, v, do, dq, lse, delta,
            *common,
            *q.stride(), *k.stride(), *v.stride(), *do.stride(),
            *dq.stride(), *lse.stride(),
            H, T, S, S - T, n_rep, ctx.window or 0, ctx.sinks,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
            CAUSAL=ctx.causal, WINDOWED=ctx.window is not None,
            num_warps=warps, num_stages=stages,
        )

        dk = dk_buf.view(B, Hk, n_rep, S, D).sum(2).to(k.dtype)
        dv = dv_buf.view(B, Hk, n_rep, S, D).sum(2).to(v.dtype)
        return dq, dk, dv, None, None, None, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True, sm_scale: float | None = None,
                    window: int | None = None, sinks: int = 0) -> torch.Tensor:
    """Causal (or full) scaled dot-product attention, ours.

    q: (B, H, T, D). k, v: (B, Hk, S, D) with H a multiple of Hk (GQA is handled inside
    the kernel -- the repeated KV heads are never materialised). Queries are taken to sit
    at the END of the key sequence, so `causal=True` with T < S means "these T new tokens,
    against S-T already cached ones".

    `window` restricts each query to the last `window` keys, and `sinks` keeps the first
    few keys visible regardless (see `transformer.sliding_window_mask` for why that second
    number is not optional). Passing them as integers rather than as a mask is the point:
    the equivalent bool tensor is 64 MB at T=8192, and long context is where it matters.

    Differentiable. Falls back to nothing: call `usable()` first if the shape might not be
    one the kernel handles.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttention.apply(q, k, v, causal, sm_scale, window, sinks)


def reference_attention(q, k, v, causal=True, sm_scale=None, window=None, sinks=0):
    """The (T, S) matrix version, written out, in fp32. Slow and memory-hungry on purpose.

    This is the definition the kernel is tested against and the thing it exists to avoid:
    it allocates the score matrix the whole file is about, which is what makes it the
    honest baseline for the memory measurement in `main()`.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    B, H, T, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    # Bottom-right alignment: query m of T is at absolute position m + (S - T).
    qi = torch.arange(T, device=q.device)[:, None] + (S - T)
    ki = torch.arange(S, device=q.device)[None, :]
    if causal:
        s = s.masked_fill(ki > qi, float("-inf"))
    if window is not None:
        s = s.masked_fill(~((ki > qi - window) | (ki < sinks)), float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


# --------------------------------------------------------------------------------------
# Bench / self-check:  python -m aksharallm.model.flash
# --------------------------------------------------------------------------------------

def _bench(fn, iters: int = 20, warmup: int = 5) -> float:
    """Milliseconds per call, median of `iters`, CUDA-synchronised."""
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _peak_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main(argv=None):
    import argparse

    import torch.nn.functional as F

    p = argparse.ArgumentParser(
        prog="python -m aksharallm.model.flash",
        description="Check and benchmark the from-scratch FlashAttention kernel.",
    )
    p.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=4, help="GQA; equal to --heads for MHA")
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no-bwd", action="store_true", help="forward only")
    args = p.parse_args(argv)

    if not available():
        print("Triton + CUDA are not available here; nothing to measure.")
        return 1

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    B, H, Hk, D = args.batch, args.heads, args.kv_heads, args.head_dim
    print(f"B={B} H={H} Hk={Hk} D={D} {args.dtype} causal, on {torch.cuda.get_device_name()}\n")

    print("correctness against the (T,S)-matrix reference, in fp32")
    print(f"{'T':>6} {'max abs err':>12} {'dq':>10} {'dk':>10} {'dv':>10}")
    torch.manual_seed(0)
    for T in args.seqlens[:2]:
        qkv = [torch.randn(B, n, T, D, device="cuda", dtype=dtype, requires_grad=True)
               for n in (H, Hk, Hk)]
        out = flash_attention(*qkv)
        ref = reference_attention(*[x.detach() for x in qkv])
        g = torch.randn_like(out)
        grads = torch.autograd.grad(out, qkv, g)
        rq = [x.detach().clone().requires_grad_(True) for x in qkv]
        rgrads = torch.autograd.grad(reference_attention(*rq), rq, g)
        errs = [(a.float() - b.float()).abs().max().item() for a, b in zip(grads, rgrads)]
        print(f"{T:>6} {(out.float() - ref.float()).abs().max().item():>12.2e} "
              + " ".join(f"{e:>10.2e}" for e in errs))

    print("\nlatency, ms/call  (ours vs F.scaled_dot_product_attention)")
    head = f"{'T':>6} {'ours fwd':>10} {'sdpa fwd':>10} {'x':>6}"
    if not args.no_bwd:
        head += f" {'ours f+b':>10} {'sdpa f+b':>10} {'x':>6}"
    print(head)
    for T in args.seqlens:
        qkv = [torch.randn(B, n, T, D, device="cuda", dtype=dtype, requires_grad=True)
               for n in (H, Hk, Hk)]
        q, k, v = qkv
        ours = _bench(lambda: flash_attention(q, k, v))
        sdpa = _bench(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=H != Hk))
        row = f"{T:>6} {ours:>10.3f} {sdpa:>10.3f} {sdpa / ours:>5.2f}x"
        if not args.no_bwd:
            g = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

            def fb(fn):
                out = fn()
                torch.autograd.grad(out, qkv, g)

            ours_fb = _bench(lambda: fb(lambda: flash_attention(q, k, v)))
            sdpa_fb = _bench(lambda: fb(lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=H != Hk)))
            row += f" {ours_fb:>10.3f} {sdpa_fb:>10.3f} {sdpa_fb / ours_fb:>5.2f}x"
        print(row)

    print("\npeak memory, MB  (the point of the exercise)")
    print(f"{'T':>6} {'ours':>10} {'sdpa':>10} {'(T,S) matrix':>14}")
    for T in args.seqlens:
        q, k, v = (torch.randn(B, n, T, D, device="cuda", dtype=dtype) for n in (H, Hk, Hk))
        ours = _peak_mb(lambda: flash_attention(q, k, v))
        sdpa = _peak_mb(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=H != Hk))
        try:
            naive = f"{_peak_mb(lambda: reference_attention(q, k, v)):>14.0f}"
        except torch.cuda.OutOfMemoryError:
            naive = f"{'OOM':>14}"
            torch.cuda.empty_cache()
        print(f"{T:>6} {ours:>10.0f} {sdpa:>10.0f} {naive}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
