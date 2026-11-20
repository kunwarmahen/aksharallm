"""Gated DeltaNet: a linear-attention layer with a fixed-size memory, written from scratch.

Softmax attention keeps every past key and value and compares the query with all of them, so
its cost per token grows with the context and its cache grows without bound. A linear
attention layer keeps a fixed-size MATRIX instead, one per head, and folds each token into it:

    S_t  = a_t * S_{t-1}                       forget a little of everything   (the gate)
    S_t += b_t * k_t (v_t - k_t^T S_t)^T       overwrite what k_t now stands for (the delta rule)
    o_t  = q_t^T S_t                           read with the query

`S` is (head_dim x head_dim), whatever the length. The delta rule is the difference from the
first linear attentions: they only ever ADDED `k v^T`, so two keys that point the same way
blurred together and the memory filled up. Here a token first reads what the memory already
returns for its key (`k^T S`) and writes only the correction, scaled by `b` in (0, 1). That is
one step of online least squares, and it is what lets a small state recall an exact value.
`a` in (0, 1) is the decay Mamba-2 added: the layer can forget on purpose.

Two ways to compute the same thing, and the test suite holds them to each other:

    `recurrent_gated_delta_rule`   the three lines above, one token at a time. The reference:
                                   obviously right, and far too slow to train with.
    `chunk_gated_delta_rule`       the chunked form. Within a chunk of C tokens the updates are
                                   unrolled into matrix products (the "WY" form: a triangular
                                   solve stands in for C sequential corrections), and the state
                                   is carried only from chunk to chunk. T/C sequential steps
                                   instead of T, each one a few C x C matmuls.

`GatedDeltaNet` is the layer around it, the shape used by Gated DeltaNet (Yang et al., 2024)
and Qwen3-Next: q/k/v projections, a short causal depthwise convolution, L2-normalised q and
k, `b = sigmoid(.)`, `log a = -exp(A) * softplus(. + dt_bias)`, and an output gate
`RMSNorm(o) * silu(W_g x)`. It takes the same arguments as `Attention`, so a block can hold
either (`model.full_attn_every`, docs/04 § "Linear attention").

It ignores RoPE: a recurrence already knows the order of what it has read. It refuses a KV
cache rather than pretending to have one; generation from a hybrid model is not built.

Read with: docs/04-model.md -- the chapter this implements; it ends with the order to read these
files in. See also docs/19-long-context.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


def recurrent_gated_delta_rule(q, k, v, g, beta):
    """Token by token. q, k: (B, H, T, Dk); v: (B, H, T, Dv); g (log decay <= 0), beta:
    (B, H, T). Returns (B, H, T, Dv). The reference the chunked form must equal."""
    B, H, T, Dk = k.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]                     # forget
        read = (S * k[:, :, t, :, None]).sum(-2)                      # k^T S: what k returns now
        delta = (v[:, :, t] - read) * beta[:, :, t, None]             # the correction
        S = S + k[:, :, t, :, None] * delta[:, :, None, :]            # write it at k
        out.append((S * q[:, :, t, :, None]).sum(-2))                 # q^T S
    return torch.stack(out, dim=2)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk: int = 64):
    """The same function as `recurrent_gated_delta_rule`, in T/chunk sequential steps.

    Inside a chunk, with cumulative log decay G_i = g_1 + ... + g_i and Gamma_ij = exp(G_i - G_j):

      A  = strictly_lower( (beta k) k^T * Gamma )     how much token i's write is changed by
                                                       the writes of the earlier tokens j < i
      T  = (I + A)^-1                                 all C sequential corrections at once
      U  = T (beta v)                                 the chunk's corrected values, if the
      W  = T (beta k * exp(G))                        incoming state were zero, and what the
                                                       incoming state contributes to them
    then per chunk, with the state S carried in:
      v_new = U - W S
      o     = (q * exp(G)) S  +  causal(q k^T * Gamma) v_new
      S     = exp(G_C) S  +  (k * exp(G_C - G))^T v_new
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:
        # Padding at the END only: a padded position has k = v = beta = g = 0, so it writes
        # nothing and forgets nothing, and its outputs are cut off below.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (x.reshape(B, H, n, chunk, x.shape[-1]) for x in (q, k, v))
    g = g.reshape(B, H, n, chunk).cumsum(-1)
    beta = beta.reshape(B, H, n, chunk)
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]

    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    # Gamma_ij for j <= i; the exponent is masked BEFORE exp, because above the diagonal it is
    # positive and can overflow to inf, and inf * 0 is nan.
    gamma = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, float("-inf")).exp()
    # Strictly lower. (Zeroing the diagonal is for the reader: `unitriangular=True` below
    # never reads it, so leaving it in changes nothing, as a mutation test confirms.)
    A = ((k_beta @ k.transpose(-1, -2)) * gamma).masked_fill(lower.logical_not() | eye.bool(), 0.0)
    Tinv = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False,
                                         unitriangular=True)
    U = Tinv @ v_beta
    W = Tinv @ (k_beta * g.exp()[..., None])
    within = (q @ k.transpose(-1, -2)) * gamma                         # already causal: gamma
    S = q.new_zeros(B, H, Dk, Dv)
    out = []
    for c in range(n):
        v_new = U[:, :, c] - W[:, :, c] @ S
        out.append((q[:, :, c] * g[:, :, c, :, None].exp()) @ S + within[:, :, c] @ v_new)
        last = g[:, :, c, -1]
        S = (S * last.exp()[..., None, None]
             + (k[:, :, c] * (last[..., None] - g[:, :, c]).exp()[..., None]).transpose(-1, -2)
             @ v_new)
    return torch.stack(out, dim=2).reshape(B, H, n * chunk, Dv)[:, :, :T]


class GatedDeltaNet(nn.Module):
    """A drop-in for `Attention` in one block: same input, same output, no KV cache."""

    def __init__(self, cfg: ModelConfig, chunk: int = 64, conv: int = 4):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        inner = self.n_heads * self.head_dim
        self.chunk = chunk
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        # The short causal convolution lets q/k/v see the last few tokens before they reach
        # the recurrence (padding on the left only, by cutting the right).
        self.conv = nn.Conv1d(3 * inner, 3 * inner, conv, groups=3 * inner, padding=conv - 1,
                              bias=False)
        self.b_proj = nn.Linear(cfg.d_model, self.n_heads, bias=False)
        self.a_proj = nn.Linear(cfg.d_model, self.n_heads, bias=False)
        # Mamba-2's decay: log a = -exp(A_log) * softplus(a_proj(x) + dt_bias), with A in
        # [1, 16] and softplus(dt_bias) in [1e-3, 1e-1] at init, so heads start with a spread
        # of memories, from a few tokens to a few thousand.
        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1, 16)))
        dt = torch.exp(torch.empty(self.n_heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1(dt)
        self.g_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.wo = nn.Linear(inner, cfg.d_model, bias=False)   # "wo": scaled at init like Attention.wo
        self.eps = cfg.norm_eps
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None, cache=None, attn_mask=None):
        if cache is not None or attn_mask is not None:
            raise ValueError("a Gated DeltaNet layer has a recurrent state, not a KV cache, and "
                             "generation from a hybrid model is not built: score whole "
                             "sequences (docs/04 § Linear attention)")
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = F.silu(self.conv(self.qkv(x).transpose(1, 2))[..., :T]).transpose(1, 2)
        q, k, v = (t.reshape(B, T, H, D).transpose(1, 2).float() for t in qkv.chunk(3, dim=-1))
        q = F.normalize(q, dim=-1) * D ** -0.5
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.b_proj(x).float()).transpose(1, 2)            # (B, H, T)
        g = (-self.A_log.float().exp()
             * F.softplus(self.a_proj(x).float() + self.dt_bias.float())).transpose(1, 2)
        # fp32 throughout the recurrence: the state is a running sum over the whole sequence,
        # and bf16's 8-bit mantissa would lose the small corrections (gotcha 14's family).
        with torch.autocast(device_type=x.device.type, enabled=False):
            o = chunk_gated_delta_rule(q, k, v, g, beta, self.chunk)            # (B, H, T, D)
        o = o.transpose(1, 2)                                                   # (B, T, H, D)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * self.o_norm_weight
        o = o.to(x.dtype) * F.silu(self.g_proj(x)).reshape(B, T, H, D)
        return self.dropout(self.wo(o.reshape(B, T, H * D)))
