"""A Llama-style decoder-only transformer, written from scratch.

Architecture choices and why they differ from the original GPT-2:

  RMSNorm instead of LayerNorm   - no mean-subtraction, no bias. Cheaper, works as well.
  RoPE instead of learned pos    - relative positions baked into attention; extrapolates better.
  SwiGLU instead of GELU MLP     - gated activation, ~1.5x params in the MLP but better loss/param.
  GQA instead of MHA             - fewer key/value heads => a much smaller KV cache at inference.
  Pre-norm residual stream       - norm *before* each sublayer. Required for stable deep training.
  No biases anywhere             - they buy nothing at scale and cost memory bandwidth.

Shapes convention used throughout:
  B = batch, T = time/sequence, C = d_model, H = n_heads, Hk = n_kv_heads, D = head_dim

Read with: docs/03-model.md -- the chapter this implements; it ends with the order to read these
files in. See also docs/06-inference.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .moe import MoEFeedForward, moe_stats

# torch >= 2.5 can do grouped-query attention inside SDPA without materialising repeated KV.
_SDPA_HAS_GQA = "enable_gqa" in F.scaled_dot_product_attention.__doc__


class RMSNorm(nn.Module):
    """x / rms(x) * weight.  rms(x) = sqrt(mean(x^2) + eps)

    Note we cast to fp32 for the normalisation itself: computing a mean of squares in
    bf16 loses precision badly, and this is cheap enough that it doesn't matter.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(
    head_dim: int, max_seq_len: int, theta: float, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotation angles for RoPE.

    Each pair of adjacent channels (i, i + head_dim/2) is treated as a 2-D vector and
    rotated by an angle proportional to the token's position. Channel pairs get
    geometrically decreasing frequencies, so early channels rotate fast (encoding local
    position) and late channels rotate slowly (encoding global position).

    Returns (cos, sin), each shaped (max_seq_len, head_dim).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (T, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, D) -- duplicated to match rotate_half
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., D) where the two halves are swapped and the first is negated."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, D). cos/sin: (T, D), or (B, T, D) when positions differ per row.

    The two shapes are the two ways this model is used. Training and single-sequence
    generation share one window of positions across the batch — every row is at the same
    place — so `(T, D)` broadcasts and costs nothing. **Serving does not**: a batch there is
    several unrelated conversations, one on its 12th token and another on its 400th, so each
    row carries its own positions and RoPE has to rotate them by different angles. Getting
    that wrong is silent: every sequence still attends to the right keys, they are just
    labelled with someone else's position.
    """
    if cos.dim() == 3:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    else:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Preallocated once so generation does no allocation in the hot loop.
    """

    def __init__(self, batch, n_kv_heads, max_seq_len, head_dim, dtype, device):
        shape = (batch, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.pos = 0  # number of tokens currently cached

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        self.k[:, :, self.pos : self.pos + t] = k
        self.v[:, :, self.pos : self.pos + t] = v
        self.pos += t
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]

    def reset(self):
        self.pos = 0

    def rewind(self, to: int) -> None:
        """Forget everything cached after position `to`.

        Nothing is erased: the entries beyond `to` are simply out of scope and the next
        `update` overwrites them. That is the whole trick — a rollback costs one integer.

        This exists for **speculative decoding**, which is a loop of "guess several tokens,
        then throw away the ones the big model disagrees with". Without a rewind, a rejected
        guess would leave its keys and values in the cache and every token after it would
        attend to a position the model never actually chose.
        """
        self.pos = max(0, min(int(to), self.pos))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # how many Q heads share one KV head

        # One fused projection for Q, K, V would be nice, but with GQA the three have
        # different output sizes, so keep them separate for clarity.
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None, attn_mask=None):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        # Causal masking, and there are three cases rather than the two it looked like:
        #
        #   no cache, T > 1      prefill / training  -> the ordinary triangular mask
        #   cache,    T == 1     one decode step     -> NO mask; every cached position is
        #                        legal, and is_causal=True here would be wrong because the
        #                        single query sits at the *end* of the sequence, not the start
        #   cache,    T > 1      several tokens against a warm cache -> `attn_mask`, built by
        #                        `Transformer.forward`, because `is_causal=True` aligns its
        #                        triangle to the TOP-LEFT when the query and key lengths
        #                        differ: query i would see keys 0..i instead of 0..(start+i),
        #                        hiding most of the prompt from every one of them. Nothing
        #                        exercised that path until speculative decoding, which
        #                        verifies a whole draft in one pass, needed it.
        #
        # A prefill into an *empty* cache gets no mask and `is_causal=True`: query and key
        # lengths are equal there, so the triangle is aligned and the fast kernel is kept.
        is_causal = attn_mask is None and T > 1

        if _SDPA_HAS_GQA:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=self.n_rep > 1,
            )
        else:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """FFN(x) = W2( silu(W1 x) * W3 x )

    The elementwise product is the "gate": W1's branch decides how much of W3's branch
    passes through. Costs a third matrix vs. a plain MLP, which is why d_ff is set to
    ~8/3*d_model instead of 4*d_model to keep the parameter count comparable.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm transformer block. The residual stream (x) is never normalised in place —
    each sublayer reads a normalised *copy* and writes an additive update. That clean
    residual path is what lets gradients reach layer 0 without vanishing."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # The ONLY thing a mixture of experts changes about the architecture. Attention,
        # RoPE, the norms and the residual path are untouched, which is why an MoE
        # checkpoint can be upcycled from a dense one (see `model/moe.py`).
        self.ffn = MoEFeedForward(cfg) if cfg.moe_layer(layer_idx) else SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Share one matrix between input lookup and output projection. Saves
            # vocab*d_model params and empirically helps at small scale.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream. Without this the
        # residual variance grows like n_layers and deep models blow up early in training.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def num_active_params(self, non_embedding: bool = False) -> int:
        """Parameters used to compute ONE token. Equals `num_params` on a dense model.

        Both numbers have to be reported for a mixture of experts or every comparison it
        appears in is misleading: total parameters say what it cost to store, active
        parameters say what it cost to run, and MoE's whole claim is that those two stop
        being the same number.
        """
        n = self.num_params(non_embedding)
        for module in self.modules():
            if isinstance(module, MoEFeedForward):
                n -= module.n_total_params() - module.n_active_params()
        return n

    def moe_aux_loss(self) -> torch.Tensor | None:
        """The summed auxiliary losses from the last forward, or None if dense."""
        terms = [m.aux_loss for m in self.modules()
                 if isinstance(m, MoEFeedForward) and m.aux_loss is not None]
        return torch.stack(terms).sum() if terms else None

    def routing(self) -> dict | None:
        """Per-expert routing shares from the last forward — see `model/moe.py`."""
        return moe_stats(self)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        loss_mask: torch.Tensor | None = None,
        full_logits: bool = False,
        positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        """idx: (B, T) int64 token ids.
        targets: (B, T) int64, -100 to ignore. If given, returns (logits, loss).
        loss_mask: (B, T) bool/float — optional extra weighting, used by SFT.
        full_logits: return every position's logits with no loss, as (logits, None).

        `full_logits` exists for evaluation. Scoring a multiple-choice answer needs the
        log-probability of each token *individually*, so the harness computes its own
        per-token cross-entropy — and asking for `targets` just to get the full logit
        tensor would make this function compute a mean loss the caller throws away, at the
        cost of a second float32 copy of a (B, T, vocab) tensor. At batch 8 x 512 tokens
        that copy is a quarter of a gigabyte, on a device that is often the CPU because a
        training run owns the card.
        """
        B, T = idx.shape
        start = caches[0].pos if caches is not None else 0

        # `positions` and `attn_mask` are the serving path: a batch of unrelated sequences at
        # different lengths, whose keys and values live in a paged pool rather than one
        # contiguous block. The caller knows where each row is and what it may attend to, so
        # it says. Everything else — training, single-sequence generation — leaves both None
        # and gets the ordinary shared-window behaviour below, unchanged.
        if positions is None:
            assert start + T <= self.cfg.max_seq_len, (
                f"sequence position {start + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
            cos = self.rope_cos[start : start + T]
            sin = self.rope_sin[start : start + T]
        else:
            cos = self.rope_cos[positions]        # (B, T, head_dim)
            sin = self.rope_sin[positions]

        # Several tokens against a WARM cache: build the mask once here rather than in each
        # of the 24 attention layers. Query j is at absolute position start+j and may see
        # every key up to and including it — the whole cached prefix, plus the part of this
        # block that precedes it. `start > 0` is the point: a prefill into an empty cache is
        # already aligned, needs no mask, and keeps the faster kernel.
        if attn_mask is None and caches is not None and T > 1 and start > 0:
            q_pos = torch.arange(start, start + T, device=idx.device)[:, None]
            k_pos = torch.arange(start + T, device=idx.device)[None, :]
            attn_mask = k_pos <= q_pos

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None, attn_mask)
        x = self.norm(x)

        if targets is None:
            if full_logits:
                return self.lm_head(x), None
            # Inference: only the last position matters. Computing the full (B,T,vocab)
            # logit tensor here would be the single biggest allocation in generation.
            return self.lm_head(x[:, -1:, :]), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )
        self.last_ce = loss.detach()
        if loss_mask is not None:
            # Recompute as a weighted mean when a mask is supplied.
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            m = loss_mask.reshape(-1).float()
            loss = (per_tok * m).sum() / m.sum().clamp(min=1)
            self.last_ce = loss.detach()

        # The balancing loss is added HERE rather than left for the trainer to remember,
        # because forgetting it does not fail — it trains a model whose experts quietly
        # collapse, and the loss curve looks fine while that happens.
        #
        # And it is added ONLY while training. A validation loss carrying an auxiliary
        # regularisation term is not a cross-entropy any more, and could not be compared
        # with the dense baseline's 1.472 — which is the entire point of the 13.8M
        # experiment. `last_ce` always holds the pure number.
        if self.training:
            aux = self.moe_aux_loss()
            if aux is not None:
                loss = loss + aux.to(loss.dtype)
        return logits, loss

    # ---- optimiser ----------------------------------------------------------------

    def configure_optimizers(self, weight_decay, lr, betas, device_type="cuda"):
        """Weight decay applies to matmul weights only. Decaying 1-D params (RMSNorm gains,
        and biases if we had any) actively hurts — they have no redundancy to regularise."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # fused AdamW keeps the whole optimiser step in one CUDA kernel; measurably faster.
        use_fused = device_type == "cuda"
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=use_fused)
        return opt, (sum(p.numel() for p in decay), sum(p.numel() for p in no_decay))

    def estimate_mfu(self, tokens_per_sec: float, peak_flops: float = 71e12) -> float:
        """Model FLOPs Utilisation. Default peak is the 3090's bf16 tensor-core number
        (~71 TFLOPs with fp32 accumulate). 30-45% is a healthy number for this setup."""
        n = self.num_params(non_embedding=True)
        cfg = self.cfg
        # 6N per token for fwd+bwd matmuls, plus 12*L*T*D for the attention score matmuls.
        flops_per_token = 6 * n + 12 * cfg.n_layers * cfg.max_seq_len * cfg.d_model
        return (flops_per_token * tokens_per_sec) / peak_flops

    def init_caches(self, batch_size, max_seq_len=None, dtype=torch.bfloat16, device="cuda"):
        max_seq_len = max_seq_len or self.cfg.max_seq_len
        return [
            KVCache(batch_size, self.cfg.n_kv_heads, max_seq_len, self.cfg.head_dim, dtype, device)
            for _ in range(self.cfg.n_layers)
        ]
