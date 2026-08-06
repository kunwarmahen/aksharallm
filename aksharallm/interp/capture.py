"""Looking inside the model while it runs: the residual stream, and what attention attends to.

Everything else in this repo asks the model *questions* — what token comes next, what loss,
what score. This package opens it up instead: which layer decided the answer, which earlier
token the decision leaned on, and what the model believed halfway through.

Two things are captured, and they are captured differently for a reason.

**The residual stream** is easy: it is the value flowing between blocks, so a forward hook on
each block gets it for free. It is also the single most useful thing to look at, because in a
pre-norm transformer every layer *adds* to it — the stream is a running total of what the
model has worked out so far, which is what makes the logit lens (`lens.py`) meaningful.

**Attention weights are not stored anywhere.** `F.scaled_dot_product_attention` computes
`softmax(QK^T/sqrt(d))V` in one fused kernel and never materialises the matrix in the middle —
that is precisely why it is fast and memory-light. So they are *recomputed* here from the same
inputs: hook the block's attention norm, take its output, apply the layer's own `wq`/`wk`,
rotate by the same RoPE angles, and do the matmul. `tests/test_interp.py` asserts that the
result, multiplied by V, reproduces the layer's real output — because a plausible-looking
attention map that does not match what the model actually did is worse than none.

Nothing here changes the model. No flags, no `capture=True` parameter threaded through
`forward`, no branch in the hot path: hooks are attached, one forward runs, hooks come off.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..model.transformer import apply_rope


@dataclass
class Capture:
    """One forward pass, opened up.

    `residual[i]` is the stream *after* block `i`, so `residual[-1]` is what the final norm
    and the output head see. `embedding` is the stream before any block ran — layer 0's input,
    and the honest starting point for a logit lens.
    """

    tokens: list[int]
    embedding: torch.Tensor | None = None                 #: (T, d_model)
    residual: list[torch.Tensor] = field(default_factory=list)
    attn_out: list[torch.Tensor] = field(default_factory=list)
    attn_in: list[torch.Tensor] = field(default_factory=list)   #: the normed input to attention
    logits: torch.Tensor | None = None                    #: (T, vocab)

    @property
    def n_layers(self) -> int:
        return len(self.residual)


@contextmanager
def hooks_on(model, cap: Capture):
    """Attach the capture hooks for the duration of one forward pass, then remove them.

    A context manager rather than a flag on the model: an interpretability run that leaves
    hooks attached quietly doubles the memory of every later forward, including a training
    one, and nothing would say so.
    """
    handles = []

    # Everything is kept at full batch shape `(B, T, d)`. `run` slices row 0 for the
    # single-prompt tools; `collect_activations` wants all of it, and an earlier version that
    # kept only row 0 quietly threw away 31 of every 32 sequences — which showed up as an
    # out-of-memory error, because reaching a target number of activations then took 32x the
    # forward passes.
    def keep_residual(_module, _inputs, output):
        cap.residual.append(output.detach())

    def keep_attn_in(_module, inputs):
        cap.attn_in.append(inputs[0].detach())

    def keep_attn_out(_module, _inputs, output):
        cap.attn_out.append(output.detach())

    for block in model.blocks:
        handles.append(block.register_forward_hook(keep_residual))
        handles.append(block.attn.register_forward_pre_hook(keep_attn_in))
        handles.append(block.attn.register_forward_hook(keep_attn_out))
    handles.append(model.tok_emb.register_forward_hook(
        lambda _m, _i, out: setattr(cap, "embedding", out.detach())))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run(model, token_ids: list[int], device: str = "cpu") -> Capture:
    """One forward pass over `token_ids`, with everything worth looking at kept."""
    model.eval()
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    cap = Capture(tokens=list(token_ids))
    with hooks_on(model, cap):
        logits, _ = model(ids, full_logits=True)
    # One prompt, so drop the batch axis here and let every reader index by position.
    cap.residual = [r[0] for r in cap.residual]
    cap.attn_in = [a[0] for a in cap.attn_in]
    cap.attn_out = [a[0] for a in cap.attn_out]
    cap.embedding = cap.embedding[0] if cap.embedding is not None else None
    cap.logits = logits.detach()[0]
    return cap


@torch.no_grad()
def attention_maps(model, cap: Capture, layer: int) -> torch.Tensor:
    """`(n_heads, T, T)` attention weights for one layer, recomputed from its own inputs.

    Row `i` is what position `i` looked at: it sums to 1 and is zero above the diagonal,
    because a token cannot attend to its future. Read a row, not a column.

    Recomputed rather than captured — see the module docstring. The arithmetic is the same
    three lines the model runs, kept side by side with it on purpose: `Q K^T / sqrt(d)`, the
    causal mask, `softmax`.
    """
    attn = model.blocks[layer].attn
    x = cap.attn_in[layer]                                # (T, d_model), already normed
    T = x.shape[0]
    q = attn.wq(x).view(T, attn.n_heads, attn.head_dim).transpose(0, 1)[None]
    k = attn.wk(x).view(T, attn.n_kv_heads, attn.head_dim).transpose(0, 1)[None]
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    q, k = apply_rope(q, cos, sin)[0], apply_rope(k, cos, sin)[0]
    if attn.n_rep > 1:                                    # grouped-query: heads share K
        k = k.repeat_interleave(attn.n_rep, dim=0)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(attn.head_dim)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), 1)
    scores = scores.masked_fill(causal, float("-inf"))
    return F.softmax(scores, dim=-1)


@torch.no_grad()
def attention_values(model, cap: Capture, layer: int) -> torch.Tensor:
    """The V of the same layer, `(n_heads, T, head_dim)` — only needed to *check* the map."""
    attn = model.blocks[layer].attn
    x = cap.attn_in[layer]
    T = x.shape[0]
    v = attn.wv(x).view(T, attn.n_kv_heads, attn.head_dim).transpose(0, 1)
    if attn.n_rep > 1:
        v = v.repeat_interleave(attn.n_rep, dim=0)
    return v


def attention_summary(weights: torch.Tensor, tokens: list[str],
                      top: int = 3) -> list[dict]:
    """Per head: how far back it looks, and what the last token attended to.

    Two numbers do most of the work when scanning 16 heads. **Distance** — the mean gap
    between a query and the positions it weights — separates a head reading the token next to
    it from one reaching back to the start. **Self weight** says how much of a head is simply
    "attend to myself", which many heads mostly are, and which is worth knowing before reading
    anything into their maps.
    """
    out = []
    T = weights.shape[-1]
    # On the CPU throughout: this is a handful of reductions over a (T, T) matrix for
    # presentation, and building the position vector on the *default* device while the
    # weights sat on the card is exactly the mismatch that took the Interp tab's attention
    # view down with a 500.
    weights = weights.detach().float().cpu()
    pos = torch.arange(T, dtype=torch.float)
    for h in range(weights.shape[0]):
        w = weights[h]
        gaps = (pos[:, None] - pos[None, :]).clamp(min=0)
        distance = float((w * gaps).sum() / max(T, 1))
        self_weight = float(w.diagonal().mean())
        last = w[-1]
        idx = torch.topk(last, min(top, T)).indices.tolist()
        out.append({
            "head": h,
            "distance": distance,
            "self_weight": self_weight,
            "attends_to": [{"pos": int(i), "token": tokens[int(i)],
                            "weight": float(last[int(i)])} for i in idx],
        })
    return out
