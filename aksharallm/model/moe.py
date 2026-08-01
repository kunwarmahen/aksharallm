"""Mixture of experts: more parameters than you compute with.

Every model in this project so far spends the same FLOPs on every token. A mixture of
experts breaks that link. The dense SwiGLU in each block is replaced by **N of them plus a
router**, and each token is sent to only the top-k. Total parameters grow by roughly N/k;
compute per token does not move.

The reason it is a good fit *here* is where our parameters already are:

|                  | 300M blended base | 13.8M TinyStories |
|------------------|-------------------|-------------------|
| embedding (tied) | 33.6M (11%)       | —                 |
| attention        | 62.9M (21%)       | —                 |
| **FFN**          | **202.9M (68%)**  | **7.1M (51%)**    |

Two thirds of the real model is the thing MoE replaces.

Two shapes, and they answer different questions
-----------------------------------------------
* **Matched *active* parameters** — each expert is `d_ff / k` wide, so top-k of them costs
  exactly what the dense FFN cost. Identical FLOPs per token, more total capacity. This is
  the honest test of what MoE actually claims, and it is what `configs/tiny-moe.yaml` runs
  against the dense val-1.472 baseline.
* **Sparse upcycling** — each expert is a *copy* of an already-trained full-width FFN.
  Active parameters go up by k, total by N. This reuses an expensive run instead of
  restarting it, which is the only affordable way to get an MoE out of the 300M given that
  the blend is 10B tokens and a 1.7B model wants ~34B. See :func:`upcycle_state_dict`.

The router is `d_model × N` per layer — 18k parameters at N=8 on the 13.8M model, 0.02% of
it. It is almost free and almost all of the difficulty.

Router collapse, which is what actually goes wrong
--------------------------------------------------
Nothing in the objective wants the experts to be *used*. A few of them win slightly early,
receive more gradient, get better, and win more; the rest never train. The model quietly
becomes a smaller dense one that carries dead weight, and **the loss curve looks fine while
it happens** — it is a little worse than it should be, which is indistinguishable from a
model that is simply a little worse.

So two things are non-negotiable here, and both are in :class:`Router`:

1. the **load-balancing auxiliary loss**, `alpha * N * sum_i f_i * P_i`, minimised when
   routing is uniform; and
2. **per-expert token counts, logged from step 1** — the number that makes collapse visible
   the moment it starts, plotted by the portal beside the loss.

Why the routing is a sort and not a mask
----------------------------------------
The obvious implementation runs every expert on every token and multiplies by a 0/1 mask.
It is correct, trivially, and it throws away the entire point: you pay N experts' compute
per token. The version here flattens the batch, sorts the (token, expert) assignments by
expert id, and does **one matmul per expert over a contiguous slice** of the tokens that
chose it. No token is dropped (there is no capacity factor), and the cost is k/N of the
masked version. Expect MFU to fall against the dense model anyway — a sort and N smaller
matmuls use the card less well than one big one, which is the honest cost of the trade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class MoEStats:
    """What the router did on the last forward, for logging. Detached, always.

    Collapse is visible here long before it is visible in the loss, so these are computed on
    every step rather than behind a flag: `max_share` climbing towards 1.0 while `min_share`
    goes to 0 *is* the failure, and it costs a `bincount` to see.
    """

    __slots__ = ("counts", "n_tokens", "aux_loss", "z_loss")

    def __init__(self, counts: torch.Tensor, n_tokens: int, aux_loss: float, z_loss: float):
        self.counts = counts
        self.n_tokens = n_tokens
        self.aux_loss = aux_loss
        self.z_loss = z_loss

    @property
    def shares(self) -> torch.Tensor:
        """Fraction of routed assignments each expert received. Sums to 1."""
        total = self.counts.sum().clamp(min=1)
        return self.counts / total

    def as_dict(self) -> dict:
        shares = self.shares.tolist()
        n = max(1, len(shares))
        return {
            "shares": [round(s, 5) for s in shares],
            "max_share": round(max(shares), 5) if shares else 0.0,
            "min_share": round(min(shares), 5) if shares else 0.0,
            # 1.0 = perfectly uniform, 1/N = one expert takes everything. The single number
            # to watch; `n * max_share` would say the same thing upside down.
            "balance": round((1.0 / n) / max(max(shares), 1e-9), 5) if shares else 0.0,
            "dead": sum(1 for s in shares if s < 0.01 / n),
            "aux_loss": round(self.aux_loss, 6),
            "z_loss": round(self.z_loss, 6),
            "n_tokens": self.n_tokens,
        }


class Router(nn.Module):
    """Chooses k experts per token, and pays for the privilege of not being uniform.

    One `nn.Linear(d_model, n_experts, bias=False)`. Bias is left out deliberately: a bias
    is a per-expert constant that shifts the argmax and is exactly the thing the balancing
    loss is trying to control, so it gives the model a cheap way to collapse that costs it
    almost nothing.

    The routing weights are **renormalised over the chosen k**. That matters beyond taste:
    it is what makes sparse upcycling an identity at initialisation. If every expert is a
    copy of the same trained FFN and the k weights sum to 1, the MoE block computes exactly
    what the dense block computed — so training starts from the trained model rather than
    from a perturbation of it, which is the same lesson `docs/11` records about LoRA's
    `B = 0`.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 aux_alpha: float = 0.01, z_alpha: float = 1e-3):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in 1..{n_experts}, got {top_k}")
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_alpha = aux_alpha
        self.z_alpha = z_alpha
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x_flat: torch.Tensor):
        """`x_flat`: (N, d_model). Returns (weights (N,k), indices (N,k), aux, stats)."""
        logits = self.gate(x_flat)                      # (N, E)
        # float32 for the softmax: the balancing signal is a difference between small
        # probabilities, and in bf16 the ones that matter round to each other.
        probs = F.softmax(logits.float(), dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        with torch.no_grad():
            counts = torch.bincount(topi.reshape(-1),
                                    minlength=self.n_experts).to(torch.float32)

        aux = self._balance_loss(probs, topi, counts)
        z = self._z_loss(logits)
        stats = MoEStats(counts.detach().cpu(), x_flat.shape[0],
                         float(aux.detach()), float(z.detach()))
        return topw.to(x_flat.dtype), topi, aux + z, stats

    def _balance_loss(self, probs: torch.Tensor, topi: torch.Tensor,
                      counts: torch.Tensor) -> torch.Tensor:
        """`alpha * N * sum_i f_i * P_i` — the Switch Transformer loss.

        `f_i` is the *fraction of tokens dispatched* to expert i and is discrete, so no
        gradient flows through it; `P_i` is the mean router probability for expert i and is
        where the gradient lives. The product is minimised (at `1/N` per term, `1` in total)
        when both are uniform, and the `N` factor makes the value scale-free so `alpha`
        means the same thing at 4 experts and at 64.

        Using `f` alone would have no gradient at all; using `P` alone lets the router keep
        a flat *average* while still sending every individual token to one expert.
        """
        if self.aux_alpha == 0:
            return probs.new_zeros(())
        f = counts / counts.sum().clamp(min=1)          # (E,) no grad, by construction
        p = probs.mean(dim=0)                           # (E,) carries the gradient
        return self.aux_alpha * self.n_experts * torch.sum(f * p)

    def _z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Keeps the router's logits from drifting large.

        Softmax is shift-invariant, so nothing in the main loss stops the logits growing
        without bound; once they do, one expert's probability saturates at 1 and the router
        stops being trainable at all. Penalising `logsumexp(logits)^2` pins the scale. Small
        (1e-3) — it is a leash, not an objective.
        """
        if self.z_alpha == 0:
            return logits.new_zeros(())
        return self.z_alpha * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


class MoEFeedForward(nn.Module):
    """N SwiGLU experts, top-k of them per token.

    The expert weights are three stacked tensors — `w1/w3: (E, d_model, d_ff)` and
    `w2: (E, d_ff, d_model)` — rather than a `ModuleList` of `SwiGLU`s. That is what makes
    the sorted dispatch below one `bmm`-shaped matmul per expert instead of a Python loop
    over modules, and it keeps the state dict flat enough for upcycling to be a slice
    assignment.

    The cost of that choice, stated because it is real: these are plain `nn.Parameter`s and
    not `nn.Linear`s, so `quant/` and `lora/` — which both walk the module tree looking for
    `nn.Linear` — do not see the experts. Quantizing or adapting an MoE model is a follow-up
    with its own decision (the **router must never** be quantized: a wrong route is a
    different expert, not a slightly wrong number), and until then the code refuses rather
    than silently doing nothing.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d_ff = cfg.moe_expert_d_ff or cfg.d_ff
        self.n_experts = cfg.n_experts
        self.top_k = cfg.moe_top_k
        self.d_model = cfg.d_model
        self.d_ff = d_ff
        self.router = Router(cfg.d_model, cfg.n_experts, cfg.moe_top_k,
                             aux_alpha=cfg.moe_aux_alpha, z_alpha=cfg.moe_z_alpha)
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, d_ff, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.stats: MoEStats | None = None
        self.aux_loss: torch.Tensor | None = None
        self.reset_parameters(cfg)

    def reset_parameters(self, cfg: ModelConfig):
        for w in (self.w1, self.w3):
            nn.init.normal_(w, mean=0.0, std=0.02)
        # Same residual-scaled init the dense `w2` gets in `Transformer.__init__`: the
        # down-projection writes into the residual stream, and without the 1/sqrt(2L) the
        # variance grows with depth.
        nn.init.normal_(self.w2, mean=0.0,
                        std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)                                   # (N, D)
        weights, indices, aux, stats = self.router(flat)
        self.aux_loss, self.stats = aux, stats

        n_tokens = flat.shape[0]
        # Every (token, expert) pair, flattened. Sorting by expert id turns "which tokens
        # chose expert e" into a contiguous slice, which is the whole trick.
        flat_expert = indices.reshape(-1)                          # (N*k,)
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        token_of = order // self.top_k                             # which row of `flat`
        weight_of = weights.reshape(-1)[order].unsqueeze(-1)       # (N*k, 1)

        counts = torch.bincount(sorted_expert, minlength=self.n_experts)
        offsets = torch.cumsum(counts, dim=0)

        out = torch.zeros_like(flat)
        start = 0
        # A Python loop over E slices, not over tokens: E is 8, and each iteration is one
        # real matmul over every token that chose that expert.
        for e in range(self.n_experts):
            end = int(offsets[e])
            if end == start:
                continue                       # an expert nobody chose this batch
            rows = token_of[start:end]
            xe = flat.index_select(0, rows)                        # (n_e, D)
            h = F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])        # (n_e, d_ff)
            ye = (h @ self.w2[e]) * weight_of[start:end]           # (n_e, D)
            out.index_add_(0, rows, ye.to(out.dtype))
            start = end

        return self.drop(out.view(B, T, D))

    # ---- reporting ----------------------------------------------------------------------
    def n_active_params(self) -> int:
        """Parameters this layer actually uses per token — `top_k` experts plus the router."""
        per_expert = 2 * self.d_model * self.d_ff + self.d_ff * self.d_model
        return self.top_k * per_expert + self.d_model * self.n_experts

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------------------
# sparse upcycling
# --------------------------------------------------------------------------------------

def upcycle_state_dict(dense: dict, n_experts: int, jitter: float = 0.0,
                       generator: torch.Generator | None = None) -> dict:
    """A dense checkpoint's weights, rearranged so an MoE model can load them.

    Each block's `ffn.w1/w2/w3` becomes `ffn.w1/w2/w3` of shape `(E, ...)` holding E copies
    of the trained matrix, and `ffn.router.gate.weight` is initialised to **zeros**.

    Those two choices together are what make this an identity at step 0. Zero gate logits
    give a uniform softmax; the top-k weights are renormalised to sum to 1; every expert
    computes the same function; so the block's output is exactly the dense block's output.
    Training then differentiates the experts from a model that already works, rather than
    from a perturbation of one — the same reason LoRA initialises `B = 0` (docs/11).

    `jitter` breaks the tie if you want it broken: identical experts receive identical
    gradients *only* while the router is uniform, and it is not uniform after one step, so
    symmetry breaks on its own. A small jitter (1e-2 relative) makes that happen sooner at
    the cost of no longer being exactly the dense model on step 0. Default off, because
    "exactly the dense model" is a property worth being able to assert in a test.
    """
    out = {}
    for key, value in dense.items():
        if key.endswith("ffn.w1.weight") or key.endswith("ffn.w3.weight"):
            # nn.Linear stores (out_features, in_features) = (d_ff, d_model); the expert
            # tensors are (E, d_model, d_ff), so the copy is transposed.
            stacked = value.t().unsqueeze(0).repeat(n_experts, 1, 1).contiguous()
            out[key[: -len(".weight")]] = _jitter(stacked, jitter, generator)
        elif key.endswith("ffn.w2.weight"):
            stacked = value.t().unsqueeze(0).repeat(n_experts, 1, 1).contiguous()
            out[key[: -len(".weight")]] = _jitter(stacked, jitter, generator)
            prefix = key[: -len("w2.weight")]
            # `w2` is Linear(d_ff -> d_model), so its weight is (d_model, d_ff) and the
            # router's input width is shape[0], not shape[1]. Getting this backwards is a
            # loud shape error rather than a silent one, which is the only reason it is
            # worth a comment instead of a test of its own.
            out[f"{prefix}router.gate.weight"] = torch.zeros(
                n_experts, value.shape[0], dtype=value.dtype)
        else:
            out[key] = value
    return out


def _jitter(t: torch.Tensor, scale: float, generator: torch.Generator | None) -> torch.Tensor:
    if not scale:
        return t
    noise = torch.randn(t.shape, dtype=t.dtype, generator=generator)
    return t * (1 + scale * noise)


def moe_stats(model: nn.Module) -> dict | None:
    """Routing stats for the whole model, averaged over its MoE layers.

    Returns None for a dense model, which is how every caller — the trainer's log line, the
    run's jsonl, the portal — decides whether there is anything to report without having to
    know what kind of model it is holding.
    """
    layers = [m for m in model.modules() if isinstance(m, MoEFeedForward) and m.stats]
    if not layers:
        return None
    counts = torch.stack([m.stats.counts for m in layers]).sum(dim=0)
    total = counts.sum().clamp(min=1)
    shares = (counts / total).tolist()
    n = len(shares)
    per_layer = [m.stats.as_dict()["balance"] for m in layers]
    return {
        "n_experts": n,
        "n_layers": len(layers),
        "shares": [round(s, 5) for s in shares],
        "max_share": round(max(shares), 5),
        "min_share": round(min(shares), 5),
        "balance": round((1.0 / n) / max(max(shares), 1e-9), 5),
        # A model-wide average can look healthy while one layer has collapsed, so the worst
        # layer is carried separately. It is the one that would show up as "the model is a
        # bit worse than it should be" and nothing else.
        "worst_layer_balance": round(min(per_layer), 5),
        "dead": sum(1 for s in shares if s < 0.01 / n),
        "aux_loss": round(sum(m.stats.aux_loss for m in layers) / len(layers), 6),
    }
