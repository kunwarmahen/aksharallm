"""Muon: momentum, orthogonalised, for the matrices -- AdamW for everything else.

AdamW scales every coordinate of the gradient by its own running RMS. For a weight *matrix*
that is a strange choice: the gradient of a linear layer is a sum of a few outer products,
so it is close to low rank, and a step along it mostly pushes the few directions that are
already moving. Muon (Jordan et al., 2024) takes the momentum of a 2-D weight's gradient,
`M = U S V^T`, and steps along `U V^T` instead -- the same directions with every singular
value set to one. Each step then moves *every* direction of the matrix by the same amount,
which is the whole idea.

Computing an SVD per matrix per step would be slow, so `U V^T` is approximated by a few
Newton-Schulz iterations: a fixed quintic polynomial applied to `X X^T`, which pushes every
singular value towards 1 and only uses matmuls. Five iterations in bf16 are enough; the
singular values come out in roughly [0.7, 1.2] rather than exactly 1, which empirically does
not matter.

```
grad G ──► momentum M = mu*M + G ──► nesterov G + mu*M ──► Newton-Schulz(5) ≈ U V^T
                                                                      │
          W ◄── W*(1 - lr*wd) - lr * 0.2*sqrt(max(rows, cols)) * U V^T ◄┘
```

Which parameters get it, and why the rest do not:

* **2-D weights inside the blocks** (attention and FFN projections) -> Muon.
* **The token embedding** (tied to the output head here) -> AdamW. Its rows are looked up
  one token at a time; it is a table, not a linear map, and orthogonalising its update
  would move the rows of tokens that did not appear in the batch.
* **1-D gains** (RMSNorm), **3-D expert stacks** and the **router** (MoE) -> AdamW. A vector has no
  singular values; an expert stack is several matrices and is left to AdamW rather than
  guessing which axis to split.

**The `0.2 * sqrt(max(rows, cols))` factor** is from Moonlight ("Muon is Scalable for LLM
Training", 2025). An orthogonal matrix has entries of RMS `1/sqrt(max)`, so this scales the
update to RMS ~0.2 -- about what AdamW's update is. That lets Muon use **the same `lr`,
schedule and weight decay as the AdamW groups beside it**, so a config switches optimizer
with one key and nothing else, and the comparison against an AdamW run is one variable.

One optimizer object holds both kinds of group, so the trainer's `for g in param_groups:
g["lr"] = lr` and the checkpoint's `state_dict()` work unchanged. A checkpoint saved under
AdamW has a different number of groups and refuses to load into this one (and vice versa),
which is the right behaviour for a resume that changed optimizer.

Read with: docs/05-pretraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math

import torch

# Keller Jordan's quintic. Chosen to push singular values in (0, 1] towards ~1 as fast as
# possible in five steps, at the price of not converging to exactly 1.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately `U V^T` for `g = U S V^T`, using only matmuls.

    Runs in bf16 on CUDA (it is the bulk of the optimizer's cost) and float32 elsewhere.
    Works on the short side: a tall matrix is transposed first, so `X X^T` is the small
    Gram matrix.
    """
    assert g.dim() == 2, "Newton-Schulz orthogonalises a matrix"
    a, b, c = NS_COEFFS
    x = g.to(torch.bfloat16 if g.is_cuda else torch.float32)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    # Dividing by the Frobenius norm puts every singular value in (0, 1], where the
    # polynomial converges. The spectral norm would be tighter and is not cheap.
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if tall:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """One optimizer, two kinds of parameter group.

    A group with `muon=True` is updated by orthogonalised Nesterov momentum; any other group
    is plain AdamW. Every group carries `lr` and `weight_decay`; the trainer's schedule sets
    `lr` on all of them alike.
    """

    def __init__(self, params, lr: float = 1e-3, *, momentum: float = 0.95,
                 ns_steps: int = 5, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, betas=betas, eps=eps,
                        weight_decay=weight_decay, muon=False)
        super().__init__(params, defaults)
        for g in self.param_groups:
            if g["muon"] and any(p.dim() != 2 for p in g["params"]):
                raise ValueError("a muon group may only hold 2-D weights")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon if group["muon"] else self._adamw)(group)
        return loss

    def _muon(self, group):
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros_like(p)
            buf = st["momentum"]
            buf.mul_(mu).add_(p.grad)
            g = p.grad.add(buf, alpha=mu)          # Nesterov
            u = newton_schulz(g, group["ns_steps"])
            scale = 0.2 * math.sqrt(max(p.shape))  # match AdamW's update RMS (Moonlight)
            p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr * scale)

    def _adamw(self, group):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        b1, b2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            p.mul_(1 - lr * wd)
            p.addcdiv_(m, (v / bc2).sqrt_().add_(eps), value=-lr / bc1)


def split_for_muon(named_params) -> tuple[list, list, list]:
    """(muon, adamw-with-decay, adamw-without-decay) parameter lists.

    Muon takes the 2-D weights that are linear maps; the embedding table (and with it the
    tied output head) is a lookup and goes to AdamW with decay; 1-D and 3-D parameters go to
    AdamW, 1-D without decay as before.
    """
    muon, decay, plain = [], [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        # The MoE router is excluded too: its rows score experts against each other, and an
        # update that moves every row equally is not what a ranking wants. A Gated DeltaNet
        # block's `a_proj` / `b_proj` (d_model -> n_heads) are the same kind of thing: each row
        # is one head's decay or write gate, a scalar per head rather than a linear map, so
        # they stay with AdamW (added for the first hybrid trained with Muon, small-student-
        # hybrid; no earlier run had both).
        if (p.dim() == 2 and "emb" not in name and not name.startswith("lm_head")
                and "router" not in name and not name.endswith(("a_proj.weight", "b_proj.weight"))):
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            plain.append(p)
    return muon, decay, plain
