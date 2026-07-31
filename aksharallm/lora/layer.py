"""LoRALinear: the whole idea, in one module.

The observation
---------------
Fine-tuning changes a weight matrix from `W` to `W + dW`. Full fine-tuning stores and
trains all of `dW`, which is the same size as `W`. But `dW` is not an arbitrary matrix --
it is the accumulated result of adapting to one narrow task, and empirically it has very
low *rank*. The directions the model actually needs to move in number in the tens, not
the thousands.

So do not store `dW` at all. Store two thin matrices whose product is `dW`:

    dW  =  B @ A          A: (r, in_features)      B: (out_features, r)

with `r` maybe 8 or 16 against an `in_features` of 1024. For our 300M model's
`w1` (1024 -> 2752) that is 2.8M numbers replaced by 30k -- about 1%.

    y  =  x @ W.T  +  (alpha / r) * (x @ A.T) @ B.T

The frozen `W` term is the base model, untouched. The second term is the adapter.

Why B starts at zero
--------------------
`B = 0` makes the whole second term zero, so at step 0 the adapted model computes exactly
what the base model computed. Training starts from the base model rather than from a
randomly perturbed version of it -- which matters because a random perturbation of a
pretrained model is much worse than the model, and the first few hundred steps would be
spent undoing it.

`A` cannot also be zero: `B @ A` would be zero, its gradient would be zero, and nothing
would ever move. So one of them is random and the other is zero, and B is the one that
gets to be zero because it is the one applied last.

What alpha is for
-----------------
`alpha / r` scales the update. The point is that if you double the rank you do not want
to double the size of the update -- you want more directions, not a bigger step. Dividing
by `r` keeps the update's magnitude roughly constant as you sweep the rank, so the
learning rate you found at r=8 is still about right at r=32. Convention is alpha = 2*r.

Why the base can be a QuantLinear
---------------------------------
`self.base` is called, not indexed. It only has to be something that maps `x` to
`x @ W.T`, and `QuantLinear` is exactly that. Swapping a frozen 4-bit base underneath is
the entire difference between LoRA and QLoRA -- there is no separate class for it.

The gradient still has to flow *through* the base's dequantization to reach `x`, and for
QuantLinear's `torch` backend it does: dequantizing is a differentiable function of
buffers that require no gradient, and `F.linear` is differentiable in `x`. The Triton
backend is a raw kernel with no backward, which is why `lora.inject` pins the backend to
`torch` while training. See `docs/11-lora.md`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def base_in_out(base: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for anything we can wrap."""
    return int(base.in_features), int(base.out_features)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank correction."""

    def __init__(
        self,
        base: nn.Module,
        r: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")
        in_f, out_f = base_in_out(base)
        self.base = base
        self.r = r
        self.alpha = float(alpha if alpha is not None else 2 * r)
        self.scaling = self.alpha / r
        self.in_features, self.out_features = in_f, out_f

        # Dropout on the *input to the adapter only*. The base path is left alone, so this
        # regularises the thing being learned without perturbing the thing being preserved.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = _device_of(base)
        # fp32 adapters even when the base is bf16. They are ~1% of the parameters, so the
        # memory is irrelevant, and they are the only thing an optimiser state exists for --
        # keeping them in fp32 is the standard master-weight pattern and costs nothing here.
        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        self.reset_parameters()

        # Set False to compute the base model's output through this same module. DPO uses
        # it to get its reference model for free; see `disable_adapters`.
        self.adapter_enabled = True
        self.freeze_base()

    def reset_parameters(self):
        # Kaiming-uniform on A is what the paper uses; any zero-mean init with sane
        # variance works, because B=0 means the product starts at zero regardless.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        for p in self.base.parameters(recurse=True):
            p.requires_grad_(False)

    # ---- use ---------------------------------------------------------------------

    def delta_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """`dW = (alpha/r) * B @ A`, materialised. Only for merging and for tests --
        the forward pass never builds this, which is the entire memory argument."""
        dw = (self.lora_B @ self.lora_A) * self.scaling
        return dw if dtype is None else dw.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.adapter_enabled:
            return out
        # Two skinny matmuls, in this order. `(x @ A.T) @ B.T` costs
        # r*(in+out) multiply-adds per token; building `B @ A` first and doing one big
        # matmul would cost in*out, i.e. the full-rank price we are here to avoid.
        h = self.lora_dropout(x).to(self.lora_A.dtype)
        delta = (h @ self.lora_A.T) @ self.lora_B.T
        return out + (delta * self.scaling).to(out.dtype)

    def extra_repr(self) -> str:
        return (f"r={self.r}, alpha={self.alpha:g}, scaling={self.scaling:g}, "
                f"in={self.in_features}, out={self.out_features}")


def _device_of(mod: nn.Module) -> torch.device | None:
    for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
        if t is not None:
            return t.device
    return None


class disable_adapters:
    """Context manager: run the *base* model, adapters off.

    ```python
    with disable_adapters(model):
        ref_logits = model(x)      # the frozen base, exactly
    ```

    This is worth more than it looks. DPO needs a reference model to measure the policy
    against, and normally that means a second full copy of the weights in memory. With
    LoRA the base *is* the reference -- the adapter is the only difference between them --
    so switching a boolean gives you the reference model for zero extra bytes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.saved: list[tuple[LoRALinear, bool]] = []

    def __enter__(self):
        for mod in self.model.modules():
            if isinstance(mod, LoRALinear):
                self.saved.append((mod, mod.adapter_enabled))
                mod.adapter_enabled = False
        return self.model

    def __exit__(self, *exc):
        for mod, prev in self.saved:
            mod.adapter_enabled = prev
        self.saved.clear()
        return False
