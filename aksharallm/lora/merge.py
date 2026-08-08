"""Folding an adapter back into the weights.

`W' = W + (alpha/r) * B @ A` and the adapter is gone -- one weight matrix again, so
inference costs exactly what the base model cost. Merging is the right move once you have
picked a specialisation and want to ship it; keeping it separate is the right move while
you still want to swap between several.

    merged, unmerged:  same output, different shape of computation
    merged:            one matmul, no adapter, cannot be turned off
    unmerged:          two matmuls, ~1% more parameters, hot-swappable

Merging into a quantized base is a different matter
---------------------------------------------------
The adapter was trained against a *specific* set of 4-bit weights. `W_q + BA` is not
representable in 4 bits, so you have two options and neither is free:

  1. Merge into float. Dequantize, add, keep it. Numerically exact, and the result is a
     bf16 checkpoint -- you have given back the memory that quantizing bought.
  2. Merge and re-quantize. You get a small model again, but re-quantizing `W_q + BA`
     rounds it *afresh*: the result is not the model you evaluated, and the difference is
     of the same order as the adapter you just trained.

`merge_lora` does (1) and says so. (2) is available by running the quantize CLI on the
merged output, deliberately as a second explicit step rather than a flag, because the
number you measured before it is not the number you have after it.

The honest third option, and usually the best one, is not to merge at all: keep the 4-bit
base and the 11 MB adapter side by side, which is how QLoRA is normally deployed.

Read with: docs/12-lora.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..quant.qlinear import QuantLinear
from .inject import _set_module, lora_layers
from .layer import LoRALinear


def merge_lora(model: nn.Module, dtype: torch.dtype = torch.float32) -> dict:
    """Fold every adapter into its base, in place. Returns a small report.

    A `QuantLinear` base becomes a float `nn.Linear`, because the sum is not
    representable in 4 bits — see the module docstring.
    """
    merged, dequantized = [], []
    for name, mod in list(lora_layers(model).items()):
        base = mod.base
        delta = mod.delta_weight(torch.float32)
        if isinstance(base, QuantLinear):
            w = base.dequantize_weight(torch.float32)
            dequantized.append(name)
        else:
            w = base.weight.data.to(torch.float32)
        lin = nn.Linear(mod.in_features, mod.out_features, bias=False,
                        device=w.device, dtype=dtype)
        lin.weight.data.copy_((w + delta).to(dtype))
        _set_module(model, name, lin)
        merged.append(name)

    for p in model.parameters():
        p.requires_grad_(True)
    return {
        "merged": merged,
        "dequantized": dequantized,
        "note": (
            f"{len(dequantized)} quantized layers were dequantized to float — the sum of "
            "4-bit weights and a float adapter is not a 4-bit tensor. Re-quantizing the "
            "result is a fresh rounding, so its perplexity is not the one you measured."
        ) if dequantized else "",
    }


def unmerged_equivalent(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run `x` through the model as it stands. Exists so tests can compare a merged model
    against its unmerged self on identical input without duplicating the plumbing."""
    with torch.no_grad():
        out, _ = model(x)
    return out


def merge_into_checkpoint(model: nn.Module, ckpt: dict, out_path, source: str | None = None):
    """Save a merged model as an ordinary checkpoint the rest of the project can load.

    Deliberately *not* carrying the source checkpoint's `quant` key: after merging there
    are no QuantLinears left, and a `quant` block would make `build_from_checkpoint` try
    to rebuild shapes that no longer exist.
    """
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_config": ckpt["model_config"],
        "config": ckpt.get("config"),
        "step": ckpt.get("step"),
        "best_val": ckpt.get("best_val"),
        "merged_from": source,
    }
    torch.save(payload, out_path)
    return out_path
