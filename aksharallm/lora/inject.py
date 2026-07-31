"""Deciding *which* layers get an adapter, and wiring them in.

Choosing targets is the one LoRA hyperparameter that reliably matters more than the rank.
The original paper adapted only the attention query and value projections. The QLoRA
paper's ablation found that at a fixed parameter budget it is better to adapt *every*
linear layer at a low rank than a few at a high rank -- so `all-linear` is the default
here, and `qv` is kept so you can reproduce the original setting and see the gap yourself.

Our transformer's linear layers, by dotted name:

    blocks.<i>.attn.wq / wk / wv / wo     attention
    blocks.<i>.ffn.w1 / w3 / w2           SwiGLU gate / up / down
    lm_head                               output projection

`lm_head` is excluded from every preset, for the same reason quantization skips it: with
`tie_embeddings` it *is* the embedding table, so adapting it silently adapts the input
lookup too. That is a different and much less predictable intervention than adapting a
projection, and it is not what anyone means by "LoRA on the head".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from ..quant.qlinear import QuantLinear
from .layer import LoRALinear

#: name -> the suffixes it expands to. Order is the order they appear in a block.
PRESETS: dict[str, tuple[str, ...]] = {
    "qv": ("wq", "wv"),
    "attn": ("wq", "wk", "wv", "wo"),
    "ffn": ("w1", "w2", "w3"),
    "all-linear": ("wq", "wk", "wv", "wo", "w1", "w2", "w3"),
}

PRESET_BLURBS = {
    "qv": "query and value only — the original LoRA paper's setting, fewest parameters",
    "attn": "all four attention projections; leaves the SwiGLU alone",
    "ffn": "the SwiGLU only — two thirds of the weights live here",
    "all-linear": "every projection in every block — the QLoRA paper's recommendation",
}


def resolve_targets(spec: str) -> tuple[str, ...]:
    """A preset name, or a comma-separated list of module suffixes."""
    key = spec.strip()
    if key in PRESETS:
        return PRESETS[key]
    parts = tuple(p.strip() for p in key.split(",") if p.strip())
    if not parts:
        raise ValueError(f"no targets in {spec!r}")
    return parts


@dataclass
class LoRAConfig:
    """Everything that defines an adapter. Saved alongside it, so loading needs no flags."""

    r: int = 8
    alpha: float | None = None  # None -> 2*r
    dropout: float = 0.0
    targets: str = "all-linear"

    @property
    def effective_alpha(self) -> float:
        return float(self.alpha if self.alpha is not None else 2 * self.r)

    def as_dict(self) -> dict:
        return {"r": self.r, "alpha": self.effective_alpha, "dropout": self.dropout,
                "targets": self.targets}

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(r=int(d["r"]), alpha=float(d["alpha"]), dropout=float(d.get("dropout", 0.0)),
                   targets=str(d.get("targets", "all-linear")))


@dataclass
class LoRAReport:
    """What injection did. The trainable/total ratio is the headline number."""

    config: LoRAConfig
    adapted: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    trainable: int = 0
    frozen: int = 0
    quantized_base: bool = False

    @property
    def total(self) -> int:
        return self.trainable + self.frozen

    @property
    def fraction(self) -> float:
        return self.trainable / max(1, self.total)

    def as_dict(self) -> dict:
        return {"config": self.config.as_dict(), "adapted": self.adapted,
                "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
                "trainable": self.trainable, "frozen": self.frozen,
                "total": self.total, "fraction": self.fraction,
                "quantized_base": self.quantized_base}

    def summary(self) -> str:
        c = self.config
        lines = [
            f"targets      {c.targets} — {len(self.adapted)} layers adapted",
            f"rank         r={c.r}, alpha={c.effective_alpha:g} "
            f"(scaling {c.effective_alpha / c.r:g}), dropout {c.dropout}",
            f"base         {'4/8-bit QuantLinear (QLoRA)' if self.quantized_base else 'float'}"
            f", frozen",
            f"trainable    {self.trainable:,} of {self.total:,} "
            f"({100 * self.fraction:.2f}%)",
        ]
        if self.skipped:
            shown = ", ".join(sorted({n.split(".")[-1] for n, _ in self.skipped}))
            lines.append(f"  not adapted: {shown} ({self.skipped[0][1]})")
        return "\n".join(lines)


def _set_module(model: nn.Module, name: str, new: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def adaptable_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Every layer an adapter could be attached to: float Linears and QuantLinears alike."""
    return {n: m for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, QuantLinear))}


def apply_lora(model: nn.Module, config: LoRAConfig) -> LoRAReport:
    """Wrap the targeted layers in `LoRALinear` and freeze everything else, in place.

    Freezing is done first and unconditionally: the adapters are then the only parameters
    with `requires_grad`, so `configure_optimizers` picks up exactly them and nothing else
    can drift. Getting this backwards -- adding adapters and forgetting to freeze -- gives
    a run that trains fine, produces a tiny adapter file, and has quietly also moved the
    base model that the adapter file does not contain.
    """
    suffixes = resolve_targets(config.targets)
    report = LoRAReport(config=config)

    for p in model.parameters():
        p.requires_grad_(False)

    for name, mod in adaptable_layers(model).items():
        leaf = name.split(".")[-1]
        if name.endswith("lm_head"):
            report.skipped.append((name, "the head is tied to the embedding table"))
            continue
        if leaf not in suffixes:
            report.skipped.append((name, f"not in targets '{config.targets}'"))
            continue
        if isinstance(mod, nn.Linear) and mod.bias is not None:
            report.skipped.append((name, "has a bias; LoRA here assumes bias=False"))
            continue
        _set_module(model, name, LoRALinear(
            mod, r=config.r, alpha=config.effective_alpha, dropout=config.dropout))
        report.adapted.append(name)
        report.quantized_base = report.quantized_base or isinstance(mod, QuantLinear)

    report.trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if report.quantized_base:
        # Frozen `parameters()` misses a quantized base entirely -- its weights are
        # buffers, not parameters -- so count the bytes back in or the "1% trainable"
        # headline is computed against a model that appears to be almost nothing.
        report.frozen += sum(
            m.qweight.numel() * (2 if m.scheme.packed else 1)
            for m in model.modules() if isinstance(m, QuantLinear))
    return report


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def set_adapters_enabled(model: nn.Module, enabled: bool) -> int:
    n = 0
    for mod in lora_layers(model).values():
        mod.adapter_enabled = enabled
        n += 1
    return n


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def prepare_for_training(model: nn.Module) -> str | None:
    """Make a QLoRA base safe to backpropagate through. Returns a note, or None.

    The fused Triton kernel is a bare `triton.jit` call with no `autograd.Function` behind
    it, so it has no backward pass: a QLoRA step that ran through it would either error or
    -- worse -- silently detach the base and train an adapter against a constant. The
    torch backend dequantizes with ordinary differentiable ops, so gradients reach `x`
    (and therefore the adapters below this layer) correctly.

    Inference is free to switch back; `QuantLinear.backend` is a class attribute, so this
    is one assignment for the whole model.
    """
    if not any(isinstance(m, QuantLinear) for m in model.modules()):
        return None
    if QuantLinear.backend == "torch":
        return None
    QuantLinear.backend = "torch"
    return ("quantized base: pinned QuantLinear.backend='torch' for training — the fused "
            "Triton kernel has no backward pass.")
