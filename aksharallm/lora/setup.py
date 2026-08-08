"""The shared "load a base and make it trainable with adapters" path.

`train/sft.py` and `train/dpo.py` both need to do the same four things in the same order,
and the order is not arbitrary:

    1. build the float model and load its weights
    2. quantize it, if this is QLoRA          <- must be after (1): you cannot load float
                                                 weights into a QuantLinear's buffers
    3. move it to the device
    4. inject adapters and freeze everything else   <- must be after (2): the adapters
                                                       wrap whatever the base ended up
                                                       being, and freezing has to come
                                                       after the base is final

Doing (4) before (2) is the interesting mistake: quantizing would replace the very
`nn.Linear` objects the adapters wrapped, and you would end up training adapters attached
to layers that are no longer in the model. It fails silently -- the loss goes down, and
the adapter does nothing when reloaded.

Read with: docs/12-lora.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..quant.convert import (
    apply_quant_metadata,
    is_quantized_checkpoint,
    quantize_model,
)
from ..quant.qtensor import QuantScheme
from .adapter import base_identity, load_adapter_file, save_adapter
from .inject import LoRAConfig, apply_lora, prepare_for_training


def add_lora_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The LoRA/QLoRA flags, shared by every trainer that supports them."""
    g = ap.add_argument_group("LoRA / QLoRA")
    g.add_argument("--lora", action="store_true",
                   help="train low-rank adapters instead of the weights")
    g.add_argument("--lora-r", type=int, default=8,
                   help="rank. 8 is a good default; sweep 4-32 and look at the val curve")
    g.add_argument("--lora-alpha", type=float, default=None,
                   help="update scaling is alpha/r (default: 2*r, so scaling=2)")
    g.add_argument("--lora-dropout", type=float, default=0.05,
                   help="dropout on the adapter's input only; the base path is untouched")
    g.add_argument("--lora-targets", default="all-linear",
                   help="preset (all-linear | attn | qv | ffn) or a comma-separated list "
                        "of module suffixes such as 'wq,wv'")
    g.add_argument("--qlora", action="store_true",
                   help="hold the frozen base in 4 bits. Implies --lora. Ignored (with a "
                        "note) if the base checkpoint is already quantized.")
    g.add_argument("--qlora-dtype", default="nf4", choices=("nf4", "int"),
                   help="the 4-bit grid for the frozen base")
    g.add_argument("--qlora-group", type=int, default=64)
    g.add_argument("--qlora-double-quant", action="store_true",
                   help="also quantize the base's scales")
    g.add_argument("--adapter", default=None,
                   help="continue training an existing adapter file instead of starting "
                        "from zero. Its rank and targets win over the flags.")
    return ap


def lora_config_from_args(args) -> LoRAConfig:
    return LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                      targets=args.lora_targets)


def wants_lora(args) -> bool:
    return bool(getattr(args, "lora", False) or getattr(args, "qlora", False)
                or getattr(args, "adapter", None))


def prepare_base(model, ckpt: dict, args, device: str) -> list[str]:
    """Step 2 of the list above: quantize the frozen base, or note that it already is.

    Returns human-readable notes for the run header. The model is modified in place and
    is *not* moved to `device` here -- the caller does that, because quantizing on the CPU
    and then moving is both faster and avoids briefly holding a float copy in VRAM.
    """
    notes = []
    if is_quantized_checkpoint(ckpt):
        # The shapes were already rebuilt before load_state_dict by the caller; just say so.
        notes.append(f"base is already quantized ({ckpt['quant'].get('label')}) — "
                     "using it as the frozen base")
        if getattr(args, "qlora", False):
            notes.append("--qlora ignored: quantizing an already-quantized base would "
                         "compound the error")
        return notes

    if not getattr(args, "qlora", False):
        return notes

    scheme = QuantScheme(
        bits=4, group_size=args.qlora_group, method="rtn",
        dtype=args.qlora_dtype,
        sym=False if args.qlora_dtype == "int" else False,
        double_quant=args.qlora_double_quant,
    )
    report = quantize_model(model, scheme)
    t = report.totals()
    notes.append(
        f"quantized the frozen base to {scheme.label()}: "
        f"{t['model_float_bytes'] / 1e6:.0f} MB -> {t['model_quant_bytes'] / 1e6:.0f} MB "
        f"({t['model_ratio']:.2f}x), {len(report.quantized)} layers")
    if report.skipped:
        notes.append(f"  {len(report.skipped)} layers left in float "
                     f"({report.skipped[0].name.split('.')[-1]}: "
                     f"{report.skipped[0].skipped.split('—')[0].strip()})")
    return notes


def rebuild_quantized_shapes(model, ckpt: dict):
    """Call before `load_state_dict` when the base checkpoint is itself quantized."""
    if is_quantized_checkpoint(ckpt):
        apply_quant_metadata(model, ckpt["quant"])


def attach(model, args, device: str) -> tuple[LoRAConfig, object, list[str]]:
    """Step 4: inject the adapters. Returns (config, report, notes)."""
    notes = []
    if args.adapter:
        payload = load_adapter_file(args.adapter)
        config = LoRAConfig.from_dict(payload["lora_config"])
        notes.append(f"continuing adapter {args.adapter} "
                     f"(r={config.r}, targets={config.targets}) — its config wins over "
                     f"the --lora-* flags")
    else:
        payload = None
        config = lora_config_from_args(args)

    report = apply_lora(model, config)
    if payload is not None:
        from .adapter import attach_adapter  # local: avoids a cycle at import time

        # Re-attaching over the injection above is wasteful by a few milliseconds and
        # keeps exactly one code path for "load an adapter into a model".
        attach_adapter(model, payload, ckpt=None, strict=False)

    note = prepare_for_training(model)
    if note:
        notes.append(note)
    return config, report, notes


def save(path, model, config: LoRAConfig, ckpt: dict, base_path: str,
         report=None, training: dict | None = None) -> Path:
    """Write the adapter, with the base's identity and this run's numbers attached."""
    return save_adapter(path, model, config, base_identity(ckpt, base_path),
                        report=report, extra={"training": training} if training else None)


def describe_memory(model) -> dict:
    """Bytes actually held, split into the parts that matter for the memory argument.

    `optimizer` is the AdamW estimate: two fp32 moments per *trainable* parameter. That is
    the number LoRA is really attacking -- the weights are usually not what runs you out
    of VRAM, the optimiser state is.
    """
    from ..quant.qlinear import QuantLinear

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() * p.element_size()
                        for p in model.parameters() if not p.requires_grad)
    quant_bytes = sum(m.nbytes() for m in model.modules() if isinstance(m, QuantLinear))
    return {
        "trainable_params": trainable,
        "trainable_bytes": trainable * 4,
        "frozen_bytes": frozen_params + quant_bytes,
        "grad_bytes": trainable * 4,
        "optimizer_bytes": trainable * 8,
        "total_bytes": frozen_params + quant_bytes + trainable * 16,
    }


def memory_line(model) -> str:
    m = describe_memory(model)
    mb = lambda b: f"{b / 1e6:.0f} MB"  # noqa: E731
    return (f"memory     base {mb(m['frozen_bytes'])} frozen + adapters "
            f"{mb(m['trainable_bytes'])} + grads {mb(m['grad_bytes'])} + Adam "
            f"{mb(m['optimizer_bytes'])} = {mb(m['total_bytes'])}")


def maybe_torch_load(path, device):
    return torch.load(path, map_location=device, weights_only=False)
