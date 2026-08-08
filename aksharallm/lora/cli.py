"""Command line for adapters.

    python -m aksharallm.lora show   tiny/sft_best.lora.pt
    python -m aksharallm.lora merge  tiny/ckpt_best.pt --adapter tiny/sft_best.lora.pt
    python -m aksharallm.lora budget tiny/ckpt_best.pt --qlora

Training an adapter is `train/sft.py --lora` (or `--qlora`), not a command here: an
adapter is the *output* of fine-tuning, so it belongs to the fine-tuner. This CLI is for
everything you do to one afterwards.

`budget` is the one to reach for first when deciding whether a fine-tune fits. It prints
what full fine-tuning, LoRA and QLoRA would each cost in memory on a given checkpoint,
without training anything, so "will this run on the 3090 while nothing else is on it" is
a question with an answer before you start rather than after an OOM.

Read with: docs/12-lora.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ..config import ModelConfig
from ..infer.checkpoints import CheckpointStore
from ..model.transformer import Transformer
from ..quant.convert import is_quantized_checkpoint, quantize_model
from ..quant.qtensor import QuantScheme
from .adapter import (
    AdapterError,
    attach_adapter,
    describe,
    load_adapter_file,
)
from .inject import PRESET_BLURBS, PRESETS, LoRAConfig, apply_lora
from .merge import merge_into_checkpoint, merge_lora
from .setup import describe_memory, rebuild_quantized_shapes


def _resolve(ref: str) -> Path:
    p = Path(ref)
    if p.is_file():
        return p
    try:
        store = CheckpointStore()
        return store.get(store.identify(ref)).path
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot find checkpoint {ref!r}: {e}")


def _load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    rebuild_quantized_shapes(model, ckpt)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def _mb(b) -> str:
    return f"{b / 1e6:,.0f} MB"


# ---- show ---------------------------------------------------------------------------


def cmd_show(args) -> int:
    payload = load_adapter_file(args.adapter)
    d = describe(payload)
    print(f"adapter    {args.adapter}")
    print(f"rank       r={d['r']}, alpha={d['alpha']:g} (scaling {d['alpha'] / d['r']:g})")
    print(f"targets    {d['targets']} — {d['layers']} layers")
    print(f"size       {d['params']:,} parameters, {_mb(d['bytes'])} on disk")
    print(f"base       {d['base_path']} (step {d['base_step']})"
          + (f", quantized {d['base_quant']}" if d.get("base_quant") else ""))
    print(f"tokenizer  {d['tokenizer']}")
    if d.get("created"):
        print(f"created    {time.strftime('%Y-%m-%d %H:%M', time.localtime(d['created']))}")
    if d.get("trained"):
        t = d["trained"]
        bits = [f"{k} {v}" for k, v in t.items() if v is not None]
        print(f"trained    {', '.join(bits)}")
    rep = payload.get("report")
    if rep:
        print(f"trainable  {rep['trainable']:,} of {rep['total']:,} "
              f"({100 * rep['fraction']:.2f}%) at training time")
    if args.json:
        Path(args.json).write_text(json.dumps({**d, "report": rep}, indent=2))
        print(f"wrote {args.json}")
    return 0


# ---- merge --------------------------------------------------------------------------


def cmd_merge(args) -> int:
    src = _resolve(args.checkpoint)
    model, ckpt = _load_model(src, args.device)
    payload = load_adapter_file(args.adapter)
    try:
        attach_adapter(model, payload, ckpt=ckpt, strict=not args.force)
    except AdapterError as e:
        raise SystemExit(str(e))

    info = merge_lora(model, dtype=torch.float32)
    print(f"merged {len(info['merged'])} adapters into the weights")
    if info["note"]:
        print(f"note: {info['note']}")

    out = Path(args.out) if args.out else src.with_name(
        f"{src.stem}-merged-{Path(args.adapter).stem.replace('.lora', '')}.pt")
    if is_quantized_checkpoint(ckpt) and args.requantize:
        scheme = QuantScheme.from_dict(ckpt["quant"]["scheme"])
        rep = quantize_model(model, scheme)
        print(f"\nre-quantized to {scheme.label()}: "
              f"{rep.totals()['model_ratio']:.2f}x")
        print("  the perplexity you measured before merging does NOT carry over — this is "
              "a fresh rounding of different weights. Re-measure it.")
        from ..quant.convert import save_quantized

        save_quantized(out, model, scheme, rep, ckpt, source_path=str(src))
    else:
        merge_into_checkpoint(model, ckpt, out, source=f"{src} + {args.adapter}")
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


# ---- budget -------------------------------------------------------------------------


def cmd_budget(args) -> int:
    """What each fine-tuning strategy would cost, measured on the real shapes."""
    src = _resolve(args.checkpoint)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])

    rows = []
    base = Transformer(mcfg)
    n = sum(p.numel() for p in base.parameters())
    rows.append(("full fine-tune", {
        "trainable_params": n, "frozen_bytes": 0, "trainable_bytes": n * 4,
        "grad_bytes": n * 4, "optimizer_bytes": n * 8, "total_bytes": n * 16}))

    for r in args.ranks:
        m = Transformer(mcfg)
        apply_lora(m, LoRAConfig(r=r, targets=args.targets))
        rows.append((f"LoRA r={r}", describe_memory(m)))

    for r in args.ranks:
        m = Transformer(mcfg)
        quantize_model(m, QuantScheme(bits=4, group_size=64, dtype="nf4",
                                      double_quant=True, method="rtn"))
        apply_lora(m, LoRAConfig(r=r, targets=args.targets))
        rows.append((f"QLoRA r={r}", describe_memory(m)))

    print(f"{src}  —  {n / 1e6:.0f}M parameters, targets '{args.targets}'\n")
    hdr = f"{'strategy':<16} {'trainable':>12} {'weights':>10} {'grads':>9} " \
          f"{'Adam':>9} {'total':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        weights = m["frozen_bytes"] + m["trainable_bytes"]
        print(f"{name:<16} {m['trainable_params']:>12,} {_mb(weights):>10} "
              f"{_mb(m['grad_bytes']):>9} {_mb(m['optimizer_bytes']):>9} "
              f"{_mb(m['total_bytes']):>10}")
    print("\nActivations are not included — they depend on batch size and sequence length,"
          "\nand they are the one term LoRA does *not* reduce: the forward pass still runs"
          "\nthrough every layer at full width. On a 24 GB card that is what you tune the"
          "\nbatch size against once the weights and optimiser fit.")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"checkpoint": str(src), "params": n, "targets": args.targets,
             "rows": [{"strategy": k, **v} for k, v in rows]}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


# ---- presets ------------------------------------------------------------------------


def cmd_presets(args) -> int:
    for name, suffixes in PRESETS.items():
        print(f"{name:<12} {', '.join(suffixes):<28} {PRESET_BLURBS[name]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.lora",
        description="Inspect, merge and budget LoRA adapters. Training one is "
                    "`train/sft.py --lora`.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="what is in an adapter file")
    s.add_argument("adapter")
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("merge", help="fold an adapter into the weights")
    s.add_argument("checkpoint", help="the base it was trained on")
    s.add_argument("--adapter", required=True)
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--force", action="store_true",
                   help="merge even if the adapter's base does not match")
    s.add_argument("--requantize", action="store_true",
                   help="if the base was quantized, quantize the merged result again. "
                        "The result is a fresh rounding — re-measure it.")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("budget", help="what full / LoRA / QLoRA fine-tuning would cost")
    s.add_argument("checkpoint")
    s.add_argument("--ranks", type=int, nargs="+", default=[8, 16])
    s.add_argument("--targets", default="all-linear")
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_budget)

    s = sub.add_parser("presets", help="the target presets and what they cover")
    s.set_defaults(fn=cmd_presets)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except AdapterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
