"""Extend a trained checkpoint's context window — without touching a single weight.

This is the part that surprises people, so it is worth stating plainly: **the file this
writes has exactly the same tensors in it as the file it read.** RoPE has no parameters.
The rotation angles are computed from `head_dim`, `rope_theta` and the position, and the
cache holding them is registered `persistent=False` precisely so it is never saved and
never loaded. Changing how a model addresses position is therefore a config edit, and this
module is a config edit with bookkeeping around it.

```mermaid
flowchart LR
    A["ckpt_best.pt<br/>max_seq_len 1024<br/>rope_scaling none"] -->|extend --method yarn --factor 4| B["ckpt_yarn4.pt<br/>max_seq_len 4096<br/>rope_scaling yarn x4<br/>original_max_seq_len 1024"]
    A -. same weights, byte for byte .-> B
```

The one piece of bookkeeping that matters is `original_max_seq_len`. Once `max_seq_len` has
been raised to 4,096 there is nothing left in the config to say the weights were trained on
1,024 — and every scaling method needs that number to compute its factor. Recording it at
extend time is what makes an extended checkpoint self-describing, so it reloads correctly
in the Playground, the eval harness and the server without anyone passing a flag.

Read with: docs/18-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..model.rope import METHODS, RopeScaling


def default_out_name(source: str | Path, method: str, factor: float) -> Path:
    """Where an extension lands when nobody says otherwise: beside the source, renamed.

    One function because two callers need to agree — the CLI writes the file and the portal
    tells the reader what it is about to write, and a confirmation dialog naming a path that
    turns out not to be the one used is worse than no dialog.
    """
    source = Path(source)
    return source.with_name(f"{source.stem}_{method}{factor:g}x.pt")


def plan_extension(model_cfg: dict, method: str, factor: float,
                   original: int | None = None, **knobs) -> dict:
    """The new `model_config` dict. Pure — it touches nothing on disk.

    Separated from `extend` so the portal and the tests can show what *would* change
    without writing a 1.2 GB file to find out.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cfg = dict(model_cfg)
    current = int(cfg["max_seq_len"])

    # The window the weights know. On a checkpoint that has never been extended that is
    # simply its max_seq_len; on one that has, it is whatever the earlier extension
    # recorded -- so extending twice compounds the factor instead of silently rebasing.
    prior = cfg.get("rope_scaling") or {}
    if isinstance(prior, RopeScaling):
        prior = prior.__dict__
    trained = original or prior.get("original_max_seq_len") or current

    if method == "none":
        cfg["rope_scaling"] = RopeScaling().__dict__
        cfg["max_seq_len"] = int(trained)
        return cfg

    scaling = RopeScaling(type=method, factor=float(factor),
                          original_max_seq_len=int(trained), **knobs)
    cfg["rope_scaling"] = scaling.__dict__
    cfg["max_seq_len"] = int(round(trained * factor))
    return cfg


def _pretty(key: str, value) -> str:
    """One config value, for a human. The scaling block is the one that matters: printed
    raw it is a seven-field dict of which two fields are interesting, and the portal showed
    the other five to somebody trying to understand what a factor is."""
    if key == "rope_scaling":
        if not value or (value.get("type") if isinstance(value, dict) else None) in (None, "none"):
            return "none"
        orig = value.get("original_max_seq_len")
        return (f"{value['type']} x{value.get('factor', 1):g}"
                + (f" (trained on {orig})" if orig else ""))
    if key == "max_seq_len":
        return f"{value:,} tokens" if isinstance(value, int) else str(value)
    return "none" if value in (None, 0) else str(value)


def describe(before: dict, after: dict) -> list[str]:
    """Human-readable diff. What the CLI prints and the portal shows."""
    labels = {"max_seq_len": "context window", "rope_scaling": "RoPE scaling",
              "attn_window": "sliding window", "attn_sinks": "attention sinks"}
    lines = []
    for key, label in labels.items():
        old, new = _pretty(key, before.get(key)), _pretty(key, after.get(key))
        # Compared *after* rendering: `None` and a default RopeScaling are different objects
        # and the same setting, and "RoPE scaling: none → none" is not a change.
        if old != new:
            lines.append(f"{label}: {old} → {new}")
    return lines or ["nothing changed"]


def extend(ckpt_path: str | Path, out_path: str | Path, method: str, factor: float,
           original: int | None = None, window: int | None = None, sinks: int = 0,
           **knobs) -> dict:
    """Write an extended copy of `ckpt_path`. Returns a summary of what changed."""
    ckpt_path, out_path = Path(ckpt_path), Path(out_path)
    if out_path.resolve() == ckpt_path.resolve():
        raise ValueError("refusing to overwrite the source checkpoint -- "
                         "an extension is a hypothesis, keep the original")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    before = dict(ckpt["model_config"])
    after = plan_extension(before, method, factor, original, **knobs)
    if window is not None:
        after["attn_window"] = int(window)
        after["attn_sinks"] = int(sinks)

    ckpt["model_config"] = after
    # A note in the checkpoint itself, because the filename will not survive being copied
    # around and "why does this one say 4096" is the question someone will ask in a month.
    ckpt.setdefault("notes", []).append(
        f"context extended from {before['max_seq_len']} to {after['max_seq_len']} "
        f"({method} x{factor:g}); weights unchanged")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    return {
        "source": str(ckpt_path), "out": str(out_path),
        "before": before, "after": after,
        "changes": describe(before, after),
        "trained_window": after["rope_scaling"]["original_max_seq_len"],
        "addressable": after["max_seq_len"],
    }
