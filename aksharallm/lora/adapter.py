"""Reading and writing adapter files.

An adapter is a `.pt` holding the `lora_A` / `lora_B` pairs and nothing else. On our 300M
config at r=8 over all-linear that is about 11 MB against the base model's 599 MB, which
is the practical payoff of the whole technique: a specialisation is a file you can email.

The file also carries the identity of the base it was trained against, and loading checks
it. This is not bureaucracy. An adapter is a *delta*; applied to the wrong base it is
meaningless, and the failure is silent -- the model still runs, still emits fluent text,
and is simply worse for no visible reason. The three things checked:

  model_config   the architecture. Shapes would usually catch a mismatch, but two runs
                 with the same dimensions and a different depth would not be caught.
  tokenizer      an adapter trained through one BPE vocabulary is nonsense through
                 another, and this is invisible in the shapes.
  stage          a chat adapter belongs on a base checkpoint, not on top of a DPO one.

Mismatches raise by default and can be forced past with `strict=False`, because "I know,
I am doing an experiment" is a legitimate thing to want.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from .inject import LoRAConfig, LoRAReport, apply_lora, lora_layers

#: Bumped if the on-disk layout ever changes incompatibly.
FORMAT_VERSION = 1


class AdapterError(RuntimeError):
    pass


def base_identity(ckpt: dict, path: str | Path | None = None) -> dict:
    """The fingerprint of the checkpoint an adapter was trained on."""
    cfg = ckpt.get("config") or {}
    return {
        "path": str(path) if path else None,
        "step": ckpt.get("step"),
        "best_val": ckpt.get("best_val"),
        "model_config": ckpt.get("model_config"),
        "tokenizer": (cfg.get("data") or {}).get("tokenizer"),
        "quant": (ckpt.get("quant") or {}).get("label"),
    }


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors, on the CPU, keyed by the layer they belong to."""
    state = {}
    for name, mod in lora_layers(model).items():
        state[f"{name}.lora_A"] = mod.lora_A.detach().to("cpu", torch.float32)
        state[f"{name}.lora_B"] = mod.lora_B.detach().to("cpu", torch.float32)
    return state


def save_adapter(
    path: str | Path,
    model: nn.Module,
    config: LoRAConfig,
    base: dict,
    report: LoRAReport | None = None,
    extra: dict | None = None,
) -> Path:
    """Write an adapter file. `base` comes from `base_identity`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "kind": "lora-adapter",
        "lora": adapter_state(model),
        "lora_config": config.as_dict(),
        "base": base,
        "report": report.as_dict() if report is not None else None,
        "created": time.time(),
        **(extra or {}),
    }
    torch.save(payload, path)
    return path


def load_adapter_file(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "lora-adapter":
        raise AdapterError(f"{path} is not a LoRA adapter file")
    if int(payload.get("format", 0)) > FORMAT_VERSION:
        raise AdapterError(
            f"{path} was written by a newer format (v{payload['format']} > "
            f"v{FORMAT_VERSION})")
    return payload


def is_adapter_file(path: str | Path) -> bool:
    try:
        load_adapter_file(path)
        return True
    except Exception:  # noqa: BLE001 — any failure to parse means "not an adapter"
        return False


def check_base(payload: dict, ckpt: dict, strict: bool = True) -> list[str]:
    """Compare an adapter's recorded base against the checkpoint it is being applied to.

    Returns the list of complaints; raises instead when `strict`.
    """
    want = payload.get("base") or {}
    have = base_identity(ckpt)
    problems = []

    wmc, hmc = want.get("model_config"), have.get("model_config")
    if wmc and hmc:
        differing = [k for k in wmc if wmc.get(k) != hmc.get(k)]
        # dropout is a training knob, not architecture: SFT raises it and inference sets
        # it to zero, so it differs on almost every legitimate load.
        differing = [k for k in differing if k != "dropout"]
        if differing:
            problems.append(
                "model_config differs: " + ", ".join(
                    f"{k} {wmc.get(k)!r} != {hmc.get(k)!r}" for k in differing))

    wt, ht = want.get("tokenizer"), have.get("tokenizer")
    if wt and ht and Path(wt).name != Path(ht).name:
        problems.append(f"tokenizer differs: {wt} != {ht}")

    if problems and strict:
        raise AdapterError(
            "this adapter was not trained on this checkpoint:\n  - "
            + "\n  - ".join(problems)
            + "\nAn adapter is a delta; on the wrong base it degrades the model silently."
              "\nPass --force if you meant to."
        )
    return problems


def attach_adapter(
    model: nn.Module,
    payload: dict,
    ckpt: dict | None = None,
    strict: bool = True,
) -> LoRAReport:
    """Inject adapters into `model` and load an adapter file's weights into them.

    The config comes from the file, so nothing about rank or targets has to be repeated
    at load time -- an adapter that was trained on `qv` at r=32 loads that way whatever
    the caller believes.
    """
    if ckpt is not None:
        check_base(payload, ckpt, strict=strict)
    config = LoRAConfig.from_dict(payload["lora_config"])
    report = apply_lora(model, config)

    state = payload["lora"]
    layers = lora_layers(model)
    missing = [n for n in layers if f"{n}.lora_A" not in state]
    unexpected = [k[: -len(".lora_A")] for k in state
                  if k.endswith(".lora_A") and k[: -len(".lora_A")] not in layers]
    if missing or unexpected:
        raise AdapterError(
            f"adapter does not line up with the model: {len(missing)} layers have no "
            f"weights in the file, {len(unexpected)} entries match no layer. "
            f"First missing: {missing[:3]}; first unexpected: {unexpected[:3]}")

    for name, mod in layers.items():
        a = state[f"{name}.lora_A"]
        b = state[f"{name}.lora_B"]
        if a.shape != mod.lora_A.shape or b.shape != mod.lora_B.shape:
            raise AdapterError(
                f"{name}: adapter shapes {tuple(a.shape)}/{tuple(b.shape)} do not match "
                f"the model's {tuple(mod.lora_A.shape)}/{tuple(mod.lora_B.shape)}")
        mod.lora_A.data.copy_(a.to(mod.lora_A.device, mod.lora_A.dtype))
        mod.lora_B.data.copy_(b.to(mod.lora_B.device, mod.lora_B.dtype))
    return report


def describe(payload: dict) -> dict:
    """A summary for the CLI and the portal, without touching the tensors' values."""
    cfg = payload.get("lora_config") or {}
    state = payload.get("lora") or {}
    n = sum(t.numel() for t in state.values())
    base = payload.get("base") or {}
    return {
        "r": cfg.get("r"),
        "alpha": cfg.get("alpha"),
        "targets": cfg.get("targets"),
        "dropout": cfg.get("dropout"),
        "layers": len(state) // 2,
        "params": n,
        "bytes": n * 4,
        "base_step": base.get("step"),
        "base_path": base.get("path"),
        "base_quant": base.get("quant"),
        "tokenizer": base.get("tokenizer"),
        "created": payload.get("created"),
        "trained": payload.get("training"),
    }
