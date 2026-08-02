"""Turn a trained model into a quantized one, and get it back off disk again.

Three jobs:

  quantize_model()  walk a Transformer, replace every nn.Linear with a QuantLinear
  save_quantized()  write a .pt the rest of the project can load
  load_quantized()  rebuild it -- which means rebuilding the *shapes* before the weights,
                    because a QuantLinear's buffers do not fit an nn.Linear's slot

The lm_head question
--------------------
Our models set `tie_embeddings: true`, so `lm_head.weight` **is** `tok_emb.weight` -- one
matrix, two names. That has a consequence people get wrong:

    quantizing a tied lm_head saves nothing.

The embedding table still has to exist in float for the input lookup (you cannot index
into a packed 4-bit matrix to fetch row 8,421), so the bytes stay. You would be paying the
full accuracy cost of quantizing the single most sensitive matrix in the model and getting
zero memory back. On our 300M config that matrix is 32k x 1024 = 33.5M weights -- 11% of
the model -- so it is not a rounding error either way.

Hence: **skip lm_head when embeddings are tied**, by default, and report it as skipped
rather than silently omitting it. `--quantize-head` overrides for the curious.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model.transformer import Transformer
from .qlinear import QuantLinear
from .qtensor import QuantScheme, resolve_group_size

#: A callable that turns one float Linear into a QuantLinear. RTN ignores `name`;
#: GPTQ and AWQ use it to find that layer's calibration statistics.
Quantizer = Callable[[str, nn.Linear, QuantScheme], QuantLinear]


@dataclass
class LayerReport:
    name: str
    in_features: int
    out_features: int
    float_bytes: int
    quant_bytes: int
    group_size: int
    requested_group: int
    skipped: str | None = None  # reason, if it was left in float

    @property
    def regrouped(self) -> bool:
        return self.skipped is None and self.group_size != self.requested_group

    def as_dict(self) -> dict:
        return {
            "name": self.name, "in_features": self.in_features,
            "out_features": self.out_features, "float_bytes": self.float_bytes,
            "quant_bytes": self.quant_bytes, "group_size": self.group_size,
            "requested_group": self.requested_group, "skipped": self.skipped,
        }


@dataclass
class QuantReport:
    """What quantizing actually did. Printed by the CLI, shown by the portal."""

    scheme: QuantScheme
    layers: list[LayerReport] = field(default_factory=list)
    seconds: float = 0.0
    other_bytes: int = 0  # embeddings, norms — everything not a quantized Linear

    @property
    def quantized(self) -> list[LayerReport]:
        return [x for x in self.layers if x.skipped is None]

    @property
    def skipped(self) -> list[LayerReport]:
        return [x for x in self.layers if x.skipped is not None]

    @property
    def regrouped(self) -> list[LayerReport]:
        return [x for x in self.layers if x.regrouped]

    def totals(self) -> dict:
        """Whole-model bytes, before and after. This is the headline number, and it
        deliberately includes the *unquantized* parts -- embeddings and norms do not
        shrink, so a 4x weight saving is never a 4x model saving."""
        lin_f = sum(x.float_bytes for x in self.layers)
        lin_q = sum(x.quant_bytes for x in self.layers)
        return {
            "linear_float_bytes": lin_f,
            "linear_quant_bytes": lin_q,
            "other_bytes": self.other_bytes,
            "model_float_bytes": lin_f + self.other_bytes,
            "model_quant_bytes": lin_q + self.other_bytes,
            "linear_ratio": lin_f / max(lin_q, 1),
            "model_ratio": (lin_f + self.other_bytes) / max(lin_q + self.other_bytes, 1),
        }

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme.as_dict(),
            "label": self.scheme.label(),
            "seconds": self.seconds,
            "totals": self.totals(),
            "layers": [x.as_dict() for x in self.layers],
        }

    def summary(self) -> str:
        t = self.totals()
        mb = lambda b: f"{b / 1e6:.1f} MB"  # noqa: E731
        lines = [
            f"scheme            {self.scheme.label()}",
            f"linear layers     {len(self.quantized)} quantized, {len(self.skipped)} skipped",
            f"  weights         {mb(t['linear_float_bytes'])} -> {mb(t['linear_quant_bytes'])}"
            f"  ({t['linear_ratio']:.2f}x)",
            f"  everything else {mb(t['other_bytes'])} (embeddings + norms, unquantized)",
            f"  whole model     {mb(t['model_float_bytes'])} -> {mb(t['model_quant_bytes'])}"
            f"  ({t['model_ratio']:.2f}x)",
            f"took              {self.seconds:.1f}s",
        ]
        for x in self.skipped:
            lines.append(f"  skipped {x.name}: {x.skipped}")
        if self.regrouped:
            names = ", ".join(sorted({x.name.split(".")[-1] for x in self.regrouped}))
            g = self.regrouped[0]
            lines.append(
                f"  note: {len(self.regrouped)} layers ({names}) use group_size "
                f"{g.group_size}, not {g.requested_group} — in_features "
                f"{g.in_features} is not divisible by it"
            )
        return "\n".join(lines)


# ---- walking the model ------------------------------------------------------------


def linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Every bias-free Linear in the model, by dotted name."""
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def _set_module(model: nn.Module, name: str, new: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _tensors(mod: nn.Module) -> list[torch.Tensor]:
    return [t for t in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False))
            if t is not None]


def _other_bytes(model: nn.Module) -> int:
    """Bytes in everything that is *not* a Linear: embeddings, RMSNorm gains, buffers.

    Storage shared with a Linear is excluded, or a tied model is counted twice: with
    `tie_embeddings` the skipped `lm_head` and `tok_emb` are one allocation, already
    charged to the layer table.
    """
    linear_storage = set()
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, QuantLinear)):
            linear_storage.update(id(t.untyped_storage()) for t in _tensors(mod))

    total, seen = 0, set()
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, QuantLinear)):
            continue
        for t in _tensors(mod):
            ident = id(t.untyped_storage())
            if ident in seen or ident in linear_storage:
                continue
            seen.add(ident)
            total += t.numel() * t.element_size()
    return total


def quantize_model(
    model: Transformer,
    scheme: QuantScheme,
    quantizer: Quantizer | None = None,
    quantize_head: bool = False,
    skip: tuple[str, ...] = (),
    progress: Callable[[str, int, int], None] | None = None,
) -> QuantReport:
    """Replace the model's Linear layers with QuantLinear, in place.

    quantizer:      how to quantize one layer. Defaults to RTN.
    quantize_head:  quantize lm_head even when embeddings are tied (see module docstring).
    skip:           substrings; any layer whose name contains one is left in float.
    """
    _refuse_moe(model)
    t0 = time.monotonic()
    tied = bool(getattr(model.cfg, "tie_embeddings", False))
    report = QuantReport(scheme=scheme)
    targets = linear_layers(model)
    total = len(targets)

    for i, (name, lin) in enumerate(targets.items()):
        out_f, in_f = lin.weight.shape
        float_bytes = out_f * in_f * 2  # bf16 is the reference point
        reason = None
        if name.endswith("lm_head") and tied and not quantize_head:
            reason = ("tied to tok_emb — quantizing it costs accuracy and saves no bytes, "
                      "because the embedding table must stay float for the input lookup")
        elif any(s in name for s in skip):
            reason = "excluded by --skip"
        elif in_f % 2 != 0 and scheme.packed:
            reason = f"in_features {in_f} is odd, cannot pack 4-bit pairs"

        if reason is not None:
            report.layers.append(LayerReport(
                name=name, in_features=in_f, out_features=out_f, float_bytes=float_bytes,
                quant_bytes=float_bytes, group_size=-1,
                requested_group=scheme.group_size, skipped=reason))
            continue

        q = (quantizer or _rtn_quantizer)(name, lin, scheme)
        _set_module(model, name, q)
        report.layers.append(LayerReport(
            name=name, in_features=in_f, out_features=out_f, float_bytes=float_bytes,
            quant_bytes=q.nbytes(), group_size=q.group_size,
            requested_group=scheme.group_size))
        if progress:
            progress(name, i + 1, total)

    report.other_bytes = _other_bytes(model)
    report.seconds = time.monotonic() - t0
    return report


def _refuse_moe(model: nn.Module) -> None:
    """A mixture of experts cannot go through this path, and failing loudly is the point.

    Two separate things would go wrong quietly. The experts are stacked `nn.Parameter`s
    rather than `nn.Linear`s, so `linear_layers()` does not see them — on the 300M that is
    68% of the model left in float while the report cheerfully claims a 2.8x saving. And the
    router's gate *is* an `nn.Linear`, so it would be quantized, which is the one layer that
    must never be: a wrong route sends the token to a different expert, not to a slightly
    wrong number, so the error is discrete and unbounded rather than small and averaged out.
    """
    from ..model.moe import MoEFeedForward

    layers = [n for n, m in model.named_modules() if isinstance(m, MoEFeedForward)]
    if layers:
        raise ValueError(
            f"this checkpoint is a mixture of experts ({len(layers)} MoE layers) and "
            "quantizing it here would silently leave every expert in float while "
            "quantizing the router, which must never be quantized. Quantizing an MoE model "
            "needs expert-aware packing — see docs/14 § 'What MoE breaks'.")


def _rtn_quantizer(name: str, lin: nn.Linear, scheme: QuantScheme) -> QuantLinear:
    return QuantLinear.from_linear(lin, scheme)


# ---- disk -------------------------------------------------------------------------


def quant_metadata(model: nn.Module, scheme: QuantScheme, report: QuantReport) -> dict:
    """The shape recipe needed to rebuild this model before loading its weights."""
    return {
        "scheme": scheme.as_dict(),
        "label": scheme.label(),
        "layers": {
            n: {"in_features": m.in_features, "out_features": m.out_features,
                "group_size": m.group_size}
            for n, m in model.named_modules() if isinstance(m, QuantLinear)
        },
        "report": report.as_dict(),
    }


def save_quantized(
    path: str | Path,
    model: nn.Module,
    scheme: QuantScheme,
    report: QuantReport,
    source: dict,
    source_path: str | None = None,
) -> Path:
    """Write a quantized checkpoint.

    `source` is the checkpoint dict this came from; we carry its `model_config` and
    `config` forward unchanged. That is not bookkeeping -- `config.data.tokenizer` is how
    the inference engine knows which BPE vocabulary this model's embedding index means,
    and it refuses to load a checkpoint that has lost it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_config": source["model_config"],
        "config": source.get("config"),
        "step": source.get("step"),
        "best_val": source.get("best_val"),
        "quant": quant_metadata(model, scheme, report),
    }
    payload["quant"]["source"] = source_path
    torch.save(payload, path)
    return path


def apply_quant_metadata(model: Transformer, meta: dict) -> Transformer:
    """Swap in *empty* QuantLinears of the right shape, so `load_state_dict` fits.

    This must happen before the weights are loaded. A QuantLinear's state is
    `qweight/scales/qzeros`, which bears no resemblance to an nn.Linear's `weight` --
    loading one into the other fails loudly, which is the good case; the bad case would
    be silently keeping the float weights and wondering why nothing got smaller.
    """
    scheme = QuantScheme.from_dict(meta["scheme"])
    for name, spec in meta["layers"].items():
        _set_module(model, name, QuantLinear(
            in_features=spec["in_features"], out_features=spec["out_features"],
            scheme=scheme, group_size=spec["group_size"], dtype=torch.bfloat16))
    return model


def is_quantized_checkpoint(ckpt: dict) -> bool:
    return isinstance(ckpt, dict) and isinstance(ckpt.get("quant"), dict)


def build_from_checkpoint(ckpt: dict, device: str = "cpu", dtype=None) -> Transformer:
    """Rebuild a (possibly quantized) model from a loaded checkpoint dict."""
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    if is_quantized_checkpoint(ckpt):
        apply_quant_metadata(model, ckpt["quant"])
    model.load_state_dict(ckpt["model"])
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device)
    # Only the *float* parts get cast: .to(dtype) on the whole module would try to turn
    # the packed uint8 weights into floats and destroy them.
    _cast_float_only(model, dtype)
    model.eval()
    return model


def _cast_float_only(model: nn.Module, dtype: torch.dtype):
    for mod in model.modules():
        if isinstance(mod, QuantLinear):
            mod.out_dtype = dtype
            # Only the plain-fp16 case has a buffer to normalise; with double quantization
            # the scales are int8 codes and `.scales` is a freshly built temporary, so
            # writing to it would change nothing.
            buf = mod._buffers.get("scales")
            if buf is not None:
                buf.data = buf.data.to(torch.float16)
            continue
        for n, p in list(mod.named_parameters(recurse=False)):
            if p.is_floating_point():
                setattr(mod, n, nn.Parameter(p.data.to(dtype), requires_grad=p.requires_grad))
        for n, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.is_floating_point():
                mod.register_buffer(n, b.to(dtype))


def load_quantized(path: str | Path, device: str = "cpu") -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not is_quantized_checkpoint(ckpt):
        raise ValueError(f"{path} is not a quantized checkpoint (no 'quant' key)")
    return build_from_checkpoint(ckpt, device=device), ckpt["quant"]


def model_nbytes(model: nn.Module) -> int:
    """Every byte the model holds: parameters and buffers, counting shared storage once."""
    total, seen = 0, set()
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None:
            continue
        ident = id(t.untyped_storage())
        if ident in seen:
            continue
        seen.add(ident)
        total += t.numel() * t.element_size()
    return total
