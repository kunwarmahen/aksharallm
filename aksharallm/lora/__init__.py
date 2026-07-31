"""LoRA and QLoRA, from scratch.

Fine-tuning a model without training the model.

The problem it solves is not "fine-tuning is slow" -- it is that fine-tuning needs room
for the *optimiser*, not just the weights. AdamW keeps two running averages per trainable
parameter, in fp32. For our 300M model, full fine-tuning costs roughly:

    weights   1.2 GB      grads   1.2 GB      Adam state   4.8 GB      = 7.2 GB, plus
                                                                        activations

which a 24 GB card can do only if nothing else is on it. Scale to 1B, which is where
Phase 4 is headed, and it stops fitting at all.

LoRA trains a low-rank correction instead of the weights (`layer.py`), so the trainable
parameter count drops by ~99% and the optimiser state with it. QLoRA goes further and
holds the *frozen* base in 4 bits, which is possible precisely because it is frozen and
never receives a gradient. The same 300M model becomes:

    base (nf4)  0.21 GB    adapters  0.011 GB    Adam state  0.09 GB   = ~0.3 GB

Two things follow that are worth stating plainly, because they are the actual payoff:

  * A specialisation is a **file**, not a model. One base plus N adapters of ~11 MB each,
    swapped at inference. The chat model and the Python model stop being two 1.2 GB
    checkpoints.
  * The reference model in DPO becomes **free** -- switch the adapters off and the base
    you are already holding *is* the reference. See `disable_adapters`.

Modules:
    layer     LoRALinear: the frozen base plus `B @ A`, and `disable_adapters`
    inject    which layers get adapted, freezing, and the trainable/total report
    adapter   the .pt format, and the base-identity checks that stop silent mismatches
    merge     folding an adapter back into the weights, and why that is lossy on a
              quantized base
    cli       `python -m aksharallm.lora`

The deep dive is `docs/11-lora.md`.
"""

from .adapter import (
    AdapterError,
    attach_adapter,
    base_identity,
    describe,
    is_adapter_file,
    load_adapter_file,
    save_adapter,
)
from .inject import (
    PRESETS,
    LoRAConfig,
    LoRAReport,
    apply_lora,
    has_lora,
    lora_layers,
    prepare_for_training,
    resolve_targets,
    set_adapters_enabled,
)
from .layer import LoRALinear, disable_adapters
from .merge import merge_lora

__all__ = [
    "AdapterError",
    "LoRAConfig",
    "LoRALinear",
    "LoRAReport",
    "PRESETS",
    "apply_lora",
    "attach_adapter",
    "base_identity",
    "describe",
    "disable_adapters",
    "has_lora",
    "is_adapter_file",
    "load_adapter_file",
    "lora_layers",
    "merge_lora",
    "prepare_for_training",
    "resolve_targets",
    "save_adapter",
    "set_adapters_enabled",
]
