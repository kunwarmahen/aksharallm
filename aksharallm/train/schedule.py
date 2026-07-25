r"""Learning-rate schedules.

The LR schedule matters more than almost any other hyperparameter at this scale. The
shape everyone converges on:

    lr
    ^        ___
    |       /   \____
    |      /         \____
    |     /               \___
    +----/---------------------\--> step
      warmup      decay      floor

  warmup  - start near zero and ramp up. At init the model's gradients are large and
            uninformative; a full-LR step there can permanently damage the embeddings.
  decay   - cosine to a floor. Large steps early explore, small steps late refine.
  floor   - never decay to exactly 0; the last few steps still do useful work.
"""

from __future__ import annotations

import math


def get_lr(step: int, *, base_lr: float, warmup_steps: int, max_steps: int,
           min_lr_ratio: float = 0.1, schedule: str = "cosine") -> float:
    min_lr = base_lr * min_lr_ratio

    if step < warmup_steps:
        # linear warmup. (step+1) so step 0 isn't a literal zero-LR no-op.
        return base_lr * (step + 1) / max(1, warmup_steps)

    if schedule == "constant":
        return base_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    if schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (base_lr - min_lr)

    if schedule == "wsd":
        # Warmup-Stable-Decay: hold base_lr flat, then decay hard over the last 20%.
        # Useful when you don't know max_steps up front -- you can stop any time by
        # running the decay phase, instead of being locked into a cosine's endpoint.
        if progress < 0.8:
            return base_lr
        p = (progress - 0.8) / 0.2
        return min_lr + (base_lr - min_lr) * (1.0 - p)

    raise ValueError(f"unknown schedule '{schedule}'")
