"""Sharper photos: a patch discriminator beside the autoencoder's pixel loss (G4c).

`ae-coco` was trained on L1 + L2 alone and keeps only about half of a real photo's fine detail
(`measure.sharpness`, 0.54). That is what a pixel loss asks for: when the decoder cannot tell
exactly where an edge or a blade of grass falls, the *average* of the places it might be costs
least, and an average of edges is a smear. Nothing in L1 + L2 prefers a sharp edge in a slightly
wrong place to a blurred one in roughly the right place; to a pixel loss the sharp guess is
punished twice.

A discriminator changes what is cheap. It is a small convolutional network that looks at
overlapping patches (each of its 8x8 outputs sees a 38-pixel neighbourhood of a 64-pixel photo)
and says, for each, whether it came from a real photo or from the decoder. The decoder is then
also trained to make its patches read as real, and a smear does not: real photos have texture.
This is the recipe Stable Diffusion's autoencoder uses (with a pretrained perceptual network as
well, which this repo does not have, since everything here is trained by us).

```mermaid
flowchart LR
    X["photo"] --> E["encoder"] --> Z["latent 8x8x16"] --> D["decoder"] --> R["reconstruction"]
    R --> L["L1 + L2 + KL<br/>(as before)"]
    R --> P["patch discriminator<br/>(8x8 real/fake scores)"]
    X --> P
    P --> G["the decoder's adversarial loss:<br/>make R's patches read as real"]
    P --> H["the discriminator's hinge loss:<br/>tell X from R"]
```

Three details that keep it from going wrong, each with a test:

* **Hinge losses** (`d_loss`, `g_loss`). The discriminator pushes real scores above +1 and fake
  ones below -1 and then stops caring, so it cannot grow without bound against a decoder that is
  still learning. The decoder's loss is minus the score of its own patches.
* **The weight adapts** (`adaptive_weight`): the adversarial gradient at the decoder's last layer
  is scaled to the size of the reconstruction gradient there, times `weight`. A fixed weight is
  either too small to matter early or, as the discriminator gets confident, large enough to
  overrule the reconstruction and paint plausible texture that is not in the photo.
* **It starts late** (`start_step`): the discriminator only switches on once the autoencoder
  already reconstructs. Against a decoder that outputs grey, it wins instantly and its gradient
  says nothing useful.
* **"Best" is the latest checkpoint** once it has started (`train_ae.py`): the pixel loss rises
  as texture replaces averages, so ranking by it picks the least sharp checkpoint there is.

`weight: 0` (the default) builds no discriminator at all, so every earlier config trains
exactly as before.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdversarialConfig:
    #: 0 = off (no discriminator is built). The adversarial term is `weight * adaptive * g_loss`.
    weight: float = 0.0
    #: Step at which the discriminator starts training and the decoder starts reading it.
    start_step: int = 10000
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    lr: float = 2.0e-4
    #: Bound on the adaptive factor, so a vanishing adversarial gradient cannot blow it up.
    max_adaptive: float = 1.0e4


class PatchDiscriminator(nn.Module):
    """Strided convolutions to an `(B, 1, h, w)` map of real/fake scores, one per patch.

    No batch norm: real and fake batches are scored separately, and batch statistics would let
    it tell them apart by the batch rather than by the picture."""

    def __init__(self, channels: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 3
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 4, stride=2, padding=1),
                       nn.GroupNorm(math.gcd(8, ch), ch), nn.LeakyReLU(0.2)]
            prev = ch
        layers += [nn.Conv2d(prev, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """The discriminator's hinge loss: real above +1, fake below -1, nothing beyond."""
    return 0.5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())


def g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """The decoder's adversarial loss: raise the score of its own patches."""
    return -fake_logits.mean()


def adaptive_weight(rec: torch.Tensor, adv: torch.Tensor, last: torch.Tensor,
                    max_value: float = 1.0e4) -> torch.Tensor:
    """`|grad rec| / |grad adv|` at the decoder's last layer, detached.

    Makes the adversarial term as strong as the reconstruction term where both reach the
    picture, whatever the discriminator's current scale; `weight` then says how much stronger
    or weaker it should be."""
    g_rec = torch.autograd.grad(rec, last, retain_graph=True)[0]
    g_adv = torch.autograd.grad(adv, last, retain_graph=True)[0]
    return (g_rec.norm() / (g_adv.norm() + 1e-4)).clamp(0.0, max_value).detach()


def last_layer(autoencoder) -> torch.Tensor:
    """The decoder's final convolution weight: where the two gradients are compared."""
    return autoencoder.decoder.net[-1].weight
