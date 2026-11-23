"""A continuous autoencoder for latent flow (G3b): no quantizer, a light KL term.

G3 on the VQ tokenizer's latent worked at 8×8 (88%) and failed at 16×16 (~50%), although both
tokenizers reconstruct every picture perfectly (docs/26 § "Is it the latent's resolution?").
A VQ latent is a sum of codebook vectors, so its values cluster; a flow model learns a smooth
path to a target. Stable Diffusion's autoencoder is built for exactly this: it has no codebook,
and a small KL penalty pulls each latent towards a unit Gaussian, so the space a generator has
to learn is smooth and roughly the shape of the noise it starts from.

    encoder -> (mean, log variance) per latent cell
    z = mean + exp(logvar / 2) * noise         (only while training)
    decoder(z) -> picture
    loss = L1 + L2 on pixels + kl_weight * KL(N(mean, var) || N(0, 1))

The encoder and decoder are the tokenizer's (`tokenizer.py`), unchanged; only what sits between
them differs. The latent is small on purpose: 8×8×8 = 512 numbers a picture, a real 24×
compression of its 12,288 pixels, where the VQ latents were 4,096 and 16,384. For drawing, the
**mean** is the latent (deterministic, as Stable Diffusion encodes for training its generator).

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .tokenizer import Decoder, Encoder, PictureConfig


@dataclass
class AEConfig:
    image_size: int = 64
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    latent_dim: int = 8
    #: Weight of the KL term (mean over latent numbers). Large enough to keep the latent
    #: smooth and near unit scale; small enough that reconstruction still wins.
    kl_weight: float = 1e-3

    @property
    def grid(self) -> int:
        return self.image_size // (2 ** len(self.channels))


class PictureAutoencoder(nn.Module):
    """Same encoder and decoder as the tokenizer; a Gaussian latent in place of the codebooks."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(PictureConfig(image_size=cfg.image_size, channels=cfg.channels,
                                             latent_dim=2 * cfg.latent_dim))
        self.decoder = Decoder(PictureConfig(image_size=cfg.image_size, channels=cfg.channels,
                                             latent_dim=cfg.latent_dim))

    def moments(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)

    def forward(self, x):
        """`(reconstruction, kl)`: a sampled latent while training, the mean otherwise."""
        mean, logvar = self.moments(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.training else mean
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
        return self.decoder(z), kl

    @torch.no_grad()
    def latent(self, x) -> torch.Tensor:
        """Pictures -> the latent G3 draws: the encoder's mean."""
        return self.moments(x)[0]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
