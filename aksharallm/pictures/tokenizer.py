"""An image tokenizer: pictures in, a small grid of integers out, and back (G0 of PLAN's
picture line).

The transformer in `model/transformer.py` models integers. The audio phase turned speech into
integers with a codec (`audio/codec.py`): a strided 1-D conv encoder, residual vector
quantization, and a mirrored decoder. A picture is the same problem in two dimensions, so this
is that codec with 2-D convolutions, and **the quantizer is imported unchanged**:

```mermaid
flowchart LR
    X["64x64x3 picture"] --> E["encoder: conv, then 3 x<br/>(res block, stride-2 conv)"]
    E --> Z["8x8 grid of<br/>D-dim vectors"]
    Z --> Q["ResidualVQ (audio/vq.py):<br/>N codebooks of K entries"]
    Q --> I["N x 8 x 8 integers"]
    Q --> D["decoder: mirror image,<br/>nearest upsample + conv"]
    D --> Y["64x64x3 picture again"]
```

`ResidualVQ` takes `(B, D, T)`. A picture's latent is `(B, D, h, w)`, so it is flattened to
`T = h*w` positions in raster order, quantized, and folded back. Nothing in the quantizer knows
it is looking at a picture, which is the claim this file exists to test: EMA codebooks, the
first-batch seeding, dead-code restart and quantizer dropout were all written for audio and
carry over without an edit. The collapse they prevent is just as silent here.

What it does not have: no adversarial loss, no perceptual (LPIPS) loss. Both need a network
that is not ours (LPIPS is a pretrained VGG), and on flat synthetic shapes a pixel loss is
enough to ask the question that matters downstream, which is whether the reconstruction still
says the same thing. `pictures/measure.py` asks it with `reader.py`.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/21-audio.md (the codec and quantizer this reuses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.vq import ResidualVQ


@dataclass
class PictureConfig:
    image_size: int = 64
    #: Encoder widths, one per downsampling step. Three steps: 64 -> 32 -> 16 -> 8.
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    #: Width of each latent vector, the thing the codebooks quantize.
    latent_dim: int = 64
    n_codebooks: int = 4
    codebook_size: int = 512
    #: Train with a random number of codebooks per batch, so every prefix decodes (audio's
    #: quantizer dropout; without it the 1-codebook picture is never optimised).
    quantizer_dropout: bool = True
    commit: float = 0.25
    #: A codebook entry unchosen for this many steps is restarted at a live encoder output.
    restart_after: int = 200

    @property
    def downsample(self) -> int:
        return 2 ** len(self.channels)

    @property
    def grid(self) -> int:
        """Codes per side: a 64-pixel picture at three halvings is an 8x8 grid."""
        return self.image_size // self.downsample

    @property
    def tokens(self) -> int:
        """Integers per picture: grid x grid positions, each with `n_codebooks` codes."""
        return self.grid * self.grid * self.n_codebooks

    @property
    def bits(self) -> float:
        return self.tokens * math.log2(self.codebook_size)


class ResBlock(nn.Module):
    """GroupNorm, SiLU, 3x3 conv, twice, plus the input. GroupNorm rather than BatchNorm so a
    picture's output never depends on what else is in its batch (the same reason the audio
    codec and the recogniser avoid batch statistics)."""

    def __init__(self, c: int):
        super().__init__()
        g = math.gcd(8, c)
        self.body = nn.Sequential(nn.GroupNorm(g, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                  nn.GroupNorm(g, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, cfg: PictureConfig):
        super().__init__()
        c = cfg.channels
        layers: list[nn.Module] = [nn.Conv2d(3, c[0], 3, padding=1)]
        prev = c[0]
        for ch in c:
            # A 4x4 kernel at stride 2 covers every input pixel exactly twice (no checkerboard
            # of pixels seen once and pixels seen twice, which a 3x3 at stride 2 would give).
            layers += [ResBlock(prev), nn.Conv2d(prev, ch, 4, stride=2, padding=1)]
            prev = ch
        layers += [ResBlock(prev), nn.GroupNorm(math.gcd(8, prev), prev), nn.SiLU(),
                   nn.Conv2d(prev, cfg.latent_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: PictureConfig):
        super().__init__()
        c = list(reversed(cfg.channels))
        layers: list[nn.Module] = [nn.Conv2d(cfg.latent_dim, c[0], 3, padding=1), ResBlock(c[0])]
        prev = c[0]
        for ch in c[1:] + [c[-1]]:
            # Nearest upsampling then a conv, not a transposed conv: a stride-2 transposed conv
            # overlaps unevenly and paints a faint checkerboard on every flat region, and a
            # shapes picture is almost all flat region.
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(prev, ch, 3, padding=1), ResBlock(ch)]
            prev = ch
        layers += [nn.GroupNorm(math.gcd(8, prev), prev), nn.SiLU(), nn.Conv2d(prev, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class PictureTokenizer(nn.Module):
    """Encoder + ResidualVQ + decoder. Pictures are `(B, 3, H, W)` floats in [-1, 1]
    (`vision.image.to_tensor`), codes are `(B, N, h, w)` longs."""

    def __init__(self, cfg: PictureConfig):
        super().__init__()
        if cfg.image_size % cfg.downsample:
            raise ValueError(f"image_size {cfg.image_size} is not divisible by "
                             f"{cfg.downsample} (2 ** {len(cfg.channels)} downsampling steps)")
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.vq = ResidualVQ(cfg.latent_dim, cfg.n_codebooks, cfg.codebook_size,
                             dropout=cfg.quantizer_dropout, commit=cfg.commit,
                             restart_after=cfg.restart_after)
        self.decoder = Decoder(cfg)

    def _quantize(self, z, n_active=None):
        B, D, h, w = z.shape
        q, idx, commit, stats = self.vq(z.reshape(B, D, h * w), n_active=n_active)
        return q.reshape(B, D, h, w), idx.reshape(B, -1, h, w), commit, stats

    def forward(self, x, n_active: int | None = None):
        """`(reconstruction, codes, commitment loss, [VQStats])`."""
        q, idx, commit, stats = self._quantize(self.encoder(x), n_active)
        return self.decoder(q), idx, commit, stats

    @torch.no_grad()
    def encode(self, x, n_active: int | None = None) -> torch.Tensor:
        """Pictures -> `(B, N, h, w)` codes. Codebooks past `n_active` are zeros."""
        n = self.cfg.n_codebooks if n_active is None else n_active
        _, idx, _, _ = self._quantize(self.encoder(x), n)
        return idx

    @torch.no_grad()
    def decode(self, idx: torch.Tensor, n_active: int | None = None) -> torch.Tensor:
        """`(B, N, h, w)` codes -> pictures, from the first `n_active` codebooks."""
        B, N, h, w = idx.shape
        z = self.vq.decode(idx.reshape(B, N, h * w), n_active=n_active)
        return self.decoder(z.reshape(B, -1, h, w))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 + L2 on pixels. L2 alone blurs edges (it forgives many small errors and punishes
    few large ones); L1 alone is slow to fix the colour of a whole flat region. A shape picture
    is mostly edges and flat regions, so it gets both."""
    return F.l1_loss(recon, target) + F.mse_loss(recon, target)
