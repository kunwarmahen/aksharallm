"""G2: drawing a picture by flow matching, in pixels, with a diffusion transformer.

G1 wrote a picture one code at a time. G2 starts from pure noise and moves the WHOLE picture
towards a drawing in a few dozen small steps, steered by the caption at every one. That is how
Stable Diffusion 3, Flux and the Veo/Sora family generate, and the training rule is short:

    x0 ~ N(0, I)                      noise
    x1                                a training picture, in [-1, 1]
    x_t = (1 - t) * x0 + t * x1       a straight line from noise (t = 0) to picture (t = 1)
    the model predicts v = x1 - x0    the direction along that line (its "velocity")
    loss = || model(x_t, t, caption) - (x1 - x0) ||^2

Drawing is then an ODE: start at noise, take `steps` Euler steps of size 1/steps along the
predicted velocity, and arrive at a picture. This is **rectified flow** / flow matching. It
needs no noise schedule tables (DDPM's betas and alphas), and its paths are straight, which is
why it samples in tens of steps rather than a thousand.

**The model is the same transformer a third time** (`model/transformer.py`, unchanged):

```
picture 64x64x3 -> 16x16 patches of 4x4x3 -> linear -> + (row, column) embedding
[time token] [count word] [colour word] [shape word] [256 patch tokens]   (inputs_embeds)
    -> Transformer(causal=False) -> hidden states of the 256 patches -> linear -> velocity
```

`causal: false` is the key masked diffusion added (docs/20): every patch sees every other
patch and the caption, because the whole picture is denoised at once. The caption and the time
step enter **in context**, as four extra tokens, and the time embedding is also added to every
patch (cheap insurance that each patch knows how noisy it is). DiT's adaLN-Zero would need a
new block; in-context conditioning needs none, which is the point of this repo's seams.

Guidance is the same line as G1's, applied to velocities instead of logits:
`v = v_uncond + w * (v_cond - v_uncond)`.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model.transformer import Transformer
from .codes import CAPTION_LEN, NULL, WORD0, WORDS, word_ids


@dataclass
class FlowConfig:
    image_size: int = 64
    patch: int = 4
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    dropout: float = 0.0
    #: Share of training captions replaced by NULL (for guidance).
    cfg_drop: float = 0.1
    #: Euler steps when drawing. Straight paths: tens, not a thousand.
    sample_steps: int = 50
    #: The tokenizer used only to SCORE drawings (copies, variety), never to make them.
    tokenizer: str = "checkpoints/imgtok-shapes/ckpt_best.pt"
    books: int = 2

    @property
    def grid(self) -> int:
        return self.image_size // self.patch

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch * self.patch


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """`(B, 3, H, W)` -> `(B, (H/p)*(W/p), 3*p*p)`, rows of patches in raster order."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(t: torch.Tensor, p: int, size: int) -> torch.Tensor:
    B, N, D = t.shape
    g = size // p
    x = t.reshape(B, g, g, 3, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 3, size, size)


def time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of t in [0, 1] (scaled by 1000, as diffusion models do)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class FlowModel(nn.Module):
    """Velocity predictor: (noisy picture, t, caption words) -> velocity, all pixels at once."""

    def __init__(self, fc: FlowConfig):
        super().__init__()
        self.fc = fc
        d = fc.d_model
        n_tokens = 1 + CAPTION_LEN + fc.grid * fc.grid
        self.cfg = ModelConfig(vocab_size=1, d_model=d, n_layers=fc.n_layers,
                               n_heads=fc.n_heads, dropout=fc.dropout, causal=False, embeds_only=True,
                               max_seq_len=n_tokens)
        self.lm = Transformer(self.cfg)
        self.patch_in = nn.Linear(fc.patch_dim, d)
        self.row_emb = nn.Embedding(fc.grid, d)
        self.col_emb = nn.Embedding(fc.grid, d)
        self.word_emb = nn.Embedding(WORD0 + len(WORDS), d)       # ids as in codes.py
        self.slot_emb = nn.Embedding(CAPTION_LEN, d)               # count / colour / shape slot
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.out = nn.Linear(d, fc.patch_dim)
        for e in (self.row_emb, self.col_emb, self.word_emb, self.slot_emb):
            nn.init.normal_(e.weight, std=0.02)
        nn.init.zeros_(self.out.weight)          # start by predicting zero velocity
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        fc = self.fc
        g = torch.arange(fc.grid, device=x_t.device)
        pos = (self.row_emb(g)[:, None] + self.col_emb(g)[None]).reshape(fc.grid * fc.grid, -1)
        te = self.time_mlp(time_features(t, fc.d_model))                    # (B, d)
        patches = self.patch_in(patchify(x_t, fc.patch)) + pos[None] + te[:, None]
        cap = self.word_emb(words) + self.slot_emb.weight[None]             # (B, 3, d)
        seq = torch.cat([te[:, None], cap, patches], dim=1)
        h, _ = self.lm(None, inputs_embeds=seq, return_hidden=True)
        v = self.out(h[:, 1 + CAPTION_LEN:])
        return unpatchify(v, fc.patch, fc.image_size)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def sample_t(n: int, gen: torch.Generator | None = None, device="cpu") -> torch.Tensor:
    """Logit-normal times (Stable Diffusion 3): most training where the picture is half
    formed, which is where the decisions are, and few at the trivial ends."""
    return torch.sigmoid(torch.randn(n, generator=gen)).to(device)


def flow_loss(model: FlowModel, x1: torch.Tensor, words: torch.Tensor,
              gen: torch.Generator | None = None) -> torch.Tensor:
    """One flow-matching loss: the MSE between predicted and true velocity on the straight
    path from noise to `x1`. With `gen` the noise and times are reproducible (validation)."""
    B = x1.shape[0]
    x0 = torch.randn(x1.shape, generator=gen).to(x1.device, x1.dtype)
    t = sample_t(B, gen, x1.device)
    tt = t[:, None, None, None]
    x_t = (1 - tt) * x0 + tt * x1
    v = model(x_t, t, words)
    return ((v.float() - (x1 - x0).float()) ** 2).mean()


@torch.no_grad()
def sample(model, captions: list[str], *, steps: int | None = None, guidance: float = 1.0,
           seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Captions -> `(B, 3, H, W)` pictures in [-1, 1], by Euler steps from noise. Seeded, so
    the same call draws the same pictures (what makes two settings pairable)."""
    fc = model.fc
    steps = steps or fc.sample_steps
    B = len(captions)
    words = torch.tensor([word_ids(c) for c in captions], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((B, 3, fc.image_size, fc.image_size), generator=gen).to(device)
    guide = guidance != 1.0
    null = torch.full_like(words, NULL)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        if guide:
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([words, null]))
            vc, vu = v[:B], v[B:]
            v = vu + guidance * (vc - vu)
        else:
            v = model(x, t, words)
        x = x + v.float() / steps
    return x.clamp(-1, 1)


@torch.no_grad()
def draw(model, captions: list[str], *, guidance: float = 1.0, steps: int | None = None,
         seed: int = 0, device: str = "cpu", batch: int = 64) -> np.ndarray:
    """Captions -> `(B, 64, 64, 3)` uint8 pictures."""
    from ..vision.image import from_tensor
    model.eval()
    out = [from_tensor(sample(model, captions[i:i + batch], steps=steps, guidance=guidance,
                              seed=seed + i, device=device).cpu().numpy())
           for i in range(0, len(captions), batch)]
    return np.concatenate(out)


def load_flow(path: str | Path, device: str = "cpu"):
    """A trained G2 checkpoint -> `(model, FlowConfig, blob)` in eval mode."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "flow":
        raise ValueError(f"{path} is not a G2 flow checkpoint")
    fc = FlowConfig(**blob["flow"])
    model = FlowModel(fc)
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), fc, blob


def public_config(fc: FlowConfig) -> dict:
    return asdict(fc)
