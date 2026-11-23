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

**G3 is this file too**, with `space: latent`: the same model draws the tokenizer's normalised
8x8x64 latent (`patch: 1`, `channels: 64`, so 64 tokens instead of 256) and `draw` hands the
result to the tokenizer's decoder (`latents.py`). Nothing else changes, which is the point:
latent diffusion is pixel diffusion in a smaller space.

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
    #: "pixels" (G2) or "latent" (G3: the tokenizer's normalised 8x8x64 latent; the tokenizer
    #: above then also DECODES every drawing, so it must be the one that wrote the latents).
    space: str = "pixels"
    #: Channels of what is being drawn: 3 for pixels, the tokenizer's latent_dim for latents.
    channels: int = 3
    #: G3: re-quantize a drawn latent onto the codebooks before decoding.
    snap: bool = False
    #: The VQ tokenizer used to SCORE drawings (copies, variety) when `tokenizer` is a
    #: continuous autoencoder, which has no codes to compare. None = `tokenizer`.
    score_tokenizer: str | None = None
    #: G3: the per-channel statistics the latents were normalised with (latents.py). Saved in
    #: the config so a checkpoint can always map its drawings back.
    latent_mean: list | None = None
    latent_std: list | None = None
    #: G4a: read the caption with this FROZEN text model (a checkpoint) instead of the three
    #: word tokens. None = the three words (G1–G3). Its tokenizer and how many of its tokens
    #: to keep; `text_dim` is filled in from the model when it is loaded.
    text: str | None = None
    text_tokenizer: str | None = None
    text_len: int = 16
    text_dim: int = 0
    #: G4c: where the frozen text model runs ("cpu", or "auto" for the card) and how many
    #: captions' vectors it keeps (0 = all)
    text_device: str = "cpu"
    text_cache: int = 0

    @property
    def grid(self) -> int:
        return self.image_size // self.patch

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch * self.patch


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """`(B, C, H, W)` -> `(B, (H/p)*(W/p), C*p*p)`, rows of patches in raster order."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(t: torch.Tensor, p: int, size: int, channels: int = 3) -> torch.Tensor:
    B, N, D = t.shape
    g = size // p
    x = t.reshape(B, g, g, channels, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, channels, size, size)


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
        self.n_cap = fc.text_len if fc.text else CAPTION_LEN
        n_tokens = 1 + self.n_cap + fc.grid * fc.grid
        self.cfg = ModelConfig(vocab_size=1, d_model=d, n_layers=fc.n_layers,
                               n_heads=fc.n_heads, dropout=fc.dropout, causal=False, embeds_only=True,
                               max_seq_len=n_tokens)
        self.lm = Transformer(self.cfg)
        self.patch_in = nn.Linear(fc.patch_dim, d)
        self.row_emb = nn.Embedding(fc.grid, d)
        self.col_emb = nn.Embedding(fc.grid, d)
        self.word_emb = nn.Embedding(WORD0 + len(WORDS), d)       # ids as in codes.py
        self.slot_emb = nn.Embedding(self.n_cap, d)                # caption position
        # G4a: the frozen text model's vectors, projected to this width. The encoder itself is
        # not a submodule (it is frozen and never saved here): `attach_text` hands it over.
        self.text_proj = nn.Linear(fc.text_dim, d) if fc.text else None
        self._text = []
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
        if self.text_proj is not None:      # (B, text_len, text_dim) vectors from text.py
            cap = self.text_proj(words.to(patches.dtype)) + self.slot_emb.weight[None]
        else:                               # (B, 3) word ids
            cap = self.word_emb(words) + self.slot_emb.weight[None]
        seq = torch.cat([te[:, None], cap, patches], dim=1)
        h, _ = self.lm(None, inputs_embeds=seq, return_hidden=True)
        v = self.out(h[:, 1 + self.n_cap:])
        return unpatchify(v, fc.patch, fc.image_size, fc.channels)

    def attach_text(self, encoder) -> None:
        self._text = [encoder]

    def condition(self, captions: list[str], device="cpu") -> torch.Tensor:
        """Captions -> what `forward` takes as `words`: ids for G1–G3's three words, the frozen
        text model's vectors for G4a."""
        if self.text_proj is None:
            return torch.tensor([word_ids(c) for c in captions], dtype=torch.long, device=device)
        if not self._text:
            raise ValueError("a G4a model needs its text encoder attached (attach_text)")
        return self._text[0].encode(captions).to(device)

    def null_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """The empty caption guidance subtracts: NULL ids, or all-zero text vectors."""
        return torch.zeros_like(cond) if self.text_proj is not None else torch.full_like(cond, NULL)

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
    words = (model.condition(captions, device) if hasattr(model, "condition") else
             torch.tensor([word_ids(c) for c in captions], dtype=torch.long, device=device))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((B, fc.channels, fc.image_size, fc.image_size), generator=gen).to(device)
    guide = guidance != 1.0
    null = (model.null_condition(words) if hasattr(model, "null_condition")
            else torch.full_like(words, NULL))
    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        if guide:
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([words, null]))
            vc, vu = v[:B], v[B:]
            v = vu + guidance * (vc - vu)
        else:
            v = model(x, t, words)
        x = x + v.float() / steps
    # Pixels live in [-1, 1]; a latent has no such range, and clamping it would distort it.
    return x.clamp(-1, 1) if fc.space == "pixels" else x


@torch.no_grad()
def draw(model, captions: list[str], *, guidance: float = 1.0, steps: int | None = None,
         seed: int = 0, device: str = "cpu", batch: int = 64, tokenizer=None,
         snap: bool | None = None) -> np.ndarray:
    """Captions -> `(B, 64, 64, 3)` uint8 pictures. A G3 model (`space: latent`) draws a
    normalised latent, which is mapped back with the saved statistics and decoded by
    `tokenizer`'s decoder (the tokenizer that wrote the latents)."""
    from ..vision.image import from_tensor
    from .latents import decode_latent, denormalise, stats_tensors
    model.eval()
    fc = model.fc
    snap = fc.snap if snap is None else snap
    out = []
    for i in range(0, len(captions), batch):
        x = sample(model, captions[i:i + batch], steps=steps, guidance=guidance,
                   seed=seed + i, device=device)
        if fc.space == "latent":
            if tokenizer is None or fc.latent_mean is None:
                raise ValueError("a latent (G3) model needs its tokenizer and saved statistics "
                                 "to turn a drawing into pixels")
            m, s = stats_tensors(fc.latent_mean, fc.latent_std, device)
            x = decode_latent(tokenizer, denormalise(x, m, s), snap=snap).float().clamp(-1, 1)
        out.append(from_tensor(x.cpu().numpy()))
    return np.concatenate(out)


def load_flow(path: str | Path, device: str = "cpu", root: str | Path | None = None):
    """A trained G2 checkpoint -> `(model, FlowConfig, blob)` in eval mode."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "flow":
        raise ValueError(f"{path} is not a G2 flow checkpoint")
    fc = FlowConfig(**blob["flow"])
    model = FlowModel(fc)
    model.load_state_dict(blob["model"])
    if fc.text:
        model.attach_text(TextEncoder_for(fc, device, root))
    return model.to(device).eval(), fc, blob


def TextEncoder_for(fc: FlowConfig, device: str = "cpu", root: str | Path | None = None):
    """The frozen text model a G4a config names (paths relative to `root` if given). A
    `text_device: auto` config runs it on the card whenever the drawing does."""
    from .clip import text_encoder_for
    dev = "cpu" if fc.text_device == "cpu" or device == "cpu" else None   # None: the config's
    return text_encoder_for(fc, dev, root)


def public_config(fc: FlowConfig) -> dict:
    return asdict(fc)
