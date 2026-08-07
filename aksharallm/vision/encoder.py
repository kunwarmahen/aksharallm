"""A vision transformer, and the two-layer projector that is the whole LLaVA idea.

The audio phase established that the transformer does not care what its tokens mean. Vision
makes the same point a second time and more cheaply, because it does not even need a codec:
an image is **already** a grid of numbers, and cutting it into patches turns it into a
sequence directly. There is nothing to quantize and nothing to learn a vocabulary for.

```mermaid
flowchart LR
    I["image<br/>3 x 64 x 64"] --> P["cut into 8x8 patches<br/>= 64 patches"]
    P --> E["linear: 192 -> d_vision<br/>+ learned positions"]
    E --> V["ViT blocks<br/>bidirectional attention"]
    V --> J["projector<br/>2-layer MLP"]
    J --> L["d_model — the SAME space<br/>the text embeddings live in"]
```

**The projector is the entire trick, and it is two matrices.** LLaVA's contribution was not
an architecture; it was noticing that a *frozen* language model will accept vectors from a
frozen vision encoder if you train a small MLP to put them in the right place. Everything
expensive is already trained; the only new parameters are the bridge.

Three things worth understanding before reading the code
---------------------------------------------------------
**Patches are a convolution.** Cutting a 64x64 image into 8x8 patches and projecting each
one is exactly `Conv2d(3, d, kernel_size=8, stride=8)` — same arithmetic, one kernel call,
and it is how every ViT implementation does it. Writing it as an explicit reshape first and
then noticing they are the same is worth doing once; `patchify` is here for that reason and
is asserted equal to the conv in the tests.

**Position embeddings are learned and absolute, not RoPE.** RoPE encodes *relative* position
along one axis, and an image has two. There is no ordering of patches for which "the patch
before this one" means anything consistent, so the ViT convention — a learned vector per grid
position — is what fits. This is the one place the vision tower deliberately does not reuse
the language model's machinery.

**Attention here is bidirectional.** A patch at the top-left should see the bottom-right;
there is nothing to predict left-to-right. That is the same `causal: false` flag the
diffusion phase added, reused unchanged.

Read with: docs/21-vision.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/03-model.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    """The tower's shape. Small: it is the cheap half of a vision-language model."""

    image_size: int = 64
    patch: int = 8
    d_vision: int = 256
    n_layers: int = 6
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    #: How many vectors the projector hands the language model per image. `None` means one
    #: per patch. Pooling to fewer is the knob that decides how much of the LM's context an
    #: image costs — 64 patches is 64 positions of a 1,024-token window, which is a lot to
    #: spend on one picture.
    n_tokens: int | None = None

    @property
    def grid(self) -> int:
        if self.image_size % self.patch:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by patch {self.patch}; "
                "a partial patch at the edge would be silently dropped"
            )
        return self.image_size // self.patch

    @property
    def n_patches(self) -> int:
        return self.grid * self.grid

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch * self.patch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """`(B, 3, H, W)` -> `(B, n_patches, 3·patch·patch)`, in row-major grid order.

    Written explicitly rather than as a conv so the shape is legible, and asserted equal to
    the conv in the tests. The order matters: row-major, so patch index `i` is at grid
    position `(i // grid, i % grid)`, which is what the position embedding assumes.
    """
    b, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} is not divisible by patch {patch}")
    gh, gw = h // patch, w // patch
    x = images.reshape(b, c, gh, patch, gw, patch)
    # (B, C, gh, p, gw, p) -> (B, gh, gw, C, p, p) -> (B, gh*gw, C*p*p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * patch * patch)
    return x


class Block(nn.Module):
    """A pre-norm transformer block. Bidirectional — there is no order to respect."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        d = cfg.d_vision
        self.n_heads = cfg.n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, d // self.n_heads)
        q, k, v = (z.reshape(shape).transpose(1, 2) for z in (q, k, v))
        # is_causal=False: a patch at the top-left must see the bottom-right.
        att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.mlp(self.norm2(x)))


class VisionEncoder(nn.Module):
    """Image in, one vector per patch out. A small ViT, written out."""

    def __init__(self, cfg: VisionConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or VisionConfig()
        self.embed = nn.Linear(cfg.patch_dim, cfg.d_vision)
        # Learned absolute positions, one per grid cell. See the module docstring for why
        # this is not RoPE.
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_vision))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_vision)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` -> `(B, n_patches, d_vision)`."""
        x = self.embed(patchify(images, self.cfg.patch)) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Projector(nn.Module):
    """`d_vision -> d_model`, two layers with a GELU between. **This is LLaVA.**

    LLaVA-1.0 used a single linear layer; LLaVA-1.5 found two with a nonlinearity measurably
    better, and it is still under a million parameters. The reason it works at all is that
    a language model's input space is not a code to be cracked — it is a space the model has
    learned to read, and any vector placed usefully within it gets read.

    `n_tokens` pools the patch sequence down before projecting. That trade is worth stating:
    64 patches is 64 positions of the language model's context spent on one image, and at
    `max_seq_len: 1024` that is 6% of everything the model can see. Pooling to 16 costs
    detail and buys context.
    """

    def __init__(self, d_vision: int, d_model: int, n_tokens: int | None = None,
                 hidden: int | None = None):
        super().__init__()
        self.n_tokens = n_tokens
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_vision, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """`(B, n_patches, d_vision)` -> `(B, n_tokens, d_model)`."""
        if self.n_tokens is not None and self.n_tokens != patches.shape[1]:
            # Average adjacent patches. Adaptive pooling over the *sequence* rather than the
            # 2-D grid: a 2-D pool would be more principled and needs the grid shape here,
            # and at these sizes the two agree to within noise. Documented rather than hidden.
            x = F.adaptive_avg_pool1d(patches.transpose(1, 2), self.n_tokens)
            patches = x.transpose(1, 2)
        return self.net(patches)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VisionTower(nn.Module):
    """Encoder plus projector: image in, language-model embeddings out.

    Kept as one module because they are trained together and saved together, and because
    "what goes into the language model" is one question with one answer.
    """

    def __init__(self, cfg: VisionConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionEncoder(cfg)
        self.projector = Projector(cfg.d_vision, d_model, cfg.n_tokens)

    @property
    def n_image_tokens(self) -> int:
        """How many positions of the language model's context one image costs."""
        return self.cfg.n_tokens or self.cfg.n_patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(images))

    def n_params(self) -> dict[str, int]:
        return {"encoder": self.encoder.n_params(),
                "projector": self.projector.n_params(),
                "total": sum(p.numel() for p in self.parameters())}
