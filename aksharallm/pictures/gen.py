"""G1: a transformer that reads "two red circles" and writes the picture's codes.

The model is `model/transformer.py`'s `Transformer`, imported unchanged: a picture is a
sequence of integers (codes.py), so drawing one is next-token prediction over a vocabulary of
words and codes. What is new is all at the edges:

* **the slot mask.** A position-major sequence alternates codebooks (cell 0 book 0, cell 0
  book 1, cell 1 book 0, ...), so at every step only one codebook's 512 ids are legal. The
  sampler sets every other logit to -inf. Training needs no mask: the model learns where each
  codebook's ids go within a few hundred steps, and a loss over the legal ids alone would hide
  it if it had not.
* **classifier-free guidance.** The model is trained with `cfg_drop` of its captions replaced
  by `NULL`, so it can also draw "a picture from this corpus" with no caption. At sampling time
  each step runs both, and the logits are pushed away from the unconditional ones:

      logits = uncond + w * (cond - uncond)

  `w = 1` is the conditional model as trained; `w > 1` trades variety for doing what the
  caption says. It is the knob every text-to-image system has (Imagen, Stable Diffusion), and
  here it can be measured exactly, because the reader says whether the picture obeys.

```
caption -> [w w w IMG] -> transformer -> code, code, ... (128 of them) -> tokenizer.decode -> picture
                                ^ cond and NULL rows side by side, mixed by w at every step
```

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..config import ModelConfig
from ..model.transformer import KVCache, Transformer
from .codes import CAPTION_LEN, CODE0, IMG, NULL, unflatten, vocab_size, word_ids


@dataclass
class PicGenConfig:
    #: The tokenizer whose codes this model writes, and whose decoder paints them.
    tokenizer: str = "checkpoints/imgtok-shapes/ckpt_best.pt"
    #: How many of the tokenizer's codebooks to model. 1 is 64 codes a picture; 2 is 128.
    books: int = 2
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    #: Share of training captions replaced by NULL, so the model can draw unconditionally
    #: (what classifier-free guidance subtracts). 0 disables guidance.
    cfg_drop: float = 0.1


def model_config(pg: PicGenConfig, codebook_size: int, grid: int) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size(pg.books, codebook_size), d_model=pg.d_model,
                       n_layers=pg.n_layers, n_heads=pg.n_heads, dropout=pg.dropout,
                       max_seq_len=CAPTION_LEN + 1 + pg.books * grid * grid)


def build(pg: PicGenConfig, codebook_size: int, grid: int) -> Transformer:
    return Transformer(model_config(pg, codebook_size, grid))


def load_generator(path: str | Path, device: str = "cpu"):
    """A trained G1 checkpoint -> `(model, PicGenConfig, blob)` in eval mode."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "picgen":
        raise ValueError(f"{path} is not a G1 picture generator checkpoint")
    pg = PicGenConfig(**blob["picgen"])
    model = Transformer(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), pg, blob


def slot_range(step: int, books: int, codebook_size: int) -> tuple[int, int]:
    """The legal token ids for code number `step` of a picture (position-major order)."""
    b = step % books
    lo = CODE0 + b * codebook_size
    return lo, lo + codebook_size


@torch.no_grad()
def sample(model: Transformer, captions: list[str], *, books: int, codebook_size: int,
           grid: int, guidance: float = 1.0, temperature: float = 1.0, top_k: int | None = None,
           seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Captions -> `(B, books, grid, grid)` codes. One KV cache over 2B rows (conditional and
    unconditional side by side, the same length, so one cache serves both).

    Seeded per call: the same captions, seed and model give the same pictures, which is what
    lets two guidance settings be compared picture by picture.
    """
    model.eval()
    B = len(captions)
    words = torch.tensor([word_ids(c) for c in captions], dtype=torch.long)
    guide = guidance != 1.0
    if guide:
        words = torch.cat([words, torch.full_like(words, NULL)])
    prompt = torch.cat([words, torch.full((len(words), 1), IMG)], dim=1).to(device)
    R = len(prompt)
    cfg = model.cfg
    n_kv = cfg.n_kv_heads or cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads
    dtype = next(model.parameters()).dtype
    caches = [KVCache(R, n_kv, cfg.max_seq_len, head_dim, dtype, device)
              for _ in range(cfg.n_layers)]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_codes = books * grid * grid
    out = torch.empty(B, n_codes, dtype=torch.long)
    logits, _ = model(prompt, caches=caches)
    for t in range(n_codes):
        lg = logits[:, -1, :].float()
        if guide:
            cond, unc = lg[:B], lg[B:]
            lg = unc + guidance * (cond - unc)
        lo, hi = slot_range(t, books, codebook_size)
        lg = lg[:, lo:hi] / max(temperature, 1e-6)
        if top_k:
            kth = lg.topk(min(top_k, lg.shape[-1]), dim=-1).values[:, -1:]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        probs = torch.softmax(lg, dim=-1).cpu()
        pick = torch.multinomial(probs, 1, generator=gen).squeeze(1) + lo
        out[:, t] = pick
        nxt = pick.repeat(2) if guide else pick
        logits, _ = model(nxt[:, None].to(device), caches=caches)
    return unflatten(out, books, codebook_size, grid)


@torch.no_grad()
def draw(model, tokenizer, captions: list[str], *, pg: PicGenConfig, guidance: float = 1.0,
         temperature: float = 1.0, top_k: int | None = None, seed: int = 0,
         device: str = "cpu", batch: int = 64):
    """Captions -> `(pictures (B,64,64,3) uint8, codes (B, books, grid, grid))`."""
    from ..vision.image import from_tensor
    tc = tokenizer.cfg
    pics, codes = [], []
    for i in range(0, len(captions), batch):
        c = sample(model, captions[i:i + batch], books=pg.books, codebook_size=tc.codebook_size,
                   grid=tc.grid, guidance=guidance, temperature=temperature, top_k=top_k,
                   seed=seed + i, device=device)
        full = torch.zeros(len(c), tc.n_codebooks, tc.grid, tc.grid, dtype=torch.long)
        full[:, :pg.books] = c
        img = tokenizer.decode(full.to(device), n_active=pg.books).float().cpu().numpy()
        pics.append(from_tensor(img))
        codes.append(c)
    import numpy as np
    return np.concatenate(pics), torch.cat(codes)


def public_config(pg: PicGenConfig) -> dict:
    return asdict(pg)
