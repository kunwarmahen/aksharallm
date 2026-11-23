"""G4a's text reader: a frozen language model of our own turns a caption into vectors.

Real text-to-image models read the prompt with a pretrained text encoder (T5 for Imagen,
CLIP's for Stable Diffusion) and never train it. G4a does the same with `tiny`, the 13.8M
TinyStories model of Phase 1: tokenize the caption with its own tokenizer, run the model, keep
the final hidden state at every token. Those vectors, projected to the flow model's width,
replace G1–G3's three word tokens in front of the picture.

The model is causal, so a token's vector has read only the words before it; the flow model,
which is bidirectional, sees all of them at once, so it can still combine "red" with "circles"
wherever they come. Captions are padded with zero vectors to `max_len` tokens; the empty
caption (for guidance) is all zeros.

Every caption's vectors are cached, because training uses a few hundred distinct sentences and
the text model never changes.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from pathlib import Path

import torch


class TextEncoder:
    def __init__(self, checkpoint: str | Path, tokenizer: str | Path, max_len: int = 16,
                 device: str = "cpu"):
        from ..infer.cli import load_model
        from ..tokenizer.tokenizer import Tokenizer
        self.model, ckpt = load_model(str(checkpoint), device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = Tokenizer(str(tokenizer))
        self.max_len, self.device = max_len, device
        self.dim = self.model.cfg.d_model
        self.cache: dict[str, tuple[torch.Tensor, int]] = {}

    @torch.no_grad()
    def _one(self, caption: str) -> tuple[torch.Tensor, int]:
        ids = self.tok.encode(caption)[: self.max_len]
        out = torch.zeros(self.max_len, self.dim)
        if ids:
            h, _ = self.model(torch.tensor([ids], device=self.device), return_hidden=True)
            out[: len(ids)] = h[0].float().cpu()
        return out, len(ids)

    def encode(self, captions: list[str], with_mask: bool = False):
        """Captions -> `(B, max_len, dim)` vectors (zeros past the caption's length). With
        `with_mask`, also the `(B, max_len)` bool mask of real tokens (G4b pools over it)."""
        for c in captions:
            if c not in self.cache:
                self.cache[c] = self._one(c)
        vecs = torch.stack([self.cache[c][0] for c in captions])
        if not with_mask:
            return vecs
        lens = torch.tensor([self.cache[c][1] for c in captions])
        return vecs, torch.arange(self.max_len)[None] < lens[:, None]
