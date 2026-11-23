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
the text model never changes. COCO (G4c) has 590,000 captions, so the cache is bounded
(`cache_limit`): past it, captions are encoded every time, the uncached ones of a batch in ONE
padded forward. Padding goes at the end, and a causal model's vector at a real token never
sees what comes after it, so a padded row is exactly the row alone (tested).

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from pathlib import Path

import torch


class TextEncoder:
    def __init__(self, checkpoint: str | Path, tokenizer: str | Path, max_len: int = 16,
                 device: str = "cpu", cache_limit: int | None = None):
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
        self.cache_limit = cache_limit

    @torch.no_grad()
    def _many(self, captions: list[str]) -> list[tuple[torch.Tensor, int]]:
        """Captions -> their vectors, in one forward padded at the end."""
        ids = [self.tok.encode(c)[: self.max_len] for c in captions]
        longest = max((len(i) for i in ids), default=0)
        outs = [(torch.zeros(self.max_len, self.dim), len(i)) for i in ids]
        if longest == 0:
            return outs
        batch = torch.zeros(len(ids), longest, dtype=torch.long)
        for r, i in enumerate(ids):
            batch[r, : len(i)] = torch.tensor(i, dtype=torch.long)
        h, _ = self.model(batch.to(self.device), return_hidden=True)
        h = h.float().cpu()
        for r, (vec, n) in enumerate(outs):
            vec[:n] = h[r, :n]
        return outs

    def _one(self, caption: str) -> tuple[torch.Tensor, int]:
        return self._many([caption])[0]

    def encode(self, captions: list[str], with_mask: bool = False):
        """Captions -> `(B, max_len, dim)` vectors (zeros past the caption's length). With
        `with_mask`, also the `(B, max_len)` bool mask of real tokens (G4b pools over it)."""
        missing = list(dict.fromkeys(c for c in captions if c not in self.cache))
        fresh = {}
        for i in range(0, len(missing), 256):
            fresh.update(zip(missing[i:i + 256], self._many(missing[i:i + 256]), strict=True))
        for c, v in fresh.items():
            if self.cache_limit is None or len(self.cache) < self.cache_limit:
                self.cache[c] = v
        got = [self.cache.get(c) or fresh[c] for c in captions]
        vecs = torch.stack([g[0] for g in got])
        if not with_mask:
            return vecs
        lens = torch.tensor([g[1] for g in got])
        return vecs, torch.arange(self.max_len)[None] < lens[:, None]
