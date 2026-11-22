"""One door for every model that draws from a caption: G1 (codes, `stage: picgen`) and G2
(flow matching in pixels, `stage: flow`). The CLI's `draw` / `gen-report` and the portal's
Draw panel all go through `load_drawer`, so a new drawing model is one more branch here and
nothing else learns its name.

Both kinds are scored identically (`score.judge_drawings`): the reader on every picture, and
the tokenizer's codes of every picture for copies and variety. For G1 those codes are what it
wrote; for G2 the drawn pixels are encoded by the same tokenizer first.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

STAGES = ("picgen", "flow")


@dataclass
class Drawer:
    kind: str                     # "G1" or "G2"
    path: Path
    model: object
    cfg: object                   # PicGenConfig or FlowConfig
    blob: dict
    tokenizer: object
    root: Path
    device: str

    @property
    def step(self):
        return self.blob.get("step")

    @property
    def books(self) -> int:
        return self.cfg.books

    def draw(self, captions: list[str], **kw) -> np.ndarray:
        return self.draw_with_tokens(captions, **kw)[0]

    def draw_with_tokens(self, captions: list[str], *, guidance: float = 1.0, seed: int = 0,
                         temperature: float = 1.0, top_k: int | None = None,
                         steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pictures and their code tokens. G1's tokens are the codes it wrote (exactly what the
        first reports used); G2's are its pixels encoded by the tokenizer."""
        from .score import tokens_of
        if self.kind == "G1":
            from .codes import flatten
            from .gen import draw
            pics, codes = draw(self.model, self.tokenizer, captions, pg=self.cfg,
                               guidance=guidance, temperature=temperature, top_k=top_k,
                               seed=seed, device=self.device)
            toks = flatten(codes.numpy().reshape(len(captions), self.books, -1).astype(np.int64),
                           self.books, self.tokenizer.cfg.codebook_size)
            return pics, toks
        from .flow import draw
        pics = draw(self.model, captions, guidance=guidance, steps=steps, seed=seed,
                    device=self.device)
        return pics, tokens_of(pics, self.tokenizer, self.books, self.device)

    def seen(self) -> set[bytes]:
        """The training pictures' codes, for the copy check (empty if the codes are absent)."""
        from .codes import CodesDataset
        from .score import train_code_set
        codes = (self.blob.get("config") or {}).get("data", {}).get("codes")
        if not codes or not (self.root / codes / "codes.npy").is_file():
            return set()
        return train_code_set(CodesDataset(self.root / codes, books=self.books, split="train"))

    def report(self, *, k: int = 8, guidance: float = 1.0, seed: int = 0,
               temperature: float = 1.0, top_k: int | None = None, steps: int | None = None,
               captions: list[str] | None = None) -> dict:
        from .codes import all_captions
        from .score import judge_drawings
        captions = captions or all_captions()
        t0 = time.time()
        pics, toks = self.draw_with_tokens([c for c in captions for _ in range(k)],
                                           guidance=guidance, seed=seed,
                                           temperature=temperature, top_k=top_k, steps=steps)
        out = judge_drawings(captions, k, pics, toks, self.seen())
        return out | {"model": self.kind, "guidance": guidance, "temperature": temperature,
                      "top_k": top_k, "steps": steps, "seed": seed,
                      "seconds": round(time.time() - t0, 1)}


def stage_of(path: str | Path) -> str | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("stage")
    except Exception:
        return None


def load_drawer(path: str | Path, root: str | Path, device: str = "cpu") -> Drawer:
    """A G1 or G2 checkpoint, with the tokenizer it is scored (and, for G1, decoded) with."""
    from .train import load_tokenizer
    root, path = Path(root), Path(path)
    stage = stage_of(path)
    if stage == "picgen":
        from .gen import load_generator
        model, cfg, blob = load_generator(path, device)
        kind = "G1"
    elif stage == "flow":
        from .flow import load_flow
        model, cfg, blob = load_flow(path, device)
        kind = "G2"
    else:
        raise ValueError(f"{path} is not a drawing model (G1 picgen or G2 flow checkpoint)")
    tok = Path(cfg.tokenizer)
    tokenizer, _ = load_tokenizer(tok if tok.is_absolute() else root / tok, device)
    return Drawer(kind, path, model, cfg, blob, tokenizer, root, device)
