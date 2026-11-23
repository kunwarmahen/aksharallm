"""One door for every model that draws from a caption: G1 (codes, `stage: picgen`), G2
(flow matching in pixels, `stage: flow`) and G3 (the same in the tokenizer's latent,
`stage: flow` with `space: latent`). The CLI's `draw` / `gen-report` and the portal's
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
    #: G3 only: re-quantize a drawn latent before decoding (None = the checkpoint's choice).
    snap: bool | None = None
    #: The VQ tokenizer that scores copies and variety (the decoder's own, unless it is an
    #: autoencoder with no codes).
    scorer: object = None

    @property
    def step(self):
        return self.blob.get("step")

    @property
    def books(self) -> int:
        return self.cfg.books

    def draw(self, captions: list[str], *, guidance: float = 1.0, seed: int = 0,
             temperature: float = 1.0, top_k: int | None = None,
             steps: int | None = None) -> np.ndarray:
        """Pictures only. A flow model's are not encoded back to codes, which also lets a
        coder with no codes (G4c's photo autoencoder, no VQ scorer) draw."""
        if self.kind == "G1":
            return self.draw_with_tokens(captions, guidance=guidance, seed=seed,
                                         temperature=temperature, top_k=top_k)[0]
        from .flow import draw
        return draw(self.model, captions, guidance=guidance, steps=steps, seed=seed,
                    device=self.device, tokenizer=self.tokenizer, snap=self.snap)

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
                    device=self.device, tokenizer=self.tokenizer, snap=self.snap)
        return pics, tokens_of(pics, self.scorer, self.books, self.device)

    def seen(self) -> set[bytes]:
        """The training pictures' codes, for the copy check (empty if the codes are absent)."""
        from .codes import CodesDataset
        from .score import train_code_set
        codes = (self.blob.get("config") or {}).get("data", {}).get("codes")
        if not codes or not (self.root / codes / "codes.npy").is_file():
            return set()
        return train_code_set(CodesDataset(self.root / codes, books=self.books, split="train"))

    def corpus_kind(self) -> str:
        """"shapes" or "scenes" (G4b): the kind of corpus this model trained on, read from the
        corpus's manifest, so a report asks it for the facts it can say."""
        from ..vision.image import manifest_kind
        corpus = (self.blob.get("config") or {}).get("data", {}).get("corpus")
        return manifest_kind(self.root / str(corpus)) if corpus else "shapes"

    def report(self, *, k: int = 8, guidance: float = 1.0, seed: int = 0,
               temperature: float = 1.0, top_k: int | None = None, steps: int | None = None,
               captions: list[str] | None = None, phrasing: str | None = None,
               clip=None) -> dict:
        """Draw `k` per caption and judge them. `phrasing` (G4a) says every fact (60 shapes,
        or 270 scenes) in that phrasing (phrases.py) instead of the canonical caption; facts,
        not captions, are what the reader checks. `clip` (G4b: `(model, checkpoint)`) also
        scores every picture with our contrastive model, beside the reader (`own_clip`)."""
        from .phrases import all_facts, phrase
        from .score import judge_drawings
        kind = self.corpus_kind()
        facts = None
        if phrasing or kind != "shapes" or clip is not None:
            facts = all_facts(kind)
            captions = [phrase(f, phrasing or "canonical") for f in facts]
        if captions is None:
            from .codes import all_captions
            captions = all_captions()
        t0 = time.time()
        pics, toks = self.draw_with_tokens([c for c in captions for _ in range(k)],
                                           guidance=guidance, seed=seed,
                                           temperature=temperature, top_k=top_k, steps=steps)
        out = judge_drawings(captions, k, pics, toks, self.seen(), facts)
        if clip is not None:
            from .clip import score_drawings
            out["own_clip"] = score_drawings(clip[0], out, all_facts(kind), device=self.device,
                                             checkpoint=str(clip[1]), kind=kind)
        return out | {"model": self.kind, "corpus_kind": kind, "phrasing": phrasing,
                      "guidance": guidance, "temperature": temperature,
                      "top_k": top_k, "steps": steps, "seed": seed,
                      "seconds": round(time.time() - t0, 1)}


def stage_of(path: str | Path) -> str | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("stage")
    except Exception:
        return None


def load_drawer(path: str | Path, root: str | Path, device: str = "cpu") -> Drawer:
    """A G1 or G2 checkpoint, with the tokenizer it is scored (and, for G1, decoded) with."""
    from .train import load_coder, load_tokenizer
    root, path = Path(root), Path(path)
    stage = stage_of(path)
    if stage == "picgen":
        from .gen import load_generator
        model, cfg, blob = load_generator(path, device)
        kind = "G1"
    elif stage == "flow":
        from .flow import load_flow
        model, cfg, blob = load_flow(path, device, root=root)
        kind = "G4" if cfg.text else ("G3" if cfg.space == "latent" else "G2")
    else:
        raise ValueError(f"{path} is not a drawing model (a G1 picgen or a G2/G3 flow checkpoint)")
    def at(p):
        p = Path(p)
        return p if p.is_absolute() else root / p
    tokenizer, _ = load_coder(at(cfg.tokenizer), device)
    scorer = tokenizer
    if getattr(cfg, "score_tokenizer", None):
        scorer, _ = load_tokenizer(at(cfg.score_tokenizer), device)
    return Drawer(kind, path, model, cfg, blob, tokenizer, root, device, scorer=scorer)
