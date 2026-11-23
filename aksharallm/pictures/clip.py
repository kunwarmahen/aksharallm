"""G4b: our own contrastive picture-caption model, and the adherence score it gives.

Every drawing so far was judged by the rule-based reader (reader.py), which only works because
we drew the corpus ourselves. Real text-to-image models are judged by a model that has learned
which captions go with which pictures: CLIP (OpenAI's contrastive model) and the scores built
on it. We cannot borrow theirs, so G4b trains one of our own, on our own pictures and captions,
and then checks it against the reader on the pictures where both apply: a learned judge is only
trusted where it agrees with an exact one, which is what lets it be used later where no exact
judge exists (photos, G4c).

```mermaid
flowchart LR
    P["picture 64x64"] --> V["ViT (docs/22's tower)<br/>mean over patches"] --> PI["linear -> 128<br/>unit length"]
    C["caption"] --> T["tiny, FROZEN<br/>one vector per token"] --> H["small text head<br/>(masked attention,<br/>masked mean)"] --> TI["linear -> 128<br/>unit length"]
    PI --> S["cosine x learned scale"]
    TI --> S
    S --> L["cross-entropy both ways:<br/>which caption is this picture's?<br/>which picture is this caption's?"]
```

**Several pictures in a batch say the same thing.** CLIP's loss treats the diagonal as the only
match: picture *i* goes with caption *i* and every other caption in the batch is a negative.
With 60 shapes facts and 256 pictures a batch, each fact appears four times, so plain CLIP
would push "two red circles" away from three other pictures of two red circles, and away from
three other phrasings of the same fact. Here the target is every caption of the same FACT,
equally (`clip_loss`, `same`), which is the version of the loss that is true of this data.
`tests/test_clip.py` checks that a model which already matches every picture to its fact gets
no gradient from it, and that the diagonal-only loss would still push it.

**It can say "none of these fit".** A closed-set judge picks the best of the captions it is
offered, so a picture of five shapes, with no caption saying "five", is called "four". A cosine
floor fixes that (`calibrate`): the 1st percentile of how well clean originals match their own
caption, so by construction about 1% of correct pictures are refused, and a picture whose best
caption fits worse than that is judged "none fits". The floor is written beside the checkpoint
(`none_fits.json`, with the step it was measured at) and ignored if the step does not match.

**The score it gives is named for what it is.** `own_clip` in a report is the share of drawn
pictures whose closest caption, among every caption the corpus can say, is the one that was
asked for, in this model's feature space. It is not CLIPScore and not FID (gotcha 12): those
are numbers in OpenAI's and Inception's feature spaces, and this is neither.

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
import torch.nn.functional as F

from ..vision.encoder import VisionConfig, VisionEncoder


@dataclass
class ClipConfig:
    image_size: int = 64
    patch: int = 8
    d_vision: int = 192
    n_layers: int = 4
    n_heads: int = 4
    #: the trainable text head over the frozen model's vectors
    d_text: int = 192
    text_layers: int = 1
    d_embed: int = 128
    #: the softmax temperature at step 0 (learned; CLIP's 0.07)
    temperature: float = 0.07
    #: the frozen text model and its tokenizer (G4a's)
    text: str = "checkpoints/tiny/ckpt_best.pt"
    text_tokenizer: str = "data/tinystories/tokenizer.json"
    text_len: int = 20
    text_dim: int = 0             # filled from the text model
    #: where the frozen text model runs: "cpu" (tiny), or "auto" (the card if there is one:
    #: the 300M reading COCO's captions)
    text_device: str = "cpu"
    #: captions whose vectors are kept (0 = all: fine for shapes' few thousand, not for COCO)
    text_cache: int = 0


def fact_key(f: dict) -> tuple:
    """What two captions must share to describe the same picture: the facts of a drawn
    picture, or the photo itself for COCO (G4c), whose five captions all describe it."""
    if "id" in f:
        return ("photo", f["id"])
    return tuple(f.get(k) for k in ("count", "size", "colour", "shape", "where"))


def text_of(f: dict, phrasing: str = "canonical", i: int = 0) -> str:
    """The caption to read for a fact: a phrasing of a drawn picture's facts, or the `i`-th
    of a COCO photo's own captions (its people-written sentences; phrasings do not apply)."""
    if "captions" in f:
        return f["captions"][i % len(f["captions"])]
    from .phrases import phrase
    return phrase(f, phrasing)


def same_fact(facts: list[dict]) -> torch.Tensor:
    """`(B, B)` float: 1 where two rows of a batch describe the same fact."""
    keys = [fact_key(f) for f in facts]
    idx = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    ids = torch.tensor([idx[k] for k in keys])
    return (ids[:, None] == ids[None, :]).float()


def clip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor,
              same: torch.Tensor) -> torch.Tensor:
    """Symmetric contrastive loss with every same-fact pair as a positive.

    `img`, `txt`: `(B, d)` unit vectors; `same`: `(B, B)` 0/1. Each row's target spreads
    evenly over its positives, so with one positive per row (`same` = identity) this is CLIP's
    loss exactly."""
    logits = scale * img @ txt.t()
    same = same.to(logits.dtype)
    to_txt = same / same.sum(1, keepdim=True)
    to_img = same.t() / same.t().sum(1, keepdim=True)
    li = -(to_txt * F.log_softmax(logits, 1)).sum(1).mean()
    lt = -(to_img * F.log_softmax(logits.t(), 1)).sum(1).mean()
    return (li + lt) / 2


class _TextBlock(nn.Module):
    """A pre-norm bidirectional block that ignores padding (the frozen vectors are zero past
    the caption, and a pad must neither be attended to nor counted in the pool)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        shape = (b, t, self.n_heads, d // self.n_heads)
        q, k, v = (z.reshape(shape).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x))


class ClipModel(nn.Module):
    def __init__(self, cc: ClipConfig):
        super().__init__()
        self.cfg = cc
        self.vision = VisionEncoder(VisionConfig(image_size=cc.image_size, patch=cc.patch,
                                                 d_vision=cc.d_vision, n_layers=cc.n_layers,
                                                 n_heads=cc.n_heads))
        self.img_out = nn.Linear(cc.d_vision, cc.d_embed)
        self.text_in = nn.Linear(cc.text_dim, cc.d_text)
        self.text_blocks = nn.ModuleList(_TextBlock(cc.d_text, cc.n_heads)
                                         for _ in range(cc.text_layers))
        self.text_norm = nn.LayerNorm(cc.d_text)
        self.txt_out = nn.Linear(cc.d_text, cc.d_embed)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / cc.temperature)))
        self._text = []

    def attach_text(self, encoder) -> None:
        """The frozen text model (text.py). Not a submodule: it is never trained or saved."""
        self._text = [encoder]

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.clamp(max=math.log(100)).exp()     # CLIP's cap at 100

    def embed_images(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, 3, 64, 64)` in [-1, 1] -> `(B, d_embed)` unit vectors."""
        return F.normalize(self.img_out(self.vision(x).mean(1)), dim=-1)

    def embed_vectors(self, vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The frozen text model's `(B, L, text_dim)` vectors -> `(B, d_embed)` unit vectors."""
        h = self.text_in(vecs)
        for blk in self.text_blocks:
            h = blk(h, mask)
        m = mask[..., None].to(h.dtype)
        pooled = (self.text_norm(h) * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.txt_out(pooled), dim=-1)

    def embed_texts(self, captions: list[str], device="cpu") -> torch.Tensor:
        if not self._text:
            raise ValueError("attach the frozen text model first (attach_text)")
        vecs, mask = self._text[0].encode(captions, with_mask=True)
        return self.embed_vectors(vecs.to(device), mask.to(device))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def resolve_device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def text_encoder_for(cc, device: str | None = None, root: str | Path | None = None):
    """The frozen text model a G4b (or G4a/G4c flow) config names. `device` overrides the
    config's `text_device`; paths are relative to `root` if given."""
    from .text import TextEncoder

    def at(p):
        return Path(root) / p if root is not None and not Path(p).is_absolute() else Path(p)
    dev = resolve_device(device or getattr(cc, "text_device", "cpu"))
    return TextEncoder(at(cc.text), at(cc.text_tokenizer), cc.text_len, dev,
                       cache_limit=getattr(cc, "text_cache", 0) or None)


NONE_FITS = "none_fits.json"


def load_clip(path: str | Path, device: str = "cpu", root: str | Path | None = None):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "clip":
        raise ValueError(f"{path} is not a G4b contrastive model (stage {blob.get('stage')!r})")
    cc = ClipConfig(**blob["clip"])
    model = ClipModel(cc)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    model.attach_text(text_encoder_for(cc, None if device == "cuda" else "cpu", root))
    model.none_fits = read_floor(path, blob.get("step"))
    return model, blob


def read_floor(path: str | Path, step) -> dict | None:
    """The none-fits floors measured for THIS checkpoint (same file name and step), or None.
    A floor measured on another step's embeddings is a different scale and is ignored."""
    f = Path(path).parent / NONE_FITS
    if not f.is_file():
        return None
    import json
    d = json.loads(f.read_text())
    mine = d.get(Path(path).name) or {}
    return mine.get("floors") if mine.get("step") == step else None


@torch.no_grad()
def calibrate(model: ClipModel, sets: list[tuple[str, np.ndarray, list[dict]]], *,
              q: float = 0.01, device: str = "cpu") -> dict:
    """Per corpus kind: the `q` quantile of the cosine between clean original pictures and
    their own canonical caption. `sets` = (kind, pictures, facts)."""
    out = {}
    for kind, pics, facts in sets:
        from .phrases import all_facts
        r = rank(model, pics, facts, all_facts(kind), device=device)
        out[kind] = float(np.quantile(np.asarray(r["cos"]), q))
    return out


def public_config(cc: ClipConfig) -> dict:
    return asdict(cc)


# ---------------------------------------------------------------------------------------
# scoring: which caption does a picture say?
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def picture_vectors(model: ClipModel, pics: np.ndarray, device="cpu", batch=256) -> torch.Tensor:
    from ..vision.image import to_tensor
    out = []
    for i in range(0, len(pics), batch):
        x = torch.from_numpy(to_tensor(np.asarray(pics[i:i + batch]))).to(device)
        out.append(model.embed_images(x).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def rank(model: ClipModel, pics: np.ndarray, asked: list[dict], candidates: list[dict], *,
         phrasing: str = "canonical", device: str = "cpu", floor: float | None = None) -> dict:
    """For each picture: is the caption it was drawn for the closest of all `candidates`
    (every fact the corpus can say, in `phrasing`)? Returns per-picture `top1` (bool),
    `best` (index of the closest candidate), `cos` (cosine with the asked caption) and
    `margin` (that cosine minus the best other one). With a `floor`, also `fits` (the best
    caption's cosine reaches it) and `top1` requires it: below the floor the answer is "none
    of these fit", whichever caption happened to be closest."""
    keys = [fact_key(f) for f in candidates]
    where = {k: i for i, k in enumerate(keys)}
    want = [where[fact_key(f)] for f in asked]
    txt = torch.cat([model.embed_texts([text_of(f, phrasing) for f in candidates[i:i + 512]],
                                       device).float().cpu()
                     for i in range(0, len(candidates), 512)])
    sims = picture_vectors(model, pics, device) @ txt.t()                # (N, C)
    best = sims.argmax(1)
    w = torch.tensor(want)
    cos = sims[torch.arange(len(w)), w]
    other = sims.clone()
    other[torch.arange(len(w)), w] = -2.0
    out = {"top1": (best == w).tolist(), "best": best.tolist(), "cos": cos.tolist(),
           "margin": (cos - other.max(1).values).tolist()}
    if floor is not None:
        fits = sims.max(1).values >= floor
        out["closed_top1"] = out["top1"]
        out["fits"] = fits.tolist()
        out["top1"] = ((best == w) & fits).tolist()
    return out


def agreement(model_ok: list[bool], reader_ok: list[bool]) -> dict:
    """How often the learned judge and the exact one give the same verdict, and Cohen's
    kappa (agreement beyond what their two rates would give by chance)."""
    n = len(model_ok)
    both = sum(a and b for a, b in zip(model_ok, reader_ok, strict=True))
    neither = sum((not a) and (not b) for a, b in zip(model_ok, reader_ok, strict=True))
    model_only = sum(a and not b for a, b in zip(model_ok, reader_ok, strict=True))
    reader_only = sum(b and not a for a, b in zip(model_ok, reader_ok, strict=True))
    agree = (both + neither) / n if n else None
    pm, pr = (both + model_only) / max(n, 1), (both + reader_only) / max(n, 1)
    chance = pm * pr + (1 - pm) * (1 - pr)
    kappa = None if not n or chance >= 1 else (agree - chance) / (1 - chance)
    return {"n": n, "both": both, "neither": neither, "model_only": model_only,
            "reader_only": reader_only, "agree": agree, "kappa": kappa}


def score_drawings(model: ClipModel, result: dict, candidates: list[dict], *,
                   device: str = "cpu", checkpoint: str = "", kind: str = "shapes") -> dict:
    """The `own_clip` block a report gains: adherence by our contrastive model, seen and
    held-out separately, beside its agreement with the reader on the same pictures. With a
    calibrated floor (`none_fits.json`), "none fits" counts as wrong; the closed-set verdict
    is kept beside it (`closed_top1`)."""
    k, pics = result["k"], result["_pictures"]
    facts = [p["fact"] for p in result["per_caption"]]
    asked = [f for f in facts for _ in range(k)]
    floor = (getattr(model, "none_fits", None) or {}).get(kind)
    r = rank(model, pics, asked, candidates, device=device, floor=floor)
    reader_ok = [v == "1" for p in result["per_caption"]
                 for v in _verdicts(p, k)]
    held = [p["held_out"] for p in result["per_caption"] for _ in range(k)]

    def part(sel):
        if not sel:
            return {"n": 0}
        return {"n": len(sel), "top1": sum(r["top1"][i] for i in sel) / len(sel),
                "cos": float(np.mean([r["cos"][i] for i in sel])),
                "correct": "".join("1" if r["top1"][i] else "0" for i in sel),
                "agreement": agreement([r["top1"][i] for i in sel],
                                       [reader_ok[i] for i in sel])}
    seen = [i for i in range(len(asked)) if not held[i]]
    out = [i for i in range(len(asked)) if held[i]]
    res = {"checkpoint": checkpoint, "candidates": len(candidates), "floor": floor,
           "seen": part(seen), "held_out": part(out),
           "agreement": agreement(r["top1"], reader_ok)}
    if floor is not None:
        res["closed"] = {"top1": float(np.mean(r["closed_top1"])),
                         "agreement": agreement(r["closed_top1"], reader_ok),
                         "none_fits": int(sum(not f for f in r["fits"]))}
    return res


def _verdicts(per_caption: dict, k: int) -> str:
    """The reader's verdict on each of a caption's k pictures, from what it read."""
    from .reader import ATTRS
    f = per_caption["fact"]
    return "".join("1" if all(r.get(a) == f[a] for a in ATTRS if a in f) else "0"
                   for r in per_caption["reads"][:k])
