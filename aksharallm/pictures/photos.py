"""G4c's report: photos drawn from COCO captions, judged in our own contrastive model's space.

There is no reader for a photo, so every number here comes from `clip-coco`, a contrastive
model trained like G4b's on COCO's photos and captions, and each one is printed beside the
same number for the REAL val photo of that caption (the ceiling) and for a mismatched pair
(the floor). A score with no ceiling and floor beside it cannot be read.

For each of the first `n` val captions (one per photo, its first caption), one picture is
drawn. Then:

* **match**: the cosine between the drawing and its own caption, against the real photo's
  cosine with the same caption, and the drawing's cosine with another photo's caption.
* **found in the top 1 / 5 / 10**: among the `n` captions, is the drawing's own caption one of
  the closest? R-precision in our own feature space; chance is 1/n, 5/n, 10/n.
* **distance to real photos**: the Fréchet distance between Gaussians fitted to the drawings'
  and the real photos' image vectors, with the same distance between two halves of the real
  photos beside it (what "indistinguishable" scores here). Our features, so **not FID**,
  which is the same formula in an Inception network's features (gotcha 12).

Per-caption verdicts (`top5` as one character each) are kept, so two reports on the same
captions pair with a sign test (`coco-compare`).

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two sets of row vectors:
    |mu1 - mu2|^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2)).

    The trace of `(S1 S2)^(1/2)` is the sum of square roots of the eigenvalues of
    `S1^(1/2) S2 S1^(1/2)`, which is symmetric, so two `eigh` calls do it with no general
    matrix square root (and no scipy)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(s1)
    r = (v * np.sqrt(np.clip(w, 0, None))) @ v.T                 # S1^(1/2)
    m = np.linalg.eigvalsh(r @ s2 @ r)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2)
                 - 2 * np.sqrt(np.clip(m, 0, None)).sum())


@torch.no_grad()
def score(clip, drawn: np.ndarray, real: np.ndarray, captions: list[str], *,
          device: str = "cpu") -> dict:
    """Every number of the report, for drawings and for the real photos of the same captions."""
    from .clip import picture_vectors
    txt = torch.cat([clip.embed_texts(captions[i:i + 512], device).float().cpu()
                     for i in range(0, len(captions), 512)])
    out = {"n": len(captions)}
    vecs = {}
    shift = np.roll(np.arange(len(captions)), 1)          # caption i paired with photo i-1's
    for name, pics in (("drawn", drawn), ("real", real)):
        img = picture_vectors(clip, pics, device)
        vecs[name] = img.numpy()
        sims = img @ txt.t()
        own = sims.diagonal()
        rank = (sims > own[:, None]).sum(1).numpy()
        out[name] = {"match": float(own.mean()),
                     "mismatch": float(sims[np.arange(len(own)), shift].mean()),
                     "top1": float(np.mean(rank < 1)), "top5": float(np.mean(rank < 5)),
                     "top10": float(np.mean(rank < 10)),
                     "correct": "".join("1" if r < 5 else "0" for r in rank)}
    half = len(captions) // 2
    out["distance"] = {"drawn_vs_real": frechet(vecs["drawn"], vecs["real"]),
                       "real_half_vs_half": frechet(vecs["real"][:half], vecs["real"][half:])}
    out["chance"] = {"top1": 1 / len(captions), "top5": 5 / len(captions),
                     "top10": 10 / len(captions)}
    return out


def report(drawer, clip, corpus, *, n: int = 1000, guidance: float = 3.0, seed: int = 0,
           steps: int | None = None, device: str = "cpu", clip_path: str = "") -> dict:
    """Draw one picture for each of the first `n` val captions and score them."""
    from ..vision.image import ImageCaptions
    ds = ImageCaptions(corpus, split="val", val_frac=1.0)
    idx = ds.index[:n]
    real = np.asarray(ds.images[idx])
    captions = [ds.manifest.facts[int(i)]["captions"][0] for i in idx]
    t0 = time.time()
    drawn = np.concatenate([drawer.draw(captions[i:i + 250], guidance=guidance, seed=seed + i,
                                        steps=steps) for i in range(0, len(captions), 250)])
    out = score(clip, drawn, real, captions, device=device)
    return out | {"captions": captions, "corpus": str(corpus), "guidance": guidance,
                  "seed": seed, "steps": steps, "clip": clip_path,
                  "seconds": round(time.time() - t0, 1), "_drawn": drawn, "_real": real}


def compare(a: dict, b: dict) -> dict:
    """Two reports on the same captions, paired per caption on `top5` (drawn), sign test."""
    from ..eval.versus import sign_test
    if a["captions"] != b["captions"]:
        raise ValueError("the reports drew different captions; they cannot be paired")
    ra, rb = a["drawn"]["correct"], b["drawn"]["correct"]
    b_only = sum(u == "0" and v == "1" for u, v in zip(ra, rb, strict=True))
    a_only = sum(u == "1" and v == "0" for u, v in zip(ra, rb, strict=True))
    return {"n": len(ra), "a": a["drawn"]["top5"], "b": b["drawn"]["top5"], "b_only": b_only,
            "a_only": a_only, "p": sign_test(b_only, a_only)}


def gallery(result: dict, path, k: int = 16) -> None:
    """Real photo above each drawing, `k` captions, scaled x2."""
    from ..vision.image import write_png
    d, r = result["_drawn"][:k], result["_real"][:k]
    gap = np.full((64, 3, 3), 255, dtype=np.uint8)
    rows = [np.concatenate(sum(([x, gap] for x in row), [])[:-1], axis=1) for row in (r, d)]
    sep = np.full((3, rows[0].shape[1], 3), 255, dtype=np.uint8)
    write_png(path, np.concatenate([rows[0], sep, rows[1]]).repeat(2, 0).repeat(2, 1))


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}
