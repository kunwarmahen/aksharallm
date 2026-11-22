"""What an image tokenizer kept, measured three ways, at every codebook count.

1. **PSNR** of the reconstruction, in dB against the 0-255 picture. The usual pixel number,
   and a weak one: a picture can lose a whole shape and keep a high PSNR, because most of a
   shapes picture is background.
2. **Does it still say the same thing?** `reader.read` on the reconstruction against the
   description it was drawn from: count, colour, shape, all three. The reader is exact on the
   originals (that row is printed, the positive control), so every miss is the tokenizer's.
   Kept per picture as a string of 0/1 so two tokenizers, or two codebook counts, pair up
   picture by picture (`compare`, sign test), the rule for every comparison in this repo.
3. **Codebook use**: distinct entries used and the perplexity of each codebook over the whole
   set. A codebook of 512 with a perplexity of 20 is a 20-entry codebook; it is the collapse
   detector, and the loss curve cannot show collapse.

The held-out combination (`data/vision/shapes-holdout`, purple triangles, which the corpus
never draws) is scored the same way: a tokenizer that only memorised the pictures it saw would
fail on a combination of colour and shape it never saw together.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..vision.image import ImageCaptions, from_tensor, write_png
from .reader import judge


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """dB between two uint8 pictures. Identical pictures are capped at 99 rather than inf, so
    a mean stays a number."""
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _marks(pictures: np.ndarray, facts: list[dict]) -> dict:
    js = [judge(p, f) for p, f in zip(pictures, facts)]
    out = {k: "".join("1" if j[k] else "0" for j in js) for k in ("count", "colour", "shape", "all")}
    out["reads"] = [j["read"] for j in js]
    return out


@torch.no_grad()
def ladder(model, ds: ImageCaptions, *, n: int = 256, device: str = "cpu",
           batch: int = 64, books=None) -> dict:
    """Reconstruct the first `n` pictures of `ds` at each codebook count in `books`."""
    model.eval()
    N = model.cfg.n_codebooks
    books = list(books or sorted({1, 2, N // 2 or 1, N}))
    n = min(n, len(ds))
    idxs = [ds.index[i] for i in range(n)]
    originals = np.asarray(ds.images[idxs])
    facts = [ds.manifest.facts[j] for j in idxs]
    rows, codes_all = [], None
    recons = {}
    for k in books:
        outs, codes = [], []
        for i in range(0, n, batch):
            x = torch.from_numpy(_to_tensor(originals[i:i + batch])).to(device)
            idx = model.encode(x, k)
            codes.append(idx.cpu())
            outs.append(from_tensor(model.decode(idx, k).float().cpu().numpy()))
        pics = np.concatenate(outs)
        recons[k] = pics
        if k == N:
            codes_all = torch.cat(codes)
        m = _marks(pics, facts)
        rows.append({"books": k, "bits": k * model.cfg.grid ** 2 * math.log2(model.cfg.codebook_size),
                     "psnr": float(np.mean([psnr(a, b) for a, b in zip(originals, pics)])),
                     **{key: m[key].count("1") / n for key in ("count", "colour", "shape", "all")},
                     "correct": m["all"]})
    orig = _marks(originals, facts)
    usage = []
    for b in range(N):
        h = torch.bincount(codes_all[:, b].reshape(-1), minlength=model.cfg.codebook_size).float()
        p = h / h.sum()
        nz = p[p > 0]
        usage.append({"book": b, "used": int((h > 0).sum()),
                      "perplexity": float(torch.exp(-(nz * nz.log()).sum()))})
    return {"n": n, "books": books, "ladder": rows, "usage": usage,
            "originals": {key: orig[key].count("1") / n for key in ("count", "colour", "shape", "all")},
            "_recons": recons, "_originals": originals, "_facts": facts}


def _to_tensor(images: np.ndarray) -> np.ndarray:
    from ..vision.image import to_tensor
    return to_tensor(images)


def gallery(result: dict, out_dir: str | Path, k: int = 8) -> list[str]:
    """One PNG per picture: the original, then each codebook count, left to right, scaled up
    x3 so a 64-pixel picture is visible. Returns the file names, in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(min(k, result["n"])):
        tiles = [result["_originals"][i]] + [result["_recons"][b][i] for b in result["books"]]
        gap = np.full((64, 4, 3), 255, dtype=np.uint8)
        row = np.concatenate(sum(([t, gap] for t in tiles), [])[:-1], axis=1)
        row = row.repeat(3, axis=0).repeat(3, axis=1)
        name = f"pic{i:02d}.png"
        write_png(out_dir / name, row)
        names.append(name)
    return names


def public(result: dict) -> dict:
    """The JSON-safe part of a `ladder` result (the pictures stay on disk as PNGs)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def compare(a: dict, b: dict, books_a: int | None = None, books_b: int | None = None) -> dict:
    """Two ladder rows, paired picture by picture on `all`: b right where a was wrong, and the
    reverse, with an exact sign test. Refuses different picture sets."""
    from ..eval.versus import sign_test
    if a["n"] != b["n"]:
        raise ValueError(f"different picture sets: {a['n']} vs {b['n']} pictures")
    ra = next(r for r in a["ladder"] if r["books"] == (books_a or a["books"][-1]))
    rb = next(r for r in b["ladder"] if r["books"] == (books_b or b["books"][-1]))
    b_only = sum(u == "0" and v == "1" for u, v in zip(ra["correct"], rb["correct"], strict=True))
    a_only = sum(u == "1" and v == "0" for u, v in zip(ra["correct"], rb["correct"], strict=True))
    return {"a": ra["all"], "b": rb["all"], "b_only": b_only, "a_only": a_only,
            "p": sign_test(b_only, a_only), "n": a["n"]}
