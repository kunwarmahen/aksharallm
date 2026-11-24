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

* **sharpness**: the drawings' fine detail (power above a quarter cycle per pixel,
  `measure.sharpness`) against the real photos'; 1.0 is as sharp, below 1 is blurrier. And a
  third row, **the real photos through the drawer's own autoencoder**: no drawing can be sharper
  or closer than that, so it is the ceiling a drawing should be read against, where the real
  photos are the ceiling the autoencoder is read against.

**Larger photos** (128 px): the judge was trained at 64, so every picture is shrunk to 64
before it is judged (`clip.picture_vectors`), and sharpness is reported twice: at the drawing's
own size against the real photos at that size, and at the judge's size (`sharpness_judged`),
which is the one that compares a 128-px model with a 64-px one on one scale.

`recon_report` is the autoencoder's own version (`pictures ae-report`): every val photo through
it, PSNR, sharpness and the same judge, per photo, so two autoencoders pair photo by photo.

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
def through(coder, pictures: np.ndarray, device: str = "cpu", batch: int = 250) -> np.ndarray:
    """Pictures encoded and decoded by an autoencoder (its latent mean, as a drawing's is)."""
    from ..vision.image import from_tensor, to_tensor
    out = []
    for i in range(0, len(pictures), batch):
        x = torch.from_numpy(to_tensor(pictures[i:i + batch])).to(device)
        out.append(from_tensor(coder.decoder(coder.latent(x)).float().clamp(-1, 1).cpu().numpy()))
    return np.concatenate(out)


@torch.no_grad()
def score(clip, drawn: np.ndarray, real: np.ndarray, captions: list[str], *,
          device: str = "cpu", coded: np.ndarray | None = None) -> dict:
    """Every number of the report, for drawings and for the real photos of the same captions
    (and, given `coded`, for the real photos through the drawer's autoencoder)."""
    from .clip import picture_vectors
    from .measure import high_frequency, shrink
    txt = torch.cat([clip.embed_texts(captions[i:i + 512], device).float().cpu()
                     for i in range(0, len(captions), 512)])
    out = {"n": len(captions)}
    vecs = {}
    shift = np.roll(np.arange(len(captions)), 1)          # caption i paired with photo i-1's
    hf_real = high_frequency(real).mean()
    side, judged = np.asarray(real).shape[1], clip.cfg.image_size
    hf_real_j = high_frequency(shrink(real, judged)).mean() if side != judged else hf_real
    sets = [("drawn", drawn), ("real", real)] + ([("coded", coded)] if coded is not None else [])
    for name, pics in sets:
        img = picture_vectors(clip, pics, device)
        vecs[name] = img.numpy()
        sims = img @ txt.t()
        own = sims.diagonal()
        rank = (sims > own[:, None]).sum(1).numpy()
        out[name] = {"match": float(own.mean()),
                     "mismatch": float(sims[np.arange(len(own)), shift].mean()),
                     "top1": float(np.mean(rank < 1)), "top5": float(np.mean(rank < 5)),
                     "top10": float(np.mean(rank < 10)),
                     "correct": "".join("1" if r < 5 else "0" for r in rank),
                     "sharpness": float(high_frequency(pics).mean() / hf_real),
                     # the same at the judge's size, so a 128-px model and a 64-px one compare
                     # on one scale: does the extra detail survive shrinking to 64?
                     "sharpness_judged": float(high_frequency(shrink(pics, judged)).mean()
                                               / hf_real_j)}
    half = len(captions) // 2
    out["distance"] = {"drawn_vs_real": frechet(vecs["drawn"], vecs["real"]),
                       "real_half_vs_half": frechet(vecs["real"][:half], vecs["real"][half:])}
    if coded is not None:
        out["distance"]["coded_vs_real"] = frechet(vecs["coded"], vecs["real"])
    out["side"], out["judged_at"] = int(side), int(judged)
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
    coder = getattr(drawer, "tokenizer", None)
    coded = through(coder, real, device) if hasattr(coder, "latent") else None
    out = score(clip, drawn, real, captions, device=device, coded=coded)
    return out | {"captions": captions, "corpus": str(corpus), "guidance": guidance,
                  "seed": seed, "steps": steps, "clip": clip_path,
                  "seconds": round(time.time() - t0, 1), "_drawn": drawn, "_real": real}


@torch.no_grad()
def recon_report(coder, clip, corpus, *, n: int = 1000, device: str = "cpu",
                 clip_path: str = "") -> dict:
    """An autoencoder on the first `n` val photos: PSNR and sharpness per photo, and our judge's
    numbers for the reconstructions beside the real photos' (`score` with them as "drawn")."""
    from ..vision.image import ImageCaptions
    from .measure import high_frequency, psnr
    ds = ImageCaptions(corpus, split="val", val_frac=1.0)
    idx = ds.index[:n]
    real = np.asarray(ds.images[idx])
    captions = [ds.manifest.facts[int(i)]["captions"][0] for i in idx]
    t0 = time.time()
    recon = through(coder, real, device)
    out = score(clip, recon, real, captions, device=device) if clip is not None else {"n": n}
    hf_r, hf_x = high_frequency(real), high_frequency(recon)
    out["per_photo"] = {"psnr": [round(psnr(a, b), 3) for a, b in zip(real, recon, strict=True)],
                        "sharpness": [round(float(x / max(r, 1e-12)), 4)
                                      for x, r in zip(hf_x, hf_r, strict=True)]}
    out["psnr"] = float(np.mean(out["per_photo"]["psnr"]))
    out["sharpness"] = float(hf_x.mean() / hf_r.mean())
    return out | {"captions": captions, "corpus": str(corpus), "clip": clip_path,
                  "seconds": round(time.time() - t0, 1), "_drawn": recon, "_real": real}


def _closer(x: float, y: float) -> int:
    """+1 if `y` is closer to 1 than `x` on a log scale (as sharp as the real photo), -1 if
    farther, 0 if as close: half the detail and twice the detail are equally wrong."""
    dx, dy = abs(float(np.log(max(x, 1e-9)))), abs(float(np.log(max(y, 1e-9))))
    return (dy < dx) - (dy > dx)


def recon_compare(a: dict, b: dict) -> dict:
    """Two autoencoders on the same photos, paired: on which photo is each one's PSNR higher,
    and whose fine detail is closer to the real photo's (|log sharpness| smaller)."""
    from ..eval.versus import sign_test
    if a["captions"] != b["captions"]:
        raise ValueError("the reports reconstructed different photos; they cannot be paired")
    out = {"n": len(a["captions"])}
    pa, pb = a["per_photo"], b["per_photo"]
    wins = {"psnr": [(y > x) - (y < x) for x, y in zip(pa["psnr"], pb["psnr"], strict=True)],
            "sharpness": [_closer(x, y) for x, y in zip(pa["sharpness"], pb["sharpness"],
                                                         strict=True)]}
    for k, w in wins.items():
        b_only, a_only = w.count(1), w.count(-1)
        out[k] = {"a": a[k], "b": b[k], "b_better": b_only, "a_better": a_only,
                  "p": sign_test(b_only, a_only)}
    return out


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
    side = r.shape[1]
    gap = np.full((side, 3, 3), 255, dtype=np.uint8)
    rows = [np.concatenate(sum(([x, gap] for x in row), [])[:-1], axis=1) for row in (r, d)]
    sep = np.full((3, rows[0].shape[1], 3), 255, dtype=np.uint8)
    scale = max(1, 128 // side)                     # 64-px galleries x2, 128-px ones as drawn
    write_png(path, np.concatenate([rows[0], sep, rows[1]]).repeat(scale, 0).repeat(scale, 1))


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}
