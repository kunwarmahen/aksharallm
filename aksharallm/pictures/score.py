"""Scoring G1: draw every caption, read each picture back, and check it is not a copy.

A drawn picture is judged the way a reconstruction is (reader.py): does it show the count,
colour and shape the caption asked for? Three more questions make the number mean something:

* **the held-out combination.** "purple triangles" was never drawn in training. Both words
  were, with other partners. A model that binds words to attributes draws them anyway; one that
  memorised (caption -> pictures) cannot, because it has no pictures to recall.
* **copies.** 8,000 training pictures is few enough to memorise. Every drawn picture's codes are
  looked up among the training pictures' codes; an exact match is a copy, not a drawing. Shapes
  are placed on a jittered 3x3 grid, so honest drawings can collide by chance; the rate is
  reported, not used to disqualify.
* **variety.** Distinct pictures among the k drawn for one caption. Guidance trades this for
  obedience, and the trade is the point of measuring both.

Every (caption, sample) keeps its verdict as one character of a string, so two reports (two
guidance settings, two checkpoints) pair picture by picture with `compare` and a sign test.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .codes import all_captions, fact_of
from .reader import judge

HELD_OUT = ("purple", "triangle")


def is_held_out(caption: str) -> bool:
    f = fact_of(caption)
    return (f["colour"], f["shape"]) == HELD_OUT


def train_code_set(ds) -> set[bytes]:
    """Every training picture's code sequence, as bytes, for the copy check."""
    return {ds.tokens[i].astype(np.int64).tobytes() for i in ds.index}


def report(model, tokenizer, pg, *, train_ds, k: int = 8, guidance: float = 1.0,
           temperature: float = 1.0, top_k: int | None = None, seed: int = 0,
           device: str = "cpu", captions: list[str] | None = None) -> dict:
    """G1: draw `k` pictures for each caption (default: all 60) and score them."""
    from .codes import flatten
    from .gen import draw
    captions = captions or all_captions()
    asked = [c for c in captions for _ in range(k)]
    t0 = time.time()
    pics, codes = draw(model, tokenizer, asked, pg=pg, guidance=guidance,
                       temperature=temperature, top_k=top_k, seed=seed, device=device)
    toks = flatten(codes.numpy().reshape(len(asked), pg.books, -1).astype(np.int64),
                   pg.books, tokenizer.cfg.codebook_size)
    out = judge_drawings(captions, k, pics, toks, train_code_set(train_ds))
    return out | {"guidance": guidance, "temperature": temperature, "top_k": top_k,
                  "seed": seed, "seconds": round(time.time() - t0, 1)}


def tokens_of(pictures: np.ndarray, tokenizer, books: int, device: str = "cpu") -> np.ndarray:
    """Pictures -> the same position-major code tokens G1 writes, so a picture drawn in
    pixels (G2) can be checked for copies and variety exactly as G1's drawings are."""
    import torch

    from ..vision.image import to_tensor
    from .codes import flatten
    out = []
    for i in range(0, len(pictures), 256):
        x = torch.from_numpy(to_tensor(np.asarray(pictures[i:i + 256]))).to(device)
        idx = tokenizer.encode(x, books)[:, :books]
        out.append(flatten(idx.reshape(len(x), books, -1).cpu().numpy().astype(np.int64),
                           books, tokenizer.cfg.codebook_size))
    return np.concatenate(out)


def judge_drawings(captions: list[str], k: int, pics: np.ndarray, toks: np.ndarray,
                   seen: set[bytes]) -> dict:
    """The scoring shared by every drawing model: `pics` holds `k` pictures per caption, in
    caption order; `toks` are their code tokens (for copies and variety)."""
    asked = [c for c in captions for _ in range(k)]
    verdicts = [judge(p, fact_of(c)) for p, c in zip(pics, asked, strict=True)]
    copies = [t.astype(np.int64).tobytes() in seen for t in toks]

    def group(sel: list[int]) -> dict:
        if not sel:
            return {"n": 0}
        n = len(sel)
        out = {key: sum(verdicts[i][key] for i in sel) / n
               for key in ("count", "colour", "shape", "all")}
        out["n"] = n
        out["copies"] = sum(copies[i] for i in sel) / n
        out["correct"] = "".join("1" if verdicts[i]["all"] else "0" for i in sel)
        return out

    held = [i for i, c in enumerate(asked) if is_held_out(c)]
    rest = [i for i, c in enumerate(asked) if not is_held_out(c)]
    per_caption = []
    for j, c in enumerate(captions):
        sel = list(range(j * k, (j + 1) * k))
        distinct = len({toks[i].tobytes() for i in sel})
        per_caption.append({"caption": c, "held_out": is_held_out(c),
                            "all": sum(verdicts[i]["all"] for i in sel) / k,
                            "distinct": distinct, "copies": sum(copies[i] for i in sel),
                            "reads": [verdicts[i]["read"] for i in sel]})
    return {"k": k, "captions": captions, "seen": group(rest), "held_out": group(held),
            "variety": float(np.mean([p["distinct"] / k for p in per_caption])),
            "per_caption": per_caption, "_pictures": pics}


def gallery(result: dict, out_dir: str | Path, per_row: int = 8) -> list[str]:
    """One PNG per caption: its k drawings side by side, scaled x2. Returns the names."""
    from ..vision.image import write_png
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k, pics, names = result["k"], result["_pictures"], []
    gap = np.full((64, 3, 3), 255, dtype=np.uint8)
    for j, c in enumerate(result["captions"]):
        tiles = list(pics[j * k:(j * k) + min(k, per_row)])
        row = np.concatenate(sum(([t, gap] for t in tiles), [])[:-1], axis=1)
        name = c.replace(" ", "-") + ".png"
        write_png(out_dir / name, row.repeat(2, axis=0).repeat(2, axis=1))
        names.append(name)
    return names


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def compare(a: dict, b: dict) -> dict:
    """Two G1 reports paired per (caption, sample) on `all`, seen and held-out separately.
    Refuses reports that drew different captions or a different k."""
    from ..eval.versus import sign_test
    if a["captions"] != b["captions"] or a["k"] != b["k"]:
        raise ValueError("the reports drew different captions or a different k per caption; "
                         "they cannot be paired")
    out = {}
    for part in ("seen", "held_out"):
        ra, rb = a[part], b[part]
        if not ra.get("n"):
            continue
        b_only = sum(u == "0" and v == "1" for u, v in zip(ra["correct"], rb["correct"], strict=True))
        a_only = sum(u == "1" and v == "0" for u, v in zip(ra["correct"], rb["correct"], strict=True))
        out[part] = {"n": ra["n"], "a": ra["all"], "b": rb["all"], "b_only": b_only,
                     "a_only": a_only, "p": sign_test(b_only, a_only)}
    return out
