"""Read a shapes picture back into its description, with no model at all.

Every piece of the picture line is scored by asking whether a picture still says what it was
meant to say: "three red circles". A learned reader (the vision model of docs/22) could answer
that, but a learned reader has its own error rate, and a tokenizer's mistakes would then be
mixed with the reader's. The shapes corpus was drawn by `vision/image.py` from a description
we chose, on a flat background, in five exact colours, so the description can be read back
by rule:

```mermaid
flowchart LR
    I["64x64 picture"] --> F["foreground:<br/>pixels far from the<br/>background grey"]
    F --> C["colour: nearest<br/>palette entry, majority"]
    F --> L["connected blobs<br/>(4-neighbour), big ones only"]
    L --> N["count = how many"]
    L --> S["shape = fill of the<br/>bounding box: square ~1,<br/>circle ~0.79, triangle ~0.5"]
    L --> Z["size = bounding box<br/>across: small ~9, big ~15"]
    L --> W["where = the column every<br/>blob's centre is in"]
```

G4b's scenes (`scenes.py`) add size and place. The reader always reads all five; `judge`
checks only the attributes the fact names, so a shapes-corpus fact (no size, no place) is
judged exactly as before.

The positive control is the point: on the corpus itself the reader must be right on every
picture (`tests/test_pictures.py` checks all of them on a fresh render), so whatever it gets
wrong on a reconstruction is the reconstruction's fault. It reads a picture the way a strict
person would, which is also its limit: a blob smeared into its neighbour counts as one, and a
colour drifted half way to another reads as whichever is nearer. Both are real damage.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..vision.image import COLOURS, NUMBERS, SHAPES

#: The background `vision/image.py` paints (near-black, 18 on every channel).
BACKGROUND = 18
#: A pixel is foreground when some channel is this far from the background. The darkest
#: palette channel above background is red's 220 / blue's 230 / ... every colour has one
#: channel at least 150 above 18, so 70 leaves a wide margin either way.
FOREGROUND = 70
#: A blob smaller than this is noise, not a shape. The smallest true shape (a triangle of
#: half-size 5.76 px) covers ~66 pixels.
MIN_AREA = 20
#: Fill-ratio boundaries between the shapes: square 1.0 > 0.90 > circle 0.79 > 0.64 > triangle 0.5.
SQUARE_ABOVE, CIRCLE_ABOVE = 0.90, 0.64
#: A blob wider (or taller) than this is big: small shapes are ~9-10 px across, big ~15.
BIG_ABOVE = 12
#: The attributes a fact can name, in the order they are reported.
ATTRS = ("count", "size", "colour", "shape", "where")

_PALETTE = np.array(list(COLOURS.values()), dtype=np.float32)
_NAMES = list(COLOURS)


def foreground(img: np.ndarray) -> np.ndarray:
    """`(h, w)` bool: pixels that are part of some shape."""
    return (np.abs(img.astype(np.int16) - BACKGROUND).max(-1)) > FOREGROUND


def blobs(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a bool mask, as arrays of `(y, x)`; written out rather than
    imported, because it is twenty lines and the judge should hide nothing."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q, pts = deque([(y0, x0)]), []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        out.append(np.array(pts))
    return out


def shape_of(pts: np.ndarray) -> str:
    """Square, circle or triangle, from how much of its bounding box a blob fills."""
    hgt = pts[:, 0].max() - pts[:, 0].min() + 1
    wid = pts[:, 1].max() - pts[:, 1].min() + 1
    fill = len(pts) / float(hgt * wid)
    return "square" if fill > SQUARE_ABOVE else "circle" if fill > CIRCLE_ABOVE else "triangle"


def size_of(pts: np.ndarray) -> str:
    """Small or big, from the larger side of the blob's bounding box."""
    hgt = pts[:, 0].max() - pts[:, 0].min() + 1
    wid = pts[:, 1].max() - pts[:, 1].min() + 1
    return "big" if max(hgt, wid) > BIG_ABOVE else "small"


def where_of(blobs_: list[np.ndarray], width: int) -> str:
    """The column (left / middle / right third) every blob's centre is in, or "spread"."""
    cols = {min(2, int(b[:, 1].mean() * 3 // width)) for b in blobs_}
    return ("left", "middle", "right")[cols.pop()] if len(cols) == 1 else "spread"


def read(img: np.ndarray) -> dict:
    """`(h, w, 3)` uint8 -> `{count, size, colour, shape, where}`. A picture with no shape in
    it reads as count 0 and None for the rest, which never equals any description."""
    mask = foreground(img)
    big = [b for b in blobs(mask) if len(b) >= MIN_AREA]
    if not big:
        return {"count": 0, "size": None, "colour": None, "shape": None, "where": None}
    px = img[mask].astype(np.float32)
    nearest = ((px[:, None, :] - _PALETTE[None]) ** 2).sum(-1).argmin(1)
    colour = _NAMES[int(np.bincount(nearest, minlength=len(_NAMES)).argmax())]
    kinds = [shape_of(b) for b in big]
    shape = max(SHAPES, key=kinds.count)
    sizes = [size_of(b) for b in big]
    size = max(("small", "big"), key=sizes.count)
    return {"count": len(big), "size": size, "colour": colour, "shape": shape,
            "where": where_of(big, img.shape[1])}


def judge(img: np.ndarray, fact: dict) -> dict:
    """Did this picture say `fact`? One boolean per attribute the fact names (count, colour
    and shape always; size and where for a scene), and `all` for all of them."""
    got = read(img)
    out = {k: got[k] == fact[k] for k in ATTRS if k in fact}
    out["all"] = all(out.values())
    out["read"] = got
    return out


def caption(fact: dict) -> str:
    """`{count, colour, shape}` -> "three red circles" (the corpus's own wording); a fact with
    size and where adds them ("three big red circles on the left")."""
    if not fact.get("count"):
        return "nothing"
    plural = "" if fact["count"] == 1 else "s"
    n = NUMBERS[fact["count"] - 1] if fact["count"] <= len(NUMBERS) else str(fact["count"])
    size = f"{fact['size']} " if fact.get("size") else ""
    where = fact.get("where")
    place = {"left": " on the left", "middle": " in the middle",
             "right": " on the right"}.get(where, f" ({where})" if where else "")
    return f"{n} {size}{fact['colour']} {fact['shape']}{plural}{place}"
