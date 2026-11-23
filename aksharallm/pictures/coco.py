"""G4c's corpus: COCO 2017, every photo cut to a 64x64 square, with its five captions.

The shapes and scenes corpora were drawn by us, so their captions were exact and a rule could
judge every picture. COCO is the first corpus here that nobody drew for us: 118,287 training
and 5,000 validation photos of everyday scenes, each with five sentences written by people
("A man riding a wave on top of a surfboard."). No rule can read a photo back, so G4c is judged
by a contrastive model trained the same way as G4b's (`clip.py`), whose trust was earned on the
shapes, where it agreed with the exact reader.

```mermaid
flowchart LR
    Z["train2017.zip / val2017.zip<br/>(never extracted)"] --> D["decode one JPEG<br/>(Pillow)"]
    D --> C["centre square crop"] --> R["resize to 64x64<br/>(area filter)"] --> B["images.bin<br/>uint8, the same format<br/>as the shapes corpus"]
    A["captions_*.json"] --> M["manifest: first caption<br/>+ all five per photo"]
    M --> B
```

The output is the same `images.bin` + `manifest.json` every picture corpus here uses, with
`kind: "coco"` and one fact per photo, `{"id": <COCO image id>, "captions": [five]}`, so
`ImageCaptions`, the autoencoder trainer and the latents encoder read it unchanged. A centre
crop loses the edges of wide photos; resizing the whole photo to a square would squash it.
Both lose something, and the crop keeps shapes honest.

The archives are read member by member with `zipfile`, never extracted, and only the members
the caption file names are opened: a downloaded archive is data, not a directory to trust.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..vision.image import ImageManifest

SIZE = 64


def square(img, size: int = SIZE) -> np.ndarray:
    """A PIL image -> `(size, size, 3)` uint8: centre square crop, then an area resize."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def _work(args) -> list[bytes]:
    zip_path, names, size = args
    from PIL import Image
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for name in names:
            with Image.open(io.BytesIO(z.read(name))) as im:
                out.append(square(im, size).tobytes())
    return out


def captions_by_image(annotations_zip: str | Path, split: str) -> list[tuple[int, str, list[str]]]:
    """`(image id, file name, its captions)` for every image of a split, sorted by id."""
    with zipfile.ZipFile(annotations_zip) as z:
        d = json.loads(z.read(f"annotations/captions_{split}2017.json"))
    caps: dict[int, list[str]] = {}
    for a in d["annotations"]:
        caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    return [(i, files[i], caps.get(i, [])) for i in sorted(files) if caps.get(i)]


def pack(raw_dir: str | Path, out_dir: str | Path, split: str, *, size: int = SIZE,
         limit: int | None = None, workers: int = 12, progress=print) -> ImageManifest:
    """Pack one split (`train` or `val`) of COCO 2017 into a picture corpus."""
    raw = Path(raw_dir)
    rows = captions_by_image(raw / "annotations_trainval2017.zip", split)
    if limit:
        rows = rows[:limit]
    zip_path = raw / f"{split}2017.zip"
    with zipfile.ZipFile(zip_path) as z:
        present = set(z.namelist())
    names = [f"{split}2017/{f}" for _, f, _ in rows]
    missing = [n for n in names if n not in present]
    if missing:
        raise ValueError(f"{zip_path} lacks {len(missing)} images the captions name, e.g. {missing[0]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunk = 500
    jobs = [(str(zip_path), names[i:i + chunk], size) for i in range(0, len(names), chunk)]
    t0 = time.time()
    pool = Pool(workers) if workers > 1 else None
    results = pool.imap(_work, jobs) if pool else map(_work, jobs)   # imap keeps the order
    with open(out / "images.bin", "wb") as f:
        for j, blobs in enumerate(results):
            for b in blobs:
                f.write(b)
            if progress and j % 20 == 0:
                progress(f"  {min((j + 1) * chunk, len(names)):,}/{len(names):,} photos "
                         f"({time.time() - t0:.0f}s)")
    if pool:
        pool.close()
        pool.join()
    written = (out / "images.bin").stat().st_size
    if written != len(rows) * size * size * 3:
        raise ValueError(f"wrote {written} bytes for {len(rows)} photos: a worker failed")
    man = ImageManifest(size=size, channels=3, captions=[c[0] for _, _, c in rows],
                        facts=[{"id": i, "captions": c} for i, _, c in rows],
                        built=time.strftime("%Y-%m-%d %H:%M:%S"), kind="coco")
    man.save(out / "manifest.json")
    if progress:
        progress(f"{len(rows):,} {split} photos at {size}x{size}, "
                 f"{sum(len(c) for _, _, c in rows):,} captions -> {out}/")
    return man
