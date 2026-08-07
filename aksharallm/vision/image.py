"""Images in, and a corpus of them that needs no download.

The same shape as `audio/io.py` and `audio/dataset.py`, for the same reasons: read the file
format ourselves, pack the corpus into one flat `uint8` file plus a manifest, and memmap it.
An image is already a tensor — there is nothing to decode once it is off the disk — so this
file is mostly about getting a *corpus* to exist.

**The synthetic corpus is the point of the file.** As with `audio/dataset.py`'s vowel babble,
it is generated from a description we chose, so **the caption is known exactly**:

    "three red circles and one blue square"

That is what makes the vision path *measurable* rather than merely demonstrable. A caption
model trained on COCO can only be judged by reading its output; a caption model trained on
this can be scored — did it get the count, the colour, the shape? — by anyone, in seconds,
with no download and no human. Real captions are a strictly harder problem, and the point of
starting here is that a wrong answer is unambiguous.

```mermaid
flowchart LR
    S["a description we chose:<br/>3 red circles"] --> R["render 64x64 RGB"]
    R --> P["pack into images.bin<br/>(uint8) + manifest"]
    S --> C["the caption, exactly"]
    P --> D["ImageCaptions<br/>random batches"]
    C --> D
```

Read with: docs/21-vision.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Everything is square and this size. A 64x64 image at patch 8 is 64 patches, which is a
#: sequence the 300M can hold beside a caption without thinking about it. Doubling this
#: quadruples the patch count, which is the cost that actually matters.
IMAGE_SIZE = 64

#: The vocabulary the synthetic corpus draws from. Small on purpose: the interesting question
#: is whether the model can *bind* an attribute to an object (three RED circles, not three
#: circles and something red), and a small vocabulary makes a failure to bind obvious.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 190, 70),
    "blue": (50, 90, 230),
    "yellow": (240, 210, 50),
    "purple": (150, 60, 200),
}
SHAPES = ("circle", "square", "triangle")
NUMBERS = ("one", "two", "three", "four")


@dataclass
class ImageManifest:
    """What is in an `images.bin`, and the caption of each image."""

    size: int
    channels: int
    captions: list[str]
    facts: list[dict] = field(default_factory=list)  # the description each image was made from
    built: str = ""

    @property
    def n_images(self) -> int:
        return len(self.captions)

    @property
    def bytes_per_image(self) -> int:
        return self.size * self.size * self.channels

    @classmethod
    def load(cls, path: str | Path) -> ImageManifest:
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))


# ---------------------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------------------


def read_image(path: str | Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Any image file to `(size, size, 3)` uint8, via Pillow.

    Pillow rather than a from-scratch decoder, and the line is worth drawing explicitly:
    this repo writes its own tokenizer, transformer, quantizer and codec because those are
    the *subject*. JPEG's entropy coder is not — it is the same category as `datasets`
    downloading a parquet file. Everything from the pixels onward is ours.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def write_png(path: str | Path, image: np.ndarray) -> None:
    """`(h, w, 3)` uint8 to a PNG, so a rendered image can actually be looked at."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def to_tensor(images: np.ndarray) -> np.ndarray:
    """`(..., h, w, 3)` uint8 -> `(..., 3, h, w)` float32 in [-1, 1].

    Channels-first because that is what a conv expects, and centred on zero rather than in
    [0, 1] because the first layer has no bias to absorb a constant offset with.
    """
    x = np.asarray(images, dtype=np.float32) / 127.5 - 1.0
    return np.moveaxis(x, -1, -3)


def from_tensor(x: np.ndarray) -> np.ndarray:
    """The inverse, for looking at what a model produced or was given."""
    x = np.moveaxis(np.asarray(x), -3, -1)
    # ROUND, not truncate. `astype(uint8)` truncates, so a pixel that comes back as
    # 17.999999 becomes 17 — every value off by one, invisible to the eye and enough to make
    # `from_tensor(to_tensor(x)) == x` false, which is the test that catches a real bug later.
    return np.clip(np.round((x + 1.0) * 127.5), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------


def _disc(size: int, cy: float, cx: float, r: float) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def _square(size: int, cy: float, cx: float, r: float) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    return (np.abs(y - cy) <= r) & (np.abs(x - cx) <= r)


def _triangle(size: int, cy: float, cx: float, r: float) -> np.ndarray:
    """A filled upward triangle: apex at `cy - r`, base at `cy + r` spanning ±r.

    The width at height `dy` grows linearly from 0 at the apex to `r` at the base, so the
    condition is `|dx| <= (dy + r) / 2`. Written that way rather than as an intersection of
    three half-planes because the half-plane version is easy to get subtly wrong — the first
    attempt here rendered a notched crown, which still looked like *a* shape and would have
    trained perfectly happily.
    """
    y, x = np.ogrid[:size, :size]
    dy, dx = y - cy, x - cx
    return (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2)


_RENDER = {"circle": _disc, "square": _square, "triangle": _triangle}


def render(fact: dict, size: int = IMAGE_SIZE, seed: int = 0) -> np.ndarray:
    """Draw one `{count, colour, shape}` description. `(size, size, 3)` uint8.

    Positions are jittered so the model cannot solve counting by memorising layouts — with
    fixed positions, "three circles" becomes "the pattern that has ink at these coordinates",
    which is a lookup rather than a count and generalises to nothing.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 18, dtype=np.uint8)  # near-black, not black
    n = fact["count"]
    r = size * 0.09
    # Place on a jittered grid: overlapping shapes make the count genuinely ambiguous, and
    # a corpus whose labels are ambiguous cannot be scored.
    slots = rng.permutation(9)[:n]
    for slot in slots:
        gy, gx = divmod(int(slot), 3)
        cy = size * (0.22 + 0.28 * gy) + rng.uniform(-2, 2)
        cx = size * (0.22 + 0.28 * gx) + rng.uniform(-2, 2)
        mask = _RENDER[fact["shape"]](size, cy, cx, r)
        img[mask] = COLOURS[fact["colour"]]
    return img


def caption_of(fact: dict) -> str:
    """The exact caption for a description. Plural handled, because "one circles" would
    teach the model that grammar is noise."""
    shape = fact["shape"] + ("" if fact["count"] == 1 else "s")
    return f"{NUMBERS[fact['count'] - 1]} {fact['colour']} {shape}"


# ---------------------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------------------


def synth_corpus(out_dir: str | Path, *, n_images: int = 4000, size: int = IMAGE_SIZE,
                 seed: int = 0, hold_out: tuple[str, str] | None = ("purple", "triangle"),
                 progress=print) -> ImageManifest:
    """Render a corpus of shape images with exact captions. ~12 MB for 4,000 at 64x64.

    `hold_out` removes one (colour, shape) *combination* from training while leaving both the
    colour and the shape well represented elsewhere. That is the compositional generalisation
    test: a model that has seen purple squares and red triangles but never a purple triangle
    should still describe one, and a model that has merely memorised pairs will not. It is
    the cheapest interesting question this corpus can ask, and it costs one tuple.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    facts, captions = [], []
    with open(out_dir / "images.bin", "wb") as f:
        i = 0
        while len(facts) < n_images:
            fact = {
                "count": int(rng.integers(1, len(NUMBERS) + 1)),
                "colour": str(rng.choice(list(COLOURS))),
                "shape": str(rng.choice(SHAPES)),
            }
            i += 1
            if hold_out and (fact["colour"], fact["shape"]) == tuple(hold_out):
                continue
            img = render(fact, size, seed=seed * 1_000_003 + i)
            f.write(img.tobytes())
            facts.append(fact)
            captions.append(caption_of(fact))
            if progress and len(facts) % 1000 == 0:
                progress(f"  {len(facts):,}/{n_images:,} images")

    man = ImageManifest(size=size, channels=3, captions=captions, facts=facts,
                        built=time.strftime("%Y-%m-%d %H:%M:%S"))
    man.save(out_dir / "manifest.json")
    if progress:
        held = f", holding out {hold_out[0]} {hold_out[1]}s" if hold_out else ""
        progress(f"{len(facts):,} images at {size}x{size}{held} -> {out_dir}/images.bin")
    return man


def holdout_set(out_dir: str | Path, combo: tuple[str, str] = ("purple", "triangle"),
                n: int = 64, size: int = IMAGE_SIZE, seed: int = 999) -> ImageManifest:
    """Only the held-out combination, for the compositional test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    facts, captions = [], []
    with open(out_dir / "images.bin", "wb") as f:
        for i in range(n):
            fact = {"count": int(rng.integers(1, len(NUMBERS) + 1)),
                    "colour": combo[0], "shape": combo[1]}
            f.write(render(fact, size, seed=seed + i).tobytes())
            facts.append(fact)
            captions.append(caption_of(fact))
    man = ImageManifest(size=size, channels=3, captions=captions, facts=facts,
                        built=time.strftime("%Y-%m-%d %H:%M:%S"))
    man.save(out_dir / "manifest.json")
    return man


class ImageCaptions:
    """A packed image corpus, as random batches of `(images, captions)`.

    Memmapped like every other corpus here, and split by *index* rather than by file: the
    last `val_frac` of the images are held out, and because the generator draws each
    description independently there is nothing sequential to leak across the boundary.
    """

    def __init__(self, path: str | Path, *, split: str = "train", val_frac: float = 0.05,
                 seed: int | None = None):
        d = Path(path)
        self.manifest = ImageManifest.load(d / "manifest.json")
        m = self.manifest
        raw = np.memmap(d / "images.bin", dtype=np.uint8, mode="r")
        expected = m.n_images * m.bytes_per_image
        if raw.size != expected:
            raise ValueError(
                f"{d}/images.bin holds {raw.size} bytes but the manifest describes "
                f"{m.n_images} x {m.size}x{m.size}x{m.channels} = {expected}. Re-render it."
            )
        self.images = raw.reshape(m.n_images, m.size, m.size, m.channels)
        n_val = max(1, int(m.n_images * val_frac))
        self.index = (list(range(m.n_images - n_val, m.n_images)) if split == "val"
                      else list(range(m.n_images - n_val)))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def batch(self, batch_size: int) -> tuple[np.ndarray, list[str]]:
        picks = self.rng.choice(self.index, size=batch_size, replace=len(self.index) < batch_size)
        return (to_tensor(self.images[picks]),
                [self.manifest.captions[int(i)] for i in picks])

    def item(self, i: int) -> tuple[np.ndarray, str, dict]:
        j = self.index[i % len(self.index)]
        return (to_tensor(self.images[j : j + 1])[0], self.manifest.captions[j],
                self.manifest.facts[j])
