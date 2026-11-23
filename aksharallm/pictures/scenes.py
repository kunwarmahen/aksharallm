"""G4b's second corpus: captions that say more than count, colour and shape.

The shapes corpus (`vision/image.py`) has three attributes, and by G4a every drawing model got
all three right 99% of the time. The next question a caption can ask is **where** and **how
big**, which is also where real text-to-image models are known to be weak (left/right, relative
size). So a scene adds two attributes:

    two big red circles on the left
    one small blue square in the middle
    three small yellow triangles on the right

* `where`: the shapes all sit in one column of the 3x3 grid the shapes corpus already uses
  (left, middle, right), so the count is 1-3.
* `size`: small (half-size 4.8 px) or big (7 px), against the shapes corpus's 5.8.

Two combinations are never drawn in training: purple triangles (G0-G4a's held-out pair, any
size and place) and **circles on the right** (any colour and size), the held-out combination
of the new attribute: circles are drawn everywhere else and other shapes on the right, but
never the two together.

```mermaid
flowchart LR
    F["fact:<br/>count, size, colour,<br/>shape, where"] --> R["render: one column,<br/>1-3 rows, jittered"]
    F --> C["caption, exactly:<br/>two big red circles<br/>on the left"]
    R --> J["the reader reads all five<br/>(must be 100% on the corpus)"]
```

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..vision.image import _RENDER, COLOURS, IMAGE_SIZE, NUMBERS, SHAPES, ImageManifest

#: Half-size of a shape, as a fraction of the picture. Small must stay well above the reader's
#: MIN_AREA as a triangle (about 2r^2 = 40 px) and big must leave a gap between rows of the
#: column (rows are 0.28 * 64 = 17.9 px apart; a big shape is 14 px across).
SIZES: dict[str, float] = {"small": 0.075, "big": 0.11}
#: Jitter (pixels) per size: big shapes get less, so two in a column never touch.
JITTER: dict[str, float] = {"small": 2.0, "big": 1.0}
WHERE = ("left", "middle", "right")
PLACE = {"left": "on the left", "middle": "in the middle", "right": "on the right"}
MAX_COUNT = 3

#: (name, predicate) for every held-out combination. A fact without `where` (the shapes
#: corpus) can only be the first.
HELD_OUT: dict[str, callable] = {
    "purple triangles": lambda f: (f.get("colour"), f.get("shape")) == ("purple", "triangle"),
    "circles on the right": lambda f: (f.get("shape"), f.get("where")) == ("circle", "right"),
}


def held_out(fact: dict) -> str | None:
    """The held-out combination a fact belongs to, or None."""
    return next((name for name, p in HELD_OUT.items() if p(fact)), None)


def caption_of(fact: dict) -> str:
    """The canonical caption. A fact without size/where is the shapes corpus's caption."""
    n = fact["count"]
    size = f"{fact['size']} " if fact.get("size") else ""
    shape = fact["shape"] + ("" if n == 1 else "s")
    place = f" {PLACE[fact['where']]}" if fact.get("where") else ""
    return f"{NUMBERS[n - 1]} {size}{fact['colour']} {shape}{place}"


def render(fact: dict, size: int = IMAGE_SIZE, seed: int = 0) -> np.ndarray:
    """Draw one scene: `count` shapes in column `where`, rows picked at random."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    r = size * SIZES[fact["size"]]
    j = JITTER[fact["size"]]
    col = WHERE.index(fact["where"])
    for row in rng.permutation(3)[: fact["count"]]:
        cy = size * (0.22 + 0.28 * int(row)) + rng.uniform(-j, j)
        cx = size * (0.22 + 0.28 * col) + rng.uniform(-j, j)
        img[_RENDER[fact["shape"]](size, cy, cx, r)] = COLOURS[fact["colour"]]
    return img


def all_facts() -> list[dict]:
    """Every scene: 3 counts x 2 sizes x 5 colours x 3 shapes x 3 places = 270."""
    return [{"count": n, "size": z, "colour": c, "shape": s, "where": w}
            for w in WHERE for s in SHAPES for c in COLOURS for z in SIZES
            for n in range(1, MAX_COUNT + 1)]


def random_fact(rng) -> dict:
    return {"count": int(rng.integers(1, MAX_COUNT + 1)), "size": str(rng.choice(list(SIZES))),
            "colour": str(rng.choice(list(COLOURS))), "shape": str(rng.choice(SHAPES)),
            "where": str(rng.choice(WHERE))}


def _write(out_dir: Path, facts: list[dict], seeds: list[int]) -> ImageManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "images.bin", "wb") as f:
        for fact, s in zip(facts, seeds, strict=True):
            f.write(render(fact, seed=s).tobytes())
    man = ImageManifest(size=IMAGE_SIZE, channels=3, captions=[caption_of(x) for x in facts],
                        facts=facts, built=time.strftime("%Y-%m-%d %H:%M:%S"), kind="scenes")
    man.save(out_dir / "manifest.json")
    return man


def scenes_corpus(out_dir: str | Path, *, n_images: int = 64000, seed: int = 0,
                  progress=print) -> ImageManifest:
    """Render `n_images` scenes, skipping every held-out combination. ~790 MB at 64k."""
    rng = np.random.default_rng(seed)
    facts, seeds, i = [], [], 0
    while len(facts) < n_images:
        fact = random_fact(rng)
        i += 1
        if held_out(fact):
            continue
        facts.append(fact)
        seeds.append(seed * 1_000_003 + i)
    man = _write(Path(out_dir), facts, seeds)
    if progress:
        progress(f"{n_images:,} scenes at {IMAGE_SIZE}x{IMAGE_SIZE}, holding out "
                 f"{' and '.join(HELD_OUT)} -> {out_dir}/images.bin")
    return man


def holdout_corpus(out_dir: str | Path, *, per_combo: int = 64, seed: int = 999) -> ImageManifest:
    """Only the held-out combinations, `per_combo` of each, for scoring a scorer (G4b's
    contrastive model is tested on pictures it never saw, like the tokenizers are)."""
    rng = np.random.default_rng(seed)
    facts, seeds, i = [], [], 0
    for name in HELD_OUT:
        got = 0
        while got < per_combo:
            fact = random_fact(rng)
            i += 1
            if held_out(fact) == name:
                facts.append(fact)
                seeds.append(seed + i)
                got += 1
    return _write(Path(out_dir), facts, seeds)
