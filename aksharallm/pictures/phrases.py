"""G4a's captions: many ways to say one picture, some never used in training.

G1–G3 read exactly three words ("two red circles"). G4a reads a sentence through a frozen text
model (text.py), so the captions have to be sentences, and the test has to include sentences
the model has never seen in that form. Each phrasing is a template over a picture's facts:

    canonical        two red circles
    a picture of     a picture of two red circles
    ...
    held out         show me two red circles       <- never in training
                     two circles in red            <- never in training

The held-out phrasings are a second held-out axis beside the held-out combination (purple
triangles): a model that learned "these exact strings make these pictures" draws nothing
useful from a new phrasing, and one that reads the words still can. Scoring never parses a
caption: the reader is checked against the facts the caption was made from.

G4b's scenes (`scenes.py`) say two more things, size and place, and the same templates say
them: the size rides with the shape word ("two big circles, all red") and the place goes at the
end ("... on the left"). A fact without them gives exactly the strings G4a trained on.

`fact_from_text` exists only for the portal's free-text box, where the facts are not known: it
looks for one count, one colour and one shape word, and says when it cannot find them.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import re

from ..vision.image import COLOURS, NUMBERS, SHAPES

DIGITS = ("1", "2", "3", "4")
_PLACE = {"left": "on the left", "middle": "in the middle", "right": "on the right"}


def _num(f: dict, digits: bool = False) -> str:
    return (DIGITS if digits else NUMBERS)[f["count"] - 1]


def _shape(f: dict) -> str:
    """The shape word, with the scene's size in front of it ("big circles") when there is one."""
    size = f"{f['size']} " if f.get("size") else ""
    return size + f["shape"] + ("" if f["count"] == 1 else "s")


def _canon(f: dict, *, digits: bool = False) -> str:
    """"two big red circles" (size only for a scene; no place)."""
    size = f"{f['size']} " if f.get("size") else ""
    return f"{_num(f, digits)} {size}{f['colour']} {f['shape']}" + ("" if f["count"] == 1 else "s")


def _place(f: dict) -> str:
    """" on the left" for a scene, "" for a shapes-corpus fact, so a shapes fact's phrasings
    are exactly the strings G4a trained on."""
    return f" {_PLACE[f['where']]}" if f.get("where") else ""


def _all_of(f):
    return (f"one {_shape(f)}, {f['colour']}" if f["count"] == 1
            else f"{_num(f)} {_shape(f)}, all {f['colour']}")


def _of_them(f):
    size = f"{f['size']} " if f.get("size") else ""
    return (f"a {size}{f['colour']} {f['shape']}, just one" if f["count"] == 1
            else f"{size}{f['colour']} {f['shape']}s, {_num(f)} of them")


#: name -> template. Order matters only for display; the first is the canonical caption. A
#: scene's place goes at the end of every one, before "on a dark background" in that one.
TRAIN: dict[str, callable] = {
    "canonical": lambda f: _canon(f) + _place(f),
    "picture_of": lambda f: "a picture of " + _canon(f) + _place(f),
    "digits": lambda f: _canon(f, digits=True) + _place(f),
    "all_of": lambda f: _all_of(f) + _place(f).replace(" ", ", ", 1),
    "of_them": lambda f: _of_them(f) + _place(f).replace(" ", ", ", 1),
    "there_are": lambda f: ("there is " if f["count"] == 1 else "there are ") + _canon(f) + _place(f),
    "coloured": lambda f: f"{_num(f)} {_shape(f)} coloured {f['colour']}" + _place(f),
    "image_showing": lambda f: "an image showing " + _canon(f) + _place(f),
    "dark_background": lambda f: _canon(f) + _place(f) + " on a dark background",
    "draw": lambda f: "draw " + _canon(f) + _place(f),
}

#: Never used in training: new phrasings of facts the model has seen.
HELD_OUT: dict[str, callable] = {
    "show_me": lambda f: "show me " + _canon(f) + _place(f),
    "in_colour": lambda f: f"{_num(f)} {_shape(f)} in {f['colour']}" + _place(f),
}


def phrase(f: dict, name: str) -> str:
    return (TRAIN.get(name) or HELD_OUT[name])(f)


def all_facts(kind: str = "shapes") -> list[dict]:
    """Every fact a corpus can draw. Shapes: 4 counts x 5 colours x 3 shapes = 60, in the
    same order as `codes.all_captions()`. Scenes (G4b): the 270 of `scenes.all_facts()`."""
    if kind == "scenes":
        from .scenes import all_facts as scene_facts
        return scene_facts()
    if kind != "shapes":
        raise ValueError(f"no corpus kind {kind!r} (shapes or scenes)")
    return [{"count": n, "colour": c, "shape": s}
            for s in SHAPES for c in COLOURS for n in range(1, len(NUMBERS) + 1)]


_COUNT = {**{w: i + 1 for i, w in enumerate(NUMBERS)}, **{d: i + 1 for i, d in enumerate(DIGITS)}}


_SIZE = {"small": "small", "little": "small", "tiny": "small", "big": "big", "large": "big"}
_WHERE = {"left": "left", "middle": "middle", "centre": "middle", "center": "middle",
          "right": "right"}


def fact_from_text(text: str) -> dict | None:
    """One count, one colour and one shape word in free text -> a fact; None if any is
    missing or ambiguous. A size word (small/big) and a place word (left/middle/right) are
    added when present, once each, so a scene sentence is judged on all five. For the
    portal's free-text box only; scoring uses known facts."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts = {_COUNT[w] for w in words if w in _COUNT}
    if not counts and any(w in ("a", "an") for w in words):
        counts = {1}
    colours = {w for w in words if w in COLOURS}
    shapes = {w.rstrip("s") for w in words if w.rstrip("s") in SHAPES}
    if len(counts) != 1 or len(colours) != 1 or len(shapes) != 1:
        return None
    fact = {"count": counts.pop(), "colour": colours.pop(), "shape": shapes.pop()}
    sizes = {_SIZE[w] for w in words if w in _SIZE}
    places = {_WHERE[w] for w in words if w in _WHERE}
    if len(sizes) == 1:
        fact["size"] = sizes.pop()
    if len(places) == 1:
        fact["where"] = places.pop()
    return fact
