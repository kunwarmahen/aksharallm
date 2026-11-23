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

`fact_from_text` exists only for the portal's free-text box, where the facts are not known: it
looks for one count, one colour and one shape word, and says when it cannot find them.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import re

from ..vision.image import COLOURS, NUMBERS, SHAPES

DIGITS = ("1", "2", "3", "4")


def _parts(f: dict) -> tuple[str, str, str, str, str]:
    n = f["count"]
    word, digit = NUMBERS[n - 1], DIGITS[n - 1]
    shape = f["shape"] + ("" if n == 1 else "s")
    return word, digit, f["colour"], f["shape"], shape


def _all_of(f):
    word, _, c, s1, s = _parts(f)
    return f"one {s1}, {c}" if f["count"] == 1 else f"{word} {s}, all {c}"


def _of_them(f):
    word, _, c, s1, s = _parts(f)
    return f"a {c} {s1}, just one" if f["count"] == 1 else f"{c} {s}, {word} of them"


#: name -> template. Order matters only for display; the first is the canonical caption.
TRAIN: dict[str, callable] = {
    "canonical": lambda f: "{0} {2} {4}".format(*_parts(f)),
    "picture_of": lambda f: "a picture of {0} {2} {4}".format(*_parts(f)),
    "digits": lambda f: "{1} {2} {4}".format(*_parts(f)),
    "all_of": _all_of,
    "of_them": _of_them,
    "there_are": lambda f: ("there is one {2} {3}" if f["count"] == 1
                            else "there are {0} {2} {4}").format(*_parts(f)),
    "coloured": lambda f: "{0} {4} coloured {2}".format(*_parts(f)),
    "image_showing": lambda f: "an image showing {0} {2} {4}".format(*_parts(f)),
    "dark_background": lambda f: "{0} {2} {4} on a dark background".format(*_parts(f)),
    "draw": lambda f: "draw {0} {2} {4}".format(*_parts(f)),
}

#: Never used in training: new phrasings of facts the model has seen.
HELD_OUT: dict[str, callable] = {
    "show_me": lambda f: "show me {0} {2} {4}".format(*_parts(f)),
    "in_colour": lambda f: "{0} {4} in {2}".format(*_parts(f)),
}


def phrase(f: dict, name: str) -> str:
    return (TRAIN.get(name) or HELD_OUT[name])(f)


def all_facts() -> list[dict]:
    """Every fact the corpus can draw: 4 counts x 5 colours x 3 shapes = 60, in the same
    order as `codes.all_captions()`."""
    return [{"count": n, "colour": c, "shape": s}
            for s in SHAPES for c in COLOURS for n in range(1, len(NUMBERS) + 1)]


_COUNT = {**{w: i + 1 for i, w in enumerate(NUMBERS)}, **{d: i + 1 for i, d in enumerate(DIGITS)}}


def fact_from_text(text: str) -> dict | None:
    """One count, one colour and one shape word in free text -> a fact; None if any is
    missing or ambiguous. For the portal's free-text box only; scoring uses known facts."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts = {_COUNT[w] for w in words if w in _COUNT}
    if not counts and any(w in ("a", "an") for w in words):
        counts = {1}
    colours = {w for w in words if w in COLOURS}
    shapes = {w.rstrip("s") for w in words if w.rstrip("s") in SHAPES}
    if len(counts) != 1 or len(colours) != 1 or len(shapes) != 1:
        return None
    return {"count": counts.pop(), "colour": colours.pop(), "shape": shapes.pop()}
