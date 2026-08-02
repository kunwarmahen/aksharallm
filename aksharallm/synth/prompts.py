"""The seed grid: where diversity actually comes from.

The obvious way to generate 5,000 samples is to send the same prompt 5,000 times at a high
temperature. It does not work, and the failure is quiet. A model asked "write a Python
exercise" writes about FizzBuzz, palindromes, and reversing a string, over and over; raising
the temperature changes the variable names and the wording of the docstring, not the *task*.
You end up with a large dataset whose effective size is a few dozen distinct problems, the
duplicate filter throws most of it away, and what survives teaches the model three functions
very thoroughly.

So the prompt is assembled from a **grid** — topic × twist × difficulty for code, subject ×
form × constraint for chat — and the grid is walked in a shuffled order rather than sampled
independently, so 200 samples use 200 different cells instead of landing on the same popular
corner twice. That is a coverage guarantee, which a temperature is not.

`TEMPLATE_VERSION` is recorded in every dataset's `meta.json`. When a prompt is edited, bump
it: two batches generated from different wordings are two datasets that happen to share a
directory, and six weeks later nothing else will tell them apart.

Read with: docs/13-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

#: v2 (2026-08-01): the Python template gained the "work out every expected value by
#: executing your own code" rule. The first real batch lost three quarters of its samples to
#: `tests_failed`, and reading the rejects showed almost all of them were a correct solution
#: with a wrong expected value in the test — a plausible-looking `sorted(key=len)` example
#: that is simply not what the function returns.
TEMPLATE_VERSION = 2


@dataclass(frozen=True)
class Seed:
    """One cell of the grid: everything that makes this request different from the last."""

    id: str
    recipe: str
    #: Substituted into the recipe's template.
    fields: dict

    def as_dict(self) -> dict:
        return {"id": self.id, "recipe": self.recipe, **self.fields}


# --------------------------------------------------------------------------------------
# python: what the exercise is about
# --------------------------------------------------------------------------------------

#: Deliberately mundane and broad. The point is not to be clever, it is to be *spread*:
#: these are the shapes of the standard library a small model will be asked about, and a
#: dataset that covers twenty of them beats one that covers three brilliantly.
PY_TOPICS = [
    "strings and text processing",
    "lists and slicing",
    "dictionaries and lookup tables",
    "sets and membership",
    "tuples and unpacking",
    "sorting with a key function",
    "searching and filtering a sequence",
    "counting and frequency",
    "integer arithmetic and number theory",
    "floating point and rounding",
    "dates, times and durations (datetime only)",
    "recursion",
    "iterators and generators",
    "nested data structures (lists of dicts)",
    "matrices as lists of lists",
    "stacks and queues built from a list",
    "simple parsing of a formatted string",
    "validation and raising errors",
    "small class with two or three methods",
    "bit manipulation",
]

#: The constraint that makes the exercise a *specific* exercise rather than the first thing
#: the model thinks of about that topic. Several of these exist to force edge cases into the
#: tests, which is where a generated dataset is usually thinnest.
PY_TWISTS = [
    "it must handle the empty input correctly",
    "it must raise ValueError on invalid input, and the tests must check that",
    "it must work on a single-element input as well as a large one",
    "it must not mutate its argument",
    "it must be case-insensitive",
    "it must preserve the original order of the input",
    "it must handle negative numbers",
    "it must have a keyword argument with a sensible default",
    "it must return a new object rather than None",
    "it must handle duplicate values in the input",
    "it must work for both str and list input",
    "it must be written without any import at all",
]

PY_DIFFICULTY = [
    ("easy", "a beginner could write it in five lines"),
    ("medium", "it needs one non-obvious step, but still fits in about fifteen lines"),
]


# --------------------------------------------------------------------------------------
# chat: what an instruction actually looks like
# --------------------------------------------------------------------------------------

CHAT_SUBJECTS = [
    "everyday science", "cooking and food", "computers and the internet",
    "history", "geography and travel", "health and the human body",
    "money and everyday economics", "plants and animals", "space and astronomy",
    "language and words", "sport and games", "music and instruments",
    "weather and climate", "tools and how things are made", "art and design",
    "transport and vehicles", "school subjects", "the natural world",
]

#: The *shape* of the instruction, which matters more than the subject: a model that has
#: only ever seen "explain X" answers "list three X" with an explanation of X.
CHAT_FORMS = [
    "ask a plain factual question",
    "ask for an explanation of why something happens",
    "ask for a comparison between two things",
    "ask for a numbered list of a specific number of items",
    "ask for step-by-step instructions",
    "ask for a definition in one sentence",
    "ask for a short piece of writing (a note, a caption, an email opening)",
    "ask the assistant to rewrite a given sentence in a different style",
    "ask the assistant to summarise a short passage that you include in the question",
    "ask a question whose honest answer is 'it depends', so the answer must say what it "
    "depends on",
    "ask for a common misconception to be corrected",
    "ask a simple arithmetic or estimation question that needs the reasoning shown",
]

#: Length and format constraints. These are what "instruction following" is measured on —
#: the eval harness's judge marks an answer down to a 3 for ignoring one, so the training
#: data has to contain them.
CHAT_CONSTRAINTS = [
    "the answer must be at most two sentences",
    "the answer must be a numbered list and nothing else",
    "the answer must be under 60 words",
    "the answer must avoid jargon entirely — a ten-year-old should follow it",
    "the answer must give one concrete example",
    "the answer must be one short paragraph",
]


# --------------------------------------------------------------------------------------
# preference: how the rejected answer is worse
# --------------------------------------------------------------------------------------

#: DPO learns the *difference* between the pair, so the difference has to be one thing. A
#: rejected answer that is worse in six ways at once teaches "prefer the first style", which
#: is not a preference, it is a formatting habit. Each of these is a single named flaw, kept
#: in the sample so the dataset can be audited by flaw type — and so a model that has learned
#: to avoid one but not another can be seen doing it.
FLAWS = [
    ("verbose", "far too long and padded with restatement, though the facts are right"),
    ("ignores_format", "ignores the format or length instruction completely"),
    ("hedged", "hedges so much that it never actually answers"),
    ("overconfident_wrong", "confidently states one specific fact that is wrong"),
    ("off_target", "answers a slightly different question than the one asked"),
    ("robotic", "correct but written as a stiff wall of jargon"),
]


# --------------------------------------------------------------------------------------
# walking the grid
# --------------------------------------------------------------------------------------

def _shuffled(grid: list[tuple], seed: int) -> list[tuple]:
    cells = list(grid)
    random.Random(seed).shuffle(cells)
    return cells


def seeds(recipe: str, n: int, seed: int = 0) -> list[Seed]:
    """`n` seeds for `recipe`, no cell repeated until every cell has been used once.

    The grids are 480 cells (python) and 1,296 (chat), so a few hundred samples never
    repeats a cell at all. Past that it wraps with a different shuffle rather than stopping:
    the same cell asked twice at temperature 0.9 does give two different problems — it is
    only the *sole* source of variety that fails, not a contributing one. The pass number
    goes into the seed id, so a duplicate is traceable to a wrap rather than a mystery.
    """
    if recipe == "python":
        grid = list(itertools.product(PY_TOPICS, PY_TWISTS, PY_DIFFICULTY))
    elif recipe in ("chat", "preference"):
        grid = list(itertools.product(CHAT_SUBJECTS, CHAT_FORMS, CHAT_CONSTRAINTS))
    else:
        raise ValueError(f"no seed grid for recipe {recipe!r}")

    out: list[Seed] = []
    lap = 0
    while len(out) < n:
        cells = _shuffled(grid, seed + lap)
        for i, cell in enumerate(cells):
            if len(out) >= n:
                break
            sid = f"{recipe}-{seed}-{lap}-{i:04d}"
            if recipe == "python":
                topic, twist, (level, level_note) = cell
                fields = {"topic": topic, "twist": twist, "difficulty": level,
                          "difficulty_note": level_note}
            else:
                subject, form, constraint = cell
                fields = {"subject": subject, "form": form, "constraint": constraint}
                if recipe == "preference":
                    # The flaw is drawn per sample rather than being a grid axis: it would
                    # multiply the grid by six for no extra coverage of what is being asked,
                    # and every flaw should appear against every kind of question.
                    flaw, flaw_note = FLAWS[(i + lap) % len(FLAWS)]
                    fields |= {"flaw": flaw, "flaw_note": flaw_note}
            out.append(Seed(id=sid, recipe=recipe, fields=fields))
        lap += 1
    return out


def grid_size(recipe: str) -> int:
    if recipe == "python":
        return len(PY_TOPICS) * len(PY_TWISTS) * len(PY_DIFFICULTY)
    if recipe in ("chat", "preference"):
        return len(CHAT_SUBJECTS) * len(CHAT_FORMS) * len(CHAT_CONSTRAINTS)
    return 0
