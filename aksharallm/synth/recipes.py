"""The three recipes: what is asked for, how the reply is read, and what makes it usable.

A recipe is four small things — a prompt, a parser, a set of checks, and an export — and the
interesting design in all three is the **output format**, which is plain section headers:

    ### PROBLEM
    ...
    ### SOLUTION
    ```python
    ...
    ```

not JSON. Asking a model for JSON containing code is asking it to escape newlines and quotes
inside a Python function by hand, and a 14B model gets that wrong often enough that a
meaningful fraction of the budget goes on parse failures — failures that are *correlated
with long, interesting functions*, which is the worst possible bias to introduce into a
dataset. Headers and fences are what a model writes naturally, so parse failures stay rare
and, when they happen, are not about the content.

The three:

| recipe | what it produces | how it is checked | feeds |
|---|---|---|---|
| `python` | problem + solution + tests | **the sandbox runs the tests, twice** | SFT |
| `chat` | one instruction, one answer | format, length, boilerplate, dedup | SFT |
| `preference` | one prompt, a good and a deliberately flawed answer | both parse, differ, named flaw | DPO |

`preference` is the only one that asks for something *bad* on purpose. DPO learns the
difference within a pair, so a rejected answer has to be plausible and worse in one named
way — and the flaw's name is kept on the sample, which is what makes the dataset auditable
later ("did we teach it to stop hedging, or only to stop rambling?").

Read with: docs/13-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import ast
import re

from .dataset import SynthError
from .prompts import TEMPLATE_VERSION, Seed

# --------------------------------------------------------------------------------------
# reading the reply
# --------------------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^[ \t]*#{2,4}\s*([A-Z][A-Z ]{2,20})\s*:?\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def sections(text: str) -> dict[str, str]:
    """`{"PROBLEM": "...", "SOLUTION": "..."}` from a reply written with `### HEADER` lines.

    Anything before the first header is dropped: models like to open with "Sure, here is an
    exercise about lists" and that sentence is not part of any section.
    """
    found = list(_SECTION_RE.finditer(text or ""))
    out: dict[str, str] = {}
    for i, match in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        out[match.group(1).strip().upper()] = text[match.end():end].strip()
    return out


def code_block(text: str) -> str:
    """The contents of the first fenced block, or the text itself if it was not fenced.

    Both happen. A model told to put code in a fence usually does; the same model at
    temperature 0.9 sometimes just writes the code. Accepting both costs three lines and
    recovers a few percent of every batch.
    """
    found = _FENCE_RE.search(text or "")
    body = found.group(1) if found else (text or "")
    return body.strip("\n").rstrip()


def entry_point(solution: str) -> str | None:
    """The name of the thing the tests are supposed to call.

    The AST is asked first, so a `def` inside a docstring or a nested helper cannot be
    mistaken for the entry point. The regex is the fallback for code that does not parse —
    which is itself a rejection, but a rejection that reads better with a name in it.
    """
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        found = re.search(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", solution or "",
                          re.MULTILINE)
        return found.group(1) if found else None
    names = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and not n.name.startswith("_")]
    return names[-1] if names else None


# --------------------------------------------------------------------------------------
# the recipes
# --------------------------------------------------------------------------------------

class Recipe:
    """One way of turning a seed into a training sample."""

    name = ""
    blurb = ""
    #: Which trainer eventually consumes it: `sft` (messages) or `dpo` (preference triples).
    consumer = "sft"
    #: Whether a sample can be *executed* to check it. Only one recipe can.
    verified = False

    def messages(self, seed: Seed) -> list[dict]:
        raise NotImplementedError

    def parse(self, text: str, seed: Seed) -> dict:
        """Reply text -> a sample dict. Raises SynthError with a REJECT_REASONS key."""
        raise NotImplementedError

    def dedup_key(self, sample: dict) -> str:
        """The text two samples are compared on. Not the whole sample: for code, comparing
        the *problem* catches "the same exercise, different variable names", which comparing
        the solution would not."""
        raise NotImplementedError

    def to_sft(self, sample: dict) -> list[dict] | None:
        return None

    def to_dpo(self, sample: dict) -> dict | None:
        return None


PY_SYSTEM = """\
You write small, self-contained Python exercises that are used as training data for a
language model. Precision matters more than creativity: the exercise is thrown away
automatically unless the tests run and pass.

Hard rules:
* Standard library only, and prefer no imports at all. No file access, no network, no
  input(), no randomness, no current time, no printing.
* The solution must be deterministic and fast — the tests get a few seconds of CPU.
* Write the tests as bare `assert` statements at module level, at least three of them,
  covering the ordinary case and the edge case the constraint asks about.
* The tests must call the function by name. They must not import anything from a module;
  your solution and your tests are pasted into one file, in that order.
* The tests must FAIL if the function's body were removed. Do not write assertions like
  `assert isinstance(result, list)` that any implementation would satisfy.
* EVERY expected value must be one you have worked out by executing your own code line by
  line on that input. This is where these exercises usually go wrong: a sorted-by-length or
  case-insensitive example that looks right and is not. If you are not certain what the
  function returns for an input, use a simpler input — short lists, small numbers, obvious
  answers. A boring test that is right is worth more than an interesting one that is wrong.
* No explanation, no commentary, no markdown outside the three sections. Nothing after the
  final fence.

Answer in exactly this format:

### PROBLEM
One paragraph stating the task as a person would ask for it. Name the function and describe
what it takes and returns. Do not include the solution.

### SOLUTION
```python
def the_function(...):
    ...
```

### TESTS
```python
assert the_function(...) == ...
```
"""


class PythonRecipe(Recipe):
    name = "python"
    blurb = ("problem + solution + tests, and the tests are executed — the only recipe here "
             "whose correctness is checked rather than assumed")
    consumer = "sft"
    verified = True

    def messages(self, seed: Seed) -> list[dict]:
        f = seed.fields
        ask = (f"Write one exercise.\n\n"
               f"Topic: {f['topic']}\n"
               f"Constraint the task must include: {f['twist']}\n"
               f"Difficulty: {f['difficulty']} — {f['difficulty_note']}\n\n"
               "Use the three sections exactly as instructed.")
        return [{"role": "system", "content": PY_SYSTEM},
                {"role": "user", "content": ask}]

    def parse(self, text: str, seed: Seed) -> dict:
        block = sections(text)
        problem = block.get("PROBLEM", "").strip()
        solution = code_block(block.get("SOLUTION", ""))
        tests = code_block(block.get("TESTS", ""))
        if not (problem and solution and tests):
            raise SynthError("unparseable")
        name = entry_point(solution)
        return {
            "kind": "python",
            "problem": problem,
            "solution": solution,
            "tests": tests,
            "entry_point": name,
            "topic": seed.fields["topic"],
            "twist": seed.fields["twist"],
            "difficulty": seed.fields["difficulty"],
        }

    def dedup_key(self, sample: dict) -> str:
        # The problem statement, not the code: two teachers asked about "count word
        # frequency" write the same exercise with different identifiers, and the identifiers
        # are exactly what a code-level comparison would latch onto.
        return sample["problem"]

    def to_sft(self, sample: dict) -> list[dict]:
        """The exercise as a conversation.

        The assistant turn is the solution in a fenced block and nothing else — no
        re-statement of the problem, no "here you go". Whatever shape is in the training
        data is the shape the model will produce, and a model that answers a coding question
        with a paragraph of preamble is harder to grade and harder to use.
        """
        return [
            {"role": "user", "content": sample["problem"]},
            {"role": "assistant", "content": f"```python\n{sample['solution']}\n```"},
        ]


CHAT_SYSTEM = """\
You write instruction/response pairs used as training data for a small language model.

Hard rules:
* Write BOTH sides: the human's question and the assistant's answer.
* The question must be self-contained — no "as we discussed", no reference to a previous
  turn, no attachment.
* The answer must obey the constraint you are given exactly. If it says two sentences, write
  two sentences.
* The answer must be correct and specific. Prefer a named example over a general claim. If
  the honest answer is that it depends, say what it depends on.
* No disclaimers, no "as an AI", no offers to help further, no sign-off.
* Plain prose or a plain list. No headings, no bold, no emoji.

Answer in exactly this format:

### PROMPT
The human's question.

### ANSWER
The assistant's answer.
"""


class ChatRecipe(Recipe):
    name = "chat"
    blurb = ("instruction/response pairs — the bootstrap for a base model that has never "
             "seen an instruction; checked by format and by diversity, not by execution")
    consumer = "sft"

    def messages(self, seed: Seed) -> list[dict]:
        f = seed.fields
        ask = (f"Write one pair.\n\n"
               f"Subject: {f['subject']}\n"
               f"Kind of request: {f['form']}\n"
               f"Constraint on the answer: {f['constraint']}\n\n"
               "Use the two sections exactly as instructed.")
        return [{"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": ask}]

    def parse(self, text: str, seed: Seed) -> dict:
        block = sections(text)
        prompt = block.get("PROMPT", "").strip()
        answer = block.get("ANSWER", "").strip()
        if not (prompt and answer):
            raise SynthError("unparseable")
        return {
            "kind": "chat",
            "prompt": prompt,
            "answer": answer,
            "subject": seed.fields["subject"],
            "form": seed.fields["form"],
            "constraint": seed.fields["constraint"],
        }

    def dedup_key(self, sample: dict) -> str:
        return sample["prompt"]

    def to_sft(self, sample: dict) -> list[dict]:
        return [{"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["answer"]}]


PREF_SYSTEM = """\
You write preference pairs used to align a small language model with DPO. Each pair is one
question, one good answer, and one worse answer.

Hard rules:
* The good answer is correct, specific, and obeys the constraint exactly.
* The worse answer must be PLAUSIBLE — the kind of thing a decent model actually produces —
  and worse in exactly the ONE way you are told. Do not make it worse in other ways as well:
  the pair teaches the difference between them, so a second difference teaches the wrong
  lesson.
* Both answers must address the same question. Neither may mention that it is good or bad,
  or refer to the other.
* No disclaimers, no "as an AI", no sign-off, no markdown headings.

Answer in exactly this format:

### PROMPT
The human's question.

### GOOD
The better answer.

### BAD
The worse answer.
"""


class PreferenceRecipe(Recipe):
    name = "preference"
    blurb = ("chosen/rejected pairs for DPO, each pair differing in one named way — the "
             "flaw is recorded so the dataset can be audited by flaw type")
    consumer = "dpo"

    def messages(self, seed: Seed) -> list[dict]:
        f = seed.fields
        ask = (f"Write one preference pair.\n\n"
               f"Subject: {f['subject']}\n"
               f"Kind of request: {f['form']}\n"
               f"Constraint on the answer: {f['constraint']}\n"
               f"The worse answer must be worse in exactly this way: {f['flaw_note']}\n\n"
               "Use the three sections exactly as instructed.")
        return [{"role": "system", "content": PREF_SYSTEM},
                {"role": "user", "content": ask}]

    def parse(self, text: str, seed: Seed) -> dict:
        block = sections(text)
        prompt = block.get("PROMPT", "").strip()
        good = block.get("GOOD", "").strip()
        bad = block.get("BAD", "").strip()
        if not (prompt and good and bad):
            raise SynthError("unparseable")
        if good.strip() == bad.strip():
            raise SynthError("identical_pair")
        return {
            "kind": "preference",
            "prompt": prompt,
            "chosen": good,
            "rejected": bad,
            "flaw": seed.fields["flaw"],
            "subject": seed.fields["subject"],
            "form": seed.fields["form"],
            "constraint": seed.fields["constraint"],
        }

    def dedup_key(self, sample: dict) -> str:
        return sample["prompt"]

    def to_dpo(self, sample: dict) -> dict:
        return {"prompt": sample["prompt"], "chosen": sample["chosen"],
                "rejected": sample["rejected"]}


RECIPES: dict[str, Recipe] = {r.name: r for r in
                              (PythonRecipe(), ChatRecipe(), PreferenceRecipe())}


def get_recipe(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError:
        raise SynthError(f"unknown recipe {name!r}. Known: {', '.join(RECIPES)}")


def catalogue() -> list[dict]:
    """What each recipe is, for the CLI's `recipes` command and the portal's picker."""
    from .prompts import grid_size

    return [{"name": r.name, "blurb": r.blurb, "consumer": r.consumer,
             "verified": r.verified, "grid": grid_size(r.name),
             "template_version": TEMPLATE_VERSION}
            for r in RECIPES.values()]
