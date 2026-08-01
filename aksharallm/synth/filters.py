"""What gets thrown away, and why — including the near-duplicates nothing else can see.

Two kinds of filter live here and they fail in different ways.

**Validity** is cheap and obvious: an answer that is four characters long, a "solution" with
no `def` in it, a chat reply that starts "As an AI language model". These are caught the
first time you look at the data.

**Duplication** is neither. An exact-duplicate check is trivially satisfied — two samples
that differ by a variable name are not equal — while the dataset quietly becomes fifty
paraphrases of the same four problems. So the check here is on *content shingles*: overlapping
runs of five words, compared by Jaccard similarity, which is high for a paraphrase and low
for a genuinely different sample.

The comparison is against every sample kept so far, which is quadratic if done naively and
would be the slowest thing in the pipeline by a wide margin at 10,000 samples. An inverted
index from shingle to sample makes it linear in practice: only samples that share at least
one five-word run are candidates, and two unrelated Python problems share none.

`REJECT_REASONS` exists because the rejection tally *is* the quality signal. A recipe whose
pass rate is 30% because the teacher writes bad tests needs a different prompt; one whose
pass rate is 30% because everything is a near-duplicate needs a bigger seed grid. The number
alone cannot tell those apart, so every reject is counted by reason and the CLI prints the
breakdown rather than a single percentage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

#: Every reason a sample can be dropped, with what it means. Kept in one place so the CLI,
#: the portal panel and `meta.json` all use the same vocabulary.
REJECT_REASONS = {
    "unparseable": "the teacher's reply did not contain the sections the template asked for",
    "too_short": "there was not enough text to be a sample",
    "too_long": "longer than the cap — usually the model rambling past the task",
    "boilerplate": "an assistant disclaimer or a refusal rather than an answer",
    "leaked_template": "the reply contained the instructions we sent it",
    "no_entry_point": "the solution defines no function to test",
    "bad_tests": "the tests do not exercise the function, or there are too few of them",
    "unsafe_code": "the code touches the filesystem, the network or the process",
    "tests_failed": "the tests were executed and did not pass",
    "vacuous_tests": "the tests passed even with the solution removed",
    "sandbox_error": "the code could not be run at all",
    "duplicate": "an exact repeat of a sample already kept",
    "near_duplicate": "a paraphrase of a sample already kept",
    "identical_pair": "the preferred and rejected answers were the same",
    "teacher_error": "the teacher failed to answer",
}

#: Phrases that mean the teacher answered *about* the task instead of doing it. A refusal or
#: a disclaimer in the training data teaches our model to produce refusals and disclaimers,
#: which is the one thing a 300M model does not need help with.
BOILERPLATE = (
    "as an ai language model", "as an ai model", "i cannot fulfill", "i can't fulfill",
    "i cannot assist", "i'm sorry, but i", "i am sorry, but i", "i do not have the ability",
    "certainly! here is the", "sure! here's a python function that solves",
)

#: Section headers from our own templates. If one comes back it means the model echoed the
#: instructions, and whatever follows is the instruction, not the answer.
TEMPLATE_MARKERS = ("### PROBLEM", "### SOLUTION", "### TESTS", "### PROMPT", "### ANSWER",
                    "### GOOD", "### BAD")

#: Imports and builtins a self-contained exercise has no business using. This is a *quality*
#: filter first — a function that reads a file cannot be tested by asserts, so it will fail
#: the sandbox anyway — and a safety filter second, since the sandbox is honest about not
#: being a security boundary (see `infer/sandbox.py`).
UNSAFE = re.compile(
    r"\b(?:import\s+(?:os|sys|subprocess|socket|shutil|requests|urllib|pathlib|ctypes|"
    r"multiprocessing|threading)\b|from\s+(?:os|sys|subprocess|socket|shutil|requests|"
    r"urllib|pathlib|ctypes)\s+import\b|__import__|eval\s*\(|exec\s*\(|open\s*\(|input\s*\()")

WORD_RE = re.compile(r"[a-z0-9_]+")


def normalise(text: str) -> str:
    """Lowercased words only — the form two paraphrases have most in common."""
    return " ".join(WORD_RE.findall((text or "").lower()))


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Overlapping runs of `k` words.

    Five is the usual choice and it is a trade: at k=3 two different problems about lists
    share "return a new list" and look similar; at k=8 a paraphrase that changes one word
    every seven shares nothing and looks new.
    """
    words = normalise(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


@dataclass
class Deduper:
    """Exact and near-duplicate detection over everything kept so far.

    `threshold` is Jaccard similarity on shingle sets. 0.6 is deliberately aggressive: in a
    generated dataset the cost of dropping a real sample is one more call to the teacher,
    and the cost of keeping a paraphrase is a permanent bias in the data.
    """

    threshold: float = 0.6
    k: int = 5
    _hashes: set[str] = field(default_factory=set)
    _shingles: list[set[str]] = field(default_factory=list)
    #: shingle -> indices into `_shingles`. The reason this is not quadratic.
    _index: dict[str, list[int]] = field(default_factory=dict)

    def check(self, text: str) -> tuple[str | None, float]:
        """`(reason, similarity)` — reason is None when the sample is new."""
        digest = hashlib.sha1(normalise(text).encode()).hexdigest()
        if digest in self._hashes:
            return "duplicate", 1.0
        mine = shingles(text, self.k)
        if not mine:
            return None, 0.0
        counts: dict[int, int] = {}
        for sh in mine:
            for idx in self._index.get(sh, ()):
                counts[idx] = counts.get(idx, 0) + 1
        best = 0.0
        for idx, shared in counts.items():
            other = self._shingles[idx]
            union = len(mine) + len(other) - shared
            if union:
                best = max(best, shared / union)
        if best >= self.threshold:
            return "near_duplicate", best
        return None, best

    def add(self, text: str) -> None:
        self._hashes.add(hashlib.sha1(normalise(text).encode()).hexdigest())
        mine = shingles(text, self.k)
        idx = len(self._shingles)
        self._shingles.append(mine)
        for sh in mine:
            self._index.setdefault(sh, []).append(idx)

    def __len__(self) -> int:
        return len(self._shingles)


# --------------------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------------------

def check_text(text: str, min_chars: int = 20, max_chars: int = 4000) -> str | None:
    """The reason to drop this piece of prose, or None."""
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return "too_short"
    if len(stripped) > max_chars:
        return "too_long"
    low = stripped.lower()
    if any(phrase in low for phrase in BOILERPLATE):
        return "boilerplate"
    if any(marker in stripped for marker in TEMPLATE_MARKERS):
        return "leaked_template"
    return None


def check_code(solution: str, tests: str, entry_point: str | None) -> str | None:
    """The reason to drop this exercise before spending a sandbox run on it.

    Everything here is cheap and everything here is about *testability*: the strongest check
    in the pipeline is executing the tests, and these exist so that the execution is
    meaningful when it happens. A test block with one assert that never mentions the function
    can pass while proving nothing.
    """
    if not entry_point:
        return "no_entry_point"
    if f"def {entry_point}" not in solution and f"class {entry_point}" not in solution:
        return "no_entry_point"
    if UNSAFE.search(solution) or UNSAFE.search(tests):
        return "unsafe_code"
    asserts = len(re.findall(r"^\s*assert\b", tests, re.MULTILINE))
    if asserts < 2:
        # One assert is one example. Two is the difference between "it returns something"
        # and "it returns the right thing for more than one input".
        return "bad_tests"
    if entry_point not in tests:
        return "bad_tests"
    if re.search(r"^\s*(?:from|import)\s+solution\b", tests, re.MULTILINE):
        # The teacher assumed the solution is an importable module. It is not — the two
        # blocks are concatenated into one file — and the import would fail every time.
        return "bad_tests"
    return None
