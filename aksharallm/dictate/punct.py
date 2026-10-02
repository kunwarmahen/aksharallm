"""Punctuation and capitals, as labels on words — the half of dictation the recogniser cannot do.

The recogniser writes what LibriSpeech's transcripts look like: lower case, no punctuation,
`"i will meet you at the station is that alright"`. A dictation app has to hand back
`"I will meet you at the station. Is that alright?"`. That is a different model's job, and
this file is the contract that model works to.

**It is a tagger, not a rewriter, and that is the design.** For every word the model picks one
of twelve labels — *what punctuation follows it* (none , . ?) times *how it is cased*
(lower, Capitalised, UPPER). It never emits a word. So it cannot invent one, drop one, or
"improve" one, which is exactly what a language model asked to tidy a transcript will
sometimes do — the same argument that chose CTC over an attention decoder for the ear
(docs/23 § day-two problem 1), one layer up. `render` takes the recogniser's words and the
labels and can only *decorate* them; `test_render_never_changes_a_word` holds it to that.

```mermaid
flowchart LR
    T["web text<br/>'Is that alright? I think so.'"] --> N["normalise like the ear:<br/>is that alright i think so"]
    T --> L["labels per word:<br/>Cap·none, lower·none, lower·?<br/>Cap·none, lower·none, lower·."]
    N --> M["bidirectional transformer<br/>(our own, causal: false)"]
    L -.->|cross-entropy| M
    M --> R["render: decorate the words<br/>never change them"]
```

**The training data is free.** Any punctuated text becomes a training pair by deleting what the
tagger has to restore. `examples_from_text` does that to FineWeb-Edu on the fly, and its rules
are the part worth reading, because each one decides what the tagger is taught:

* Text is normalised to **exactly the recogniser's alphabet** (`a-z` and `'`, see `asr/vocab.py`).
  A model trained on input the ear never produces is trained for a different job.
* A sentence containing anything the ear cannot write (a digit, `e.g.`, a URL, `C++`) is
  **dropped whole**, not cleaned — deleting "1984" from "In 1984, he left." teaches a comma after
  "in". A dropped sentence costs data; a cleaned one teaches a wrong rule.
* A line that does not end in `.`, `?` or `!` is a heading, a list item or a caption, and is
  dropped: its missing full stop is not a fact about English.
* The first sentence of a window is dropped: the window started somewhere inside it.
* Four punctuation classes, not eight. `!` and `;` become `.`, `:` and dashes become `,`.
  An exclamation mark is a choice the speaker makes, and a model guessing it wrong is worse
  than one that never does.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PUNCT = ["", ",", ".", "?"]               # what follows the word
CASES = ["lower", "cap", "upper"]
N_PUNCT, N_CASE = len(PUNCT), len(CASES)
N_LABELS = N_PUNCT * N_CASE               # 12 — the tagger's whole output vocabulary
NONE, COMMA, PERIOD, QUESTION = range(N_PUNCT)
LOWER, CAP, UPPER = range(N_CASE)
IGNORE = -100

_CORE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_LEAD = "\"'([{“‘«*_"
_TRAIL_STRIP = "\"')]}”’»*_"
#: Characters that are a word boundary in writing and a pause in speech. Turned into ", "
#: before splitting, so "well—maybe" becomes "well, maybe": two words and a comma.
_DASHES = re.compile(r"\s*(?:—|–|--|\s-\s)\s*")
_INNER_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")


def label(punct: int, case: int) -> int:
    return punct * N_CASE + case


def split_label(lab: int) -> tuple[int, int]:
    return divmod(int(lab), N_CASE)


def case_of(core: str) -> int:
    if core == core.lower():
        return LOWER
    if len(core) > 1 and core == core.upper():
        return UPPER
    # "I", "Paris", "McDonald": capitalised. Inner capitals are lost — that is what the
    # personal dictionary's spellings are for (dictate/personal.py).
    return CAP if core[0].isupper() else LOWER


def punct_of(trail: str) -> int:
    if "?" in trail:
        return QUESTION
    if any(c in trail for c in ".!;"):
        return PERIOD
    if any(c in trail for c in ",:"):
        return COMMA
    return NONE


@dataclass
class Word:
    text: str      # normalised, as the recogniser would write it
    label: int


def _word(raw: str) -> Word | None:
    """One whitespace-separated piece of written text -> the word the ear would hear, or None
    if it is something the ear cannot write."""
    raw = raw.replace("’", "'").replace("‘", "'")
    core = raw.lstrip(_LEAD)
    end = len(core)
    while end and not core[end - 1].isalnum():
        end -= 1
    trail = core[end:]
    core = core[:end]
    if not _CORE.match(core):
        return None
    return Word(core.lower(), label(punct_of(trail), case_of(core)))


def sentences(line: str) -> list[list[Word] | None]:
    """A line of text -> its sentences, each a list of words, or None for one that holds
    something the recogniser could not have written."""
    line = _INNER_HYPHEN.sub(" ", _DASHES.sub(", ", line))
    out: list[list[Word] | None] = []
    cur: list[Word] = []
    bad = False
    for raw in line.split():
        w = _word(raw)
        if w is None:
            bad = True
            # A dropped piece may still end the sentence ("in 1984."): keep the boundary.
            stripped = raw.rstrip(_TRAIL_STRIP)
            if stripped.endswith((".", "?", "!")):
                out.append(None)
                cur, bad = [], False
            continue
        cur.append(w)
        if split_label(w.label)[0] in (PERIOD, QUESTION):
            out.append(None if bad else cur)
            cur, bad = [], False
    if cur:
        out.append(None if bad else cur)
    return out


def examples_from_text(text: str, drop_first: bool = True) -> list[Word]:
    """Written text -> the word sequence a recogniser would produce, with the labels to
    restore. Sentences that cannot be represented are dropped (see the module docstring)."""
    words: list[Word] = []
    first = drop_first
    for line in text.splitlines():
        s = line.strip().rstrip(_TRAIL_STRIP)
        if not s:
            continue
        if not s.endswith((".", "?", "!")):
            first = False     # a heading also ends whatever the window started inside
            continue
        for sent in sentences(line):
            if first:
                first = False
                continue
            if sent:
                words.extend(sent)
    return words


# ---------------------------------------------------------------------------------------
# words -> text
# ---------------------------------------------------------------------------------------

def _cased(word: str, case: int) -> str:
    if case == UPPER:
        return word.upper()
    if case == CAP:
        return word[:1].upper() + word[1:]
    return word


def render(words: list[str], labels: list[int], *, force_sentence_case: bool = True,
           spellings: dict[str, str] | None = None) -> str:
    """Decorate `words` with `labels`. Never adds, removes or changes a word's letters.

    Three rules sit on top of the model's labels, because a model that gets them wrong
    occasionally is worse than a rule that gets them right always: the first word and every
    word after a `.` or `?` are capitalised, `i` (and `i'm`, `i'll`…) is always `I`, and the
    text ends in a full stop if the model left it open. `spellings` maps a word to how this
    user writes it ("shaun" -> "Shaun", "github" -> "GitHub") and beats the model's casing.
    """
    if len(words) != len(labels):
        raise ValueError(f"{len(words)} words but {len(labels)} labels")
    out = []
    start = True
    for k, (w, lab) in enumerate(zip(words, labels)):
        p, c = split_label(lab)
        if force_sentence_case and start and c == LOWER:
            c = CAP
        if w == "i" or w.startswith("i'"):
            c = CAP
        s = (spellings or {}).get(w)
        if s is not None and s.lower() == w:
            # A personal spelling — but sentence case still wins for its first letter.
            s = s[:1].upper() + s[1:] if (force_sentence_case and start) else s
        else:
            s = _cased(w, c)
        if k == len(words) - 1 and force_sentence_case and p in (NONE, COMMA):
            p = PERIOD
        out.append(s + PUNCT[p])
        start = p in (PERIOD, QUESTION)
    return " ".join(out)


def words_of(text: str) -> list[str]:
    """What `render` was given, recovered from what it wrote. The inverse that the
    never-changes-a-word property is tested with."""
    return [w.strip(",.?").lower() for w in text.split()]
