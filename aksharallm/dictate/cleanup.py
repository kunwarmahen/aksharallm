"""Recogniser words -> text you would send: fillers out, spoken commands obeyed, punctuation in.

The order is the design, and every step that removes or changes a word is **listed in the
result** (`Cleaned.steps`) — a cleanup pass you cannot audit is one you cannot trust, and the
one property this whole layer is built around is that nothing appears in your text that you
did not say.

```mermaid
flowchart LR
    W["recogniser words<br/>'um i met shawn new line scratch that'"] --> R["personal replacements<br/>shawn → shaun (learned)"]
    R --> F["drop fillers<br/>um, uh, erm…"]
    F --> C["spoken commands<br/>new line · new paragraph · scratch that"]
    C --> P["punctuation tagger<br/>labels, never words"]
    P --> S["personal spellings<br/>shaun → Shaun"]
    S --> T["text"]
```

**Fillers** are a closed list of sounds that are never words in a dictated sentence (`um`,
`uh`, `erm`…). Not `like`, not `you know`, not `so`: those are words a person sometimes means,
and deleting a meant word is the worse error.

**Spoken commands** are three, matched as whole phrases:

* `new line` / `new paragraph` — a line break; each paragraph is punctuated on its own, so a
  sentence never runs across one.
* `scratch that` — delete back to the start of the current sentence (as the tagger sees it),
  or the previous one if nothing has been said since a full stop. "Meet at five, scratch that,
  six" is two sentences to a tagger, so this needs sentence boundaries — which is why it runs
  the tagger first and again afterwards.

Spoken punctuation ("comma", "full stop") is deliberately **not** a command: "the comma
splice", "a full stop sign" — every one of them is also a word, and the tagger already
punctuates.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import punct
from .personal import Personal

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "hmm", "mm", "mhm"}
BREAKS = {("new", "line"): "\n", ("new", "paragraph"): "\n\n"}
SCRATCH = ("scratch", "that")


@dataclass
class Cleaned:
    text: str
    words: list[str]                       # what was punctuated (after removals)
    steps: list[dict] = field(default_factory=list)


def _labels(words: list[str], punctuator) -> list[int]:
    if punctuator is None:
        from .tagger import rules_only
        return rules_only(words)
    return punctuator.labels(words)


def _scratch(words: list[str], punctuator, steps: list[dict]) -> list[str]:
    """Resolve every `scratch that`, left to right."""
    while True:
        at = next((i for i in range(len(words) - 1) if tuple(words[i : i + 2]) == SCRATCH), None)
        if at is None:
            return words
        before = words[:at]
        cut = 0
        if before:
            labs = _labels(before, punctuator)
            ends = [i for i, lab in enumerate(labs[:-1])
                    if punct.split_label(lab)[0] in (punct.PERIOD, punct.QUESTION)]
            cut = ends[-1] + 1 if ends else 0
        steps.append({"step": "scratch that", "removed": " ".join(before[cut:])})
        words = before[:cut] + words[at + 2 :]


def clean(transcript: str, punctuator=None, personal: Personal | None = None) -> Cleaned:
    """The whole pass. `punctuator` None means rules only (capital first, full stop last)."""
    steps: list[dict] = []
    words = transcript.lower().split()
    if personal is not None:
        words, done = personal.rewrite(words)
        steps += [{"step": "replacement", **d} for d in done]
    kept = [w for w in words if w not in FILLERS]
    if len(kept) != len(words):
        steps.append({"step": "fillers", "removed": " ".join(w for w in words if w in FILLERS)})
    words = kept

    # Split into paragraphs at the break commands, keeping which break ended each.
    paras: list[tuple[list[str], str]] = []
    cur: list[str] = []
    i = 0
    while i < len(words):
        pair = tuple(words[i : i + 2])
        if pair in BREAKS:
            paras.append((cur, BREAKS[pair]))
            steps.append({"step": " ".join(pair)})
            cur = []
            i += 2
        else:
            cur.append(words[i])
            i += 1
    paras.append((cur, ""))

    spellings = personal.spellings() if personal is not None else None
    out, all_words = [], []
    for ws, brk in paras:
        ws = _scratch(ws, punctuator, steps)
        if ws:
            out.append(punct.render(ws, _labels(ws, punctuator), spellings=spellings))
            all_words += ws
        out.append(brk)
    text = "".join(_join(out))
    return Cleaned(text=text.strip(), words=all_words, steps=steps)


def _join(parts: list[str]) -> list[str]:
    """Paragraph texts and breaks -> pieces that concatenate cleanly (no space before a
    break, one space between two paragraphs that had no break between them)."""
    out: list[str] = []
    for p in parts:
        if p in ("\n", "\n\n"):
            while out and out[-1] == " ":
                out.pop()
            out.append(p)
        elif p:
            if out and out[-1] not in ("\n", "\n\n"):
                out.append(" ")
            out.append(p)
    return out
