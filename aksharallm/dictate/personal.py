"""What this person has taught the dictation — day-two problem 5, "it never learns".

A dictation app that writes "Sean" every time you say "Shaun", however often you fix it, is
the one that gets uninstalled. This file turns a **correction** — the text the app produced
and the text you changed it to — into three kinds of memory, each with its own threshold
because each can do a different amount of damage when it is wrong:

| memory | example | learned after | what it does | worst case if wrong |
|---|---|---|---|---|
| **dictionary word** | `shaun` | 1 correction | beam search bias (`asr/decode.py`) | a rare word is a little likelier |
| **spelling** | `github` → `GitHub` | 1 correction | casing at render time | one word cased oddly |
| **replacement** | heard `shawn` → `shaun` | 2 identical corrections | rewrites the words before cleanup | a real word replaced everywhere |

The replacement is the strong one and the dangerous one, which is why it needs to be seen
twice and why it is **never learned between two words the language model already knows**:
"their" → "there" corrected twice is a grammar fix in two sentences, not a rule to apply to
every "their" from now on.

```mermaid
flowchart LR
    S["shown: 'I met sean at the get hub office.'"] --> A["align words<br/>(difflib)"]
    C["corrected: 'I met Shaun at the GitHub office.'"] --> A
    A -->|sean → shaun| D["dictionary + replacement count"]
    A -->|get hub → github| D
    A -->|github, Shaun| P["spellings"]
    D --> B["next decode:<br/>beam bias + rewrites"]
    P --> R["next render"]
```

**Only small edits are learned.** A changed span longer than `MAX_SPAN` words on either side
is a rewrite — the person changed their mind, not the app's mistake — and is logged as such
and learned from not at all. Learning from rewrites would teach the dictionary whatever the
person decided to say instead.

Everything is plain files in one directory (`logs/dictate/` by default):
`personal.json` (the three memories, with counts and dates) and `corrections.jsonl` (every
correction ever made, with what was learned from it — the audit trail, and what the day-two
measurement replays).

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

#: A changed span longer than this, on either side, is a rewrite and is not learned from.
MAX_SPAN = 3
#: Identical corrections needed before a replacement rule is applied.
REPLACE_AFTER = 2
_WORD = re.compile(r"^[a-z']{1,40}$")
_TOKEN = re.compile(r"[A-Za-z']+")


def tokens(text: str) -> list[str]:
    """Written text -> its words with their case, punctuation dropped."""
    return _TOKEN.findall(text.replace("’", "'"))


class Personal:
    def __init__(self, root: str | Path = "logs/dictate", known: Callable[[str], bool] | None = None):
        self.dir = Path(root)
        self.path = self.dir / "personal.json"
        self.log = self.dir / "corrections.jsonl"
        #: Does the word LM know this word? Replacements between two known words are refused.
        self.known = known or (lambda w: False)
        self.words: dict[str, dict] = {}
        self.replacements: dict[str, dict] = {}
        self.load()

    # ---- storage --------------------------------------------------------------------------

    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            d = {}
        self.words = d.get("words", {})
        self.replacements = d.get("replacements", {})

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"words": self.words, "replacements": self.replacements},
                                  indent=1, sort_keys=True))
        tmp.replace(self.path)

    # ---- what the decoder and renderer read ------------------------------------------------

    def dictionary(self) -> set[str]:
        return {w for w in self.words if _WORD.match(w)}

    def spellings(self) -> dict[str, str]:
        return {w: v["spelling"] for w, v in self.words.items() if v.get("spelling")}

    def active_replacements(self) -> dict[tuple[str, ...], list[str]]:
        return {tuple(k.split()): v["to"].split() for k, v in self.replacements.items()
                if v.get("count", 0) >= REPLACE_AFTER}

    def rewrite(self, words: list[str]) -> tuple[list[str], list[dict]]:
        """Apply active replacements, longest match first. Returns the words and what changed."""
        rules = self.active_replacements()
        if not rules:
            return words, []
        longest = max(len(k) for k in rules)
        out, done, i = [], [], 0
        while i < len(words):
            for n in range(min(longest, len(words) - i), 0, -1):
                key = tuple(words[i : i + n])
                if key in rules:
                    out.extend(rules[key])
                    done.append({"heard": " ".join(key), "wrote": " ".join(rules[key])})
                    i += n
                    break
            else:
                out.append(words[i])
                i += 1
        return out, done

    # ---- editing by hand ---------------------------------------------------------------

    def add_word(self, typed: str, source: str = "manual") -> str:
        w = typed.strip().replace("’", "'")
        low = w.lower()
        if not _WORD.match(low):
            raise ValueError(f"{typed!r}: a dictionary word may only use letters and '")
        e = self.words.setdefault(low, {"added": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "source": source, "count": 0})
        e["count"] = e.get("count", 0) + 1
        if w != low:
            e["spelling"] = w
        return low

    def remove(self, word: str) -> bool:
        low = word.strip().lower()
        hit = self.words.pop(low, None) is not None
        for k in [k for k, v in self.replacements.items() if k == low or v["to"] == low]:
            del self.replacements[k]
            hit = True
        return hit

    # ---- learning from a correction -----------------------------------------------------

    def learn(self, shown: str, corrected: str, heard: str | None = None, save: bool = True) -> dict:
        """Compare what was shown with what the person changed it to, and remember the small
        differences. `heard` is the recogniser's raw words, kept in the log for replay."""
        a, b = tokens(shown), tokens(corrected)
        al, bl = [w.lower() for w in a], [w.lower() for w in b]
        learned = {"words": [], "spellings": [], "replacements": [], "ignored": []}
        sm = difflib.SequenceMatcher(a=al, b=bl, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                # Same letters, different case: a spelling ("github" -> "GitHub"). Only when
                # the person typed a capital somewhere other than a sentence start.
                for k in range(i2 - i1):
                    x, y = a[i1 + k], b[j1 + k]
                    if x != y and y != y.lower() and not _sentence_start(b, j1 + k, corrected):
                        self._spell(y)
                        learned["spellings"].append(y)
                continue
            if op == "delete":
                continue    # the person removed words: a choice, not a recognition error
            old, new = al[i1:i2], bl[j1:j2]
            if len(old) > MAX_SPAN or len(new) > MAX_SPAN:
                learned["ignored"].append({"from": " ".join(old), "to": " ".join(new),
                                           "why": f"longer than {MAX_SPAN} words: a rewrite"})
                continue
            if (op == "replace" and all(self.known(w) for w in old)
                    and all(self.known(w) for w in new)):
                # "their" -> "there": a grammar fix in one sentence. Neither a rule to apply
                # everywhere nor a word to bias the decoder towards.
                learned["ignored"].append({"from": " ".join(old), "to": " ".join(new),
                                           "why": "both are ordinary words: an edit, not a rule"})
                continue
            for k in range(j1, j2):
                typed = b[k]
                if not _WORD.match(typed.lower()):
                    continue
                low = self.add_word(typed if not _sentence_start(b, k, corrected) else typed.lower(),
                                    source="correction")
                learned["words"].append(low)
                if self.words[low].get("spelling"):
                    learned["spellings"].append(self.words[low]["spelling"])
            if op == "replace":
                key, to = " ".join(old), " ".join(new)
                r = self.replacements.setdefault(key, {"to": to, "count": 0})
                if r["to"] != to:
                    # Corrected to something else this time: the old evidence no longer counts.
                    r.update(to=to, count=0)
                r["count"] += 1
                r["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
                learned["replacements"].append({"from": key, "to": to, "count": r["count"],
                                                "active": r["count"] >= REPLACE_AFTER})
        if save:
            self.save()
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.log, "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "heard": heard,
                                    "shown": shown, "corrected": corrected,
                                    "learned": learned}) + "\n")
        return learned

    def _spell(self, typed: str) -> None:
        low = typed.lower()
        if _WORD.match(low):
            e = self.words.setdefault(low, {"added": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "source": "correction", "count": 0})
            e["spelling"] = typed

    def corrections(self, limit: int = 50) -> list[dict]:
        try:
            lines = self.log.read_text().splitlines()[-limit:]
        except OSError:
            return []
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def summary(self) -> dict:
        return {"words": len(self.words), "spellings": len(self.spellings()),
                "replacements": len(self.replacements),
                "active_replacements": len(self.active_replacements())}


def _sentence_start(words: list[str], k: int, text: str) -> bool:
    """Is word k the first of a sentence in `text`? Its capital then says nothing about it."""
    if k == 0:
        return True
    # Find the k-th word's position and look at what precedes it.
    pos = 0
    for i, m in enumerate(_TOKEN.finditer(text.replace("’", "'"))):
        if i == k:
            pos = m.start()
            break
    before = text[:pos].rstrip()
    return before.endswith((".", "?", "!", "\n")) or not before
