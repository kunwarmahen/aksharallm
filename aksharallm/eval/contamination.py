"""Did the benchmark leak into the training data?

A benchmark number nobody has checked for leakage is a rumour. This module checks.

The method, and why it is in token space
-----------------------------------------
The published approach (GPT-3, Llama, and everyone since) is **n-gram overlap**: an eval
item is "dirty" if any contiguous run of *n* tokens from it also appears anywhere in the
training corpus. `n = 13` is the usual choice and the default here — shorter over-reports,
because ordinary English produces plenty of shared 8-grams, and longer under-reports,
because a paraphrase or a reformatted whitespace run breaks the streak.

Everything happens on **token ids**, not on text, and that is the one design decision worth
defending. Our training corpus exists only as `.bin` files of `uint16` ids, and decoding ten
billion of them back to strings to do substring matching would take hours and answer a
slightly different question. Both sides go through *our own* tokenizer, so a match is a
match — no normalisation to argue about, no case folding, no whitespace policy.

How it runs in a sensible amount of time
-----------------------------------------
The corpus is ~10 billion tokens and the eval suites are a few hundred thousand, so the
asymmetry decides the algorithm: build the small side into a lookup, stream the big side
past it once.

```mermaid
flowchart LR
    E["eval suites<br/>~10^5 tokens"] --> H["rolling 13-gram hashes<br/>sorted uint64 array"]
    T["train bins<br/>~10^10 tokens"] -->|"chunks of 32M"| R["rolling hashes"]
    R --> S["np.searchsorted"]
    H --> S
    S --> D["which items were hit"]
```

Both sides use the same rolling polynomial hash, computed with numpy over whole chunks
rather than per n-gram — a Python loop over ten billion positions is not a program that
finishes. Membership is `searchsorted` against a sorted array, which is a binary search per
position and vectorises.

**On hash collisions.** Two different n-grams can share a 64-bit hash. With ~10^6 probe
hashes and ~10^10 lookups the expected number of false positives is
`10^10 × 10^6 / 2^64 ≈ 5×10^-4` — i.e. this will report a spurious hit roughly once every
two thousand full scans. That is small enough to ignore and too large to leave unstated,
which is why `--verify` exists: it re-checks every reported hit against the actual token
sequence, and costs one seek per hit.

What the answer is actually for
--------------------------------
Not a pass/fail. The useful output is the **clean score**: re-read a benchmark result,
drop the contaminated items, and see whether the number moves. A suite that is 8% dirty and
scores the same either way is fine. One that is 3% dirty and gains four points on those
three percent is telling you something.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The published default. See the module docstring for why it is not 8 and not 25.
DEFAULT_N = 13

#: Tokens per chunk when streaming a training bin. 32M uint16 is 64 MB in, and about
#: 256 MB of uint64 hashes out — comfortable, and big enough that per-chunk overhead
#: disappears against the work.
CHUNK = 32_000_000

#: Rolling-hash constants. A large odd multiplier over the natural uint64 ring; the ring is
#: the point, because numpy's uint64 overflow is exactly the modulus we want and costs
#: nothing.
_MULT = np.uint64(1_000_003)


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Rolling hash of every length-`n` window of `tokens`, as uint64.

    Computed as a Horner polynomial over the whole array at once. The loop below runs `n`
    times, not `len(tokens)` times — that is the entire trick, and it is why this can cross
    ten billion tokens instead of ten million.
    """
    tokens = np.asarray(tokens, dtype=np.uint64)
    if tokens.size < n:
        return np.empty(0, dtype=np.uint64)
    out = np.zeros(tokens.size - n + 1, dtype=np.uint64)
    for i in range(n):
        # `out * MULT + tokens[i:...]`, wrapping. Done in place to avoid n temporaries the
        # size of the corpus chunk.
        out *= _MULT
        out += tokens[i: tokens.size - n + 1 + i]
    return out


@dataclass
class Probe:
    """The small side, built once: every n-gram of every eval item, ready to look up."""

    n: int
    #: Sorted unique hashes. `searchsorted` against this is the membership test.
    hashes: np.ndarray
    #: `owners[i]` lists the item keys whose text produced `hashes[i]`.
    owners: list[list[str]]
    #: Item key -> how many n-grams it contributed. Zero means "too short to check".
    sizes: dict[str, int] = field(default_factory=dict)
    #: Item key -> the token ids it was built from, for `--verify`.
    tokens: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.hashes.size)


def _key(suite: str, item_id: str, part: str) -> str:
    return f"{suite}\t{item_id}\t{part}"


@dataclass
class Text:
    """One thing to check, and where its answer starts."""

    key: str
    part: str
    text: str
    #: For `answered`, the question this answer was appended to. Only n-grams reaching past
    #: it are kept — see `build_probe`. None for anything with no question to speak of.
    context: str | None = None


def item_texts(suite: str, items: list) -> list[Text]:
    """Everything in a suite worth checking, as `Text`.

    **The question and the answer are checked separately, and that distinction is the whole
    value of the report.** A benchmark's questions turning up in a web crawl is common and
    mostly harmless — they are public text. The *answer* turning up next to the question is
    what makes a score meaningless. Collapsing the two into one "contaminated" flag throws
    away the only part a reader needs.
    """
    out: list[Text] = []
    for item in items:
        iid = getattr(item, "id", None) or str(len(out))
        ctx = getattr(item, "context", None) or getattr(item, "prompt", None) or ""
        if ctx:
            out.append(Text(_key(suite, iid, "question"), "question", ctx))
        choices = getattr(item, "choices", None)
        gold = getattr(item, "gold", None)
        answer = None
        if choices is not None and isinstance(gold, int) and 0 <= gold < len(choices):
            answer = choices[gold]
        elif isinstance(gold, str) and gold:
            answer = gold
        if answer:
            out.append(Text(_key(suite, iid, "answered"), "answered",
                            f"{ctx} {answer}".strip(), context=ctx))
        tests = getattr(item, "tests", None)
        if tests:
            out.append(Text(_key(suite, iid, "answered"), "answered", tests))
    return out


def build_probe(texts: list[Text], tok, n: int = DEFAULT_N,
                keep_tokens: bool = False) -> Probe:
    """Hash the n-grams of every text.

    For an `answered` text the n-grams are **trimmed to those that reach into the answer**,
    and that trim is not a refinement — without it the check is worthless. "Question plus
    answer" contains every n-gram of the question, so a corpus holding only the public
    question would light up `answered` too and the two columns would be the same column.
    Keeping the last `n-1` tokens of the question and everything after leaves exactly the
    windows that span the join or sit inside the answer.

    The boundary is `len(encode(question))`, which BPE can move by a token when a merge
    crosses the join — the same trap the scoring path documents. One token of slop at the
    edge of a 13-token window does not change whether an item is dirty.
    """
    by_hash: dict[int, list[str]] = {}
    sizes: dict[str, int] = {}
    tokens: dict[str, np.ndarray] = {}
    for t in texts:
        ids = np.asarray(tok.encode(t.text), dtype=np.uint64)
        if t.part == "answered" and t.context:
            start = max(0, len(tok.encode(t.context)) - (n - 1))
            ids = ids[start:]
        hs = ngram_hashes(ids, n)
        sizes[t.key] = int(hs.size)
        if keep_tokens:
            tokens[t.key] = ids
        for h in np.unique(hs).tolist():
            by_hash.setdefault(h, []).append(t.key)

    if not by_hash:
        return Probe(n, np.empty(0, dtype=np.uint64), [], sizes, tokens)
    order = np.array(sorted(by_hash), dtype=np.uint64)
    owners = [by_hash[int(h)] for h in order]
    return Probe(n, order, owners, sizes, tokens)


def scan_bin(path: str | Path, probe: Probe, chunk: int = CHUNK,
             max_tokens: int | None = None, progress=None,
             where: dict[str, tuple[str, int]] | None = None) -> dict[str, int]:
    """Stream one `.bin` past the probe. Returns `{item key: times hit}`.

    Chunks **overlap by n-1 tokens**, because an n-gram that straddles a chunk boundary is
    still an n-gram in the corpus. Forgetting that loses one window per chunk — invisible,
    and it would make the report quietly optimistic, which is the wrong direction for a
    contamination check to be wrong in.

    `where`, if given, is filled with `{key: (path, position)}` for the **first** place each
    key was hit. That is what makes verification cheap: a hit remembers where it came from,
    so checking it is a seek rather than another pass over ten billion tokens.
    """
    hits: dict[str, int] = {}
    if len(probe) == 0:
        return hits
    data = np.memmap(Path(path), dtype=np.uint16, mode="r")
    total = int(data.size if max_tokens is None else min(data.size, max_tokens))
    step = max(chunk, probe.n)
    pos = 0
    while pos < total:
        end = min(pos + step, total)
        window = np.asarray(data[pos: min(end + probe.n - 1, total)])
        hs = ngram_hashes(window, probe.n)
        if hs.size:
            idx = np.searchsorted(probe.hashes, hs)
            idx = np.clip(idx, 0, probe.hashes.size - 1)
            found = np.nonzero(probe.hashes[idx] == hs)[0]
            for f in found.tolist():
                for key in probe.owners[int(idx[f])]:
                    hits[key] = hits.get(key, 0) + 1
                    if where is not None and key not in where:
                        where[key] = (str(path), pos + f)
        pos = end
        if progress:
            progress(pos, total, Path(path).name)
    return hits


def verify(hits: dict[str, int], probe: Probe, where: dict[str, tuple[str, int]],
           n: int) -> dict[str, int]:
    """Re-check each reported hit against the real token stream, dropping collisions.

    Needed because a 64-bit hash match is *probably* an n-gram match. See the arithmetic in
    the module docstring: about one spurious hit per two thousand full scans, which is rare
    enough to ignore in a summary and not rare enough to leave unchecked in a finding
    somebody is going to quote.

    Costs **one seek per hit**, because `scan_bin` recorded where each one came from. The
    first version re-scanned the whole corpus once per hit instead, which is fine on a clean
    corpus (nothing to check) and turns a half-hour job into an overnight one the moment
    anything is found — i.e. it was slowest exactly when it mattered. A verification step
    slower than the scan it verifies is a sign the design is wrong.
    """
    if not hits:
        return hits
    confirmed: dict[str, int] = {}
    cache: dict[str, np.memmap] = {}
    for key, count in hits.items():
        spot, ids = where.get(key), probe.tokens.get(key)
        if spot is None or ids is None:
            confirmed[key] = count      # nothing recorded to check against; keep the hit
            continue
        path, pos = spot
        data = cache.setdefault(path, np.memmap(Path(path), dtype=np.uint16, mode="r"))
        found = np.asarray(data[pos: pos + n]).astype(np.uint64)
        if found.size == n and _contains(ids.astype(np.uint64), found):
            confirmed[key] = count
    return confirmed


def _contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    """Is `needle` a contiguous run inside `haystack`? Small arrays; a plain scan is fine."""
    n = needle.size
    if haystack.size < n:
        return False
    for i in range(haystack.size - n + 1):
        if np.array_equal(haystack[i: i + n], needle):
            return True
    return False


def summarise(hits: dict[str, int], probe: Probe, n: int) -> dict:
    """Per suite and per part: how many items were hit, and which."""
    suites: dict[str, dict] = {}
    for key, size in probe.sizes.items():
        suite, item_id, part = key.split("\t")
        s = suites.setdefault(suite, {"suite": suite, "parts": {}})
        p = s["parts"].setdefault(part, {"n": 0, "checkable": 0, "dirty": 0, "items": []})
        p["n"] += 1
        if size == 0:
            continue                       # shorter than one n-gram; nothing to check
        p["checkable"] += 1
        if hits.get(key):
            p["dirty"] += 1
            p["items"].append(item_id)

    out = []
    for suite, s in sorted(suites.items()):
        for part, p in s["parts"].items():
            p["rate"] = (p["dirty"] / p["checkable"]) if p["checkable"] else None
            p["too_short"] = p["n"] - p["checkable"]
        out.append(s)
    # `dirty_ids` drives the re-score, so it is **answer leaks only**. Dropping every item
    # whose *question* appears in a web crawl would discard most of a public benchmark and
    # report a "clean" score computed on whatever happened to be left — the wrong number,
    # arrived at confidently. `question_ids` is kept separately for the report.
    return {"n": n, "suites": out,
            "dirty_ids": sorted({k.split("\t")[1] for k in hits
                                 if k.split("\t")[2] == "answered"}),
            "question_ids": sorted({k.split("\t")[1] for k in hits
                                    if k.split("\t")[2] == "question"})}


def coverage(total_tokens: int, max_tokens: int | None, items_per_suite: int | None,
             texts: int, verified: bool) -> dict:
    """How much of the benchmark and how much of the corpus this report actually looked at.

    Without it a partial scan is **indistinguishable from a complete one**: identical
    fields, identical shape, and a smaller dirty count that reads as good news. Both ways of
    going faster under-count, so both are recorded even when unset —

    - `max_tokens` shrinks the *scan*: less corpus read, so leaks in the unread part are
      invisible. Degrades evenly across every benchmark item.
    - `items_per_suite` shrinks the *probe*: fewer benchmark items checked at all. Worse
      than it sounds, because the loader takes the **first** N rows rather than a sample, so
      the unchecked items are not a random remainder.

    `partial` is the flag every renderer should key on. A contamination number is only a
    finding when it is `False`.
    """
    scanned = int(min(total_tokens, max_tokens) if max_tokens else total_tokens)
    return {
        "scanned_tokens": scanned,
        "total_tokens": int(total_tokens),
        "items_per_suite": items_per_suite,
        "texts": int(texts),
        "verified": bool(verified),
        "partial": scanned < int(total_tokens) or items_per_suite is not None,
    }


def clean_score(result: dict, dirty_ids: set[str]) -> dict | None:
    """A benchmark result re-scored with the contaminated items removed.

    This is the number the whole module exists to produce, and it is only computable when
    the run recorded its **per-item** outcomes. `reported` vs `clean` moving apart is the
    finding; them agreeing is the reassurance.
    """
    items = result.get("items")
    if not items:
        return None
    kept = [i for i in items if str(i.get("id")) not in dirty_ids]
    dropped = len(items) - len(kept)
    if not kept:
        return {"reported": result.get("score"), "clean": None, "dropped": dropped,
                "kept": 0, "note": "every item was contaminated"}
    scored = [float(bool(i.get("correct"))) for i in kept]
    return {
        "reported": result.get("score"),
        "clean": sum(scored) / len(scored),
        "dropped": dropped,
        "kept": len(kept),
    }


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
