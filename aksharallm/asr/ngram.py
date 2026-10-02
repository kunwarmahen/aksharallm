"""A word trigram language model, from scratch — the thing that knows "carrots" is a word and "karots" is not.

The recogniser hears well and spells badly: on test-clean its greedy output is 4% wrong by
character and 13% wrong by word, and the word errors are things like "stew" → "stoo" and
"carrots" → "karots". It decodes one character at a time and has no idea which words exist.
A **word language model** is that idea: given the previous two words, how likely is this one?

```mermaid
flowchart LR
    T["LibriSpeech LM text<br/>~800M words of books"] --> C["count every<br/>(u, v, w) triple"]
    C --> K["Kneser-Ney:<br/>discount + continuation counts"]
    K --> A["sorted arrays<br/>searchsorted lookups"]
    A --> P["log P(w | u, v)"]
```

**Why Kneser-Ney, in one example.** "Francisco" is a common word, but almost only after "San".
Counting raw frequency, a model that has never seen "...in Francisco" would back off to a
unigram estimate that says Francisco is common — and be wrong. Kneser-Ney's lower orders count
**how many different words a word follows** (its *continuation* count), not how often it
occurs, so Francisco's unigram estimate is small: it is common in one context only. That one
idea is why KN is still the default n-gram smoothing thirty years on.

The model, interpolated, at each order *n* with an absolute discount *D_n*:

    P(w | u v) = max(c(uvw) − D3, 0) / c(uv·)  +  D3 · N1+(uv·) / c(uv·) · P_kn(w | v)
    P_kn(w | v) = max(N1+(·vw) − D2, 0) / N1+(·v·) + D2 · N1+(v·) / N1+(·v·) · P_kn(w)
    P_kn(w)     = N1+(·w) / N1+(··)

where `N1+(·vw)` is the number of distinct words seen before "v w". The discounts come from
the data (`D = n1 / (n1 + 2·n2)`, Ney's estimate from the count-of-counts).

**Why numpy and not dictionaries.** A hundred million words give tens of millions of distinct
trigrams. As Python dict entries that is gigabytes and minutes; as one `int64` per trigram —
the three word ids packed into one number — it is `np.unique` on an array, which sorts it in
seconds, and a lookup is `np.searchsorted` on a sorted array. Every count above is a
`np.unique` over packed keys.

**`<unk>` and the penalty for it.** Words outside the vocabulary are counted as one token,
`<unk>`, which therefore has the summed probability of every rare word — far more than any one
misspelling deserves. The decoder adds `unk_penalty` to it. Set that too mild and "karots"
survives; too harsh and a real name the LM has never seen cannot be written at all — which is
exactly what the personal dictionary (day-two problem 4) exists to override.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

UNK, BOS, EOS = 0, 1, 2
SPECIALS = ["<unk>", "<s>", "</s>"]
#: 21 bits per word id, three ids packed into one int64. Two million words is ten times the
#: vocabulary anyone uses for LibriSpeech, and it keeps every key positive.
BITS = 21
MAX_VOCAB = (1 << BITS) - 1


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def _lines(path: Path, max_words: int | None, every: int = 1):
    """Normalised lines of a corpus. The LibriSpeech LM text is already upper-case words, one
    sentence per line; lower-casing matches `asr/vocab.py`.

    **That file is sorted alphabetically** — line 10 is "A A A A A AH THE CRY…", line 20M is
    "JULIA PERSISTED" — so `max_words` (a prefix) is a biased sample: every sentence starting
    with "a". `every` keeps one line in N across the whole file, which is the sample to use.
    """
    n = 0
    with _open_text(path) as f:
        for i, line in enumerate(f):
            if every > 1 and i % every:
                continue
            words = line.lower().split()
            if not words:
                continue
            yield words
            n += len(words)
            if max_words and n >= max_words:
                return


def _discount(counts: np.ndarray) -> float:
    """Ney's estimate, D = n1 / (n1 + 2·n2), from how many n-grams were seen once and twice."""
    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    if n1 == 0 or n2 == 0:
        return 0.5
    return min(max(n1 / (n1 + 2 * n2), 0.05), 0.95)


class TrigramLM:
    """Interpolated Kneser-Ney over words. Build with `build`, persist with `save` / `load`."""

    def __init__(self, words: list[str], arrays: dict, meta: dict):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.V = len(words)
        self.meta = meta
        a = arrays
        self.tri_keys, self.tri_counts = a["tri_keys"], a["tri_counts"]
        self.ctx2_keys, self.ctx2_total, self.ctx2_types = a["ctx2_keys"], a["ctx2_total"], a["ctx2_types"]
        self.bi_keys, self.bi_cont = a["bi_keys"], a["bi_cont"]
        self.ctx1_cont_total, self.ctx1_types = a["ctx1_cont_total"], a["ctx1_types"]
        self.uni_cont = a["uni_cont"]
        self.D2, self.D3 = float(meta["D2"]), float(meta["D3"])
        self.uni_total = float(self.uni_cont.sum())
        self._cache: dict = {}

    # ---- building ------------------------------------------------------------------------

    @classmethod
    def build(cls, corpus: str | Path, *, vocab_size: int = 200_000, max_words: int | None = None,
              every: int = 1, chunk_words: int = 20_000_000, progress=print) -> TrigramLM:
        """Two passes over the text: words first (to fix the vocabulary), then trigrams.

        Trigrams are counted a chunk at a time (`np.unique` on that chunk's packed keys) and
        the per-chunk tables merged, so memory follows the number of *distinct* trigrams rather
        than the size of the corpus.
        """
        corpus = Path(corpus)
        t0 = time.time()
        freq: Counter = Counter()
        n_words = n_lines = 0
        for words in _lines(corpus, max_words, every):
            freq.update(words)
            n_words += len(words)
            n_lines += 1
            if progress and n_lines % 2_000_000 == 0:
                progress(f"  pass 1: {n_words / 1e6:.0f}M words, {len(freq):,} types")
        if vocab_size > MAX_VOCAB - len(SPECIALS):
            raise ValueError(f"vocab_size {vocab_size} is past the {MAX_VOCAB} a packed key holds")
        words = SPECIALS + [w for w, _ in freq.most_common(vocab_size)]
        index = {w: i for i, w in enumerate(words)}
        covered = sum(freq[w] for w in words[len(SPECIALS):])
        if progress:
            progress(f"  vocabulary {len(words) - 3:,} of {len(freq):,} types covers "
                     f"{covered / max(n_words, 1):.2%} of {n_words / 1e6:.0f}M words")
        del freq

        # pass 2: trigram counts, chunked
        tables: list[tuple[np.ndarray, np.ndarray]] = []
        buf: list[int] = []

        def flush():
            if not buf:
                return
            ids = np.asarray(buf, dtype=np.int64)
            buf.clear()
            # Each sentence was written as <s> <s> w1 .. wn </s>; a trigram ending at position i
            # is (i-2, i-1, i), and none may start inside the previous sentence: a triple whose
            # LAST id is <s> straddles a boundary, so it is dropped.
            u, v, w = ids[:-2], ids[1:-1], ids[2:]
            keep = w != BOS
            key = (u[keep] << (2 * BITS)) | (v[keep] << BITS) | w[keep]
            k, c = np.unique(key, return_counts=True)
            tables.append((k, c.astype(np.int64)))

        n = 0
        for ws in _lines(corpus, max_words, every):
            buf += [BOS, BOS] + [index.get(x, UNK) for x in ws] + [EOS]
            n += len(ws)
            if n >= chunk_words:
                flush()
                n = 0
                if progress:
                    progress(f"  pass 2: {len(tables)} chunks counted")
        flush()
        keys = np.concatenate([t[0] for t in tables])
        cnts = np.concatenate([t[1] for t in tables])
        del tables
        tri_keys, inv = np.unique(keys, return_inverse=True)
        tri_counts = np.bincount(inv, weights=cnts).astype(np.int64)
        del keys, cnts, inv
        if progress:
            progress(f"  {len(tri_keys):,} distinct trigrams")
        arrays, meta = cls._statistics(tri_keys, tri_counts, len(words))
        meta.update({"corpus": str(corpus), "every": every, "words": n_words, "sentences": n_lines,
                     "vocab": len(words) - 3, "oov_rate": 1 - covered / max(n_words, 1),
                     "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "seconds": round(time.time() - t0, 1)})
        return cls(words, arrays, meta)

    @staticmethod
    def _statistics(tri_keys: np.ndarray, tri_counts: np.ndarray, V: int):
        """Every count Kneser-Ney needs, as sorted arrays. See the module docstring's formula."""
        mask = (1 << BITS) - 1
        u = tri_keys >> (2 * BITS)
        v = (tri_keys >> BITS) & mask
        w = tri_keys & mask
        uv = (u << BITS) | v
        # c(uv·) and N1+(uv·): trigram keys are sorted, so uv runs are contiguous.
        ctx2_keys, start = np.unique(uv, return_index=True)
        ctx2_total = np.add.reduceat(tri_counts, start)
        ctx2_types = np.diff(np.append(start, len(uv)))
        # N1+(·vw): distinct u before each (v, w)
        vw = (v << BITS) | w
        bi_keys, bi_cont = np.unique(vw, return_counts=True)
        bv = bi_keys >> BITS
        bw = bi_keys & mask
        ctx1_cont_total = np.bincount(bv, weights=bi_cont, minlength=V)   # N1+(·v·)
        ctx1_types = np.bincount(bv, minlength=V)                         # N1+(v·)
        uni_cont = np.bincount(bw, minlength=V).astype(np.float64)         # N1+(·w)
        uni_cont[BOS] = 0.0  # never predicted
        meta = {"D3": _discount(tri_counts), "D2": _discount(bi_cont)}
        arrays = {"tri_keys": tri_keys, "tri_counts": tri_counts, "ctx2_keys": ctx2_keys,
                  "ctx2_total": ctx2_total, "ctx2_types": ctx2_types, "bi_keys": bi_keys,
                  "bi_cont": bi_cont, "ctx1_cont_total": ctx1_cont_total,
                  "ctx1_types": ctx1_types, "uni_cont": uni_cont}
        return arrays, meta

    # ---- persistence ---------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in (
            "tri_keys", "tri_counts", "ctx2_keys", "ctx2_total", "ctx2_types", "bi_keys",
            "bi_cont", "ctx1_cont_total", "ctx1_types", "uni_cont")}
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({"meta": self.meta, "words": self.words}))
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrigramLM:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        side = json.loads(path.with_suffix(".json").read_text())
        return cls(side["words"], arrays, side["meta"])

    # ---- scoring -------------------------------------------------------------------------

    def id(self, word: str) -> int:
        return self.index.get(word, UNK)

    @staticmethod
    def _find(keys: np.ndarray, key: int) -> int:
        i = int(np.searchsorted(keys, key))
        return i if i < len(keys) and keys[i] == key else -1

    def logprob(self, u: int, v: int, w: int) -> float:
        """Natural-log P(w | u, v), interpolated Kneser-Ney. Cached: a beam asks the same
        question many times."""
        k = (u, v, w)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        p1 = self.uni_cont[w] / self.uni_total if self.uni_total else 0.0
        # order 2
        cont_total = self.ctx1_cont_total[v]
        if cont_total > 0:
            i = self._find(self.bi_keys, (v << BITS) | w)
            cvw = float(self.bi_cont[i]) if i >= 0 else 0.0
            p2 = (max(cvw - self.D2, 0.0) + self.D2 * self.ctx1_types[v] * p1) / cont_total
        else:
            p2 = p1
        # order 3
        j = self._find(self.ctx2_keys, (u << BITS) | v)
        if j >= 0:
            total = float(self.ctx2_total[j])
            t = self._find(self.tri_keys, (u << (2 * BITS)) | (v << BITS) | w)
            c = float(self.tri_counts[t]) if t >= 0 else 0.0
            p3 = (max(c - self.D3, 0.0) + self.D3 * self.ctx2_types[j] * p2) / total
        else:
            p3 = p2
        out = math.log(p3) if p3 > 0 else -50.0
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[k] = out
        return out

    def sentence_logprob(self, words: list[str]) -> tuple[float, int]:
        """Sum of log P over a sentence including `</s>`, and how many predictions that was."""
        ids = [self.id(x) for x in words] + [EOS]
        u, v, total = BOS, BOS, 0.0
        for w in ids:
            total += self.logprob(u, v, w)
            u, v = v, w
        return total, len(ids)

    def perplexity(self, sentences: list[str]) -> dict:
        """Per-word perplexity on normalised sentences, and the share of words that are OOV."""
        lp = n = oov = words = 0
        for s in sentences:
            ws = s.split()
            a, b = self.sentence_logprob(ws)
            lp += a
            n += b
            oov += sum(1 for x in ws if x not in self.index)
            words += len(ws)
        return {"perplexity": math.exp(-lp / max(n, 1)), "oov_rate": oov / max(words, 1),
                "words": words}

    def describe(self) -> str:
        m = self.meta
        return (f"trigram KN, vocab {m['vocab']:,}, {len(self.tri_keys):,} trigrams from "
                f"{m['words'] / 1e6:.0f}M words (OOV {m['oov_rate']:.2%}), D2 {m['D2']:.2f} D3 {m['D3']:.2f}")


def verbatim_overlap(corpus: str | Path, sentences: list[str], max_words: int | None = None) -> dict:
    """How many `sentences` occur, word for word, as a line of the LM corpus.

    The LibriSpeech LM text was built by *excluding the books the dev and test audio comes
    from*. This checks that claim instead of trusting it: a test sentence in the LM's training
    text would let the LM, not the recogniser, supply the answer.
    """
    want = {" ".join(s.split()) for s in sentences if len(s.split()) >= 5}
    hits = set()
    for ws in _lines(Path(corpus), max_words):
        if len(ws) >= 5:
            line = " ".join(ws)
            if line in want:
                hits.add(line)
    return {"checked": len(want), "found": len(hits), "rate": len(hits) / max(len(want), 1),
            "examples": sorted(hits)[:5]}
