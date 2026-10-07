"""Mix several tokenized SFT sets into one, and remove anything that overlaps the benchmark.

`prepare_sft` turns ONE dataset into packed `(N, seq_len)` token and mask blocks. The Python
specialist's code SFT wants several at once — two Python instruction sets plus a share of
the chat data the general SFT used, so the model still converses — and it wants them
**decontaminated against HumanEval**, because HumanEval is the number this whole phase is
judged on and instruction sets scraped from the web do copy benchmark problems.

```mermaid
flowchart LR
    A["sft-parts/self-oss-instruct<br/>(blocks)"] -->|"fraction"| M["sample, concatenate"]
    B["sft-parts/magicoder-python"] -->|"fraction"| M
    C["data/sft (SmolTalk)"] -->|"fraction"| M
    M --> D{"any 13-gram shared with a<br/>HumanEval prompt, solution or test?"}
    D -->|"yes"| X["dropped, and the task id recorded"]
    D -->|"no"| O["data/sft-code/<br/>train/val tokens + masks + manifest.json"]
```

Two choices worth stating:

* **Mixing happens at the block level.** Every block is already a packed, self-contained
  1,024-token window with its own loss mask, so taking a fraction of one set's blocks is
  exactly "a fraction of that set", and no re-tokenizing is needed.
* **Decontamination drops whole blocks**, not single examples inside them. A block holds a
  few packed conversations, so a hit costs its neighbours too — a handful of blocks out of
  tens of thousands, against the alternative of re-packing. The hits are listed by
  HumanEval task id in the manifest, so a suspicious count can be looked at rather than
  trusted.

The overlap test is the same 13-gram, token-space test `eval/contamination.py` runs against
the pretraining corpus, and it reuses that module's probe: for a solution, only n-grams that
reach past the prompt count, or every prompt hit would be counted twice. One addition, found
on the first real run: only n-grams with **content** count (`informative`). This tokenizer
splits numbers into digit groups, so 13 tokens can be `1, 2, 3, 4, 5`, and counting every
window flagged a fifth of a Python dataset through list literals in HumanEval's tests.

Read with: docs/08-scaling.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/06-posttraining.md for what SFT data is for.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..eval.contamination import Probe, Text, build_probe, ngram_hashes

#: n-gram length for the overlap test -- the standard, and the one contamination.py uses.
N = 13


def parse_part(spec: str) -> tuple[Path, float]:
    """`dir:fraction` -> (dir, fraction). A bare dir means all of it."""
    path, _, frac = spec.rpartition(":")
    if not path:
        return Path(spec), 1.0
    try:
        f = float(frac)
    except ValueError:
        return Path(spec), 1.0
    if not 0 < f <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {f} in '{spec}'")
    return Path(path), f


def humaneval_texts(rows: list[dict]) -> list[Text]:
    """What counts as HumanEval: each task's prompt, its prompt + canonical solution (only
    n-grams reaching into the solution, via `context`), and its tests."""
    out = []
    for r in rows:
        tid = str(r.get("task_id"))
        prompt = str(r.get("prompt") or "")
        if prompt:
            out.append(Text(f"{tid}\tprompt", "question", prompt))
        sol = str(r.get("canonical_solution") or "")
        if sol:
            out.append(Text(f"{tid}\tsolution", "answered", prompt + sol, context=prompt))
        test = str(r.get("test") or "")
        if test:
            out.append(Text(f"{tid}\ttest", "question", test))
    return out


#: An n-gram must carry at least this many distinct words (3+ letters) to count. Measured on
#: the first real mix: the blend tokenizer splits numbers digit-group by digit-group, so 13
#: tokens can be just `1, 2, 3, 4, 5` -- and with every window counted, 19% of a Python
#: instruction set and 5% of SmolTalk "matched" HumanEval through counting lists and matrix
#: literals in its tests. Contamination is shared *content*, not shared punctuation.
MIN_WORDS = 3

_WORD = re.compile(r"[A-Za-z_]{3,}")


def informative(text: str, min_words: int = MIN_WORDS) -> bool:
    """Does this window say something, or is it digits, brackets and whitespace?"""
    return len(set(_WORD.findall(text))) >= min_words


def content_probe(texts: list[Text], tok, n: int = N, min_words: int = MIN_WORDS) -> Probe:
    """`build_probe`, keeping only the n-grams whose text passes `informative`."""
    full = build_probe(texts, tok, n=n, keep_tokens=True)
    keep: dict[int, list[str]] = {}
    for key, ids in full.tokens.items():
        hs = ngram_hashes(ids, n)
        for i, h in enumerate(hs.tolist()):
            if h in keep and key in keep[h]:
                continue
            if informative(tok.decode([int(x) for x in ids[i:i + n]]), min_words):
                keep.setdefault(h, []).append(key)
    order = np.array(sorted(keep), dtype=np.uint64)
    return Probe(n, order, [keep[int(h)] for h in order], full.sizes)


#: A block sharing at least this many distinct n-grams with HumanEval is a STRONG hit -- a
#: shared run of ~20+ tokens, i.e. a copied problem or solution. Below it, the hits measured
#: on the real mix are idioms (the `is_prime` loop several HumanEval solutions use). Both
#: are dropped; the count of strong ones is what says whether the source was contaminated.
STRONG = 8


def contaminated_rows(tokens: np.ndarray, probe, n: int = N,
                      chunk: int = 4096) -> tuple[np.ndarray, dict[int, set[str]], np.ndarray]:
    """`(N,)` bool -- which blocks share an n-gram with the probe --, for those the probe
    keys they hit, and `(N,)` int: how many distinct probe n-grams each block shares."""
    hit = np.zeros(len(tokens), dtype=bool)
    shared = np.zeros(len(tokens), dtype=np.int64)
    owners: dict[int, set[str]] = {}
    if probe.hashes.size == 0:
        return hit, owners, shared
    for start in range(0, len(tokens), chunk):
        block = tokens[start:start + chunk]
        for i, row in enumerate(block):
            hs = ngram_hashes(row, n)
            pos = np.searchsorted(probe.hashes, hs)
            pos = np.minimum(pos, probe.hashes.size - 1)
            m = probe.hashes[pos] == hs
            if m.any():
                hit[start + i] = True
                distinct = np.unique(pos[m])
                shared[start + i] = distinct.size
                owners[start + i] = {k for p in distinct for k in probe.owners[p]}
    return hit, owners, shared


def _load(part: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.load(part / f"{split}_tokens.npy")
    m = np.load(part / f"{split}_mask.npy")
    if t.shape != m.shape:
        raise ValueError(f"{part}: {split} tokens {t.shape} and mask {m.shape} disagree")
    return t, m


def mix(parts: list[tuple[Path, float]], out_dir: Path, tok, eval_rows: list[dict],
        seed: int = 0, n: int = N) -> dict:
    """Build `out_dir` from `parts`, decontaminated against `eval_rows`. Returns the manifest."""
    rng = np.random.default_rng(seed)
    probe = content_probe(humaneval_texts(eval_rows), tok, n=n)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"n": n, "min_words": MIN_WORDS, "seed": seed, "decontaminated_against": "humaneval",
                      "humaneval_tasks": len(eval_rows), "parts": [], "splits": {}}
    seq_len = None
    for split in ("train", "val"):
        toks, masks = [], []
        for part, frac in parts:
            t, m = _load(part, split)
            if seq_len is None:
                seq_len = t.shape[1]
            elif t.shape[1] != seq_len:
                raise ValueError(f"{part} has seq_len {t.shape[1]}, others {seq_len}")
            take = rng.permutation(len(t))[: max(1, round(len(t) * frac))] if len(t) else []
            t, m = t[np.sort(take)], m[np.sort(take)]
            dirty, owners, shared = contaminated_rows(t, probe, n)
            tasks = sorted({k.split("\t")[0] for o in owners.values() for k in o})
            manifest["parts"].append({
                "split": split, "dir": str(part), "fraction": frac,
                "blocks_available": int(len(_load(part, split)[0])),
                "blocks_taken": int(len(t)), "blocks_dropped": int(dirty.sum()),
                "blocks_strong": int((shared >= STRONG).sum()),
                "humaneval_tasks_strong": sorted({k.split("\t")[0]
                                                  for i in np.flatnonzero(shared >= STRONG)
                                                  for k in owners[int(i)]}),
                "humaneval_tasks_hit": tasks,
            })
            toks.append(t[~dirty])
            masks.append(m[~dirty])
        T, M = np.concatenate(toks), np.concatenate(masks)
        order = rng.permutation(len(T))
        T, M = T[order], M[order]
        np.save(out_dir / f"{split}_tokens.npy", T)
        np.save(out_dir / f"{split}_mask.npy", M)
        manifest["splits"][split] = {
            "blocks": int(len(T)), "tokens": int(T.size),
            "trainable_share": float(M.mean()) if M.size else 0.0,
            "max_id": int(T.max()) if T.size else None,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    from ..eval import sources
    from ..tokenizer.tokenizer import Tokenizer

    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.data.mix_sft",
        description="Combine prepare_sft outputs by fraction and drop HumanEval overlap.")
    ap.add_argument("--part", action="append", required=True, metavar="DIR[:FRACTION]",
                    help="a prepare_sft output directory, optionally with the fraction of "
                         "its blocks to take (default 1.0). Repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=N, help="n-gram length for the overlap test")
    args = ap.parse_args(argv)

    tok = Tokenizer(args.tokenizer)
    rows = sources.load("humaneval")
    man = mix([parse_part(p) for p in args.part], Path(args.out_dir), tok, rows,
              seed=args.seed, n=args.n)
    for p in man["parts"]:
        print(f"  {p['split']:<5} {p['dir']:<36} x{p['fraction']:<5} "
              f"{p['blocks_taken']:>7,} blocks, {p['blocks_dropped']:>4} dropped, "
              f"{p['blocks_strong']:>3} strong"
              f"{'  (strong: ' + ', '.join(p['humaneval_tasks_strong'][:8]) + ')' if p['blocks_strong'] else ''}")
    vocab = tok.vocab_size
    for split, s in man["splits"].items():
        print(f"  {split}: {s['blocks']:,} blocks, {s['tokens']:,} tokens, "
              f"{s['trainable_share'] * 100:.1f}% trainable, max id {s['max_id']} (< {vocab})")
        if s["max_id"] is not None and s["max_id"] >= vocab:
            print("  ERROR: a token id is outside the tokenizer's vocabulary")
            return 1
    print(f"  wrote {args.out_dir} (+ manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
