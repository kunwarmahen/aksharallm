"""Associative recall: can a model look up what it was told, and how far back?

The position curve (curve.py) asks whether a model stays *fluent* far into a long context;
the needle (haystack.py) asks whether it can *retrieve* one planted fact. At 13.8M on
TinyStories the needle is uninformative (every model is at chance), and fluency is not
retrieval: tiny-hybrid showed no loss cliff past its 512-token window, which says nothing
about whether its DeltaNet memory can actually give back a value it was told. This file is
the version that can tell, at any size, because the task is all retrieval and nothing else.

One document:

    <|endoftext|>  K17 V3  K40 V9  ...  K2 V51     F F F ... F       ?   K40 V9  K2 V51 ...
                   \\______ P pairs: a key, its value ___/  \\_ gap _/      \\_ Q queries _/

Keys are distinct within a document and values are random, so a query's value can be
predicted only by finding its key earlier in the context and reading what followed. Every
other token is unpredictable noise (that is fine: the loss on them is a constant, and the
gradient that matters comes from the query values). The vocabulary is words, as in the
world model: 3 specials + '?' + 64 keys + 64 values + 16 filler tokens = 148.

Training documents are short (`TRAIN`: 4-48 pairs, gaps mostly under 32 tokens, up to 200), so the trainer's
random 512-token windows hold whole documents most of the time. Scoring builds documents to
order and reads the argmax at each query's value position:

  by pairs   how many facts it can hold at once (16 to 48 pairs, short gap)
  by gap     how far back it can reach: the gap grows to 4x the trained window, past the
             512 positions the attention layers' RoPE ever saw. A dense model is expected to
             fall off there; a pure DeltaNet has no positions at all; the hybrid is both.

Chance is 1/64 (any value) and a model that has learned "the answer is one of this
document's values" without retrieval scores about 1/P: both are printed.

Read with: docs/19-long-context.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/04-model.md (the hybrid and Gated DeltaNet it tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SPECIALS = ("<|endoftext|>", "<|pad|>", "<unk>")
N_KEYS, N_VALUES, N_FILLER = 64, 64, 16
VOCAB = (SPECIALS + ("?",) + tuple(f"K{i}" for i in range(N_KEYS))
         + tuple(f"V{i}" for i in range(N_VALUES)) + tuple(f"F{i}" for i in range(N_FILLER)))
EOS, QUERY = 0, 3
KEY0 = 4
VALUE0 = KEY0 + N_KEYS
FILLER0 = VALUE0 + N_VALUES

#: The training distribution. Pairs are uniform in their range; the gap is uniform up to
#: `short_gap` most of the time and up to the full range with `p_long_gap`. Retrieval is
#: learned suddenly (an induction circuit forms) and late, so the documents carry as many
#: queries as they can and spend little on filler: the signal per token is what decides
#: whether it forms at all inside the budget. Every pair is queried (`queries` >= the most
#: pairs), which puts ~18% of tokens behind a lookup; at 16 queries it was 11%, and a 4-layer
#: model on the CPU was still at chance after 12M tokens. Easy documents (2-8 pairs, no
#: gap) go to their own `easy.bin`: on those alone a 2-layer model formed the lookup in
#: ~1,500 steps (52% at 4 pairs), where on the full range it had not formed by 4,500. Mixed
#: in as 30% of DOCUMENTS they were only ~4.5% of tokens (they are short) and nothing formed,
#: so the configs mix them by TOKENS with `data.train_sources`, the loader's exact per-batch
#: share. The easy documents are the foothold; the rest teach it to reach.
TRAIN = {"pairs": (4, 48), "gap": (0, 200), "short_gap": 32, "p_long_gap": 0.3, "queries": 48,
         "easy_pairs": (2, 8)}
#: What the scorer sweeps. Gaps past ~460 put the pairs beyond the trained window.
BY_PAIRS = (8, 16, 32, 48)
BY_GAP = (0, 128, 384, 768, 1536)
#: The capacity sweep (`longctx recall --pairs 32,48,56,64`): every key in the vocabulary at
#: the top, so nothing past 64 can be asked without a bigger vocabulary (and a new model).
#: Training stops at 48 pairs, so 56 and 64 are also a step past the trained range.
CAPACITY = (32, 48, 56, 64)
EVAL_SEED = 40_000


def document(rng: np.random.Generator, pairs: int, gap: int, queries: int) -> tuple[list[int], list[int]]:
    """`(tokens, answer_positions)`: the document, and the index of every query's value
    (the position the model must predict, i.e. the target at `pos`, read from logits at
    `pos - 1`)."""
    keys = rng.choice(N_KEYS, size=pairs, replace=False)
    values = rng.integers(0, N_VALUES, size=pairs)
    toks = [EOS]
    for k, v in zip(keys, values):
        toks += [KEY0 + int(k), VALUE0 + int(v)]
    toks += [FILLER0 + int(x) for x in rng.integers(0, N_FILLER, size=gap)]
    toks.append(QUERY)
    answers = []
    for j in rng.choice(pairs, size=min(queries, pairs), replace=False):
        toks.append(KEY0 + int(keys[j]))
        answers.append(len(toks))
        toks.append(VALUE0 + int(values[j]))
    return toks, answers


def write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(list(SPECIALS))
    tok.save(str(path))


def prepare(out: str | Path, *, train_tokens: int = 240_000_000, val_tokens: int = 2_000_000,
            easy_tokens: int = 60_000_000, seed: int = 0, progress=print) -> dict:
    """`train.bin` (the full range), `easy.bin` (2-8 pairs, no gap: the foothold), `val.bin`
    (full range), `tokenizer.json`, `meta.json`. A config mixes the two training files by
    token share with `data.train_sources`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_tokenizer(out / "tokenizer.json")
    meta = {"task": "recall", "vocab": list(VOCAB), "train": TRAIN, "seed": seed,
            "tokens": {}, "documents": {}}
    lo_p, hi_p = TRAIN["pairs"]
    lo_g, hi_g = TRAIN["gap"]
    lo_e, hi_e = TRAIN["easy_pairs"]
    for split, n, s in (("train", train_tokens, seed), ("easy", easy_tokens, seed + 5_000),
                        ("val", val_tokens, seed + 10_000)):
        rng = np.random.default_rng(s)
        arr = np.empty(n, dtype=np.uint16)
        i = docs = 0
        while i < n:
            if split == "easy":
                toks, _ = document(rng, int(rng.integers(lo_e, hi_e + 1)), 0, TRAIN["queries"])
            else:
                hi = hi_g if rng.random() < TRAIN["p_long_gap"] else TRAIN["short_gap"]
                toks, _ = document(rng, int(rng.integers(lo_p, hi_p + 1)),
                                   int(rng.integers(lo_g, hi + 1)), TRAIN["queries"])
            k = min(len(toks), n - i)
            arr[i:i + k] = toks[:k]
            i += k
            docs += 1
            if progress and docs % 200_000 == 0:
                progress(f"[recall] {split} {i:,}/{n:,} tokens")
        arr.tofile(out / f"{split}.bin")
        meta["tokens"][split], meta["documents"][split] = int(n), docs
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


@torch.no_grad()
def accuracy(model, *, pairs: int, gap: int, n: int = 64, queries: int = 8,
             seed: int = EVAL_SEED, device: str = "cpu", batch: int = 16) -> dict:
    """Recall accuracy on `n` documents of one shape: the share of query values the model's
    argmax gets right. All documents of one shape have one length, so no padding."""
    rng = np.random.default_rng(seed + 7919 * pairs + gap)
    docs = [document(rng, pairs, gap, queries) for _ in range(n)]
    marks = []          # one 0/1 per query, in document order: the same queries for every
                        # model (the documents are seeded by shape), so two models pair up
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False))
    for i in range(0, n, batch):
        chunk = docs[i:i + batch]
        x = torch.tensor([t for t, _ in chunk], device=device)
        with amp:
            logits, _ = model(x, full_logits=True)
        pred = logits.argmax(-1).cpu()
        for row, (toks, answers) in enumerate(chunk):
            for pos in answers:
                marks.append("1" if int(pred[row, pos - 1]) == toks[pos] else "0")
    right = marks.count("1")
    return {"pairs": pairs, "gap": gap, "length": len(docs[0][0]), "n": len(marks),
            "accuracy": right / len(marks), "chance_any": 1 / N_VALUES,
            "chance_in_doc": 1 / pairs, "correct": "".join(marks)}


def compare(a: dict, b: dict) -> list[dict]:
    """Two `score` results, paired query by query, cell by cell. Refuses results scored on
    different documents (a different `--docs`, or a file from before per-query records), since
    unpaired accuracies would be all that is left and a difference of two percentages hides
    which queries moved. `b_only` = b right where a was wrong; `p` = exact sign test."""
    from ..eval.versus import sign_test
    if a.get("n_docs") != b.get("n_docs"):
        raise ValueError(f"scored on different documents: {a.get('n_docs')} vs "
                         f"{b.get('n_docs')} per shape")
    rows = []
    for sweep in ("by_pairs", "by_gap"):
        for x, y in zip(a[sweep], b[sweep], strict=True):
            if (x["pairs"], x["gap"]) != (y["pairs"], y["gap"]):
                raise ValueError(f"different cells: {x['pairs']}/{x['gap']} vs "
                                 f"{y['pairs']}/{y['gap']}")
            if "correct" not in x or "correct" not in y:
                raise ValueError("a result has no per-query record; re-score it with "
                                 "`longctx recall` (files before 2026-11-22 have none)")
            ca, cb = x["correct"], y["correct"]
            b_only = sum(u == "0" and v == "1" for u, v in zip(ca, cb, strict=True))
            a_only = sum(u == "1" and v == "0" for u, v in zip(ca, cb, strict=True))
            rows.append({"sweep": sweep, "pairs": x["pairs"], "gap": x["gap"],
                         "length": x["length"], "n": len(ca), "a": x["accuracy"],
                         "b": y["accuracy"], "b_only": b_only, "a_only": a_only,
                         "p": sign_test(b_only, a_only)})
    return rows


def score(model, *, n: int = 64, device: str = "cpu", by_pairs=BY_PAIRS, by_gap=BY_GAP,
          gap_pairs: int = 16, pairs_gap: int = 32, progress=None) -> dict:
    """The two sweeps. `model` must have room for the longest document (`max_len`)."""
    rows_p, rows_g = [], []
    for p in by_pairs:
        rows_p.append(accuracy(model, pairs=p, gap=pairs_gap, n=n, device=device))
        if progress:
            progress(f"[recall] {p} pairs: {rows_p[-1]['accuracy'] * 100:.1f}%")
    for g in by_gap:
        rows_g.append(accuracy(model, pairs=gap_pairs, gap=g, n=n, device=device))
        if progress:
            progress(f"[recall] gap {g}: {rows_g[-1]['accuracy'] * 100:.1f}%")
    return {"by_pairs": rows_p, "by_gap": rows_g, "n_docs": n}


def max_len(by_pairs=BY_PAIRS, by_gap=BY_GAP, gap_pairs: int = 16, pairs_gap: int = 32,
            queries: int = 8) -> int:
    """The longest document `score` builds, so the model's RoPE table can be sized for it."""
    longest = max([1 + 2 * p + pairs_gap + 1 + 2 * queries for p in by_pairs]
                  + [1 + 2 * gap_pairs + g + 1 + 2 * queries for g in by_gap])
    return int(longest)


def load_for_recall(path: str | Path, device: str, by_pairs=BY_PAIRS, by_gap=BY_GAP):
    """A checkpoint with its position table sized for the longest scored document. The RoPE
    cache is derived from the config and never saved, so this changes no weight: positions
    past the trained window are simply asked for, with no scaling (that is the test)."""
    from ..config import ModelConfig
    from ..model.transformer import Transformer
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    trained = int(mcfg["max_seq_len"])
    mcfg["max_seq_len"] = max(trained, max_len(by_pairs, by_gap))
    model = Transformer(ModelConfig(**mcfg))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck, trained
