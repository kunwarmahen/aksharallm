"""Build a *blended* pretraining corpus from several datasets.

Real base models don't pretrain on one source -- they mix. Ours blends general web text
with code so the single expensive pretraining run yields a base that both chats and codes.

This orchestrator:
  1. trains ONE tokenizer on a weighted mix of all sources (so it handles both prose *and*
     code -- a prose-only tokenizer chops Python badly),
  2. tokenizes each source to its own `<name>.bin` (capped by its share of the token
     budget),
  3. writes a single combined `val.bin`,
  4. prints the `data.train_sources` block to paste into your config.

The ratio itself lives in the *config* (MixedTokenDataset samples the bins by weight at
train time), so you can retune it later without re-tokenizing.

Example -- 85% FineWeb-Edu, 15% Python, 10B tokens total:

    python -m aksharallm.data.prepare_blend \
        --out-dir data/blend --vocab-size 32768 \
        --source fineweb-edu-10bt:0.85 --source codeparrot-python:0.15 \
        --val-tokens 10000000 --max-train-tokens 10000000000

Read with: docs/02-data.md -- the chapter this implements; it ends with the order to read these
files in. See also docs/08-scaling.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

from ..tokenizer.tokenizer import Tokenizer, train_bpe
from .prepare import RECIPES, stream_texts, tokenize_to_bin


def parse_source(spec: str) -> tuple[str, float]:
    """'recipe:weight' -> (recipe, weight). Weight defaults to 1.0."""
    name, _, w = spec.partition(":")
    if name not in RECIPES:
        raise SystemExit(f"unknown recipe '{name}'. Known: {', '.join(RECIPES)}")
    return name, float(w) if w else 1.0


def blended_corpus(sources: list[tuple[str, float]], total_docs: int):
    """Yield up to `total_docs` documents, drawn from each source in proportion to weight.
    Used only to fit the tokenizer -- a representative sample of every source."""
    rng = random.Random(0)
    iters = {}
    for name, _ in sources:
        repo, cfg, col = RECIPES[name]
        iters[name] = stream_texts(repo, cfg, col, "train")
    names = [n for n, _ in sources]
    weights = [w for _, w in sources]
    produced = 0
    pbar = tqdm(total=total_docs, desc="fitting blended BPE")
    while produced < total_docs and iters:
        name = rng.choices(names, weights=weights, k=1)[0]
        if name not in iters:
            continue
        try:
            yield next(iters[name])
            produced += 1
            pbar.update(1)
        except StopIteration:
            del iters[name]
            i = names.index(name)
            names.pop(i)
            weights.pop(i)
    pbar.close()


def main():
    ap = argparse.ArgumentParser(description="Build a blended multi-source pretraining corpus.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="recipe:weight, repeatable (e.g. fineweb-edu-10bt:0.85)")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--tokenizer-train-docs", type=int, default=300_000)
    ap.add_argument("--max-train-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--n-proc", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    assert args.vocab_size <= 65536, "vocab must fit in uint16"
    sources = [parse_source(s) for s in args.source]
    total_w = sum(w for _, w in sources)
    sources = [(n, w / total_w) for n, w in sources]  # normalise weights
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"

    print(f"blending {len(sources)} sources:")
    for name, w in sources:
        print(f"    {name:<22} weight {w:.3f} -> {int(args.max_train_tokens * w):,} train tokens")

    # ---- 1. one tokenizer over the weighted mix ----------------------------------
    if not tok_path.exists():
        print(f"\n[1/3] training {args.vocab_size}-token BPE on the blend")
        train_bpe(blended_corpus(sources, args.tokenizer_train_docs),
                  args.vocab_size, tok_path)
    else:
        print(f"\n[1/3] reusing tokenizer at {tok_path}")
    tok = Tokenizer(tok_path)
    print(f"      vocab_size={tok.vocab_size}")

    # ---- 2. combined validation split --------------------------------------------
    # Take each source's share from the *start* of its stream; training skips those docs.
    print(f"\n[2/3] writing combined val.bin ({args.val_tokens:,} tokens)")
    val_parts = []
    for name, w in sources:
        repo, cfg, col = RECIPES[name]
        part = out_dir / f"_val_{name}.bin"
        tokenize_to_bin(stream_texts(repo, cfg, col, "train"), str(tok_path), part,
                        args.n_proc, max_tokens=max(1, int(args.val_tokens * w)),
                        desc=f"val:{name}")
        val_parts.append(part)
    # The combined file is a plain concatenation, so where one source ends and the next
    # begins exists only as arithmetic over the weights. Record it instead: a derived
    # boundary nobody checked is how a per-domain report becomes confidently wrong, and
    # `eval domains` otherwise has to reconstruct this and then verify its own guess.
    spans, offset = [], 0
    with open(out_dir / "val.bin", "wb") as f:
        for (name, w), p in zip(sources, val_parts):
            blob = p.read_bytes()
            f.write(blob)
            p.unlink()
            spans.append({"name": name, "start": offset, "end": offset + len(blob) // 2,
                          "weight": w})
            offset += len(blob) // 2
    (out_dir / "val.manifest.json").write_text(json.dumps(
        {"val_bin": "val.bin", "tokens": offset, "spans": spans}, indent=2))
    print(f"  val.manifest.json: " + ", ".join(
        f"{s['name']} {s['start']:,}-{s['end']:,}" for s in spans))

    # ---- 3. per-source train bins ------------------------------------------------
    print("\n[3/3] writing per-source train bins")
    skip_docs = args.val_tokens // 100  # generous skip so val docs never leak into train
    train_bins = []
    for name, w in sources:
        repo, cfg, col = RECIPES[name]

        def skipped_stream(repo=repo, cfg=cfg, col=col):
            for i, t in enumerate(stream_texts(repo, cfg, col, "train")):
                if i >= skip_docs:
                    yield t

        out_bin = out_dir / f"{name}.bin"
        n = tokenize_to_bin(skipped_stream(), str(tok_path), out_bin, args.n_proc,
                            max_tokens=max(1, int(args.max_train_tokens * w)), desc=f"train:{name}")
        train_bins.append((out_bin, w, n))

    # ---- report + config snippet -------------------------------------------------
    total = sum(n for _, _, n in train_bins)
    print(f"\ndone. {total:,} train tokens across {len(train_bins)} sources "
          f"({total * 2 / 1e9:.1f} GB)")
    print("\npaste this into your config's `data:` block:\n")
    print("data:")
    print(f"  val_bin: {out_dir}/val.bin")
    print(f"  tokenizer: {tok_path}")
    print("  train_sources:")
    for out_bin, w, n in train_bins:
        print(f"    - {{bin: {out_bin}, weight: {w:.3f}}}   # {n:,} tokens")
    sys.stdout.flush()
    os._exit(0)  # see prepare.py: the streaming reader crashes on normal shutdown


if __name__ == "__main__":
    main()
