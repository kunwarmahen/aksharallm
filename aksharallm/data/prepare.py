"""Turn a text dataset into flat uint16 token files.

The output format is deliberately dumb: one file, no headers, no records --
just every document's tokens concatenated, each followed by <|endoftext|>.

    [doc1 tokens] 0 [doc2 tokens] 0 [doc3 tokens] 0 ...

Why this works. During pretraining we don't care about document boundaries; we slice
random windows of `seq_len` out of the stream. A window that straddles two documents
teaches the model that <|endoftext|> means "topic over, start fresh", which is exactly
what we want it to learn. The 0 token is the boundary marker.

Why uint16: our vocab is <= 65536, so 2 bytes/token. A 10B-token corpus is 20 GB, which
np.memmap can serve straight from the OS page cache with no dataloader workers.

Everything streams: we never hold the raw text and the tokens in memory at once.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..tokenizer.tokenizer import Tokenizer, train_bpe

# Recipes: name -> (hf_repo, config, text_column)
# All of these stream without authentication and are parquet-native (no dataset scripts).
# Before adding one, verify with: load_dataset(repo, streaming=True); next(iter(ds)).
RECIPES = {
    # general / prose
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    "fineweb-edu-10bt": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "fineweb-edu-350bt": ("HuggingFaceFW/fineweb-edu", "sample-350BT", "text"),
    # code (for the blended base and the Python specialist)
    "codeparrot-python": ("codeparrot/codeparrot-clean", None, "content"),  # pure Python
    "stack-smol-xl": ("bigcode/the-stack-smol-xl", None, "content"),        # multi-language
}

_worker_tok: Tokenizer | None = None


def _init_worker(tok_path: str):
    global _worker_tok
    _worker_tok = Tokenizer(tok_path)


def _tokenize_batch(texts: list[str]) -> np.ndarray:
    """Encode a batch of documents, appending EOS to each, return one flat uint16 array."""
    assert _worker_tok is not None
    eos = _worker_tok.eos_id
    out: list[int] = []
    for ids in _worker_tok.encode_batch(texts):
        out.extend(ids)
        out.append(eos)
    return np.asarray(out, dtype=np.uint16)


def _batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def stream_texts(repo: str, config: str | None, column: str, split: str, limit: int | None = None):
    """Yield raw document strings, streaming from the Hub (no full download)."""
    from datasets import load_dataset

    ds = load_dataset(repo, name=config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        text = row[column]
        if text:
            yield text


def tokenize_to_bin(
    texts,
    tok_path: str,
    out_file: Path,
    n_proc: int,
    batch_size: int = 1024,
    max_tokens: int | None = None,
    desc: str = "tokenizing",
) -> int:
    """Consume a text iterator, write tokens to `out_file`. Returns tokens written."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    pbar = tqdm(unit="tok", unit_scale=True, desc=desc, total=max_tokens)

    batches = _batched(texts, batch_size)
    pool = mp.Pool(n_proc, initializer=_init_worker, initargs=(tok_path,))
    pending: deque = deque()
    # Enough queued work to keep every worker busy, bounded so we don't read the whole
    # dataset into RAM ahead of the tokenizers.
    max_pending = n_proc * 4
    done = False

    def drain_one(f) -> bool:
        """Pop the oldest result, write it. Returns True when the token budget is hit."""
        nonlocal written
        arr = pending.popleft().get()
        if max_tokens is not None and written + len(arr) > max_tokens:
            arr = arr[: max_tokens - written]
        arr.tofile(f)
        written += len(arr)
        pbar.update(len(arr))
        return max_tokens is not None and written >= max_tokens

    try:
        with open(out_file, "wb") as f:
            # NOTE: we deliberately do *not* hand `batches` to pool.imap. imap would
            # iterate it on multiprocessing's internal task-handler thread, and the HF
            # streaming reader raises from a non-main thread there. That exception is
            # swallowed, imap ends early, and you silently get a truncated (or empty)
            # dataset with a zero exit code. Iterating here keeps the stream on the main
            # thread where errors actually propagate.
            for batch in batches:
                pending.append(pool.apply_async(_tokenize_batch, (batch,)))
                if len(pending) >= max_pending and drain_one(f):
                    done = True
                    break
            if not done:
                while pending:  # flush the tail
                    if drain_one(f):
                        break
            f.flush()
            os.fsync(f.fileno())
    finally:
        # We usually stop early (max_tokens), leaving workers mid-task and the HF
        # streaming iterator holding an open HTTP response. Tear both down explicitly --
        # letting the GC do it at interpreter shutdown segfaults.
        pool.terminate()
        pool.join()
        batches.close()
        if hasattr(texts, "close"):
            texts.close()
        pbar.close()

    if written == 0:
        raise RuntimeError(
            f"tokenized 0 tokens into {out_file} -- the source stream yielded nothing. "
            "Check network access and the dataset name."
        )
    return written


def main():
    ap = argparse.ArgumentParser(description="Download, tokenize and pack a dataset.")
    ap.add_argument("recipe", choices=list(RECIPES), help="which dataset")
    ap.add_argument("--out-dir", required=True, help="e.g. data/tinystories")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer-train-docs", type=int, default=200_000,
                    help="docs sampled to fit the BPE merges (more is slower, not better)")
    ap.add_argument("--max-train-tokens", type=int, default=None,
                    help="stop after N training tokens (disk budget)")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--n-proc", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--tokenizer", default=None, help="reuse an existing tokenizer.json")
    args = ap.parse_args()

    assert args.vocab_size <= 65536, "vocab must fit in uint16"
    repo, config, column = RECIPES[args.recipe]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = args.tokenizer or str(out_dir / "tokenizer.json")

    # ---- 1. tokenizer -------------------------------------------------------------
    if not Path(tok_path).exists():
        print(f"[1/3] training {args.vocab_size}-token BPE on "
              f"{args.tokenizer_train_docs:,} docs from {repo}")
        corpus = stream_texts(repo, config, column, "train", limit=args.tokenizer_train_docs)
        train_bpe(tqdm(corpus, total=args.tokenizer_train_docs, desc="fitting BPE"),
                  args.vocab_size, tok_path)
    else:
        print(f"[1/3] reusing tokenizer at {tok_path}")
    tok = Tokenizer(tok_path)
    print(f"      vocab_size={tok.vocab_size}  eos={tok.eos_id}")

    # ---- 2. validation split ------------------------------------------------------
    # Taken from the *start* of the stream and skipped for training, so there is no
    # overlap between train and val. Contaminated val loss is worse than no val loss.
    print(f"[2/3] writing validation split ({args.val_tokens:,} tokens)")
    val_texts = stream_texts(repo, config, column, "train")
    n_val = tokenize_to_bin(val_texts, tok_path, out_dir / "val.bin", args.n_proc,
                            max_tokens=args.val_tokens, desc="val")

    # ---- 3. train split -----------------------------------------------------------
    # Skip roughly the documents consumed by val (approximate is fine at this scale;
    # we skip generously to guarantee no leakage).
    skip_docs = args.val_tokens // 100
    print(f"[3/3] writing train split (skipping first {skip_docs:,} docs to avoid val overlap)")

    def train_stream():
        for i, t in enumerate(stream_texts(repo, config, column, "train")):
            if i < skip_docs:
                continue
            yield t

    n_train = tokenize_to_bin(train_stream(), tok_path, out_dir / "train.bin", args.n_proc,
                              max_tokens=args.max_train_tokens, desc="train")

    gb = (n_train + n_val) * 2 / 1e9
    print(f"\ndone. train={n_train:,} tok  val={n_val:,} tok  ({gb:.2f} GB on disk)")
    print(f"  {out_dir}/train.bin  {out_dir}/val.bin  {tok_path}")
    sys.stdout.flush()

    # We almost always stop the stream early (--max-train-tokens), which leaves the
    # `datasets` streaming backend with a live HTTP reader thread. Normal interpreter
    # shutdown then finalizes the GIL out from under it and the process dumps core --
    # after the data is already written, but with a nonzero exit that breaks scripting.
    # Every file is closed and flushed by this point, so skip finalization entirely.
    os._exit(0)


if __name__ == "__main__":
    main()
