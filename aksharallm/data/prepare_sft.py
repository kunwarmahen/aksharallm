r"""Build a supervised fine-tuning dataset from a chat corpus.

Pretraining and SFT differ in exactly one way: *which tokens count towards the loss*.

  pretrain:  every token is a target. The model learns "what text looks like".
  SFT:       only the assistant's tokens are targets. The user's turn is context the
             model must condition on but must never be rewarded for predicting.

So alongside the token stream we write a parallel mask stream:

    tokens  <|im_start|>user \n What is 2+2? <|im_end|> <|im_start|>assistant \n Four <|im_end|>
    mask         0      0   0  0   0  0  0  0     0          0          0     0   1   1
                 \___________ context, no loss ___________/              \_ trained on _/

Examples are *packed* end-to-end into fixed seq_len blocks rather than padded. Padding a
1024-token window to hold a 60-token exchange wastes 94% of the compute; packing wastes
none. The cost is that one window can contain the tail of one conversation and the head
of the next, which the model sees as an <|im_end|> followed by a fresh <|im_start|> --
exactly the boundary it needs to learn anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..tokenizer.tokenizer import Tokenizer

# name -> (repo, config, split, converter)
# Each converter turns a dataset row into a [{"role":..., "content":...}, ...] list.
def _smoltalk(row):
    return row["messages"]


def _ultrachat(row):
    return row["messages"]


def _openhermes(row):
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    out = []
    for turn in row["conversations"]:
        role = role_map.get(turn.get("from"), turn.get("from"))
        if role in ("user", "assistant", "system"):
            out.append({"role": role, "content": turn["value"]})
    return out


def _identity(row):
    return row["messages"]


RECIPES = {
    "smoltalk": ("HuggingFaceTB/smoltalk", "all", "train", _smoltalk),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", _ultrachat),
    "openhermes": ("teknium/OpenHermes-2.5", None, "train", _openhermes),
    # A local JSONL of {"messages": [...]} rows -- which is exactly what
    # `python -m aksharallm.synth export` writes. Generated data goes through the same
    # tokenizing, packing and mask code as a downloaded corpus; the only thing that differs
    # is where the rows came from, and that is recorded in data/synth/<name>/meta.json
    # rather than here. Point it at a file with --file.
    "jsonl": (None, None, None, _identity),
    # Generated locally, no download. Exists so the SFT/DPO/LoRA machinery can be smoke
    # tested end to end in seconds — the shapes, the mask alignment, the packing and the
    # trainers are all exercised for real. It teaches the model a trivially learnable
    # mapping, which is the point: if the loss does not fall on this, the bug is in the
    # code, not in the data or the hyperparameters.
    "synthetic": (None, None, None, _identity),
}

#: The synthetic task: answer with the arithmetic. Small vocabulary, exact answers, and a
#: model that has learned it is obvious from a single generation.
_SYNTH_TEMPLATES = [
    ("What is {a} plus {b}?", "{a} plus {b} is {c}."),
    ("Add {a} and {b}.", "The sum of {a} and {b} is {c}."),
    ("Tell me {a} + {b}.", "It is {c}."),
    ("Can you add {a} to {b}?", "Yes. {a} + {b} = {c}."),
]


def synthetic_rows(n: int, seed: int = 0):
    """A stream of chat rows in the same shape the real recipes produce."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = int(rng.integers(0, 50)), int(rng.integers(0, 50))
        q, ans = _SYNTH_TEMPLATES[int(rng.integers(0, len(_SYNTH_TEMPLATES)))]
        yield {"messages": [
            {"role": "user", "content": q.format(a=a, b=b)},
            {"role": "assistant", "content": ans.format(a=a, b=b, c=a + b)},
        ]}


def jsonl_rows(path: Path):
    """Rows from a local JSONL file, one JSON object per line.

    A malformed line is skipped rather than fatal: a generation run that was killed
    mid-write leaves a truncated final line, and losing the last sample is a better outcome
    than refusing to tokenize the other 4,999.
    """
    import json

    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def is_valid(messages) -> bool:
    """Reject rows that would teach the model nothing or teach it something wrong."""
    if not messages or len(messages) < 2:
        return False
    if not any(m.get("role") == "assistant" and m.get("content", "").strip()
               for m in messages):
        return False  # nothing to train on
    return all(m.get("role") in ("system", "user", "assistant") and m.get("content")
               for m in messages)


def main():
    ap = argparse.ArgumentParser(description="Tokenize a chat dataset for SFT.")
    ap.add_argument("recipe", choices=list(RECIPES))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--val-examples", type=int, default=2000)
    ap.add_argument("--synthetic-examples", type=int, default=20000,
                    help="how many rows the 'synthetic' recipe generates")
    ap.add_argument("--file", default=None,
                    help="JSONL of {\"messages\": [...]} rows, for the 'jsonl' recipe "
                         "(e.g. data/synth/<name>/sft.jsonl)")
    args = ap.parse_args()

    repo, config, split, convert = RECIPES[args.recipe]
    tok = Tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recipe == "synthetic":
        ds = synthetic_rows(args.synthetic_examples)
    elif args.recipe == "jsonl":
        if not args.file:
            raise SystemExit("the 'jsonl' recipe needs --file <path.jsonl>")
        ds = jsonl_rows(Path(args.file))
    else:
        from datasets import load_dataset

        ds = load_dataset(repo, name=config, split=split, streaming=True)

    # Packing buffers: we accumulate tokens until we have seq_len of them, emit a block,
    # and keep the remainder for the next block.
    buf_tok: list[int] = []
    buf_mask: list[int] = []
    blocks_tok: list[np.ndarray] = []
    blocks_mask: list[np.ndarray] = []

    n_seen = n_used = n_skipped = 0
    n_trainable = 0

    pbar = tqdm(desc="packing", unit="ex")
    for row in ds:
        if args.max_examples is not None and n_used >= args.max_examples:
            break
        n_seen += 1
        try:
            messages = convert(row)
        except (KeyError, TypeError):
            n_skipped += 1
            continue
        if not is_valid(messages):
            n_skipped += 1
            continue

        ids, mask = tok.render_chat(messages, add_generation_prompt=False)
        if len(ids) > args.seq_len:
            # A conversation longer than the window gets dropped rather than truncated:
            # a truncated example ends mid-sentence and teaches the model to stop early.
            n_skipped += 1
            continue

        buf_tok.extend(ids)
        buf_mask.extend(mask)
        n_used += 1
        pbar.update(1)

        while len(buf_tok) >= args.seq_len:
            blocks_tok.append(np.asarray(buf_tok[:args.seq_len], dtype=np.uint16))
            blocks_mask.append(np.asarray(buf_mask[:args.seq_len], dtype=np.uint8))
            n_trainable += int(sum(buf_mask[:args.seq_len]))
            buf_tok = buf_tok[args.seq_len:]
            buf_mask = buf_mask[args.seq_len:]
    pbar.close()

    if not blocks_tok:
        raise RuntimeError("no blocks produced -- check the tokenizer and dataset")

    toks = np.stack(blocks_tok)     # (N, seq_len)
    masks = np.stack(blocks_mask)   # (N, seq_len)

    n_val = min(args.val_examples, max(1, len(toks) // 20))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(toks))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    np.save(out_dir / "train_tokens.npy", toks[train_idx])
    np.save(out_dir / "train_mask.npy", masks[train_idx])
    np.save(out_dir / "val_tokens.npy", toks[val_idx])
    np.save(out_dir / "val_mask.npy", masks[val_idx])

    total = toks.size
    print(f"\nexamples: {n_used:,} used, {n_skipped:,} skipped (of {n_seen:,} seen)")
    print(f"blocks:   {len(toks):,} x {args.seq_len} = {total:,} tokens")
    print(f"trained on {n_trainable:,} tokens ({100*n_trainable/total:.1f}% of the total)")
    print(f"split:    {len(train_idx):,} train / {len(val_idx):,} val blocks")
    print(f"wrote to  {out_dir}")
    sys.stdout.flush()
    os._exit(0)  # see prepare.py: the streaming reader crashes on normal shutdown


if __name__ == "__main__":
    main()
