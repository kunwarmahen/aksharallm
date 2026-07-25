"""Build a preference dataset for DPO.

Each row is a triple: one prompt, one response a human (or a stronger model) preferred,
and one they didn't.

    prompt:   "Explain gravity to a six-year-old."
    chosen:   "Imagine the Earth is giving everything a big hug..."
    rejected: "Gravity is a fundamental interaction described by general relativity..."

Neither response is *wrong*. SFT can't express this distinction at all -- it only ever
says "here is the correct answer, imitate it". Preference data is how you teach the
things that are matters of degree: tone, length, hedging, when to refuse.

Stored padded (not packed) because a DPO example is an indivisible triple: the chosen
and rejected responses must be scored against the same prompt, so we can't let a block
boundary cut through one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..tokenizer.tokenizer import Tokenizer


def _ultrafeedback(row):
    """HuggingFaceH4/ultrafeedback_binarized: chosen/rejected are message lists."""
    chosen, rejected = row["chosen"], row["rejected"]
    if not chosen or not rejected:
        return None
    prompt = [m for m in chosen[:-1]]
    return prompt, chosen[-1]["content"], rejected[-1]["content"]


def _helpsteer(row):
    return ([{"role": "user", "content": row["prompt"]}],
            row["chosen"], row["rejected"])


RECIPES = {
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", None,
                      "train_prefs", _ultrafeedback),
    "orca-dpo": ("Intel/orca_dpo_pairs", None, "train", None),  # converter below
}


def _orca(row):
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["question"]})
    return msgs, row["chosen"], row["rejected"]


RECIPES["orca-dpo"] = ("Intel/orca_dpo_pairs", None, "train", _orca)


def encode_pair(tok: Tokenizer, prompt_msgs, response: str, seq_len: int):
    """Return (ids, mask) padded to seq_len, or None if it doesn't fit.

    mask marks the response tokens -- the only ones DPO scores.
    """
    prompt_ids, _ = tok.render_chat(prompt_msgs, add_generation_prompt=True)
    resp_ids = tok.encode(response) + [tok.im_end_id]
    ids = prompt_ids + resp_ids
    if len(ids) > seq_len or not resp_ids:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
    pad = seq_len - len(ids)
    ids = ids + [tok.pad_id] * pad
    mask = mask + [0] * pad
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Tokenize a preference dataset for DPO.")
    ap.add_argument("recipe", choices=list(RECIPES))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--val-examples", type=int, default=500)
    args = ap.parse_args()

    from datasets import load_dataset

    repo, config, split, convert = RECIPES[args.recipe]
    tok = Tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo, name=config, split=split, streaming=True)

    ct, cm, rt, rm = [], [], [], []
    n_seen = n_skipped = 0
    pbar = tqdm(desc="encoding", unit="pair")
    for row in ds:
        if args.max_examples is not None and len(ct) >= args.max_examples:
            break
        n_seen += 1
        try:
            parsed = convert(row)
        except (KeyError, TypeError, IndexError):
            n_skipped += 1
            continue
        if parsed is None:
            n_skipped += 1
            continue
        prompt_msgs, chosen, rejected = parsed
        if not chosen or not rejected or chosen.strip() == rejected.strip():
            n_skipped += 1
            continue

        a = encode_pair(tok, prompt_msgs, chosen, args.seq_len)
        b = encode_pair(tok, prompt_msgs, rejected, args.seq_len)
        if a is None or b is None:
            n_skipped += 1  # too long for the context window
            continue
        ct.append(a[0]); cm.append(a[1])
        rt.append(b[0]); rm.append(b[1])
        pbar.update(1)
    pbar.close()

    if not ct:
        raise RuntimeError("no usable preference pairs -- check --seq-len and the recipe")

    arrays = {
        "chosen_tokens": np.stack(ct), "chosen_mask": np.stack(cm),
        "rejected_tokens": np.stack(rt), "rejected_mask": np.stack(rm),
    }
    n = len(ct)
    n_val = min(args.val_examples, max(1, n // 20))
    perm = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    for key, arr in arrays.items():
        np.save(out_dir / f"train_{key}.npy", arr[train_idx])
        np.save(out_dir / f"val_{key}.npy", arr[val_idx])

    print(f"\npairs: {n:,} kept, {n_skipped:,} skipped (of {n_seen:,} seen)")
    print(f"split: {len(train_idx):,} train / {len(val_idx):,} val")
    print(f"mean response length: chosen {arrays['chosen_mask'].sum(1).mean():.0f} tok, "
          f"rejected {arrays['rejected_mask'].sum(1).mean():.0f} tok")
    print(f"wrote to {out_dir}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
