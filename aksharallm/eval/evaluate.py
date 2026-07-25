"""Evaluation.

Two kinds of number, and they answer different questions:

  perplexity  - "how surprised is the model by real text?" Cheap, smooth, great for
                watching a training run. Useless for comparing models with different
                tokenizers, because it's per-token and tokens differ.

  multiple-choice accuracy (HellaSwag) - "can the model tell a sensible continuation
                from a nonsense one?" Comparable across tokenizers. Noisy below ~1B
                params: a 400M model scores near the 25% random baseline, so don't
                panic if Phase 2 looks bad here. It becomes useful as you scale.

HellaSwag is scored the standard way: for each of 4 endings, compute the model's total
log-probability of the ending tokens given the context, normalise by token count (so
long endings aren't unfairly penalised), and pick the argmax.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data.loader import TokenDataset
from ..infer.cli import load_model, resolve_tokenizer
from ..tokenizer.tokenizer import Tokenizer


@torch.no_grad()
def perplexity(model, bin_path: str, seq_len: int, n_batches: int = 200,
               batch_size: int = 16, device: str = "cuda") -> dict:
    ds = TokenDataset(bin_path, seq_len, device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else None
    total_nll, total_tok = 0.0, 0
    for x, y in tqdm(ds.iter_eval_batches(batch_size, n_batches, seed=1234),
                     total=n_batches, desc="perplexity"):
        if ctx:
            with ctx:
                logits, _ = model(x, targets=y)
        else:
            logits, _ = model(x, targets=y)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                              y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += y.numel()
    mean_nll = total_nll / total_tok
    return {"loss": mean_nll, "perplexity": math.exp(mean_nll), "tokens": total_tok}


@torch.no_grad()
def _score_continuation(model, tok: Tokenizer, context: str, endings: list[str],
                        device: str) -> int:
    """Return the index of the ending with the highest length-normalised logprob."""
    scores = []
    for ending in endings:
        ctx_ids = tok.encode(context, bos=True)
        end_ids = tok.encode(ending)
        ids = ctx_ids + end_ids
        if len(ids) > model.cfg.max_seq_len:
            # Truncate from the left; keep the ending intact.
            ids = ids[-model.cfg.max_seq_len:]
            ctx_len = max(1, len(ids) - len(end_ids))
        else:
            ctx_len = len(ctx_ids)

        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else None
        if amp:
            with amp:
                logits, _ = model(x, targets=y)
        else:
            logits, _ = model(x, targets=y)

        logprobs = F.log_softmax(logits.float(), dim=-1)
        tok_lp = logprobs[0].gather(-1, y[0, :, None]).squeeze(-1)
        # Only the ending's tokens count. Position i of tok_lp scores ids[i+1], so the
        # ending starts at index ctx_len-1.
        ending_lp = tok_lp[ctx_len - 1:]
        scores.append(ending_lp.sum().item() / max(1, ending_lp.numel()))
    return int(max(range(len(scores)), key=lambda i: scores[i]))


@torch.no_grad()
def hellaswag(model, tok: Tokenizer, limit: int = 1000, device: str = "cuda") -> dict:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct = total = 0
    for row in tqdm(ds, total=limit, desc="hellaswag"):
        if total >= limit:
            break
        ctx_text = row["ctx"]
        endings = row["endings"]
        gold = int(row["label"])
        pred = _score_continuation(model, tok, ctx_text, endings, device)
        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / max(1, total), "n": total, "random_baseline": 0.25}


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--val-bin", default=None, help="defaults to the run's own val split")
    ap.add_argument("--tasks", default="perplexity",
                    help="comma-separated: perplexity,hellaswag,samples")
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--limit", type=int, default=1000, help="hellaswag examples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    tok = Tokenizer(resolve_tokenizer(ckpt, args.tokenizer))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results: dict = {"checkpoint": args.checkpoint, "step": ckpt.get("step")}

    if "perplexity" in tasks:
        val_bin = args.val_bin or ckpt["config"]["data"]["val_bin"]
        seq_len = ckpt["config"]["train"]["seq_len"]
        results["perplexity"] = perplexity(model, val_bin, seq_len,
                                           n_batches=args.n_batches, device=args.device)
        print(f"perplexity: {results['perplexity']['perplexity']:.2f} "
              f"(loss {results['perplexity']['loss']:.4f})")

    if "hellaswag" in tasks:
        results["hellaswag"] = hellaswag(model, tok, args.limit, args.device)
        print(f"hellaswag: {results['hellaswag']['accuracy']*100:.1f}% "
              f"(random = 25%, n={results['hellaswag']['n']})")

    if "samples" in tasks:
        from ..infer.generate import generate
        prompts = ["Once upon a time", "The most important thing about", "In 1969,"]
        results["samples"] = []
        for p in prompts:
            out = generate(model, tok.encode(p, bos=True), max_new_tokens=100,
                           temperature=0.8, top_k=50, device=args.device, eos_id=tok.eos_id)
            text = tok.decode(out)
            results["samples"].append({"prompt": p, "text": text})
            print(f"\n--- {p!r}\n{text}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
