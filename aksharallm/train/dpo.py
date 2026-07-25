r"""Direct Preference Optimization.

The problem DPO solves: SFT can only say "imitate this answer". It has no way to express
"this answer is better than that one". Classic RLHF handles that by training a reward
model and then running PPO against it -- four models in memory and a famously fragile
training loop.

DPO's insight is that for the KL-regularised objective RLHF optimises, the optimal policy
has a closed form, and you can rearrange it so the reward model cancels out entirely. What
remains is a plain classification loss on preference pairs:

    L = -log sigmoid( beta * [ (log pi(chosen) - log pi_ref(chosen))
                             - (log pi(reject) - log pi_ref(reject)) ] )

Read it as: "push up the chance of the chosen response and down the chance of the
rejected one -- but measure both *relative to where the frozen reference model started*."

That reference term is the whole safety mechanism. Without it the model could drive the
chosen response's probability up by wrecking its general language ability. The reference
anchors it: drifting far from the SFT model is penalised. `beta` sets how hard the anchor
pulls (0.1 typical; higher = stay closer to the reference).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..model.transformer import Transformer
from .pretrain import human, save_checkpoint
from .schedule import get_lr
from .sft import _rebuild_cfg


class DPODataset:
    def __init__(self, data_dir: Path, prefix: str, device="cuda"):
        self.ct = np.load(data_dir / f"{prefix}_chosen_tokens.npy", mmap_mode="r")
        self.cm = np.load(data_dir / f"{prefix}_chosen_mask.npy", mmap_mode="r")
        self.rt = np.load(data_dir / f"{prefix}_rejected_tokens.npy", mmap_mode="r")
        self.rm = np.load(data_dir / f"{prefix}_rejected_mask.npy", mmap_mode="r")
        self.device = device
        self.n, self.seq_len = self.ct.shape

    def batch(self, idx):
        def prep(tokens, mask):
            t = torch.from_numpy(tokens[idx].astype(np.int64))
            m = torch.from_numpy(mask[idx].astype(np.int64))
            x, y, mm = t[:, :-1], t[:, 1:], m[:, 1:]
            if self.device.startswith("cuda"):
                x, y, mm = (v.pin_memory().to(self.device, non_blocking=True)
                            for v in (x, y, mm))
            else:
                x, y, mm = x.to(self.device), y.to(self.device), mm.to(self.device)
            return x, y, mm

        return prep(self.ct, self.cm), prep(self.rt, self.rm)

    def epoch_batches(self, batch_size, rng):
        order = rng.permutation(self.n)
        for i in range(0, self.n - batch_size + 1, batch_size):
            yield self.batch(np.sort(order[i : i + batch_size]))


def sequence_logprob(model, x, y, mask, ctx) -> torch.Tensor:
    """Sum of log p(y_t | y_<t) over the response tokens only. Returns (B,)."""
    with ctx:
        logits, _ = model(x, targets=y)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return (tok_lp * mask).sum(dim=-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float):
    """Returns (loss, accuracy, mean_margin)."""
    # How much the policy has moved relative to the reference, for each response.
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    # Accuracy = fraction of pairs where the policy already prefers the chosen response
    # more than the reference does. Starts at ~50% by construction and should climb.
    acc = (logits > 0).float().mean()
    return loss, acc, logits.mean()


@torch.no_grad()
def evaluate(model, ref, ds, batch_size, n_batches, beta, ctx):
    model.eval()
    losses, accs = [], []
    rng = np.random.default_rng(0)
    for i, ((cx, cy, cm), (rx, ry, rm)) in enumerate(ds.epoch_batches(batch_size, rng)):
        if i >= n_batches:
            break
        pc = sequence_logprob(model, cx, cy, cm, ctx)
        pr = sequence_logprob(model, rx, ry, rm, ctx)
        rc = sequence_logprob(ref, cx, cy, cm, ctx)
        rr = sequence_logprob(ref, rx, ry, rm, ctx)
        loss, acc, _ = dpo_loss(pc, pr, rc, rr, beta)
        losses.append(loss.item())
        accs.append(acc.item())
    model.train()
    return sum(losses) / max(1, len(losses)), sum(accs) / max(1, len(accs))


def main():
    ap = argparse.ArgumentParser(description="DPO preference tuning.")
    ap.add_argument("--sft", required=True, help="SFT checkpoint -- becomes both the "
                                                 "starting policy and the frozen reference")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-7,
                    help="DPO wants a *very* low LR -- 10-50x below SFT")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.sft, map_location=args.device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    mcfg.dropout = 0.0  # dropout would make the policy/reference comparison noisy

    policy = Transformer(mcfg).to(args.device)
    policy.load_state_dict(ckpt["model"])

    # The reference is a frozen snapshot of the same weights. It never trains; it exists
    # only to say "here is where you started".
    ref = Transformer(mcfg).to(args.device)
    ref.load_state_dict(ckpt["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    data_dir = Path(args.data_dir)
    train_ds = DPODataset(data_dir, "train", args.device)
    val_ds = DPODataset(data_dir, "val", args.device)

    optimizer, _ = policy.configure_optimizers(0.0, args.lr, (0.9, 0.95), args.device)
    steps_per_epoch = train_ds.n // (args.batch_size * args.grad_accum)
    max_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.1 * max_steps))

    print("=" * 78)
    print(f"policy+ref {args.sft}  ({human(policy.num_params())} params each)")
    print(f"data       {train_ds.n:,} train / {val_ds.n:,} val pairs of {train_ds.seq_len}")
    print(f"schedule   {max_steps:,} steps ({args.epochs} epochs), beta={args.beta}, lr={args.lr}")
    print("=" * 78)

    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    logf = open(out_dir / "dpo_log.jsonl", "a")
    policy.train()
    rng = np.random.default_rng(1234)
    step = 0
    t0 = time.time()
    best_val = float("inf")

    for epoch in range(args.epochs):
        batches = train_ds.epoch_batches(args.batch_size, rng)
        exhausted = False
        while not exhausted:
            lr = get_lr(step, base_lr=args.lr, warmup_steps=warmup, max_steps=max_steps,
                        min_lr_ratio=0.1, schedule="cosine")
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            agg_loss = agg_acc = 0.0
            n_micro = 0
            for _ in range(args.grad_accum):
                try:
                    (cx, cy, cm), (rx, ry, rm) = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                # Reference logprobs need no graph -- it's frozen.
                with torch.no_grad():
                    rc = sequence_logprob(ref, cx, cy, cm, ctx)
                    rr = sequence_logprob(ref, rx, ry, rm, ctx)
                pc = sequence_logprob(policy, cx, cy, cm, ctx)
                pr = sequence_logprob(policy, rx, ry, rm, ctx)
                loss, acc, margin = dpo_loss(pc, pr, rc, rr, args.beta)
                (loss / args.grad_accum).backward()
                agg_loss += loss.item() / args.grad_accum
                agg_acc += acc.item() / args.grad_accum
                n_micro += 1
            if n_micro == 0:
                break

            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                t0 = time.time()
                print(f"epoch {epoch} step {step:>5}/{max_steps} | loss {agg_loss:.4f} | "
                      f"acc {agg_acc*100:.1f}% | lr {lr:.2e} | gnorm {gnorm:.2f} | {dt:.1f}s")
                logf.write(json.dumps({"step": step, "loss": agg_loss,
                                       "acc": agg_acc, "lr": lr}) + "\n")
                logf.flush()

            if step > 0 and step % args.eval_every == 0:
                vl, va = evaluate(policy, ref, val_ds, args.batch_size, 20, args.beta, ctx)
                print(f"  >> val loss {vl:.4f} acc {va*100:.1f}%"
                      f"{'  * best' if vl < best_val else ''}")
                logf.write(json.dumps({"step": step, "val_loss": vl, "val_acc": va}) + "\n")
                logf.flush()
                if vl < best_val:
                    best_val = vl
                    save_checkpoint(out_dir / "dpo_best.pt", policy, optimizer,
                                    _rebuild_cfg(ckpt, mcfg, args), step, best_val)
                t0 = time.time()
            step += 1

    vl, va = evaluate(policy, ref, val_ds, args.batch_size, 20, args.beta, ctx)
    print(f"\nfinal val loss {vl:.4f} acc {va*100:.1f}%")
    cfg_obj = _rebuild_cfg(ckpt, mcfg, args)
    save_checkpoint(out_dir / "dpo_last.pt", policy, optimizer, cfg_obj, step, min(best_val, vl))
    if vl < best_val:
        save_checkpoint(out_dir / "dpo_best.pt", policy, optimizer, cfg_obj, step, vl)
    print(f"checkpoints in {out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
