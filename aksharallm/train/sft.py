"""Supervised fine-tuning: turn a base model into one that answers questions.

Differences from pretraining, all of them deliberate:

  loss masking     only assistant tokens count (that's the whole point)
  lower LR         ~10-30x lower. The base model already knows language; we're adjusting
                   style, not re-learning it. A pretrain-sized LR erases the pretraining.
  fewer steps      1-3 epochs. SFT datasets are small and overfit fast.
  dropout on       now we *are* overfitting-limited rather than data-limited.
  shuffled epochs  we iterate the dataset rather than sampling random windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig, config_to_dict, load_config
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .pretrain import fmt_dur, human, save_checkpoint, stamp
from .schedule import get_lr


class SFTDataset:
    """Fixed-size packed blocks with a per-token loss mask."""

    def __init__(self, tokens_path, mask_path, device="cuda"):
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        assert self.tokens.shape == self.mask.shape
        self.device = device
        self.n, self.seq_len = self.tokens.shape

    def batch(self, idx: np.ndarray):
        # x is the block minus its last token; y is the block shifted left by one.
        blk = self.tokens[idx].astype(np.int64)
        msk = self.mask[idx].astype(np.int64)
        x = torch.from_numpy(blk[:, :-1])
        y = torch.from_numpy(blk[:, 1:]).clone()
        # The mask is aligned to the *target*: position i predicts blk[i+1], so we want
        # mask[i+1]. Non-assistant targets become -100, which cross_entropy ignores.
        m = torch.from_numpy(msk[:, 1:])
        y[m == 0] = -100
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def epoch_batches(self, batch_size, rng):
        order = rng.permutation(self.n)
        for i in range(0, self.n - batch_size + 1, batch_size):
            yield self.batch(order[i : i + batch_size])


@torch.no_grad()
def evaluate(model, ds: SFTDataset, batch_size, n_batches, ctx):
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    for i, (x, y) in enumerate(ds.epoch_batches(batch_size, rng)):
        if i >= n_batches:
            break
        with ctx:
            _, loss = model(x, targets=y)
        if not math.isnan(loss.item()):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser(description="Supervised fine-tune a base checkpoint.")
    ap.add_argument("--base", required=True, help="pretrained checkpoint (.pt)")
    ap.add_argument("--data-dir", required=True, help="output of prepare_sft")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load the base model ------------------------------------------------------
    ckpt = torch.load(args.base, map_location=args.device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    mcfg.dropout = args.dropout  # re-enable dropout for fine-tuning
    model = Transformer(mcfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    tok = Tokenizer(args.tokenizer)
    data_dir = Path(args.data_dir)
    train_ds = SFTDataset(data_dir / "train_tokens.npy", data_dir / "train_mask.npy", args.device)
    val_ds = SFTDataset(data_dir / "val_tokens.npy", data_dir / "val_mask.npy", args.device)

    if train_ds.seq_len > mcfg.max_seq_len:
        raise ValueError(
            f"SFT blocks are {train_ds.seq_len} tokens but the model's context is "
            f"{mcfg.max_seq_len}. Re-run prepare_sft with --seq-len {mcfg.max_seq_len}."
        )

    optimizer, _ = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95),
                                              args.device)
    steps_per_epoch = train_ds.n // (args.batch_size * args.grad_accum)
    max_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(max_steps * args.warmup_ratio))

    print("=" * 78)
    print(f"base       {args.base} (step {ckpt.get('step')})")
    print(f"params     {human(model.num_params())}")
    print(f"data       {train_ds.n:,} train / {val_ds.n:,} val blocks of {train_ds.seq_len}")
    print(f"batch      {args.batch_size} x {args.grad_accum} accum = "
          f"{args.batch_size*args.grad_accum*train_ds.seq_len:,} tokens/step")
    print(f"schedule   {max_steps:,} steps ({args.epochs} epochs), {warmup} warmup, lr {args.lr}")
    print("=" * 78)

    if args.compile:
        model = torch.compile(model)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False))

    logf = open(out_dir / "sft_log.jsonl", "a")
    model.train()
    rng = np.random.default_rng(1234)
    best_val = float("inf")
    step = 0
    t0 = time.time()  # current log window
    run_t0 = t0  # whole invocation
    prev_log_step = -1
    print(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    for epoch in range(args.epochs):
        batches = train_ds.epoch_batches(args.batch_size, rng)
        exhausted = False
        while not exhausted:
            lr = get_lr(step, base_lr=args.lr, warmup_steps=warmup,
                        max_steps=max_steps, min_lr_ratio=0.1, schedule="cosine")
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_sum, n_micro = 0.0, 0
            for _ in range(args.grad_accum):
                try:
                    x, y = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                with ctx:
                    _, loss = model(x, targets=y)
                    loss = loss / args.grad_accum
                # A packed block can (rarely) contain zero assistant tokens; its loss is
                # then NaN from a 0/0 mean. Skip it rather than poisoning the gradients.
                if torch.isnan(loss):
                    continue
                loss.backward()
                loss_sum += loss.item()
                n_micro += 1
            if n_micro == 0:
                if exhausted:
                    break
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                t0 = time.time()
                s_per_step = dt / (step - prev_log_step)   # measured, not assumed
                prev_log_step = step
                up = time.time() - run_t0
                print(f"[{stamp()}] epoch {epoch} step {step:>5}/{max_steps} | "
                      f"loss {loss_sum:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                      f"{s_per_step:.2f}s/step | up {fmt_dur(up)} | "
                      f"eta {fmt_dur((max_steps - step) * s_per_step)}")
                logf.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_sum,
                                       "lr": lr, "time": time.time(),
                                       "s_per_step": s_per_step, "elapsed": up}) + "\n")
                logf.flush()

            if step > 0 and step % args.eval_every == 0:
                te = time.time()
                vl = evaluate(model, val_ds, args.batch_size, 20, ctx)
                print(f"  >> val {vl:.4f}{'  * best' if vl < best_val else ''}"
                      f"  ({fmt_dur(time.time() - te)})")
                logf.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                logf.flush()
                if vl < best_val:
                    best_val = vl
                    cfg_obj = _rebuild_cfg(ckpt, mcfg, args)
                    save_checkpoint(out_dir / "sft_best.pt", model, optimizer,
                                    cfg_obj, step, best_val)
                t0 = time.time()
            step += 1

    vl = evaluate(model, val_ds, args.batch_size, 20, ctx)
    print(f"\nfinal val {vl:.4f}")
    cfg_obj = _rebuild_cfg(ckpt, mcfg, args)
    save_checkpoint(out_dir / "sft_last.pt", model, optimizer, cfg_obj, step, min(best_val, vl))
    if vl < best_val:
        save_checkpoint(out_dir / "sft_best.pt", model, optimizer, cfg_obj, step, vl)
    print(f"ran {step} steps in {fmt_dur(time.time() - run_t0)}, "
          f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"checkpoints in {out_dir}")
    print(f"\ntry it:  python -m aksharallm.infer.cli {out_dir}/sft_best.pt --mode chat")
    logf.close()


def _rebuild_cfg(ckpt, mcfg, args):
    """Carry the base run's config forward so downstream tools still find the tokenizer."""
    from ..config import Config, DataConfig, TrainConfig

    base = ckpt.get("config", {})
    data = base.get("data", {})
    cfg = Config(
        name="sft",
        model=mcfg,
        data=DataConfig(
            train_bin=data.get("train_bin", ""),
            val_bin=data.get("val_bin", ""),
            tokenizer=args.tokenizer,
        ),
        train=TrainConfig(out_dir=args.out_dir, seq_len=mcfg.max_seq_len),
    )
    return cfg


if __name__ == "__main__":
    main()
