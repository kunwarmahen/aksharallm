"""Pretraining loop: next-token prediction over the token stream.

The whole of pretraining is this:

    x = tokens[i : i+T]        y = tokens[i+1 : i+1+T]
    loss = cross_entropy(model(x), y)

Everything else in this file is the machinery that makes it survive days of wall-clock:
mixed precision, gradient accumulation, LR scheduling, checkpoint/resume, and throughput
measurement so you can tell when you've made things slower.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import Config, config_to_dict, load_config
from ..data.loader import TokenDataset
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .schedule import get_lr


def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def save_checkpoint(path: Path, model, optimizer, cfg: Config, step: int, best_val: float, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_to_dict(cfg),
        "model_config": asdict(cfg.model),
        "step": step,
        "best_val": best_val,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the previous checkpoint


def load_checkpoint(path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batch_size: int, n_batches: int, ctx) -> float:
    model.eval()
    losses = []
    for x, y in dataset.iter_eval_batches(batch_size, n_batches, seed=1234):
        with ctx:
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model, tok: Tokenizer, prompt: str, max_new: int = 100, device="cuda") -> str:
    from ..infer.generate import generate

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.eval()
    ids = tok.encode(prompt, bos=True)
    out = generate(raw, ids, max_new_tokens=max_new, temperature=0.8, top_k=50,
                   device=device, eos_id=tok.eos_id)
    raw.train()
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser(description="Pretrain a language model.")
    ap.add_argument("config", help="path to a YAML config")
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="dotted.key=value, repeatable")
    args = ap.parse_args()

    # Python block-buffers stdout when it isn't a TTY, so piping a multi-day run to a
    # log file would show nothing for hours. Force line buffering.
    sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TF32 for the fp32 matmuls that autocast leaves alone (mostly the optimizer and
    # anything outside the autocast region). Free ~2x on Ampere, no accuracy cost here.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------------
    train_ds = TokenDataset(cfg.data.train_bin, cfg.train.seq_len, device)
    val_ds = TokenDataset(cfg.data.val_bin, cfg.train.seq_len, device)
    tok = Tokenizer(cfg.data.tokenizer)
    if tok.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"config says vocab_size={cfg.model.vocab_size} but tokenizer has {tok.vocab_size}"
        )

    # ---- model -------------------------------------------------------------------
    model = Transformer(cfg.model).to(device)
    optimizer, (n_decay, n_nodecay) = model.configure_optimizers(
        cfg.optim.weight_decay, cfg.optim.lr, (cfg.optim.beta1, cfg.optim.beta2), device
    )

    start_step, best_val = 0, float("inf")
    resume = cfg.train.resume
    if resume == "auto":
        cand = out_dir / "ckpt_last.pt"
        resume = str(cand) if cand.exists() else None
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, device)
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {resume} at step {start_step}")

    tokens_per_step = cfg.train.batch_size * cfg.train.grad_accum * cfg.train.seq_len

    print("=" * 78)
    print(f"run          {cfg.name}")
    print(f"params       {human(model.num_params())} total, "
          f"{human(model.num_params(True))} non-embedding")
    print(f"             decay={human(n_decay)} no-decay={human(n_nodecay)}")
    print(f"arch         d={cfg.model.d_model} L={cfg.model.n_layers} H={cfg.model.n_heads} "
          f"KV={cfg.model.n_kv_heads} ff={cfg.model.d_ff} ctx={cfg.model.max_seq_len}")
    print(f"data         {train_ds.n_tokens:,} train / {val_ds.n_tokens:,} val tokens")
    print(f"batch        {cfg.train.batch_size} x {cfg.train.grad_accum} accum x "
          f"{cfg.train.seq_len} ctx = {tokens_per_step:,} tokens/step")
    print(f"budget       {cfg.train.max_steps:,} steps = "
          f"{human(tokens_per_step * cfg.train.max_steps)} tokens "
          f"({tokens_per_step * cfg.train.max_steps / train_ds.n_tokens:.2f} epochs)")
    print(f"lr           {cfg.optim.lr} {cfg.optim.schedule}, "
          f"{cfg.optim.warmup_steps} warmup -> {cfg.optim.lr * cfg.optim.min_lr_ratio:.2e}")
    print("=" * 78)

    if cfg.train.compile:
        print("compiling model (first step will be slow)...")
        model = torch.compile(model)

    # bf16 autocast: activations and matmuls run in bf16, the master weights and the
    # optimizer state stay fp32. bf16 has fp32's exponent range, so unlike fp16 it
    # needs no loss scaling -- there is nothing to overflow.
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else nullcontext())

    use_wandb = cfg.train.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.train.wandb_project, name=cfg.train.wandb_run or cfg.name,
                   config=config_to_dict(cfg))

    # ---- graceful interruption ----------------------------------------------------
    # A 6-day run needs to be stoppable without losing progress. Three ways to stop:
    #   Ctrl-C, or `kill <pid>` (SIGTERM)  -> finishes the current step, then saves
    #   `touch <out_dir>/STOP`             -> same, but doesn't need the terminal
    # In every case we save ckpt_last.pt at the *exact* current step and exit cleanly, so
    # rerunning the same command with resume:auto picks up with zero lost work.
    stop = {"now": False}

    def _request_stop(signum, frame):
        if stop["now"]:
            print("\n[stop] second signal -- exiting immediately (may lose this step)")
            raise KeyboardInterrupt
        print(f"\n[stop] signal {signum} received -- will save and exit after this step")
        stop["now"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    stop_file = out_dir / "STOP"

    logf = open(out_dir / "train_log.jsonl", "a")
    model.train()
    t0 = time.time()
    running_loss = None

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- gradient accumulation ------------------------------------------------
        # We want a large batch (good gradient estimates) but 24 GB won't hold one.
        # So run `grad_accum` micro-batches, summing gradients, then step once.
        # Each micro-loss is divided by grad_accum so the total is a *mean*, not a sum --
        # otherwise the effective LR would scale with grad_accum.
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(cfg.train.grad_accum):
            x, y = train_ds.get_batch(cfg.train.batch_size)
            with ctx:
                _, loss = model(x, targets=y)
                loss = loss / cfg.train.grad_accum
            loss.backward()
            loss_sum += loss.item()

        # Clip by global norm. This is the single most effective guard against a bad
        # batch (or a loss spike) destroying a run that's been going for days.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        running_loss = loss_sum if running_loss is None else 0.9 * running_loss + 0.1 * loss_sum

        # ---- logging --------------------------------------------------------------
        if step % cfg.train.log_every == 0:
            torch.cuda.synchronize() if device == "cuda" else None
            dt = time.time() - t0
            t0 = time.time()
            steps_done = cfg.train.log_every if step > start_step else 1
            tok_per_sec = tokens_per_step * steps_done / dt
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            mfu = raw.estimate_mfu(tok_per_sec)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            eta = (cfg.train.max_steps - step) * dt / steps_done / 3600
            print(f"step {step:>6} | loss {loss_sum:.4f} (ema {running_loss:.4f}) | "
                  f"ppl {math.exp(min(running_loss, 20)):>7.1f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.2f} | {tok_per_sec/1e3:.1f}k tok/s | "
                  f"mfu {mfu*100:.1f}% | {mem:.1f}GB | eta {eta:.1f}h")
            rec = {"step": step, "loss": loss_sum, "ema": running_loss, "lr": lr,
                   "grad_norm": float(grad_norm), "tok_per_sec": tok_per_sec, "mfu": mfu}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if use_wandb:
                import wandb
                wandb.log(rec, step=step)

        # ---- eval -----------------------------------------------------------------
        if step > 0 and step % cfg.train.eval_every == 0:
            val_loss = evaluate(model, val_ds, cfg.train.batch_size,
                                cfg.train.eval_batches, ctx)
            print(f"  >> val loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}"
                  f"{'  * best' if val_loss < best_val else ''}")
            logf.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            logf.flush()
            if use_wandb:
                import wandb
                wandb.log({"val_loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "ckpt_best.pt", model, optimizer, cfg, step, best_val)
            t0 = time.time()  # don't bill eval time to the next step's throughput

        if step > 0 and step % cfg.train.sample_every == 0:
            print("  >> sample:", repr(sample_text(model, tok, "Once upon a time", 80, device)))
            t0 = time.time()

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, cfg, step, best_val)
            t0 = time.time()

        # ---- honour a stop request ------------------------------------------------
        if stop["now"] or stop_file.exists():
            print(f"[stop] saving ckpt_last.pt at step {step} and exiting")
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, cfg, step, best_val)
            if stop_file.exists():
                stop_file.unlink()  # so the next run doesn't stop immediately
            logf.close()
            print(f"[stop] done. resume with the same command "
                  f"(resume:auto picks up step {step + 1}).")
            return

    # final
    val_loss = evaluate(model, val_ds, cfg.train.batch_size, cfg.train.eval_batches, ctx)
    print(f"\nfinal val loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}")
    save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, cfg,
                    cfg.train.max_steps - 1, min(best_val, val_loss))
    if val_loss < best_val:
        save_checkpoint(out_dir / "ckpt_best.pt", model, optimizer, cfg,
                        cfg.train.max_steps - 1, val_loss)
    logf.close()
    print(f"checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
