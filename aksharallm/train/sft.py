"""Supervised fine-tuning: turn a base model into one that answers questions.

Differences from pretraining, all of them deliberate:

  loss masking     only assistant tokens count (that's the whole point)
  lower LR         ~10-30x lower. The base model already knows language; we're adjusting
                   style, not re-learning it. A pretrain-sized LR erases the pretraining.
  fewer steps      1-3 epochs. SFT datasets are small and overfit fast.
  dropout on       now we *are* overfitting-limited rather than data-limited.
  shuffled epochs  we iterate the dataset rather than sampling random windows.

Adapters
--------
`--lora` trains a low-rank correction instead of the weights, and `--qlora` additionally
holds the frozen base in 4 bits. Both change what is saved: an adapter file of a few MB
next to the base, rather than a full checkpoint. Everything else in this file -- the loss
mask, the schedule, the evaluation -- is identical either way, which is the point.
See `aksharallm/lora/` and `docs/11-lora.md`.

Read with: docs/05-posttraining.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/11-lora.md.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig, config_to_dict, load_config
from ..lora import setup as lora_setup
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from . import stopfile
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
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-5 for full fine-tuning, 2e-4 with --lora. The gap is "
                         "real: the adapter starts at zero and has ~1%% of the parameters, "
                         "so a full-fine-tuning LR barely moves it.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Bounded stops, the same contract pretraining uses (aksharallm/train/stopfile.py). The
    # file is NOT called STOP: SFT writes into the base model's run directory, where a file
    # by that name would be read by a pretraining run sharing it.
    ap.add_argument("--stop-file", default=None,
                    help="poll this file for a stop request (default <out-dir>/SFT_STOP). "
                         "Empty = stop now, a number = stop at that step, @<epoch> = stop "
                         "at that wall-clock time.")
    ap.add_argument("--stop-in", default=None, metavar="DURATION",
                    help="train for this long, then save and exit: 30m / 90s / 2h / 1h30m, "
                         "or a bare number read as minutes.")
    lora_setup.add_lora_args(ap)
    args = ap.parse_args()
    use_lora = lora_setup.wants_lora(args)
    if args.lr is None:
        args.lr = 2e-4 if use_lora else 1e-5

    sys.stdout.reconfigure(line_buffering=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load the base model ------------------------------------------------------
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    mcfg.dropout = args.dropout  # re-enable dropout for fine-tuning
    model = Transformer(mcfg)
    lora_setup.rebuild_quantized_shapes(model, ckpt)
    model.load_state_dict(ckpt["model"])

    # Quantize on the CPU *before* moving: doing it after would put a full float copy in
    # VRAM first, which on the 300M model is the difference between 1.2 GB and 0.2 GB at
    # the peak — and the peak is what makes a run fit or not.
    lora_notes = lora_setup.prepare_base(model, ckpt, args, args.device) if use_lora else []
    model = model.to(args.device)
    lora_config = lora_report = None
    if use_lora:
        lora_config, lora_report, notes = lora_setup.attach(model, args, args.device)
        lora_notes += notes

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
    if lora_report is not None:
        print("-" * 78)
        print(lora_report.summary())
        print(lora_setup.memory_line(model))
        for n in lora_notes:
            print(f"note       {n}")
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

    # ---- stopping early ------------------------------------------------------------
    # A fine-tune is minutes to hours, not days, but it is stopped for the same reasons and
    # the machinery is the same: a signal, or a request in a file. Whichever arrives, the
    # loop breaks out to the tail below, which evaluates and saves `sft_last`/`sft_best` --
    # so a stopped fine-tune still leaves a usable adapter rather than nothing.
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "SFT_STOP"
    stop_by = run_t0 + stopfile.parse_duration(args.stop_in) if args.stop_in else None
    stop = {"now": False}

    def _request_stop(signum, frame):
        if stop["now"]:
            print("\n[stop] second signal -- exiting immediately (this step is lost)")
            raise KeyboardInterrupt
        print(f"\n[stop] signal {signum} received -- will save and exit after this step")
        stop["now"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    print(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")
    if stop_by is not None:
        print(f"budget  {fmt_dur(stop_by - run_t0)} of training, then save and exit")

    why = None
    announced = None  # the stop request already printed, so it is logged once, not per step
    for epoch in range(args.epochs):
        if why:
            break
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

            # Decided before the log line, so the step you stop on always gets logged.
            request = None if stop["now"] else stopfile.read(stop_file)
            if stop["now"]:
                why = "signal"
            else:
                why = stopfile.reached(request, step)
                if why is None and request is not None and request != announced:
                    announced = request
                    print(f"[stop] {stop_file.name} asks to {request.describe(step)}")
            if why is None and stop_by is not None and time.time() >= stop_by:
                why = f"reached the {fmt_dur(stop_by - run_t0)} budget for this fine-tune"

            if step % args.log_every == 0 or why:
                dt = time.time() - t0
                t0 = time.time()
                s_per_step = dt / max(1, step - prev_log_step)   # measured, not assumed
                prev_log_step = step
                up = time.time() - run_t0
                eta = (max_steps - step) * s_per_step
                deadlines = [d for d in (stop_by, request.deadline if request else None)
                             if d is not None]
                if deadlines:
                    eta = min(eta, max(0.0, min(deadlines) - time.time()))
                print(f"[{stamp()}] epoch {epoch} step {step:>5}/{max_steps} | "
                      f"loss {loss_sum:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                      f"{s_per_step:.2f}s/step | up {fmt_dur(up)} | "
                      f"eta {fmt_dur(eta)}")
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
                    _save(out_dir, "best", model, optimizer, ckpt, mcfg, args, step,
                          best_val, use_lora, lora_config, lora_report)
                t0 = time.time()
            step += 1
            if why:
                print(f"[stop] {why} -- evaluating and saving at step {step}")
                break

    vl = evaluate(model, val_ds, args.batch_size, 20, ctx)
    print(f"\nfinal val {vl:.4f}")
    _save(out_dir, "last", model, optimizer, ckpt, mcfg, args, step, min(best_val, vl),
          use_lora, lora_config, lora_report)
    if vl < best_val:
        _save(out_dir, "best", model, optimizer, ckpt, mcfg, args, step, vl,
              use_lora, lora_config, lora_report)
    # Clear the request now that it has been honoured: a stop file left behind would end the
    # *next* fine-tune at step 0, and that failure looks like a broken script, not a stale file.
    if why and stop_file.exists():
        stop_file.unlink(missing_ok=True)
    print(f"ran {step} steps in {fmt_dur(time.time() - run_t0)}"
          f"{f' (stopped early: {why})' if why else ''}, "
          f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"output in {out_dir}")
    if use_lora:
        print(f"\ntry it:  python -m aksharallm.infer.cli {args.base} "
              f"--adapter {out_dir}/sft_best.lora.pt --mode chat")
    else:
        print(f"\ntry it:  python -m aksharallm.infer.cli {out_dir}/sft_best.pt --mode chat")
    logf.close()


def _save(out_dir: Path, which: str, model, optimizer, ckpt, mcfg, args, step, val,
          use_lora: bool, lora_config, lora_report):
    """Write an adapter (~MB) or a full checkpoint (~GB), depending on how we trained.

    The naming is deliberately different -- `sft_best.lora.pt` rather than `sft_best.pt` --
    because the two are not interchangeable. A full checkpoint is a model; an adapter is
    useless without the base it names in its metadata, and a filename that hid that
    difference would be an easy way to lose a model.
    """
    if use_lora:
        return lora_setup.save(
            out_dir / f"sft_{which}.lora.pt", model, lora_config, ckpt, args.base,
            report=lora_report,
            training={"stage": "sft", "step": step, "val_loss": val, "lr": args.lr,
                      "epochs": args.epochs, "data_dir": args.data_dir})
    cfg_obj = _rebuild_cfg(ckpt, mcfg, args)
    return save_checkpoint(out_dir / f"sft_{which}.pt", model, optimizer, cfg_obj, step, val)


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
