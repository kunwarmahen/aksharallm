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
import atexit
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ..config import Config, config_to_dict, load_config
from ..data.loader import MixedTokenDataset, TokenDataset
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from . import stopfile
from .schedule import get_lr


def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def fmt_dur(seconds: float) -> str:
    """Compact duration: 45.2s / 12m30s / 6h05m / 3d04h.

    Multi-day runs are read at a glance from a log file days later, so every timing we
    print goes through this: "2d04h" is instantly meaningful, "187214.6s" is not.
    """
    if seconds < 0:
        return "-" + fmt_dur(-seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def stamp() -> str:
    """Wall-clock HH:MM:SS, prefixed to every step line."""
    return datetime.now().strftime("%H:%M:%S")


def resolve_stop_step(start_step: int, stop_after: int | None,
                      stop_at: int | None) -> int | None:
    """The last step a bounded stop will train, or None for "run to max_steps".

    Both bounds are INCLUSIVE: the step returned is trained, logged and checkpointed, and
    the resume picks up the one after it. That is deliberately not `max_steps` semantics
    (`max_steps=N` makes the last step N-1) -- asking to stop at step 700 and finding the
    checkpoint at 699, with no step-700 line in the log, is a surprise every time.
    """
    if stop_after is not None:
        after = start_step + stop_after - 1
        stop_at = after if stop_at is None else min(stop_at, after)
    if stop_at is not None and stop_at < start_step:
        raise ValueError(
            f"stop_at/stop_after resolves to step {stop_at}, but this run starts at "
            f"{start_step} -- there is nothing to do. Raise it or drop the flag."
        )
    return stop_at


def claim_pid_file(out_dir: Path) -> Path:
    """Record "this process is training into this directory" in `<out_dir>/train.pid`.

    The pid belongs to the *run directory*, not to a command line. That distinction is
    load-bearing: the 50-step smoke test runs the identical command line with a throwaway
    `out_dir`, so anything that identifies a run by `pgrep -f "pretrain configs/x.yaml"`
    will happily find the smoke test and aim a stop request at it. Writing the pid here
    means `scripts/stop.sh` and the portal both get an unambiguous answer, whoever launched
    the run -- `phase2.sh`, the portal, or a bare command in a terminal.

    Released on any clean exit (including a stop or a crash), so a missing file really does
    mean "nothing is training here". A `kill -9` leaves it behind; readers check liveness.
    """
    path = out_dir / "train.pid"
    path.write_text(f"{os.getpid()}\n")

    def release():
        try:
            if int(path.read_text().strip()) == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass  # someone else's pid, or already gone: leave it alone

    atexit.register(release)
    return path


def stop_file_target(path: Path) -> int | None:
    """The step a STOP file asks for, or None for "stop now" / no step in it.

    Kept as a thin reading of `stopfile.read` because a step is what most callers want.
    The full contract -- empty, a step, or an `@epoch` deadline -- lives in
    `aksharallm.train.stopfile`, which the SFT and QAT loops share with this one.
    """
    req = stopfile.read(path)
    return req.step if req else None


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
    claim_pid_file(out_dir)

    # ---- data --------------------------------------------------------------------
    if cfg.data.train_sources:
        train_ds = MixedTokenDataset(cfg.data.train_sources, cfg.train.seq_len, device)
        print(f"blended training: {train_ds}")
    else:
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
    stop_at = resolve_stop_step(start_step, cfg.train.stop_after, cfg.train.stop_at)
    if stop_at is not None:
        print(f"stop         after step {stop_at:,} "
              f"({stop_at - start_step + 1:,} steps this run, then save and exit)")
    if cfg.train.stop_after_s is not None:
        print(f"stop         after {fmt_dur(cfg.train.stop_after_s)} of training "
              "(measured from the first step, so pre-flight and compilation are free)")
    print(f"started      {datetime.now():%Y-%m-%d %H:%M:%S}")
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
    # A 6-day run needs to be stoppable without losing progress. Ways to stop:
    #   Ctrl-C, or `kill <pid>` (SIGTERM)  -> finishes the current step, then saves
    #   `touch <out_dir>/STOP`             -> same, but doesn't need the terminal
    #   `echo N > <out_dir>/STOP`          -> keeps going, then stops at step N
    #   `echo @<epoch> > <out_dir>/STOP`   -> keeps going, then stops at that wall-clock time
    #   train.stop_after / train.stop_at   -> the same step bound, decided at launch
    #   train.stop_after_s                 -> a time budget for this session, ditto
    # In every case we save ckpt_last.pt at the *exact* current step and exit cleanly, so
    # rerunning the same command with resume:auto picks up with zero lost work.
    # (scripts/stop.sh drives all of these from the pid file phase2.sh writes.)
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

    # train_log.jsonl is append-only across sessions, so a run trained over many evenings
    # ends up as one file. Bracketing each launch with session_start/session_end records is
    # what makes those sessions separable afterwards -- otherwise you cannot tell a resume
    # from a throughput change. scripts/sessions.py reads exactly these markers.
    logf = open(out_dir / "train_log.jsonl", "a")
    model.train()
    t0 = time.time()  # start of the current log window (reset after eval/sample/ckpt)
    run_t0 = t0  # start of this invocation; never reset, so "up" is true wall-clock
    prev_log_step = start_step - 1  # so the first window measures the steps it really covers
    running_loss = None
    # The time budget starts here, not at launch: "run it for 30 minutes" means 30 minutes
    # of training, and pre-flight plus torch.compile can eat ten of them before step one.
    stop_by = None if cfg.train.stop_after_s is None else run_t0 + cfg.train.stop_after_s
    announced = None  # the STOP request already printed, so a queued stop is logged once

    def log_session(event: str, **kw):
        rec = {"event": event, "time": time.time(),
               "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "run": cfg.name, **kw}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    log_session("session_start", pid=os.getpid(), start_step=start_step,
                max_steps=cfg.train.max_steps, stop_at=stop_at, stop_by=stop_by,
                tokens_per_step=tokens_per_step)

    if start_step >= cfg.train.max_steps:
        # Resuming a finished run. It is not an error -- the checkpoint is intact and this
        # exits without touching it -- but "ran 0 steps" after a full pre-flight looks like
        # a launch that failed, so say which of the two it is.
        print(f"\nnothing to do: this run has already trained its full budget of "
              f"{cfg.train.max_steps:,} steps (the last was {cfg.train.max_steps - 1:,}).")
        print("to keep training it, raise train.max_steps:\n"
              f"    python -m aksharallm.train.pretrain {args.config} "
              f"-o train.max_steps={cfg.train.max_steps * 2}")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0,
                    elapsed=time.time() - run_t0, final_val_loss=best_val)
        logf.close()
        return

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

        # ---- is this the last step? -------------------------------------------------
        # Decided *before* logging, so a bounded stop's final step always gets a log line
        # even when it isn't a multiple of log_every. Stopping at 699 with log_every=50
        # used to leave no loss or throughput reading at the step you stopped on.
        # Four ways to be asked to stop, checked in the order they take effect: a signal,
        # the STOP file (empty / a step / an `@epoch` deadline), the step bound this launch
        # was given, and its time budget. The file is re-read every step and never copied
        # into `stop_at`, so `stop.sh --cancel` -- which only removes the file -- really does
        # put the run back on the budget it launched with.
        why = None
        request = None if stop["now"] else stopfile.read(stop_file)
        if stop["now"]:
            why = "signal"
        else:
            why = stopfile.reached(request, step)
            if why is None and request is not None and request != announced:
                announced = request
                print(f"[stop] STOP file asks to {request.describe(step)}")
        if why is None and stop_at is not None and step >= stop_at:
            why = f"reached stop step {stop_at}"
        if why is None and stop_by is not None and time.time() >= stop_by:
            why = f"reached this session's {fmt_dur(cfg.train.stop_after_s)} time budget"

        # ---- logging --------------------------------------------------------------
        # `step == max_steps - 1` is there so a run that finishes *normally* gets a line for
        # its final step, the way a bounded stop always has. Without it the last line lands
        # on the last multiple of log_every -- a run of 8,000 steps ends its log at 7,980 and
        # reads, on a dashboard, as though it stopped 20 steps early.
        if (cfg.train.log_every and step % cfg.train.log_every == 0) or why \
                or step == cfg.train.max_steps - 1:
            torch.cuda.synchronize() if device == "cuda" else None
            dt = time.time() - t0
            t0 = time.time()
            # Steps actually covered by this window -- not `log_every`. On a resumed run the
            # first window is a *partial* one (resume at 620, first log at 650 = 31 steps),
            # and assuming a full window there inflates tok/s by 50/31 and reports the
            # impossible "mfu 112%". Measure the window instead of assuming it.
            steps_done = step - prev_log_step
            prev_log_step = step
            tok_per_sec = tokens_per_step * steps_done / dt
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            mfu = raw.estimate_mfu(tok_per_sec)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            s_per_step = dt / steps_done
            up = time.time() - run_t0
            # ETA counts to whichever comes first: the budget, a bounded stop, or a queued
            # one. Both last-step numbers are inclusive (max_steps=N means the last is N-1).
            # A deadline is compared in seconds rather than converted to steps -- it is
            # already the answer the eta is trying to estimate.
            last_step = stop_at if stop_at is not None else cfg.train.max_steps - 1
            if request is not None and request.step is not None:
                last_step = min(last_step, request.step)
            eta = max(last_step - step, 0) * s_per_step
            deadlines = [d for d in (stop_by, request.deadline if request else None)
                         if d is not None]
            if deadlines:
                eta = min(eta, max(0.0, min(deadlines) - time.time()))
            # Routing, for a mixture of experts. Printed on the step line rather than
            # buried in the jsonl because router collapse is the failure this model has
            # that the dense one does not, it starts within the first few hundred steps,
            # and the loss curve does not show it -- a collapsed MoE is simply a slightly
            # worse model. `balance` is 1.0 when every expert gets an equal share and 1/N
            # when one takes everything.
            routing = raw.routing()
            moe_line = ""
            if routing:
                moe_line = (f" | experts {routing['balance']:.2f} bal"
                            f" ({routing['min_share']*100:.0f}-{routing['max_share']*100:.0f}%"
                            + (f", {routing['dead']} dead" if routing["dead"] else "") + ")")
            print(f"[{stamp()}] step {step:>6} | loss {loss_sum:.4f} "
                  f"(ema {running_loss:.4f}) | ppl {math.exp(min(running_loss, 20)):>7.1f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tok_per_sec/1e3:.1f}k tok/s | "
                  f"mfu {mfu*100:.1f}% | {mem:.1f}GB | {s_per_step:.2f}s/step | "
                  f"up {fmt_dur(up)} | eta {fmt_dur(eta)}{moe_line}")
            rec = {"step": step, "loss": loss_sum, "ema": running_loss, "lr": lr,
                   "grad_norm": float(grad_norm), "tok_per_sec": tok_per_sec, "mfu": mfu,
                   "time": time.time(), "s_per_step": s_per_step, "elapsed": up,
                   "eta_s": eta}
            if routing:
                rec["moe"] = routing
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if use_wandb:
                import wandb
                wandb.log(rec, step=step)

        # ---- eval -----------------------------------------------------------------
        if step > 0 and cfg.train.eval_every and step % cfg.train.eval_every == 0:
            te = time.time()
            val_loss = evaluate(model, val_ds, cfg.train.batch_size,
                                cfg.train.eval_batches, ctx)
            print(f"  >> val loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}"
                  f"{'  * best' if val_loss < best_val else ''}"
                  f"  ({fmt_dur(time.time() - te)})")
            logf.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            logf.flush()
            if use_wandb:
                import wandb
                wandb.log({"val_loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "ckpt_best.pt", model, optimizer, cfg, step, best_val)
            t0 = time.time()  # don't bill eval time to the next step's throughput

        # `0` reads as "never" for every one of these cadences, which is what a person
        # writing `sample_every: 0` means. It used to be a ZeroDivisionError on step 1.
        if step > 0 and cfg.train.sample_every and step % cfg.train.sample_every == 0:
            print("  >> sample:", repr(sample_text(model, tok, "Once upon a time", 80, device)))
            t0 = time.time()

        if step > 0 and cfg.train.ckpt_every and step % cfg.train.ckpt_every == 0:
            tc = time.time()
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, cfg, step, best_val)
            print(f"  >> saved ckpt_last.pt at step {step}  ({fmt_dur(time.time() - tc)})")
            t0 = time.time()

        # ---- honour a stop request ------------------------------------------------
        # `why` was decided above, before the log line. Several ways in, one way out:
        # save at the exact current step, then exit 0.
        if why:
            print(f"[stop] {why} -- saving ckpt_last.pt at step {step} and exiting")
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, cfg, step, best_val)
            if stop_file.exists():
                stop_file.unlink()  # so the next run doesn't stop immediately
            log_session("session_end", reason=why, last_step=step,
                        steps=step - start_step + 1, elapsed=time.time() - run_t0)
            logf.close()
            print(f"[stop] ran {step - start_step + 1} steps in "
                  f"{fmt_dur(time.time() - run_t0)}, finished {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"[stop] resume with the same command "
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
    log_session("session_end", reason="max_steps", last_step=cfg.train.max_steps - 1,
                steps=cfg.train.max_steps - start_step, elapsed=time.time() - run_t0,
                final_val_loss=val_loss)
    logf.close()
    print(f"ran {cfg.train.max_steps - start_step} steps in {fmt_dur(time.time() - run_t0)}, "
          f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
