"""Train G1, the caption-to-picture transformer, on the tokenizer's codes.

The same contract as every trainer here (`train.pid`, `STOP`, session records,
`ckpt_last.pt` / `ckpt_best.pt`, resume by default), launched by `scripts/audio.sh <run>`
for a config with a `picgen:` section.

**What to watch:**

1. **val loss** is the cross-entropy per code (captions and the IMG marker carry no loss).
   Step 0 reads ~ln(vocab) = 6.95 (uniform over words and both codebooks); ln(512) = 6.24 is
   a model that has learned only which codebook each slot holds.
2. **`seen all`** at every sample: the reader's verdict on pictures drawn from all 56 seen
   captions. Loss can fall for a long time on the background before a single shape is right.
3. **`held-out all`**: the four purple-triangle captions, never drawn in training. This is
   the number the experiment exists for.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from ..config import OptimConfig, config_to_dict, load_into
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .codes import CodesDataset
from .gen import PicGenConfig, build


@dataclass
class PicGenDataConfig:
    #: `pictures codes` output: codes.npy + meta.json, written from the tokenizer.
    codes: str = "data/pictures/codes-imgtok-shapes"


@dataclass
class PicGenTrainConfig:
    out_dir: str = "checkpoints/imggen-shapes"
    batch_size: int = 64
    max_steps: int = 8000
    eval_every: int = 500
    eval_batches: int = 8
    #: Draw `sample_k` pictures for each of the 60 captions and read them back.
    sample_every: int = 2000
    sample_k: int = 2
    sample_guidance: float = 3.0
    ckpt_every: int = 1000
    log_every: int = 50
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class PicGenRunConfig:
    name: str = "imggen-shapes"
    picgen: PicGenConfig = field(default_factory=PicGenConfig)
    data: PicGenDataConfig = field(default_factory=PicGenDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: PicGenTrainConfig = field(default_factory=PicGenTrainConfig)


def load_picgen_config(path, overrides=None) -> PicGenRunConfig:
    return load_into(PicGenRunConfig, path, overrides)


@torch.no_grad()
def evaluate(model, ds, device, batch_size, batches, amp) -> float:
    was = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = ds.batch(batch_size)
        with amp():
            _, loss = model(x.to(device), targets=y.to(device))
        total += float(loss)
    model.train(was)
    return total / max(batches, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.pictures.train_gen",
                                description="Train G1, caption -> picture codes (docs/26).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_picgen_config(args.config, args.override)
    pg = cfg.picgen
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda"
           else nullcontext)

    train_ds = CodesDataset(cfg.data.codes, books=pg.books, split="train", seed=cfg.train.seed)
    val_ds = CodesDataset(cfg.data.codes, books=pg.books, split="val", seed=cfg.train.seed + 1)
    meta = train_ds.meta
    if Path(meta["tokenizer"]).resolve() != Path(pg.tokenizer).resolve():
        raise SystemExit(f"{cfg.data.codes} was encoded by {meta['tokenizer']}, but the config "
                         f"names {pg.tokenizer}: the codes and the decoder must be the same "
                         "tokenizer's")
    model = build(pg, meta["codebook_size"], meta["grid"]).to(device)

    print(f"=== {cfg.name} ===")
    print(f"model      {model.num_params() / 1e6:.2f}M params, d {pg.d_model} x {pg.n_layers} "
          f"layers, vocab {model.cfg.vocab_size}")
    print(f"sequence   3 words + IMG + {pg.books} codebooks x {meta['grid']}x{meta['grid']} = "
          f"{train_ds.seq_len} tokens; caption dropped {pg.cfg_drop:.0%} of the time "
          "(for guidance)")
    print(f"codes      {cfg.data.codes} (tokenizer {meta['tokenizer']}, step "
          f"{meta['tokenizer_step']})")
    print(f"corpus     {len(train_ds):,} train / {len(val_ds):,} val pictures")
    V = model.cfg.vocab_size
    print(f"device     {device}   (step 0 should read ~ln {V} = {math.log(V):.4f}, uniform over "
          f"the vocabulary; a model that has learned only which codebook goes where reads "
          f"ln {meta['codebook_size']} = {math.log(meta['codebook_size']):.4f})")

    opt, _ = model.configure_optimizers(cfg.optim.weight_decay, cfg.optim.lr,
                                        (cfg.optim.beta1, cfg.optim.beta2), device,
                                        name=cfg.optim.name)

    start_step, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = str(ckpt_last) if cfg.train.resume == "auto" and ckpt_last.is_file() else (
        cfg.train.resume if cfg.train.resume != "auto" else None)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            opt.load_state_dict(saved["optimizer"])
        if "data_rng" in saved:
            train_ds.rng.bit_generator.state = saved["data_rng"]
        start_step = int(saved.get("step", -1)) + 1
        best = float(saved.get("best_val", float("inf")))
        del saved
        print(f"resumed {resume} at step {start_step}")

    claim_pid_file(out_dir)
    stop_file = out_dir / "STOP"
    stop_now = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop_now.__setitem__("now", True))
    logf = open(out_dir / "train_log.jsonl", "a")

    def log_session(event: str, **kw):
        logf.write(json.dumps({"event": event, "time": time.time(),
                               "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               "run": cfg.name, **kw}) + "\n")
        logf.flush()

    stop_at = cfg.train.stop_at
    if cfg.train.stop_after is not None:
        stop_at = start_step + cfg.train.stop_after - 1
    run_t0 = time.time()
    stop_by = None if cfg.train.stop_after_s is None else run_t0 + cfg.train.stop_after_s
    log_session("session_start", pid=os.getpid(), start_step=start_step,
                max_steps=cfg.train.max_steps, stop_at=stop_at, stop_by=stop_by,
                params=model.num_params(), objective="picgen", metric="ce per code",
                batch_size=cfg.train.batch_size)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    def save(path: Path, step: int, **extra):
        torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                    "picgen": asdict(pg), "codes_meta": meta, "config": config_to_dict(cfg),
                    "step": step, "best_val": best, "stage": "picgen",
                    "data_rng": train_ds.rng.bit_generator.state, **extra}, path)

    tokenizer = None

    def sample(step: int) -> dict:
        nonlocal tokenizer
        from .score import report
        from .train import load_tokenizer
        if tokenizer is None:
            tokenizer, _ = load_tokenizer(pg.tokenizer, device)
        r = report(model, tokenizer, pg, train_ds=train_ds, k=cfg.train.sample_k,
                   guidance=cfg.train.sample_guidance, seed=0, device=device)
        s = {"seen_all": r["seen"]["all"], "held_out_all": r["held_out"]["all"],
             "copies": r["seen"]["copies"], "variety": r["variety"],
             "guidance": cfg.train.sample_guidance}
        print(f"           drawn at guidance {cfg.train.sample_guidance:g}: seen all "
              f"{s['seen_all'] * 100:.0f}%  |  held-out all {s['held_out_all'] * 100:.0f}%  |  "
              f"copies {s['copies'] * 100:.0f}%  variety {s['variety'] * 100:.0f}%")
        model.train()
        return s

    model.train()
    t0 = time.time()
    ema, why, step = None, None, start_step
    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_ds.batch(cfg.train.batch_size, drop=pg.cfg_drop)
        with amp():
            _, loss = model(x.to(device), targets=y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()
        lv = float(loss.detach())
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv

        if cfg.train.log_every and (step % cfg.train.log_every == 0
                                    or step == cfg.train.max_steps - 1):
            dt = time.time() - t0
            n_steps = max(cfg.train.log_every, 1) if step > start_step else 1
            logf.write(json.dumps({"step": step, "loss": lv, "ema": ema, "lr": lr,
                                   "grad_norm": float(grad_norm), "s_per_step": dt / n_steps,
                                   "time": time.time(), "elapsed": time.time() - run_t0}) + "\n")
            logf.flush()
            tag = (f"  <- uniform is {math.log(model.cfg.vocab_size):.4f}"
                   if step == start_step == 0 else "")
            print(f"step {step:>6}  loss {lv:.4f} (ema {ema:.4f})  lr {lr:.2e}  "
                  f"gnorm {float(grad_norm):.2f}  {dt / n_steps:.3f}s/step{tag}")
            t0 = time.time()

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            val = evaluate(model, val_ds, device, cfg.train.batch_size, cfg.train.eval_batches, amp)
            improved = val < best
            if improved:
                best = val
                save(out_dir / "ckpt_best.pt", step)
            logf.write(json.dumps({"step": step, "time": time.time(), "val_loss": val}) + "\n")
            logf.flush()
            print(f"           val {val:.4f}{'  *best*' if improved else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            s = sample(step)
            logf.write(json.dumps({"step": step, "time": time.time(), "score": s}) + "\n")
            logf.flush()
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            save(ckpt_last, step, optimizer=opt.state_dict())
            t0 = time.time()

        request = None if stop_now["now"] else stopfile.read(stop_file)
        if stop_now["now"]:
            why = "SIGTERM"
        elif request is not None and (r := stopfile.reached(request, step)):
            why = r
        elif stop_at is not None and step >= stop_at:
            why = f"stop_at/stop_after reached step {stop_at}"
        elif stop_by is not None and time.time() >= stop_by:
            why = "wall-clock budget spent"
        if why:
            break

    save(ckpt_last, step, optimizer=opt.state_dict())
    val = evaluate(model, val_ds, device, cfg.train.batch_size, cfg.train.eval_batches, amp)
    if val < best:
        best = val
        save(out_dir / "ckpt_best.pt", step)
    final = sample(step) if cfg.train.sample_k else None
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0,
                final_val_loss=val, best_val=best, score=final)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}")
    print(f"    measure: python -m aksharallm.pictures gen-report {cfg.name}")
    print(f"    draw:    python -m aksharallm.pictures draw {cfg.name} \"two red circles\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
