"""Train G2, the flow-matching picture model, on pixels.

The same contract as every trainer here (`train.pid`, `STOP`, session records,
`ckpt_last.pt` / `ckpt_best.pt`, resume by default), launched by `scripts/audio.sh <run>`
for a config with a `flow:` section.

**What to watch:**

1. **val loss** is the velocity MSE on a FIXED set of val pictures, noises and times (seeded),
   so two evaluations differ only because the model did. Predicting zero velocity scores
   about 1 + E[x1^2] (the output layer starts at zero, so step 0 reads that).
2. **`seen all` / `held-out all`** at every sample: the reader on drawings of all 60
   captions, exactly as G1 is scored, so the two are compared drawing for drawing.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..config import OptimConfig, config_to_dict, load_into
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from ..vision.image import ImageCaptions
from .codes import NULL, CodesDataset, word_ids
from .flow import FlowConfig, FlowModel, flow_loss


@dataclass
class FlowDataConfig:
    corpus: str = "data/vision/shapes64k"
    #: G3: the same corpus as normalised tokenizer latents (`pictures latents`). Used when
    #: `flow.space` is "latent"; the corpus above still supplies the captions.
    latents: str = "data/pictures/latents-imgtok-shapes-64k"
    #: The same corpus as codes (`pictures codes`): only for the copy check when scoring.
    codes: str = "data/pictures/codes-imgtok-shapes-64k"


@dataclass
class FlowTrainConfig:
    out_dir: str = "checkpoints/flow-shapes"
    batch_size: int = 64
    max_steps: int = 30000
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
class FlowRunConfig:
    name: str = "flow-shapes"
    flow: FlowConfig = field(default_factory=FlowConfig)
    data: FlowDataConfig = field(default_factory=FlowDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: FlowTrainConfig = field(default_factory=FlowTrainConfig)


def load_flow_config(path, overrides=None) -> FlowRunConfig:
    return load_into(FlowRunConfig, path, overrides)


def batch_of(ds, n: int, drop: float, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """Random training examples (pictures, or G3 latents) and their caption ids, `drop` of
    captions replaced by NULL. Both datasets' `batch` return `(float array, captions)`."""
    imgs, caps = ds.batch(n)
    words = np.asarray([word_ids(c) for c in caps], dtype=np.int64)
    if drop > 0:
        words[rng.random(n) < drop] = NULL
    return torch.from_numpy(imgs), torch.from_numpy(words)


def rows(ds, idx: list[int]) -> tuple[np.ndarray, list[str]]:
    """Fixed examples by index, from either dataset."""
    if hasattr(ds, "rows"):
        return ds.rows(idx)
    from ..vision.image import to_tensor
    return to_tensor(np.asarray(ds.images[idx])), [ds.manifest.captions[j] for j in idx]


@torch.no_grad()
def evaluate(model, ds, device, batch_size, batches, amp) -> float:
    """Velocity MSE on the FIRST `batch_size * batches` val pictures, with noise and times
    from a fixed seed: a moving target would let "best" record the kindest noise."""
    was = model.training
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total, n = 0.0, 0
    for b in range(batches):
        idx = [ds.index[i % len(ds)] for i in range(b * batch_size, (b + 1) * batch_size)]
        x, caps = rows(ds, idx)
        x1 = torch.from_numpy(x).to(device)
        words = torch.tensor([word_ids(c) for c in caps], device=device)
        with amp():
            total += float(flow_loss(model, x1, words, gen))
        n += 1
    model.train(was)
    return total / max(n, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.pictures.train_flow",
                                description="Train G2, flow matching in pixels (docs/26).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_flow_config(args.config, args.override)
    fc = cfg.flow
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda"
           else nullcontext)

    if fc.space == "latent":
        # G3: the tokenizer's normalised latents. Their shape and statistics come from the
        # data, and are written into the config so every checkpoint carries them.
        from .latents import LatentDataset
        train_ds = LatentDataset(cfg.data.latents, split="train", seed=cfg.train.seed)
        val_ds = LatentDataset(cfg.data.latents, split="val", seed=cfg.train.seed + 1)
        meta = train_ds.meta
        if Path(meta["tokenizer"]).resolve() != Path(fc.tokenizer).resolve():
            raise SystemExit(f"{cfg.data.latents} was encoded by {meta['tokenizer']}, but the "
                             f"config names {fc.tokenizer}: the latents and the decoder must be "
                             "the same tokenizer's")
        fc.channels, fc.image_size = meta["shape"][0], meta["shape"][1]
        fc.latent_mean, fc.latent_std = meta["mean"], meta["std"]
    elif fc.space == "pixels":
        train_ds = ImageCaptions(cfg.data.corpus, split="train", seed=cfg.train.seed)
        val_ds = ImageCaptions(cfg.data.corpus, split="val", seed=cfg.train.seed + 1)
    else:
        raise SystemExit(f"flow.space is 'pixels' or 'latent', not {fc.space!r}")
    drop_rng = np.random.default_rng(cfg.train.seed + 2)
    model = FlowModel(fc).to(device)

    print(f"=== {cfg.name} ===")
    print(f"model      {model.n_params() / 1e6:.2f}M params, d {fc.d_model} x {fc.n_layers} "
          f"layers, {fc.space}: {fc.channels}x{fc.image_size}x{fc.image_size} as "
          f"{fc.grid}x{fc.grid} patches of {fc.patch}")
    print(f"sequence   time + 3 words + {fc.grid * fc.grid} patches, bidirectional; caption "
          f"dropped {fc.cfg_drop:.0%} of the time (for guidance)")
    src = cfg.data.latents if fc.space == "latent" else cfg.data.corpus
    print(f"corpus     {len(train_ds):,} train / {len(val_ds):,} val pictures ({src})")
    print(f"device     {device}   (step 0 predicts zero velocity: loss ~ 1 + mean(x1^2))")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)

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
        if "drop_rng" in saved:
            drop_rng.bit_generator.state = saved["drop_rng"]
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
                params=model.n_params(), objective="flow", metric="velocity mse",
                batch_size=cfg.train.batch_size)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    def save(path: Path, step: int, **extra):
        torch.save({"model": model.state_dict(), "flow": asdict(fc),
                    "config": config_to_dict(cfg), "step": step, "best_val": best,
                    "stage": "flow", "data_rng": train_ds.rng.bit_generator.state,
                    "drop_rng": drop_rng.bit_generator.state, **extra}, path)

    tokenizer = None
    scorer = [None]

    def sample(step: int) -> dict:
        nonlocal tokenizer
        from .codes import all_captions
        from .flow import draw
        from .score import judge_drawings, tokens_of, train_code_set
        from .train import load_coder, load_tokenizer
        if tokenizer is None:
            tokenizer, _ = load_coder(fc.tokenizer, device)
            scorer[0] = (load_tokenizer(fc.score_tokenizer, device)[0] if fc.score_tokenizer
                         else tokenizer)
        caps = all_captions()
        k = cfg.train.sample_k
        pics = draw(model, [c for c in caps for _ in range(k)],
                    guidance=cfg.train.sample_guidance, seed=0, device=device,
                    tokenizer=tokenizer)
        seen = (train_code_set(CodesDataset(cfg.data.codes, books=fc.books, split="train"))
                if Path(cfg.data.codes, "codes.npy").is_file() else set())
        r = judge_drawings(caps, k, pics, tokens_of(pics, scorer[0], fc.books, device), seen)
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
        x1, words = batch_of(train_ds, cfg.train.batch_size, fc.cfg_drop, drop_rng)
        with amp():
            loss = flow_loss(model, x1.to(device), words.to(device))
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
            tag = "  <- zero velocity" if step == start_step == 0 else ""
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
    print(f"    measure: python -m aksharallm.pictures gen-report {cfg.name} --guidance 3")
    print(f"    draw:    python -m aksharallm.pictures draw {cfg.name} \"two red circles\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
