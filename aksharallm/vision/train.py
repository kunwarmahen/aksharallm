"""Training the bridge — the cheapest training run in this repo.

Nothing here trains a language model or a vision encoder from scratch. Stage one of a
LLaVA-style build trains **the projector**, which is two matrices, against a frozen language
model and (optionally) a frozen vision tower. That is why it is minutes rather than hours,
and why "add vision to an existing model" is a fundamentally different proposition from
"train a vision-language model".

It obeys the same contract as every other trainer here — `train.pid`, the `STOP` file,
`train_log.jsonl` bracketed by session records, `ckpt_last.pt` / `ckpt_best.pt` — so
`scripts/stop.sh`, `scripts/sessions.py` and the portal drive it unchanged.

**What to watch:**

1. **the step-0 loss lands near or a little above `ln(vocab_size)`.** Measured here:
   **11.27 against ln(8192) = 9.01**. Above, not equal, and that is expected — an untrained
   projector feeds the frozen language model vectors from nowhere in its input distribution,
   so it is briefly *worse* than uniform. What matters is that it falls immediately; a loss
   that starts at 9.0 and sits there means the image tokens are being ignored.
2. **`all_three`**, not the loss. The loss falls smoothly while the model learns to emit the
   *shape* of a caption without getting the facts right; `score_batch` asks whether it named
   the count, the colour and the shape, and reports them separately because a model that
   never counts is a specific, diagnosable failure that one accuracy would average away.
3. **the held-out combination.** The corpus deliberately omits one (colour, shape) pair.
   Scoring on it is the compositional question, and it is the only number here that a model
   cannot reach by memorising.

Read with: docs/21-vision.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from ..config import ModelConfig, OptimConfig, config_to_dict, load_into
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .encoder import VisionConfig
from .image import ImageCaptions
from .lm import VisionLanguageModel, caption, score_batch


@dataclass
class VisionDataConfig:
    corpus: str = "data/vision/shapes"
    holdout: str = "data/vision/shapes-holdout"
    tokenizer: str = "data/tinystories/tokenizer.json"
    max_caption_tokens: int = 16


@dataclass
class VisionTrainConfig:
    out_dir: str = "checkpoints/vision-shapes"
    base: str = "checkpoints/tiny/ckpt_best.pt"
    freeze_language_model: bool = True
    freeze_encoder: bool = False
    batch_size: int = 32
    max_steps: int = 3000
    eval_every: int = 250
    eval_batches: int = 8
    #: Caption a few held-out images and score them. The number that actually says whether
    #: it works, and it is cheap because greedy decoding of 16 tokens is nothing.
    sample_every: int = 500
    sample_images: int = 32
    ckpt_every: int = 1000
    log_every: int = 25
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class VisionRunConfig:
    name: str = "vision-shapes"
    vision: VisionConfig = field(default_factory=VisionConfig)
    data: VisionDataConfig = field(default_factory=VisionDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: VisionTrainConfig = field(default_factory=VisionTrainConfig)


def load_vision_config(path, overrides=None) -> VisionRunConfig:
    return load_into(VisionRunConfig, path, overrides)


def make_batch(model: VisionLanguageModel, tok: Tokenizer, images, captions, device: str,
               max_tokens: int):
    """`(images, text, targets)` — the caption padded, with `-100` where the loss must not look."""
    rows = [tok.encode(c, bos=True)[:max_tokens] + [tok.eos_id] for c in captions]
    width = max(len(r) for r in rows)
    text = torch.full((len(rows), width), tok.eos_id, dtype=torch.long)
    targets = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        text[i, : len(r)] = torch.tensor(r)
        # **Targets are NOT shifted here, and that is the whole of a bug worth remembering.**
        # `VisionLanguageModel.forward` already shifts, by starting the text slice one
        # position early so the LAST IMAGE token is what predicts caption token 0. Shifting
        # again in the targets — the ordinary `r[1:]` that every other trainer in this repo
        # uses — stacks two shifts, and the model then learns to emit the token *after* next.
        # It trains beautifully (loss 0.003) and generates `'w green'`, because generation
        # reads the last position expecting the next token. Trains fine, generates garbage:
        # gotcha #2's family, and only `score_batch` caught it.
        targets[i, 1 : len(r)] = torch.tensor(r[1:])
    return (torch.from_numpy(images).to(device), text.to(device), targets.to(device))


@torch.no_grad()
def evaluate(model, tok, ds, device, batch_size, batches, max_tokens) -> float:
    was = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        images, caps = ds.batch(batch_size)
        img, text, targets = make_batch(model, tok, images, caps, device, max_tokens)
        _, loss = model(img, text, targets=targets)
        total += float(loss)
    model.train(was)
    return total / max(batches, 1)


@torch.no_grad()
def score(model, tok, ds, device, n: int, max_tokens: int) -> dict:
    was = model.training
    model.eval()
    pairs = []
    for i in range(min(n, len(ds))):
        image, _, fact = ds.item(i)
        said = caption(model, torch.from_numpy(image), tok,
                       max_tokens=max_tokens + 2, device=device)
        pairs.append((fact, said))
    model.train(was)
    out = score_batch(pairs)
    out["example"] = {"truth": pairs[0][0], "said": pairs[0][1]} if pairs else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m aksharallm.vision.train",
        description="Train the vision projector against a frozen language model (docs/21).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_vision_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(cfg.train.base, map_location=device, weights_only=False)
    if "model_config" not in blob:
        raise SystemExit(f"{cfg.train.base} is not a language-model checkpoint")
    lm = Transformer(ModelConfig(**blob["model_config"])).to(device)
    lm.load_state_dict(blob["model"])
    tok = Tokenizer(cfg.data.tokenizer)

    model = VisionLanguageModel(
        lm, cfg.vision,
        freeze_language_model=cfg.train.freeze_language_model,
        freeze_encoder=cfg.train.freeze_encoder,
    ).to(device)

    train_ds = ImageCaptions(cfg.data.corpus, split="train", seed=cfg.train.seed)
    val_ds = ImageCaptions(cfg.data.corpus, split="val", seed=cfg.train.seed + 1)
    holdout = None
    if Path(cfg.data.holdout, "manifest.json").is_file():
        holdout = ImageCaptions(cfg.data.holdout, split="train", val_frac=0.0,
                                seed=cfg.train.seed + 2)

    counts = model.n_params()
    uniform = math.log(lm.cfg.vocab_size)
    print(f"=== {cfg.name} ===")
    print(f"base       {cfg.train.base} ({counts['language_model'] / 1e6:.2f}M, "
          f"{'FROZEN' if cfg.train.freeze_language_model else 'trainable'})")
    print(f"tower      encoder {counts['encoder'] / 1e6:.2f}M + "
          f"projector {counts['projector'] / 1e6:.2f}M")
    print(f"trainable  {counts['trainable'] / 1e6:.2f}M of "
          f"{(counts['trainable'] + counts['language_model']) / 1e6:.2f}M "
          f"({counts['trainable'] / (counts['trainable'] + counts['language_model']) * 100:.1f}%)")
    print(f"image      {cfg.vision.image_size}px / patch {cfg.vision.patch} = "
          f"{cfg.vision.n_patches} patches -> {model.n_image_tokens} tokens of context")
    print(f"corpus     {len(train_ds):,} train / {len(val_ds):,} val"
          + (f" / {len(holdout):,} held-out combination" if holdout else ""))
    print(f"expect     step-0 loss ~= ln({lm.cfg.vocab_size}) = {uniform:.4f}")
    print(f"device     {device}")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)

    start_step, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = str(ckpt_last) if cfg.train.resume == "auto" and ckpt_last.is_file() else (
        cfg.train.resume if cfg.train.resume != "auto" else None)
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.tower.load_state_dict(saved["tower"])
        if "optimizer" in saved:
            opt.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("step", -1)) + 1
        best = float(saved.get("best_val", float("inf")))
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
                params=counts["trainable"], objective="vision", metric="ce",
                tokens_per_step=cfg.train.batch_size * cfg.data.max_caption_tokens)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    def save(path: Path, step: int, **extra):
        torch.save({"tower": model.tower.state_dict(), "vision": asdict(cfg.vision),
                    "config": config_to_dict(cfg), "step": step, "best_val": best,
                    "stage": "vision", "base": cfg.train.base,
                    "tokenizer": cfg.data.tokenizer, **extra}, path)

    model.train()
    t0 = time.time()
    ema, why, step = None, None, start_step

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr

        images, caps = train_ds.batch(cfg.train.batch_size)
        img, text, targets = make_batch(model, tok, images, caps, device,
                                        cfg.data.max_caption_tokens)
        _, loss = model(img, text, targets=targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                                   cfg.optim.grad_clip)
        opt.step()
        ema = float(loss) if ema is None else 0.9 * ema + 0.1 * float(loss)

        if cfg.train.log_every and (step % cfg.train.log_every == 0
                                    or step == cfg.train.max_steps - 1):
            dt = time.time() - t0
            rec = {"step": step, "loss": float(loss), "ema": ema, "lr": lr,
                   "grad_norm": float(grad_norm), "s_per_step": dt / cfg.train.log_every,
                   "time": time.time(), "elapsed": time.time() - run_t0}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {float(loss):>7.4f} (ema {ema:>7.4f})  lr {lr:.2e}"
                  + ("   <- uniform" if abs(float(loss) - uniform) < 0.05 else ""))
            t0 = time.time()

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            val = evaluate(model, tok, val_ds, device, cfg.train.batch_size,
                           cfg.train.eval_batches, cfg.data.max_caption_tokens)
            improved = val < best
            if improved:
                best = val
                save(out_dir / "ckpt_best.pt", step)
            logf.write(json.dumps({"step": step, "time": time.time(), "val_loss": val}) + "\n")
            logf.flush()
            print(f"           val {val:.4f}{'  *best*' if improved else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            s = score(model, tok, val_ds, device, cfg.train.sample_images,
                      cfg.data.max_caption_tokens)
            line = (f"           count {s['count'] * 100:.0f}%  colour {s['colour'] * 100:.0f}%  "
                    f"shape {s['shape'] * 100:.0f}%  ALL {s['all_three'] * 100:.0f}%")
            if holdout:
                h = score(model, tok, holdout, device, min(16, len(holdout)),
                          cfg.data.max_caption_tokens)
                line += f"   |  held-out combination ALL {h['all_three'] * 100:.0f}%"
                s["holdout"] = h
            print(line)
            print(f"           e.g. {s['example']['truth']} -> {s['example']['said']!r}")
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
    final = score(model, tok, val_ds, device, cfg.train.sample_images,
                  cfg.data.max_caption_tokens)
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0,
                best_val=best, score=final)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}")
    print(f"  count {final['count'] * 100:.0f}%  colour {final['colour'] * 100:.0f}%  "
          f"shape {final['shape'] * 100:.0f}%  ALL {final['all_three'] * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
