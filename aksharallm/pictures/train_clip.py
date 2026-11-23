"""Train G4b, our own contrastive picture-caption model (clip.py).

The same contract as every trainer here (`train.pid`, `STOP`, session records, `ckpt_last.pt`
/ `ckpt_best.pt`, resume by default), launched by `scripts/audio.sh <run>` for a config with a
`clip:` section. Small enough to train on the CPU (`CUDA_VISIBLE_DEVICES=`).

Each batch is drawn evenly from every corpus in `data.corpora` (shapes and scenes), each
picture with its caption in one of the ten TRAINING phrasings (phrases.py), and every caption
in the batch that states the same fact counts as a match (`clip.clip_loss`).

**What to watch** (every `eval_every` steps, on fixed pictures):

1. **val loss**, the same loss on fixed val pictures and phrasings. Step 0 sits near
   ln(batch) less what duplicates of a fact give away.
2. **top-1 per corpus**: for each val picture, is its own caption the closest of every
   caption that corpus can say (60 for shapes, 270 for scenes)? Canonical wording, and a
   held-out phrasing (`show_me`) the model never read.
3. **held-out pictures**: the same on the combinations no training picture shows (purple
   triangles; circles on the right). A scorer that cannot recognise them cannot judge a
   generator on them.

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
from ..vision.image import ImageCaptions, to_tensor
from .clip import ClipConfig, ClipModel, clip_loss, rank, same_fact, text_encoder_for


@dataclass
class ClipDataConfig:
    corpora: list = field(default_factory=lambda: ["data/vision/shapes64k",
                                                   "data/vision/scenes64k"])
    #: pictures of the held-out combinations only, one folder per corpus (may be missing)
    holdouts: list = field(default_factory=lambda: ["data/vision/shapes-holdout",
                                                    "data/vision/scenes64k-holdout"])


@dataclass
class ClipTrainConfig:
    out_dir: str = "checkpoints/clip-shapes"
    batch_size: int = 256
    max_steps: int = 4000
    eval_every: int = 250
    eval_batches: int = 4
    eval_pictures: int = 512
    ckpt_every: int = 500
    log_every: int = 50
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class ClipRunConfig:
    name: str = "clip-shapes"
    clip: ClipConfig = field(default_factory=ClipConfig)
    data: ClipDataConfig = field(default_factory=ClipDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: ClipTrainConfig = field(default_factory=ClipTrainConfig)


def load_clip_config(path, overrides=None) -> ClipRunConfig:
    return load_into(ClipRunConfig, path, overrides)


def kind_of(ds: ImageCaptions) -> str:
    return getattr(ds.manifest, "kind", "shapes") or "shapes"


def draw_batch(sets: list[ImageCaptions], n: int, rng, *, phrasing: str | None = None,
               picks: list[np.ndarray] | None = None):
    """`n` pictures split evenly over `sets`, with their facts and a caption for each: a
    random TRAINING phrasing, or `phrasing` for all (val)."""
    from .clip import text_of
    from .phrases import TRAIN
    names = list(TRAIN)
    imgs, facts = [], []
    per = [n // len(sets) + (i < n % len(sets)) for i in range(len(sets))]
    for j, (ds, m) in enumerate(zip(sets, per, strict=True)):
        idx = picks[j] if picks is not None else rng.choice(ds.index, size=m)
        imgs.append(to_tensor(np.asarray(ds.images[np.sort(idx)])))
        facts += [ds.manifest.facts[int(i)] for i in np.sort(idx)]
    # a drawn picture: a random training phrasing of its facts; a COCO photo: one of its own
    # five captions at random (val: the first)
    caps = [text_of(f, phrasing or names[int(rng.integers(len(names)))],
                    0 if phrasing else int(rng.integers(5))) for f in facts]
    return torch.from_numpy(np.concatenate(imgs)), facts, caps


@torch.no_grad()
def evaluate(model, sets, device, cfg, amp) -> dict:
    """Val loss on fixed pictures and phrasings, and top-1 per corpus (see the docstring)."""
    from .phrases import all_facts
    was = model.training
    model.eval()
    tc = cfg.train
    rng = np.random.default_rng(4321)
    losses = []
    for b in range(tc.eval_batches):
        picks = [np.asarray(ds.index[(b * tc.batch_size) % len(ds):][: tc.batch_size // len(sets)])
                 for ds in sets]
        x, facts, caps = draw_batch(sets, sum(len(p) for p in picks), rng, picks=picks,
                                    phrasing=["canonical", "picture_of", "all_of", "draw"][b % 4])
        with amp():
            img = model.embed_images(x.to(device))
            txt = model.embed_texts(caps, device)
            losses.append(float(clip_loss(img.float(), txt.float(), model.scale,
                                          same_fact(facts).to(device))))
    out = {"val_loss": float(np.mean(losses))}
    for ds in sets:
        kind = kind_of(ds)
        n = min(tc.eval_pictures, len(ds))
        idx = ds.index[:n]
        pics = np.asarray(ds.images[idx])
        facts = [ds.manifest.facts[int(i)] for i in idx]
        if kind == "coco":
            # photos: is a photo's own (first) caption the closest of the n val photos'
            # captions, and is it in the closest five?
            r = rank(model, pics, facts, facts, device=device)
            ranks = _ranks(model, pics, facts, device)
            out["coco_top1"] = float(np.mean(r["top1"]))
            out["coco_top5"] = float(np.mean(ranks < 5))
            continue
        cands = all_facts(kind)
        out[f"{kind}_top1"] = float(np.mean(rank(model, pics, facts, cands, device=device)["top1"]))
        out[f"{kind}_show_me"] = float(np.mean(
            rank(model, pics, facts, cands, phrasing="show_me", device=device)["top1"]))
    for path in cfg.data.holdouts:
        if not Path(path, "manifest.json").is_file():
            continue
        hd = ImageCaptions(path, split="val", val_frac=1.0)
        facts = list(hd.manifest.facts)
        r = rank(model, np.asarray(hd.images), facts, all_facts(kind_of(hd)), device=device)
        out[f"{kind_of(hd)}_held_out_top1"] = float(np.mean(r["top1"]))
    model.train(was)
    return out


@torch.no_grad()
def _ranks(model, pics, facts, device) -> np.ndarray:
    """For each photo, where its own first caption ranks among all `facts`' first captions."""
    from .clip import picture_vectors, text_of
    img = picture_vectors(model, pics, device)
    txt = torch.cat([model.embed_texts([text_of(f) for f in facts[i:i + 512]], device).float().cpu()
                     for i in range(0, len(facts), 512)])
    sims = img @ txt.t()
    own = sims.diagonal()[:, None]
    return (sims > own).sum(1).numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.pictures.train_clip",
                                description="Train G4b, our own contrastive picture-caption model (docs/26).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_clip_config(args.config, args.override)
    cc = cfg.clip
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda"
           else nullcontext)

    rng = np.random.default_rng(cfg.train.seed)
    train_sets = [ImageCaptions(c, split="train", seed=cfg.train.seed + i)
                  for i, c in enumerate(cfg.data.corpora)]
    val_sets = [ImageCaptions(c, split="val") for c in cfg.data.corpora]
    encoder = text_encoder_for(cc, None if device == "cuda" else "cpu")
    cc.text_dim = encoder.dim
    model = ClipModel(cc).to(device)
    model.attach_text(encoder)

    print(f"=== {cfg.name} ===")
    print(f"model      {model.n_params() / 1e6:.2f}M trainable params: ViT d {cc.d_vision} x "
          f"{cc.n_layers}, text head d {cc.d_text} x {cc.text_layers} over {cc.text} (frozen), "
          f"{cc.d_embed}-d embeddings")
    for ds, c in zip(train_sets, cfg.data.corpora, strict=True):
        print(f"corpus     {c}: {len(ds):,} train pictures ({kind_of(ds)})")
    print(f"batch      {cfg.train.batch_size} pictures, every same-fact caption a match")
    print(f"device     {device}")

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
        if "rng" in saved:
            rng.bit_generator.state = saved["rng"]
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
                params=model.n_params(), objective="clip", metric="contrastive loss",
                batch_size=cfg.train.batch_size)
    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    def save(path: Path, step: int, **extra):
        torch.save({"model": model.state_dict(), "clip": asdict(cc),
                    "config": config_to_dict(cfg), "step": step, "best_val": best,
                    "stage": "clip", "rng": rng.bit_generator.state, **extra}, path)

    def show(ev: dict) -> str:
        return "  ".join(f"{k.replace('_top1', '')} {v * 100:.1f}%" for k, v in ev.items()
                         if k != "val_loss")

    model.train()
    t0 = time.time()
    ema, why, step = None, None, start_step
    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        x, facts, caps = draw_batch(train_sets, cfg.train.batch_size, rng)
        with amp():
            img = model.embed_images(x.to(device))
            txt = model.embed_texts(caps, device)
        loss = clip_loss(img.float(), txt.float(), model.scale, same_fact(facts).to(device))
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
                                   "scale": float(model.scale.detach()), "grad_norm": float(grad_norm),
                                   "s_per_step": dt / n_steps, "time": time.time(),
                                   "elapsed": time.time() - run_t0}) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {lv:.4f} (ema {ema:.4f})  scale {float(model.scale.detach()):.1f}  "
                  f"lr {lr:.2e}  gnorm {float(grad_norm):.2f}  {dt / n_steps:.3f}s/step")
            t0 = time.time()

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            ev = evaluate(model, val_sets, device, cfg, amp)
            improved = ev["val_loss"] < best
            if improved:
                best = ev["val_loss"]
                save(out_dir / "ckpt_best.pt", step)
            logf.write(json.dumps({"step": step, "time": time.time(), **ev}) + "\n")
            logf.flush()
            print(f"           val {ev['val_loss']:.4f}{'  *best*' if improved else ''}  |  "
                  f"top-1: {show(ev)}")
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
    ev = evaluate(model, val_sets, device, cfg, amp)
    if ev["val_loss"] < best:
        best = ev["val_loss"]
        save(out_dir / "ckpt_best.pt", step)
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0,
                final_val_loss=ev["val_loss"], best_val=best, score=ev)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}  |  top-1: {show(ev)}")
    print(f"    measure: python -m aksharallm.pictures clip-report {cfg.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
