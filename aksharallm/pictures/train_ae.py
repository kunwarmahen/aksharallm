"""Train the continuous picture autoencoder (G3b): reconstruct shapes pictures through a
Gaussian latent with a light KL term, no codebooks (autoencoder.py).

The same contract as every trainer here, launched by `scripts/audio.sh <run>` for a config with
an `autoencoder:` section. **What to watch:** `kl` should settle small but not at zero (zero
means the latent ignores the picture), and the reader's `all` on the val pictures and on the
held-out purple triangles should reach 100%, as the tokenizer's did.

For photos, an `adversarial:` section adds a patch discriminator from `start_step` on
(adversarial.py): the line then also shows its loss and how often it is right about real and
reconstructed patches, the report shows `sharpness`, and "best" is chosen only among checkpoints
the adversary has trained.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
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
from . import adversarial as adv
from .autoencoder import AEConfig, PictureAutoencoder
from .measure import _marks, psnr, sharpness
from .tokenizer import reconstruction_loss


@dataclass
class PictureDataConfig:
    corpus: str = "data/vision/shapes"
    holdout: str = "data/vision/shapes-holdout"


@dataclass
class PictureTrainConfig:
    out_dir: str = "checkpoints/ae-shapes"
    batch_size: int = 64
    max_steps: int = 20000
    eval_every: int = 1000
    eval_batches: int = 8
    #: Reconstruct and judge this many val pictures at every count of codebooks.
    sample_every: int = 2000
    sample_images: int = 128
    ckpt_every: int = 2000
    log_every: int = 50
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class AERunConfig:
    name: str = "ae-shapes"
    autoencoder: AEConfig = field(default_factory=AEConfig)
    #: Sharper photos (adversarial.py): weight 0 = off, the default, as every earlier run.
    adversarial: adv.AdversarialConfig = field(default_factory=adv.AdversarialConfig)
    data: PictureDataConfig = field(default_factory=PictureDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: PictureTrainConfig = field(default_factory=PictureTrainConfig)


def load_ae_config(path, overrides=None) -> AERunConfig:
    return load_into(AERunConfig, path, overrides)


def load_autoencoder(path, device: str = "cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "autoencoder":
        raise ValueError(f"{path} is not a picture autoencoder checkpoint")
    model = PictureAutoencoder(AEConfig(**blob["autoencoder"]))
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), blob


@torch.no_grad()
def evaluate(model, ds, device, batch_size, batches) -> float:
    """Mean reconstruction loss on val pictures through the latent MEAN (what G3 draws)."""
    was = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x = torch.from_numpy(ds.batch(batch_size)[0]).to(device)
        recon, _ = model(x)
        total += float(reconstruction_loss(recon, x))
    model.train(was)
    return total / max(batches, 1)


@torch.no_grad()
def judge_set(model, ds, n: int, device) -> dict:
    """Reconstruct the first `n` pictures and read them back: PSNR and the reader's verdicts."""
    from ..vision.image import from_tensor, to_tensor
    model.eval()
    n = min(n, len(ds))
    idx = [ds.index[i] for i in range(n)]
    orig = np.asarray(ds.images[idx])
    recon = from_tensor(model(torch.from_numpy(to_tensor(orig)).to(device))[0]
                        .float().clamp(-1, 1).cpu().numpy())
    out = {"n": n, "psnr": float(np.mean([psnr(a, b) for a, b in zip(orig, recon, strict=True)])),
           "sharpness": sharpness(recon, orig)}
    if getattr(ds.manifest, "kind", "shapes") == "coco":
        return out | {"all": None}       # a photo has no facts to read back: PSNR only
    m = _marks(recon, [ds.manifest.facts[j] for j in idx])
    return out | {k: m[k].count("1") / n for k in ("count", "colour", "shape", "all")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.pictures.train_ae",
                                description="Train the continuous picture autoencoder (docs/26).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_ae_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PictureAutoencoder(cfg.autoencoder).to(device)
    train_ds = ImageCaptions(cfg.data.corpus, split="train", seed=cfg.train.seed)
    val_ds = ImageCaptions(cfg.data.corpus, split="val", seed=cfg.train.seed + 1)
    holdout = None
    if Path(cfg.data.holdout, "manifest.json").is_file():
        holdout = ImageCaptions(cfg.data.holdout, split="val", val_frac=1.0)   # every picture

    ac = cfg.autoencoder
    print(f"=== {cfg.name} ===")
    print(f"model      {model.n_params() / 1e6:.2f}M params, widths {ac.channels}")
    print(f"latent     {ac.image_size}px -> {ac.grid}x{ac.grid}x{ac.latent_dim} = "
          f"{ac.grid * ac.grid * ac.latent_dim} numbers (the picture is "
          f"{3 * ac.image_size ** 2:,}), KL weight {ac.kl_weight:g}")
    print(f"corpus     {len(train_ds):,} train / {len(val_ds):,} val"
          + (f" / {len(holdout):,} held-out combination" if holdout else ""))
    print(f"device     {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)
    ad = cfg.adversarial
    disc = d_opt = None
    if ad.weight > 0:
        disc = adv.PatchDiscriminator(ad.channels).to(device)
        # beta1 0.5, as GAN discriminators are usually trained: momentum lags a moving target.
        d_opt = torch.optim.AdamW(disc.parameters(), lr=ad.lr, betas=(0.5, 0.9), weight_decay=0.0)
        print(f"adversary  patch discriminator {disc.n_params() / 1e6:.2f}M params, weight "
              f"{ad.weight:g} (adaptive), from step {ad.start_step:,}")

    start_step, best, best_step = 0, float("inf"), -1
    # With an adversary, "best" is chosen only among checkpoints it has trained: the pixel loss
    # rises a little once the decoder trades exact averages for texture, so the plain rule would
    # keep the blurry checkpoint from before the adversary started, forever.
    best_from = cfg.adversarial.start_step if cfg.adversarial.weight > 0 else 0
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
        if disc is not None and "disc" in saved:
            disc.load_state_dict(saved["disc"])
            d_opt.load_state_dict(saved["disc_optimizer"])
        elif disc is not None and int(saved.get("step", -1)) >= ad.start_step:
            raise SystemExit(f"{resume} has no discriminator but is past adversarial.start_step: "
                             "its adversary would restart from nothing mid-run")
        start_step = int(saved.get("step", -1)) + 1
        best = float(saved.get("best_val", float("inf")))
        best_step = int(saved.get("best_step", -1))
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
                params=model.n_params(), objective="autoencoder", metric="l1+l2",
                batch_size=cfg.train.batch_size)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    def save(path: Path, step: int, **extra):
        torch.save({"model": model.state_dict(), "autoencoder": asdict(cfg.autoencoder),
                    "config": config_to_dict(cfg), "step": step, "best_val": best,
                    "best_step": best_step,
                    "stage": "autoencoder", "data_rng": train_ds.rng.bit_generator.state,
                    **({"disc": disc.state_dict(), "disc_optimizer": d_opt.state_dict()}
                       if disc is not None and "optimizer" in extra else {}),
                    **extra}, path)

    def sample(step: int) -> dict:
        s = {"val": judge_set(model, val_ds, cfg.train.sample_images, device)}
        line = (f"val {s['val']['psnr']:.1f}dB sharpness {s['val']['sharpness']:.2f}") + ("" if s["val"]["all"] is None
                                                  else f" all {s['val']['all'] * 100:.0f}%")
        if holdout:
            s["holdout"] = judge_set(model, holdout, 64, device)
            line += f"  |  held-out combination all {s['holdout']['all'] * 100:.0f}%"
        print(f"           {line}")
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
        x = torch.from_numpy(train_ds.batch(cfg.train.batch_size)[0]).to(device)
        recon, kl = model(x)
        rec = reconstruction_loss(recon, x)
        loss = rec + cfg.autoencoder.kl_weight * kl
        a_stats = {}
        adversary_on = disc is not None and step >= ad.start_step
        if adversary_on:
            g = adv.g_loss(disc(recon))
            lam = adv.adaptive_weight(rec, g, adv.last_layer(model), ad.max_adaptive)
            loss = loss + ad.weight * lam * g
            a_stats = {"g_loss": float(g.detach()), "adaptive": float(lam)}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()
        if adversary_on:
            # The discriminator's own step, on the reconstruction as it was (detached: this
            # loss must not move the autoencoder).
            real_l, fake_l = disc(x), disc(recon.detach())
            dl = adv.d_loss(real_l, fake_l)
            d_opt.zero_grad(set_to_none=True)
            dl.backward()
            d_opt.step()
            a_stats |= {"d_loss": float(dl.detach()),
                        "d_real": float((real_l > 0).float().mean()),
                        "d_fake": float((fake_l < 0).float().mean())}
        r = float(rec.detach())
        ema = r if ema is None else 0.9 * ema + 0.1 * r

        if cfg.train.log_every and (step % cfg.train.log_every == 0
                                    or step == cfg.train.max_steps - 1):
            dt = time.time() - t0
            n_steps = max(cfg.train.log_every, 1) if step > start_step else 1
            rec_ = {"step": step, "loss": float(loss.detach()), "recon": r, "ema": ema,
                    "kl": float(kl.detach()), "lr": lr, "grad_norm": float(grad_norm), **a_stats,
                    "s_per_step": dt / n_steps,
                    "img_per_sec": n_steps * cfg.train.batch_size / max(dt, 1e-9),
                    "time": time.time(), "elapsed": time.time() - run_t0}
            logf.write(json.dumps(rec_) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {float(loss.detach()):.4f}  recon {r:.4f} "
                  f"(ema {ema:.4f})  kl {float(kl.detach()):.4f}  lr {lr:.2e}"
                  + (f"  | adv g {a_stats['g_loss']:+.3f} x{a_stats['adaptive']:.3g} "
                     f"d {a_stats['d_loss']:.3f} (real {a_stats['d_real'] * 100:.0f}% "
                     f"fake {a_stats['d_fake'] * 100:.0f}% right)" if a_stats else ""))
            t0 = time.time()

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            val = evaluate(model, val_ds, device, cfg.train.batch_size, cfg.train.eval_batches)
            improved = step >= best_from and (val < best or best_step < best_from)
            if improved:
                best, best_step = val, step
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
    val = evaluate(model, val_ds, device, cfg.train.batch_size, cfg.train.eval_batches)
    if step >= best_from and (val < best or best_step < best_from):
        best, best_step = val, step
        save(out_dir / "ckpt_best.pt", step)
    final = sample(step) if cfg.train.sample_images else None
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0,
                final_val_loss=val, best_val=best, score=final)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}")
    print(f"    then:    python -m aksharallm.pictures latents {cfg.name} (for G3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
