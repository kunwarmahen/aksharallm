"""Training the speaker embedding (audio/speaker.py) on LibriSpeech's ~1,170 speakers.

It keeps the run contract every trainer here keeps -- `train.pid`, the `STOP` file,
`train_log.jsonl` bracketed by `session_start`/`session_end`, `ckpt_last.pt` and
`ckpt_best.pt` -- so `scripts/stop.sh`, the scheduler and the portal drive it unchanged.

"Best" is chosen on **EER over dev-clean's speakers**, none of whom are in training. The
classifier's own accuracy is logged too, and is the number not to be fooled by: it measures
how well 1,170 known people are told apart, and those 1,170 weights are thrown away.
test-clean's 40 speakers are kept for the final number (`audio speaker-eval`), so the
checkpoint is not chosen on the speakers it is reported on.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from ..config import OptimConfig, config_to_dict, load_into
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .config import AudioTrainConfig
from .speaker import (AMSoftmax, SpeakerCorpus, SpeakerEncoder, SpeakerModelConfig,
                      save_speaker, verification)
from .transfer import pick_clips


@dataclass
class SpeakerDataConfig:
    train: list = field(default_factory=lambda: ["data/asr/train-clean-100"])
    #: Unseen speakers, for choosing the best checkpoint. Not test-clean (see above).
    eval: str = "data/asr/dev-clean"
    eval_per_speaker: int = 4
    crop_seconds: float = 2.0
    val_clips: int = 1


@dataclass
class SpeakerRunConfig:
    name: str = "speaker-libri"
    speaker: SpeakerModelConfig = field(default_factory=SpeakerModelConfig)
    data: SpeakerDataConfig = field(default_factory=SpeakerDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: AudioTrainConfig = field(default_factory=AudioTrainConfig)


def load_speaker_config(path, overrides=None) -> SpeakerRunConfig:
    cfg = load_into(SpeakerRunConfig, path, overrides)
    if isinstance(cfg.speaker, dict):
        cfg.speaker = SpeakerModelConfig(**cfg.speaker)
    if cfg.train.init:
        raise ValueError("train.init is a codec-trainer option; the speaker trainer does not read it")
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.audio.train_speaker",
                                description="Train the speaker embedding (docs/21).")
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)
    cfg = load_speaker_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = cfg.speaker.encoder.sample_rate

    crop = int(cfg.data.crop_seconds * sr)
    train_ds = SpeakerCorpus(cfg.data.train, crop, seed=cfg.train.seed,
                             val_clips=cfg.data.val_clips)
    val_ds = SpeakerCorpus(cfg.data.train, crop, seed=cfg.train.seed + 1,
                           val_clips=cfg.data.val_clips, split="val")
    unseen = pick_clips(cfg.data.eval, per_speaker=cfg.data.eval_per_speaker, sample_rate=sr)

    model = SpeakerEncoder(cfg.speaker).to(device)
    head = AMSoftmax(cfg.speaker.emb_dim, train_ds.n_speakers, cfg.speaker.margin,
                     cfg.speaker.scale).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(
        [{"params": [q for q in params if q.dim() >= 2], "weight_decay": cfg.optim.weight_decay},
         {"params": [q for q in params if q.dim() < 2], "weight_decay": 0.0}],
        lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2))

    start, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = cfg.train.resume
    if resume == "auto":
        resume = str(ckpt_last) if ckpt_last.is_file() else None
    if resume:
        blob = torch.load(resume, map_location=device, weights_only=False)
        if blob.get("speakers") != train_ds.speakers:
            # The classifier's rows are speakers by index; a changed list is a different task.
            raise SystemExit(f"{resume} was trained on a different speaker list")
        model.load_state_dict(blob["model"])
        head.load_state_dict(blob["head"])
        opt.load_state_dict(blob["optimizer"])
        start, best = int(blob["step"]) + 1, float(blob.get("best_eer", best))
        print(f"resumed {resume} at step {start}, best EER {best:.4f}")

    n_params = sum(q.numel() for q in model.parameters())
    print(f"=== {cfg.name} ===")
    print(f"encoder    {cfg.speaker.encoder.describe()} -> {cfg.speaker.emb_dim}-d, "
          f"{n_params / 1e6:.2f}M (+{head.weight.numel() / 1e6:.2f}M classifier, discarded)")
    print(f"speakers   {train_ds.n_speakers} training, {len({c['speaker'] for c in unseen})} "
          f"unseen in {cfg.data.eval} ({len(unseen)} clips)")
    print(f"batch      {cfg.train.batch_size} x {cfg.data.crop_seconds:g} s crops, "
          f"AM-softmax m={cfg.speaker.margin} s={cfg.speaker.scale}  (chance EER 50%)")
    print(f"device     {device}")

    claim_pid_file(out_dir)
    stop_file = out_dir / "STOP"
    stop_now = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop_now.__setitem__("now", True))
    logf = open(out_dir / "train_log.jsonl", "a")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    def session(event, **kw):
        log({"event": event, "time": time.time(), "run": cfg.name,
             "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **kw})

    def save(path, step):
        save_speaker(model, path, head=head.state_dict(), optimizer=opt.state_dict(),
                     step=step, best_eer=best, speakers=train_ds.speakers,
                     config=config_to_dict(cfg))

    stop_at = cfg.train.stop_at
    if cfg.train.stop_after is not None:
        stop_at = start + cfg.train.stop_after - 1
    run_t0 = time.time()
    stop_by = None if cfg.train.stop_after_s is None else run_t0 + cfg.train.stop_after_s
    session("session_start", pid=os.getpid(), start_step=start, max_steps=cfg.train.max_steps,
            stop_at=stop_at, stop_by=stop_by, params=n_params, objective="speaker",
            metric="eer")
    if start >= cfg.train.max_steps:
        session("session_end", reason="already_complete", last_step=cfg.train.max_steps - 1,
                steps=0)
        return 0

    why, step, t0, prev, ema = None, start, time.time(), start - 1, None
    for step in range(start, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        x, y = train_ds.batch(cfg.train.batch_size, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            emb = model(x)
        loss, acc = head(emb.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
        opt.step()
        ema = float(loss) if ema is None else 0.9 * ema + 0.1 * float(loss)

        if cfg.train.log_every and (step % cfg.train.log_every == 0
                                    or step == cfg.train.max_steps - 1):
            dt = (time.time() - t0) / max(1, step - prev)
            log({"step": step, "loss": float(loss), "ema": ema, "acc": float(acc), "lr": lr,
                 "grad_norm": float(gnorm), "s_per_step": dt, "time": time.time(),
                 "elapsed": time.time() - run_t0})
            print(f"step {step:>6}  loss {float(loss):.4f} (ema {ema:.4f})  "
                  f"acc {float(acc) * 100:5.1f}%  gnorm {float(gnorm):.2f}  "
                  f"{dt:.3f} s/step  lr {lr:.2e}")
            t0, prev = time.time(), step

        if cfg.train.eval_every and step > start and step % cfg.train.eval_every == 0:
            with torch.no_grad():
                model.eval()
                xv, yv = val_ds.batch(cfg.train.batch_size * cfg.train.eval_batches, device)
                _, vacc = head(torch.cat([model(c) for c in xv.split(cfg.train.batch_size)]), yv)
            ver = verification(model, unseen, device)
            improved = ver["eer"] < best
            if improved:
                best = ver["eer"]
                save(out_dir / "ckpt_best.pt", step)
            log({"step": step, "time": time.time(), "val_loss": ver["eer"],
                 "val_eer": ver["eer"], "val_acc": float(vacc),
                 "same_speaker": ver["same_speaker"],
                 "different_speaker": ver["different_speaker"]})
            print(f"           unseen EER {ver['eer'] * 100:.2f}%{'  *best*' if improved else ''}"
                  f"  (same {ver['same_speaker']:.3f} / different {ver['different_speaker']:.3f})"
                  f"  known-speaker acc {float(vacc) * 100:.1f}%")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start and step % cfg.train.ckpt_every == 0:
            save(ckpt_last, step)
            t0 = time.time()

        req = None if stop_now["now"] else stopfile.read(stop_file)
        if stop_now["now"]:
            why = "SIGTERM"
        elif req is not None and (r := stopfile.reached(req, step)):
            why = r
        elif stop_at is not None and step >= stop_at:
            why = f"stop_at/stop_after reached step {stop_at}"
        elif stop_by is not None and time.time() >= stop_by:
            why = "wall-clock budget spent"
        if why:
            break

    save(ckpt_last, step)
    reason = why or "max_steps"
    session("session_end", reason=reason, last_step=step, trained_to=step,
            steps=step - start + 1, elapsed=time.time() - run_t0, best_val=best)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best unseen EER {best * 100:.2f}%")
    print(f"  final number: python -m aksharallm.audio speaker-eval {out_dir / 'ckpt_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
