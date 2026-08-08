"""Training the audio language model — next-token prediction, on sound.

Unlike `train_codec.py`, there is almost nothing new here. The data is integers, the loss is
cross-entropy, the model is the repo's transformer. What differs from `train/pretrain.py` is
only that each position carries `n_codebooks` integers rather than one, so the batch is
`(B, N, T)` and the loss averages over `N` heads.

**The number to watch is `ln(codebook_size)`.** At step 0 the model knows nothing and a
uniform distribution over 1,024 codes costs exactly `ln 1024 = 6.931` nats. Seeing that on
the first line is the cheapest possible check that the delay pattern, the target masking and
the eight heads are all wired the way they are supposed to be — the same argument as the DPO
loop's `ln 2 = 0.6931` at step 0.

**And the number that says whether it is working is not the loss.** A codec LM's loss falls
smoothly while it generates plausible-sounding gibberish, because most of the entropy is in
the high codebooks, which are nearly noise and cannot be predicted by anyone. Sample audio
every `sample_every` steps and listen.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
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
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .codec import load_codec
from .config import AudioTrainConfig
from .dataset import CodeDataset
from .delay import delay
from .io import write_wav
from .lm import AudioLM, AudioLMConfig, generate, make_targets


@dataclass
class AudioLMDataConfig:
    codes: str = "data/audio/synth-codes"
    codec: str = "checkpoints/codec-synth/ckpt_best.pt"
    window_frames: int = 200  # 4 s at 50 frames/s
    val_clips: int = 16


@dataclass
class AudioLMRunConfig:
    name: str = "audiolm-synth"
    audiolm: AudioLMConfig = field(default_factory=AudioLMConfig)
    data: AudioLMDataConfig = field(default_factory=AudioLMDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: AudioTrainConfig = field(default_factory=AudioTrainConfig)


def load_audiolm_config(path, overrides=None) -> AudioLMRunConfig:
    cfg = load_into(AudioLMRunConfig, path, overrides)
    if isinstance(cfg.audiolm.model, dict):  # nested twice; `_build` handles one level
        cfg.audiolm.model = ModelConfig(**cfg.audiolm.model)
    return cfg


@torch.no_grad()
def evaluate(model: AudioLM, ds: CodeDataset, pad_id: int, batch_size: int, batches: int) -> float:
    was = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        codes = ds.batch(batch_size)
        d = delay(codes, pad_id)
        _, loss = model(d, targets=make_targets(d, codes.shape[-1], pad_id))
        total += float(loss)
    model.train(was)
    return total / max(batches, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m aksharallm.audio.train_lm",
        description="Train the audio language model over codec tokens (docs/21).",
    )
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_audiolm_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The codec is loaded to WRITE SAMPLES, and to check that the codes were made by it.
    codec = load_codec(cfg.data.codec, device)
    if codec.cfg.n_codebooks != cfg.audiolm.n_codebooks or (
        codec.cfg.codebook_size != cfg.audiolm.codebook_size
    ):
        raise SystemExit(
            f"the codec is {codec.cfg.n_codebooks} x {codec.cfg.codebook_size} but the LM "
            f"config says {cfg.audiolm.n_codebooks} x {cfg.audiolm.codebook_size}. An audio "
            "LM run against a different codec produces confident nonsense — the integers "
            "mean different sounds."
        )

    train_ds = CodeDataset(cfg.data.codes, cfg.data.window_frames, device,
                           n_codebooks=cfg.audiolm.n_codebooks, seed=cfg.train.seed,
                           split="train", val_clips=cfg.data.val_clips)
    val_ds = CodeDataset(cfg.data.codes, cfg.data.window_frames, device,
                         n_codebooks=cfg.audiolm.n_codebooks, seed=cfg.train.seed + 1,
                         split="val", val_clips=cfg.data.val_clips)

    model = AudioLM(cfg.audiolm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)

    start_step, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = str(ckpt_last) if cfg.train.resume == "auto" and ckpt_last.is_file() else (
        cfg.train.resume if cfg.train.resume != "auto" else None
    )
    if resume:
        blob = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", -1)) + 1
        best = float(blob.get("best_val", float("inf")))
        print(f"resumed {resume} at step {start_step}, best val {best:.4f}")

    uniform = math.log(cfg.audiolm.codebook_size)
    print(f"=== {cfg.name} ===")
    print(f"model      {model.num_params() / 1e6:.2f}M params, "
          f"{cfg.audiolm.n_codebooks} heads x {cfg.audiolm.codebook_size} codes")
    print(f"codec      {cfg.data.codec} ({codec.cfg.describe()})")
    print(f"window     {cfg.data.window_frames} frames = "
          f"{cfg.data.window_frames / codec.cfg.frames_per_second:.1f}s -> "
          f"{cfg.data.window_frames + cfg.audiolm.n_codebooks - 1} positions after the delay")
    print(f"corpus     {len(train_ds)} train clips / {len(val_ds)} val")
    print(f"expect     step-0 loss ~= ln({cfg.audiolm.codebook_size}) = {uniform:.4f}")
    print(f"device     {device}")

    claim_pid_file(out_dir)
    stop_file = out_dir / "STOP"
    stop_now = {"now": False}

    def _request_stop(signum, frame):  # noqa: ARG001
        stop_now["now"] = True

    signal.signal(signal.SIGTERM, _request_stop)
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
    tokens_per_step = cfg.train.batch_size * cfg.data.window_frames * cfg.audiolm.n_codebooks

    log_session("session_start", pid=os.getpid(), start_step=start_step,
                max_steps=cfg.train.max_steps, stop_at=stop_at, stop_by=stop_by,
                params=model.num_params(), objective="audiolm", metric="ce",
                tokens_per_step=tokens_per_step)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    model.train()
    t0 = time.time()
    prev_log_step = start_step - 1
    ema, why, step = None, None, start_step

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr

        codes = train_ds.batch(cfg.train.batch_size)
        d = delay(codes, model.pad_id)
        targets = make_targets(d, codes.shape[-1], model.pad_id)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            _, loss = model(d, targets=targets)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()
        ema = float(loss) if ema is None else 0.9 * ema + 0.1 * float(loss)

        if cfg.train.log_every and (
            step % cfg.train.log_every == 0 or step == cfg.train.max_steps - 1
        ):
            dt = time.time() - t0
            n_steps = max(1, step - prev_log_step)
            rec = {"step": step, "loss": float(loss), "ema": ema, "lr": lr,
                   "grad_norm": float(grad_norm), "s_per_step": dt / n_steps,
                   "tok_per_sec": n_steps * tokens_per_step / max(dt, 1e-9),
                   "ppl": math.exp(min(float(loss), 20)),
                   "time": time.time(), "elapsed": time.time() - run_t0}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {float(loss):>7.4f} (ema {ema:>7.4f})  "
                  f"ppl {rec['ppl']:>7.1f}  {rec['tok_per_sec']:>8.0f} tok/s  lr {lr:.2e}"
                  + ("   <- uniform" if abs(float(loss) - uniform) < 0.02 else ""))
            t0, prev_log_step = time.time(), step

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            val = evaluate(model, val_ds, model.pad_id, cfg.train.batch_size,
                           cfg.train.eval_batches)
            improved = val < best
            if improved:
                best = val
                _save(model, cfg, step, best, out_dir / "ckpt_best.pt")
            logf.write(json.dumps({"step": step, "time": time.time(), "val_loss": val}) + "\n")
            logf.flush()
            print(f"           val {val:.4f}{'  *best*' if improved else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            path = _sample(model, codec, out_dir, step, cfg)
            print(f"           sample -> {path}")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            _save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict())
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

    _save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict())
    _sample(model, codec, out_dir, step, cfg)
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0, best_val=best)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}")
    print(f"  listen  {out_dir / 'samples'}/")
    return 0


def _save(model: AudioLM, cfg: AudioLMRunConfig, step: int, best: float, path: Path, **extra):
    torch.save({"model": model.state_dict(), "audiolm": asdict(cfg.audiolm),
                "config": config_to_dict(cfg), "step": step, "best_val": best,
                "stage": "audiolm", "codec": cfg.data.codec, **extra}, path)


@torch.no_grad()
def _sample(model: AudioLM, codec, out_dir: Path, step: int, cfg: AudioLMRunConfig) -> Path:
    """Generate a few seconds and decode it through the codec, so it can be listened to."""
    frames = min(cfg.audiolm.max_frames, int(3 * codec.cfg.frames_per_second))
    device = next(model.parameters()).device
    codes = generate(model, frames, device=device)
    wave = codec.decode(codes.to(device))[0].float().cpu().numpy()
    path = out_dir / "samples" / f"step{step:06d}.wav"
    write_wav(path, wave, codec.cfg.sample_rate)
    model.train()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
