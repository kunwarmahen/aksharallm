"""Training the codec — the one loop in this repo that is not next-token prediction.

`train/pretrain.py` grew an *objective* seam so masked diffusion could reuse it, and that
was the right call there: diffusion is a different loss over the same token data, the same
dataloader, the same everything. The codec is not. It reads waveforms rather than token
shards, its loss is a bank of STFTs, and its "embedding table" is updated by an EMA that no
optimizer ever sees. Bending the text trainer around that would leave two loops tangled in
one file instead of one loop each.

What it *does* share is the **contract**, and that is what makes the existing tooling work
unchanged:

* `<out_dir>/train.pid`, claimed by this process and released on exit;
* `<out_dir>/STOP`, read fresh every step, with the same three forms (`train/stopfile.py`);
* `<out_dir>/train_log.jsonl`, append-only, bracketed by `session_start`/`session_end`;
* `<out_dir>/ckpt_last.pt` and `ckpt_best.pt`.

So `scripts/stop.sh`, `scripts/sessions.py`, the GPU/cost samplers and the portal's run
machinery all drive a codec run without being told it is one.

**What to watch, in order of how badly it goes wrong:**

1. **`usage` / `perplexity`** — the effective codebook size. Codebook collapse is trap 1 of
   the phase, it is the same failure as MoE router collapse, and *it is invisible in the
   loss curve*, which plateaus for a dozen innocent reasons. If perplexity is 40 out of
   1,024 by step 2,000, stop and fix it; nothing downstream can recover from it.
2. **the sample WAVs** in `<out_dir>/samples/`. A codec's loss does not say whether it is
   intelligible. A three-second file does.
3. `stft2048` vs `stft128` — if the long scale improves while the short one does not, the
   model has the vowels and not the consonants, which is what a codec sounds like when the
   frame rate is too low.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ..config import config_to_dict
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .codec import Codec, ReconstructionLoss, codebook_report
from .config import CodecRunConfig, load_codec_config
from .dataset import AudioDataset
from .io import write_wav


def build_loss(cfg: CodecRunConfig) -> ReconstructionLoss:
    return ReconstructionLoss(
        scales=tuple(cfg.loss.scales),
        log_weight=cfg.loss.log_weight,
        convergence_weight=cfg.loss.convergence_weight,
        wave_weight=cfg.loss.wave_weight,
    )


@torch.no_grad()
def evaluate(model: Codec, ds: AudioDataset, recon, batch_size: int, batches: int) -> dict:
    """Validation loss at the **full** bitrate.

    Quantizer dropout is a training-time device; evaluating under it would make the val
    curve partly a record of which bitrates were drawn, exactly the way a fresh diffusion
    mask would. `model.eval()` turns it off (see `ResidualVQ.forward`) and this asserts
    nothing about the low-bitrate prefixes — `codec reconstruct --codebooks` is where those
    are measured, on purpose, because they are a different question.
    """
    was = model.training
    model.eval()
    total, parts_sum, n = 0.0, {}, 0
    for _ in range(batches):
        x = ds.batch(batch_size)
        y, _, vq, stats = model(x)
        loss, parts = recon(x, y)
        total += float(loss)
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v
        n += 1
    model.train(was)
    out = {"val_loss": total / max(n, 1)}
    out.update({f"val_{k}": v / max(n, 1) for k, v in parts_sum.items()})
    return out


@torch.no_grad()
def write_sample(model: Codec, ds: AudioDataset, out_dir: Path, step: int, seconds: float) -> Path:
    """Reconstruct one held-out clip and write original + reconstruction side by side.

    Both files, every time. A reconstruction on its own is impossible to judge — the ear has
    no memory for this — and going back to find the source clip is enough friction that
    nobody does it.
    """
    was = model.training
    model.eval()
    sr = ds.manifest.sample_rate
    want = int(seconds * sr)
    x = ds.clip(0)[:want].unsqueeze(0).to(next(model.parameters()).device)
    y, _, _, _ = model(x)
    samples = out_dir / "samples"
    write_wav(samples / "original.wav", x[0].float().cpu().numpy(), sr)
    path = samples / f"step{step:06d}.wav"
    write_wav(path, y[0].float().cpu().numpy(), sr)
    model.train(was)
    return path


def save(model: Codec, cfg: CodecRunConfig, step: int, best: float, path: Path, **extra) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "codec": asdict(cfg.codec),
            "config": config_to_dict(cfg),
            "step": step,
            "best_val": best,
            "stage": "codec",
            "sample_rate": cfg.codec.sample_rate,
            **extra,
        },
        path,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m aksharallm.audio.train_codec",
        description="Train the RVQ-VAE audio codec (docs/21).",
    )
    p.add_argument("config", help="configs/<run>.yaml")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    args = p.parse_args(argv)

    cfg = load_codec_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = AudioDataset(
        cfg.data.corpus, cfg.window, device, seed=cfg.train.seed,
        split="train", val_clips=cfg.data.val_clips,
    )
    val_ds = AudioDataset(
        cfg.data.corpus, cfg.window, device, seed=cfg.train.seed + 1,
        split="val", val_clips=cfg.data.val_clips,
    )
    if train_ds.manifest.sample_rate != cfg.codec.sample_rate:
        # Asserted, not repaired. Resampling here would work and would also mean the corpus
        # on disk and the checkpoint disagree about what a frame is.
        raise SystemExit(
            f"corpus is {train_ds.manifest.sample_rate} Hz but codec.sample_rate is "
            f"{cfg.codec.sample_rate}. Re-pack the corpus, or fix the config."
        )

    model = Codec(cfg.codec).to(device)
    recon = build_loss(cfg)
    counts = model.n_params()

    # No weight decay on the Snake alphas or any 1-d parameter: decaying a per-channel gain
    # towards zero is not regularisation, it is a slow deletion of the activation.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.optim.weight_decay},
         {"params": plain, "weight_decay": 0.0}],
        lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2),
    )

    start_step, best = 0, float("inf")
    resume = cfg.train.resume
    ckpt_last = out_dir / "ckpt_last.pt"
    if resume == "auto":
        resume = str(ckpt_last) if ckpt_last.is_file() else None
    if resume:
        blob = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", -1)) + 1
        best = float(blob.get("best_val", float("inf")))
        print(f"resumed {resume} at step {start_step}, best val {best:.4f}")

    print(f"=== {cfg.name} ===")
    print(f"codec      {cfg.codec.describe()}")
    print(f"params     {counts['total'] / 1e6:.2f}M "
          f"(encoder {counts['encoder'] / 1e6:.2f}M, decoder {counts['decoder'] / 1e6:.2f}M) "
          f"+ {counts['codebooks'] / 1e6:.2f}M of codebook (not trained by the optimizer)")
    print(f"corpus     {train_ds.seconds / 3600:.2f} h train / {val_ds.seconds / 60:.1f} min val, "
          f"{len(train_ds)} clips")
    print(f"window     {cfg.window} samples = {cfg.window / cfg.codec.sample_rate:.2f}s "
          f"= {cfg.window // cfg.codec.hop} frames, batch {cfg.train.batch_size}")
    print(f"device     {device}")

    claim_pid_file(out_dir)
    stop_file = out_dir / "STOP"
    stop_now = {"now": False}

    def _request_stop(signum, frame):  # noqa: ARG001
        stop_now["now"] = True
        print("\nSIGTERM: finishing this step, then saving and exiting.")

    signal.signal(signal.SIGTERM, _request_stop)

    logf = open(out_dir / "train_log.jsonl", "a")

    def log_session(event: str, **kw):
        rec = {"event": event, "time": time.time(),
               "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "run": cfg.name, **kw}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    stop_at = cfg.train.stop_at
    if cfg.train.stop_after is not None:
        stop_at = start_step + cfg.train.stop_after - 1
    run_t0 = time.time()
    stop_by = None if cfg.train.stop_after_s is None else run_t0 + cfg.train.stop_after_s

    log_session(
        "session_start", pid=os.getpid(), start_step=start_step, max_steps=cfg.train.max_steps,
        stop_at=stop_at, stop_by=stop_by, params=counts["total"], objective="codec",
        metric="recon", sample_rate=cfg.codec.sample_rate,
        # A codec's "tokens per step" is genuinely audio seconds; recording it under the
        # name the shared tooling already charts keeps one column meaning one thing.
        tokens_per_step=cfg.train.batch_size * cfg.window // cfg.codec.hop * cfg.codec.n_codebooks,
    )

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has already trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    model.train()
    t0 = time.time()
    prev_log_step = start_step - 1
    ema = None
    why = None
    step = start_step

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr

        x = train_ds.batch(cfg.train.batch_size)
        # bf16 for the convolutions; the loss itself runs in float32 because an STFT
        # magnitude spans six orders of magnitude and a log of a bf16 near the floor is a
        # coarse number pretending to be a gradient.
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            y, _, vq_loss, stats = model(x)
        loss, parts = recon(x, y.float())
        total = loss + cfg.loss.vq_weight * vq_loss.float()

        opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()

        ema = float(total) if ema is None else 0.9 * ema + 0.1 * float(total)

        if cfg.train.log_every and (step % cfg.train.log_every == 0 or step == cfg.train.max_steps - 1):
            dt = time.time() - t0
            n_steps = max(1, step - prev_log_step)
            audio_s = n_steps * cfg.train.batch_size * cfg.window / cfg.codec.sample_rate
            book = codebook_report(stats, cfg.codec.codebook_size)
            frames = n_steps * cfg.train.batch_size * cfg.window // cfg.codec.hop
            rec = {
                "step": step, "loss": float(total), "recon": float(loss),
                "vq": float(vq_loss), "ema": ema, "lr": lr,
                "grad_norm": float(grad_norm), "s_per_step": dt / n_steps,
                # `tok_per_sec` under that exact name, because the portal's throughput chart
                # and `runlog.SERIES_KEYS` read that key and nothing else -- a codec logging
                # only its own natural unit drew an empty chart. A codec "token" is one
                # codebook entry for one frame, which is what `tokens_per_step` in the
                # session record already declares, so the two agree.
                "tok_per_sec": frames * cfg.codec.n_codebooks / max(dt, 1e-9),
                # Audio-seconds reconstructed per wall-clock second. The codec's equivalent
                # of tok/s, and the only throughput number that means anything here — MFU
                # would be a fiction over a stack of strided convolutions.
                "audio_s_per_s": audio_s / max(dt, 1e-9),
                "codebook_usage": book["usage"], "codebook_perplexity": book["perplexity"],
                "codebook_used": book["used"], "codebook_dead": book["dead"],
                "per_book": book["per_book"], "n_active": len(stats),
                "time": time.time(), "elapsed": time.time() - run_t0,
                **{k: round(v, 5) for k, v in parts.items()},
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(
                f"step {step:>6}  loss {float(total):>7.4f} (ema {ema:>7.4f})  "
                f"recon {float(loss):>6.3f}  vq {float(vq_loss):>7.5f}  "
                f"book {book['perplexity']:>6.1f}/{cfg.codec.codebook_size} "
                f"({book['usage'] * 100:>4.1f}%)  dead {book['dead']:>3}  "
                f"{audio_s / max(dt, 1e-9):>6.1f} audio-s/s  lr {lr:.2e}"
            )
            t0, prev_log_step = time.time(), step

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            metrics = evaluate(model, val_ds, recon, cfg.train.batch_size, cfg.train.eval_batches)
            improved = metrics["val_loss"] < best
            if improved:
                best = metrics["val_loss"]
                save(model, cfg, step, best, out_dir / "ckpt_best.pt")
            logf.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
            logf.flush()
            print(f"           val {metrics['val_loss']:.4f}{'  *best*' if improved else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            path = write_sample(model, val_ds, out_dir, step, cfg.train.sample_seconds)
            print(f"           sample -> {path}  (against samples/original.wav)")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict())
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

    save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict())
    write_sample(model, val_ds, out_dir, step, cfg.train.sample_seconds)
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0, best_val=best)
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val {best:.4f}")
    print(f"  checkpoint {ckpt_last}")
    print(f"  listen     {out_dir / 'samples'}/  (original.wav against step{step:06d}.wav)")
    print(f"  measure    {sys.executable} -m aksharallm.audio report {out_dir / 'ckpt_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
