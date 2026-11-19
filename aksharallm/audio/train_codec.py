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
from .content import ContentHead, content_loss, load_teacher, teacher_states
from .config import CodecRunConfig, load_codec_config
from .dataset import AudioDataset, MixedAudioDataset
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


def evaluate_sources(model: Codec, val_sets: dict, weights: dict, recon, batch_size: int,
                     batches: int) -> dict:
    """`evaluate` on every corpus's own held-out tail, and the weighted mean of them.

    One corpus: exactly `evaluate`, key for key. Several: `val_loss` is the weighted mean
    (what "best" is chosen on) and each corpus adds `val_loss_<name>`, because the question
    a many-voices codec has to answer is "did the new voices improve WITHOUT the old one
    getting worse", and a blended number cannot say that.
    """
    if len(val_sets) == 1:
        (ds,) = val_sets.values()
        return evaluate(model, ds, recon, batch_size, batches)
    out, total = {}, 0.0
    for name, ds in val_sets.items():
        m = evaluate(model, ds, recon, batch_size, batches)
        out[f"val_loss_{name}"] = m["val_loss"]
        total += weights[name] * m["val_loss"]
    out["val_loss"] = total
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
    samples.mkdir(parents=True, exist_ok=True)
    write_wav(samples / "original.wav", x[0].float().cpu().numpy(), sr)
    path = samples / f"step{step:06d}.wav"
    write_wav(path, y[0].float().cpu().numpy(), sr)
    model.train(was)
    return path


def warm_start(model: Codec, cfg: CodecRunConfig, path: str, device: str) -> None:
    """Load another codec run's weights into `model`, refusing any architecture change.

    `load_state_dict` would catch a changed width, but not a changed `strides` order or
    `codebook_size` stored only in buffers of the same shape, so the configs are compared
    first, field by field. A different `quantizer_dropout` or `restart_after` is a training
    choice, not a shape, and is allowed.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("stage") != "codec":
        raise SystemExit(f"{path} is not a codec checkpoint (stage {blob.get('stage')!r})")
    theirs, ours = blob["codec"], asdict(cfg.codec)
    training_only = {"quantizer_dropout", "restart_after", "commit", "decay"}
    diff = {k: (theirs.get(k), v) for k, v in ours.items()
            if k not in training_only and theirs.get(k) != v
            and not (isinstance(v, (list, tuple)) and list(theirs.get(k) or []) == list(v))}
    if diff:
        raise SystemExit(f"train.init {path} has a different codec: "
                         + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in diff.items()))
    model.load_state_dict(blob["model"])
    model.to(device)


def save(model: Codec, cfg: CodecRunConfig, step: int, best: float, path: Path,
         head: ContentHead | None = None, **extra) -> None:
    if head is not None:
        # Beside the codec, never inside its state dict: every reader of a codec checkpoint
        # loads `model` into a plain `Codec`, and the projection is training-only.
        extra["content_head"] = head.state_dict()
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
    corpora = cfg.data.corpora()
    if len(corpora) == 1:
        train_ds = AudioDataset(
            corpora[0]["corpus"], cfg.window, device, seed=cfg.train.seed,
            split="train", val_clips=cfg.data.val_clips,
        )
    else:
        train_ds = MixedAudioDataset(corpora, cfg.window, device, seed=cfg.train.seed,
                                     split="train", val_clips=cfg.data.val_clips)
    # Every corpus keeps its own held-out tail, evaluated on its own.
    val_sets = {
        Path(c["corpus"]).name: AudioDataset(
            c["corpus"], cfg.window, device, seed=cfg.train.seed + 1 + i,
            split="val", val_clips=cfg.data.val_clips)
        for i, c in enumerate(corpora)
    }
    wsum = sum(float(c.get("weight", 1.0)) for c in corpora)
    val_weights = {Path(c["corpus"]).name: float(c.get("weight", 1.0)) / wsum for c in corpora}
    if len(val_weights) != len(corpora):
        raise SystemExit("two data.sources share a directory name; val keys would collide")
    val_ds = next(iter(val_sets.values()))
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

    teacher = head = None
    if cfg.loss.content_weight > 0:
        if not cfg.loss.content_teacher:
            raise SystemExit("loss.content_weight > 0 needs loss.content_teacher (a recogniser .pt)")
        teacher = load_teacher(cfg.loss.content_teacher, device)
        if teacher.cfg.sample_rate != cfg.codec.sample_rate:
            raise SystemExit(f"the content teacher hears {teacher.cfg.sample_rate} Hz, the codec "
                             f"{cfg.codec.sample_rate} Hz")
        head = ContentHead(cfg.codec.dim, teacher.cfg.d_model).to(device)

    # No weight decay on the Snake alphas or any 1-d parameter: decaying a per-channel gain
    # towards zero is not regularisation, it is a slow deletion of the activation.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.optim.weight_decay},
              {"params": plain, "weight_decay": 0.0}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2))
    trained = list(model.parameters()) + (list(head.parameters()) if head is not None else [])

    start_step, best = 0, float("inf")
    resume = cfg.train.resume
    ckpt_last = out_dir / "ckpt_last.pt"
    if resume == "auto":
        resume = str(ckpt_last) if ckpt_last.is_file() else None
    if resume:
        blob = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if head is not None:
            if "content_head" not in blob:
                # Turning the content loss on in the middle of a run changes the optimizer's
                # groups and the objective at once; start it as its own run (train.init).
                raise SystemExit(f"{resume} was trained without the content loss; resume it "
                                 "without it, or warm-start a new run from it with train.init")
            head.load_state_dict(blob["content_head"])
        elif "content_head" in blob:
            raise SystemExit(f"{resume} was trained with the content loss; resume it with it")
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", -1)) + 1
        best = float(blob.get("best_val", float("inf")))
        print(f"resumed {resume} at step {start_step}, best val {best:.4f}")
    elif cfg.train.init:
        warm_start(model, cfg, cfg.train.init, device)
        print(f"init       weights from {cfg.train.init} (fresh optimizer, step 0)")

    print(f"=== {cfg.name} ===")
    print(f"codec      {cfg.codec.describe()}")
    print(f"params     {counts['total'] / 1e6:.2f}M "
          f"(encoder {counts['encoder'] / 1e6:.2f}M, decoder {counts['decoder'] / 1e6:.2f}M) "
          f"+ {counts['codebooks'] / 1e6:.2f}M of codebook (not trained by the optimizer)")
    print(f"corpus     {train_ds.seconds / 3600:.2f} h train / "
          f"{sum(v.seconds for v in val_sets.values()) / 60:.1f} min val, {len(train_ds)} clips")
    if len(corpora) > 1:
        for (name, w), part in zip(val_weights.items(), train_ds.parts):
            print(f"  source   {name:<18} weight {w:.2f}  {part.seconds / 3600:7.2f} h")
    print(f"window     {cfg.window} samples = {cfg.window / cfg.codec.sample_rate:.2f}s "
          f"= {cfg.window // cfg.codec.hop} frames, batch {cfg.train.batch_size}")
    print(f"device     {device}")
    if head is not None:
        print(f"content    codebook 0 -> {cfg.loss.content_teacher} "
              f"(d={teacher.cfg.d_model}), weight {cfg.loss.content_weight:g}")

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
            if head is None:
                y, _, vq_loss, stats = model(x)
            else:
                y, _, vq_loss, stats, q0 = model(x, first=True)
                target = teacher_states(teacher, x, q0.shape[-1])
        loss, parts = recon(x, y.float())
        total = loss + cfg.loss.vq_weight * vq_loss.float()
        if head is not None:
            closs = content_loss(head, q0, target)
            total = total + cfg.loss.content_weight * closs
            parts["content"] = float(closs)

        opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trained, cfg.optim.grad_clip)
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
                + (f"content {parts['content']:.3f}  " if "content" in parts else "")
                + f"book {book['perplexity']:>6.1f}/{cfg.codec.codebook_size} "
                f"({book['usage'] * 100:>4.1f}%)  dead {book['dead']:>3}  "
                f"{audio_s / max(dt, 1e-9):>6.1f} audio-s/s  lr {lr:.2e}"
            )
            t0, prev_log_step = time.time(), step

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            metrics = evaluate_sources(model, val_sets, val_weights, recon,
                                       cfg.train.batch_size, cfg.train.eval_batches)
            improved = metrics["val_loss"] < best
            if improved:
                best = metrics["val_loss"]
                save(model, cfg, step, best, out_dir / "ckpt_best.pt", head)
            logf.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
            logf.flush()
            per = "  ".join(f"{k[9:]} {v:.4f}" for k, v in metrics.items()
                            if k.startswith("val_loss_"))
            print(f"           val {metrics['val_loss']:.4f}{'  *best*' if improved else ''}"
                  f"{'  (' + per + ')' if per else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            path = write_sample(model, val_ds, out_dir, step, cfg.train.sample_seconds)
            print(f"           sample -> {path}  (against samples/original.wav)")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            save(model, cfg, step, best, ckpt_last, head, optimizer=opt.state_dict())
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

    save(model, cfg, step, best, ckpt_last, head, optimizer=opt.state_dict())
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
