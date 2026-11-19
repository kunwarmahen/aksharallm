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

import numpy as np
import torch

from ..config import ModelConfig, OptimConfig, config_to_dict, load_into
from ..train import stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from .codec import load_codec
from .config import AudioTrainConfig
from .dataset import CodeDataset, UtteranceDataset, normalise_text
from .io import write_wav
from .lm import AudioLM, AudioLMConfig, generate, training_pair
from .speech import speak, transcribe, utterance_loss
from .text import CharTokenizer, load_transcripts


@dataclass
class AudioLMDataConfig:
    codes: str = "data/audio/synth-codes"
    codec: str = "checkpoints/codec-synth/ckpt_best.pt"
    window_frames: int = 200  # 4 s at 50 frames/s
    val_clips: int = 16
    #: "audio" = continue sound (windows of codes); "tts" = text -> audio and "asr" = audio
    #: -> text, both over whole utterances with their transcripts (`speech.utterance_loss`).
    task: str = "audio"
    #: Where the transcripts are found for tts/asr (`text.load_transcripts`): the packed
    #: corpus the codes were made from. Empty = the codes directory.
    corpus: str = ""


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
    if cfg.train.init:
        # The field is shared with the codec trainer's config; this trainer has no warm start,
        # and a key that is read by nothing would look like it had worked.
        raise ValueError("train.init is a codec-trainer option; the audio LM does not read it")
    return cfg


@torch.no_grad()
def evaluate(model: AudioLM, ds, pad_id: int, batch_size: int, batches: int,
             task: str = "audio", seed: int = 0) -> float:
    """Mean loss over `batches` validation batches -- the SAME batches every time.

    The sampler is re-seeded on each call. Without it every evaluation drew different clips
    and "best val" partly recorded the easiest draw (the diffusion validation paid for this
    lesson first: docs/20).
    """
    was = model.training
    model.eval()
    ds.rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(batches):
        if task == "audio":
            inputs, targets = training_pair(ds.batch(batch_size), pad_id, model.bos_id)
            _, loss = model(inputs, targets=targets)
        else:
            clips, texts = ds.batch(batch_size)
            loss = utterance_loss(model, clips, texts, text_first=task == "tts", with_stop=False)
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

    task = cfg.data.task
    if task not in ("audio", "tts", "asr"):
        raise SystemExit(f"data.task must be audio, tts or asr, not {task!r}")
    start_step, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = str(ckpt_last) if cfg.train.resume == "auto" and ckpt_last.is_file() else (
        cfg.train.resume if cfg.train.resume != "auto" else None
    )
    blob = torch.load(resume, map_location=device, weights_only=False) if resume else None

    tok = None
    if task == "audio":
        train_ds = CodeDataset(cfg.data.codes, cfg.data.window_frames, device,
                               n_codebooks=cfg.audiolm.n_codebooks, seed=cfg.train.seed,
                               split="train", val_clips=cfg.data.val_clips)
        val_ds = CodeDataset(cfg.data.codes, cfg.data.window_frames, device,
                             n_codebooks=cfg.audiolm.n_codebooks, seed=cfg.train.seed + 1,
                             split="val", val_clips=cfg.data.val_clips)
    else:
        transcripts = load_transcripts(cfg.data.corpus or cfg.data.codes)
        # The tokenizer IS the text embedding's index (gotcha 3), so a resume reuses the
        # one it trained with rather than rebuilding it from whatever the corpus says today.
        tok = (CharTokenizer.from_dict(blob["tokenizer"]) if blob and "tokenizer" in blob
               else CharTokenizer.from_texts(normalise_text(t) for t in transcripts.values()))
        cfg.audiolm.text_vocab_size = tok.vocab_size
        kw = dict(n_codebooks=cfg.audiolm.n_codebooks, max_frames=cfg.audiolm.max_frames,
                  max_text=cfg.audiolm.max_text, val_clips=cfg.data.val_clips)
        train_ds = UtteranceDataset(cfg.data.codes, transcripts, tok, split="train",
                                    seed=cfg.train.seed, **kw)
        val_ds = UtteranceDataset(cfg.data.codes, transcripts, tok, split="val",
                                  seed=cfg.train.seed + 1, **kw)

    model = AudioLM(cfg.audiolm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)

    if resume:
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", -1)) + 1
        best = float(blob.get("best_val", float("inf")))
        print(f"resumed {resume} at step {start_step}, best val {best:.4f}")

    # ASR supervises text, so "knows nothing" is uniform over the characters instead.
    uniform = math.log(tok.vocab_size if task == "asr" else cfg.audiolm.codebook_size)
    if task == "tts" and cfg.audiolm.predict_stop:
        uniform += math.log(2)   # the stop head's BCE starts at ln 2 too; val excludes it
    print(f"=== {cfg.name} ===")
    print(f"model      {model.num_params() / 1e6:.2f}M params, "
          f"{cfg.audiolm.n_codebooks} heads x {cfg.audiolm.codebook_size} codes")
    print(f"codec      {cfg.data.codec} ({codec.cfg.describe()})")
    if task == "audio":
        print(f"window     {cfg.data.window_frames} frames = "
              f"{cfg.data.window_frames / codec.cfg.frames_per_second:.1f}s -> "
              f"{cfg.data.window_frames + cfg.audiolm.n_codebooks} positions with the delay + BOS")
    else:
        print(f"task       {task}: whole utterances up to {cfg.audiolm.max_frames} frames "
              f"({cfg.audiolm.max_frames / codec.cfg.frames_per_second:.1f}s) and "
              f"{cfg.audiolm.max_text} characters; {tok.vocab_size} text ids")
        print(f"dropped    train " + ", ".join(f"{v} {k.replace('_', ' ')}"
                                               for k, v in train_ds.dropped.items()))
    print(f"corpus     {len(train_ds)} train clips / {len(val_ds)} val")
    print(f"expect     step-0 loss ~= ln({tok.vocab_size if task == 'asr' else cfg.audiolm.codebook_size})"
          + (" + ln 2 (stop head)" if task == "tts" and cfg.audiolm.predict_stop else "")
          + f" = {uniform:.4f}" + ("; val is the audio alone" if cfg.audiolm.predict_stop else ""))
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
    # Utterances vary in length, so a text task counts the codes it actually saw.
    tokens_per_step = cfg.train.batch_size * cfg.audiolm.n_codebooks * (
        cfg.data.window_frames if task == "audio" else cfg.audiolm.max_frames)
    seen = 0

    log_session("session_start", pid=os.getpid(), start_step=start_step,
                max_steps=cfg.train.max_steps, stop_at=stop_at, stop_by=stop_by,
                params=model.num_params(), objective="audiolm", metric="ce",
                tokens_per_step=tokens_per_step)

    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    # Saved with every checkpoint: a text model is unusable without the ids it learned.
    extra = {"task": task, **({"tokenizer": tok.to_dict()} if tok else {})}
    if task == "tts":
        # How many frames a character takes in this corpus: what `speak` falls back on for
        # a model with no stop head, saved so the checkpoint can speak without the corpus.
        extra["frames_per_char"] = train_ds.frames_per_char()
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

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            if task == "audio":
                # [BOS] + delayed codes: the layout generation starts from (lm.training_pair).
                d, targets = training_pair(train_ds.batch(cfg.train.batch_size),
                                           model.pad_id, model.bos_id)
                _, loss = model(d, targets=targets)
                seen += tokens_per_step
            else:
                clips, texts = train_ds.batch(cfg.train.batch_size)
                loss = utterance_loss(model, clips, texts, text_first=task == "tts")
                seen += sum(int(c.numel()) for c in clips)

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
                   "tok_per_sec": seen / max(dt, 1e-9),
                   "ppl": math.exp(min(float(loss), 20)),
                   "time": time.time(), "elapsed": time.time() - run_t0}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {float(loss):>7.4f} (ema {ema:>7.4f})  "
                  f"ppl {rec['ppl']:>7.1f}  {rec['tok_per_sec']:>8.0f} tok/s  lr {lr:.2e}"
                  + ("   <- uniform" if abs(float(loss) - uniform) < 0.02 else ""))
            t0, prev_log_step, seen = time.time(), step, 0

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            val = evaluate(model, val_ds, model.pad_id, cfg.train.batch_size,
                           cfg.train.eval_batches, task, cfg.train.seed + 1)
            improved = val < best
            if improved:
                best = val
                _save(model, cfg, step, best, out_dir / "ckpt_best.pt", **extra)
            logf.write(json.dumps({"step": step, "time": time.time(), "val_loss": val}) + "\n")
            logf.flush()
            print(f"           val {val:.4f}{'  *best*' if improved else ''}")
            t0 = time.time()

        if cfg.train.sample_every and step > start_step and step % cfg.train.sample_every == 0:
            path = _sample(model, codec, out_dir, step, cfg, val_ds, tok)
            print(f"           sample -> {path}")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            _save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict(), **extra)
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

    _save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict(), **extra)
    _sample(model, codec, out_dir, step, cfg, val_ds, tok)
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
def _sample(model: AudioLM, codec, out_dir: Path, step: int, cfg: AudioLMRunConfig,
            val_ds=None, tok=None) -> Path:
    """Write something to listen to (or read): the loss says nothing about either.

    audio: a few seconds of continuation. tts: the first validation sentence, spoken at its
    real length, with the text beside it. asr: that clip transcribed, printed under the truth.
    """
    device = next(model.parameters()).device
    task = cfg.data.task
    if task in ("tts", "asr"):
        clip, ids = val_ds.get(0)
        ref = tok.decode(ids[1:-1])
        if task == "asr":
            hyp = transcribe(model, clip.unsqueeze(0), tok, device=device)
            path = out_dir / "samples" / f"step{step:06d}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"truth: {ref}\nheard: {hyp}\n")
            print(f"           truth  {ref}\n           heard  {hyp}")
            model.train()
            return path
        # Its own length when it can choose one (`predict_stop`): that is what is on trial.
        codes = speak(model, ids, None if model.stop_head is not None else clip.shape[-1],
                      device=device)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "samples" / f"step{step:06d}.txt").write_text(ref + "\n")
    else:
        frames = min(cfg.audiolm.max_frames, int(3 * codec.cfg.frames_per_second))
        codes = generate(model, frames, device=device)
    wave = codec.decode(codes.to(device))[0].float().cpu().numpy()
    path = out_dir / "samples" / f"step{step:06d}.wav"
    write_wav(path, wave, codec.cfg.sample_rate)
    model.train()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
