"""Training the recogniser — waveforms and transcripts in, a Conformer that spells what it hears out.

`python -m aksharallm.asr.train configs/asr-synth.yaml` (or `scripts/audio.sh asr-synth`).

It is its own loop for the codec's reason: the data is whole utterances of varying length,
the loss is CTC, and the number that decides "best" is a **word error rate** rather than a
loss. What it shares is the **contract** every trainer here publishes, which is what lets the
existing tooling drive it unchanged:

* `<out_dir>/train.pid`, claimed on start and released on exit;
* `<out_dir>/STOP`, read every step, in the three forms of `train/stopfile.py`;
* `<out_dir>/train_log.jsonl`, append-only, bracketed by `session_start` / `session_end`;
* `ckpt_last.pt` (with the optimizer and the batch sampler's generator *state*) and
  `ckpt_best.pt`;
* `report.md` on every clean exit.

**What to watch, in order:**

1. **`val_wer`, not `val_loss`** (trap 4). CTC loss keeps falling long after greedy WER has
   stopped improving — the model grows more confident about paths it already gets right.
   `ckpt_best.pt` is chosen by WER, and `session_start` records `metric: "wer"` so nothing
   downstream mistakes `best_val` for a loss.
2. **`silence_chars`** — characters written across five clips with no speech in them,
   measured at every eval (day-two problem 1). Should reach 0 early and stay there.
3. **`skipped`** — utterances dropped from a batch because CTC had too few frames for their
   transcript. Should be 0: `Utterances` already filters them, so a non-zero count means the
   filter and the encoder disagree about frame rates.
4. **The first eval's transcripts.** An untrained CTC model writes *nothing* — every frame's
   argmax is the blank, because the blank is in every path. Empty output with a falling loss
   is the normal first phase, not a bug. It then learns spaces and common letters, then words.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
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

import numpy as np
import torch
import torch.nn.functional as F

from ..config import config_to_dict
from ..train import report, stopfile
from ..train.pretrain import claim_pid_file
from ..train.schedule import get_lr
from . import vocab
from .config import AsrRunConfig, load_asr_config
from .ctc import ctc_loss, greedy_decode
from .data import BatchSampler, Utterances
from .measure import score, silence_check, wer_interval
from .model import AsrModelConfig, Recognizer, seconds_to_frames
from .noise import replace_with_no_speech


def feasibility(cfg: AsrModelConfig, sample_rate: int):
    """`(n_samples, ids) -> bool`: does CTC have enough frames for this transcript?

    A CTC path needs one frame per letter plus one blank between each doubled letter (trap 2).
    """
    def ok(n: int, ids: list[int]) -> bool:
        repeats = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a == b)
        return seconds_to_frames(n / sample_rate, cfg) >= len(ids) + repeats
    return ok


class Mixture:
    """Several corpora, sampled in proportion to their hours."""

    def __init__(self, samplers: list[BatchSampler], seed: int, weights: list[float] | None = None):
        self.samplers = samplers
        if weights is not None:
            if len(weights) != len(samplers) or min(weights) <= 0:
                raise ValueError(f"data.weights needs one positive weight per training corpus "
                                 f"({len(samplers)}), got {weights}")
            w = np.array(weights, dtype=np.float64)
        else:
            w = np.array([s.utts.seconds for s in samplers], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def next(self, device: str) -> dict:
        i = int(self.rng.choice(len(self.samplers), p=self.p)) if len(self.samplers) > 1 else 0
        return self.samplers[i].next(device)

    def state(self) -> dict:
        return {"mix": self.rng.bit_generator.state, "each": [s.state() for s in self.samplers]}

    def load_state(self, st: dict) -> None:
        self.rng.bit_generator.state = st["mix"]
        for s, x in zip(self.samplers, st["each"], strict=True):
            s.load_state(x)


def compute_loss(lp: torch.Tensor, out_len: torch.Tensor, batch: dict, impl: str):
    """Mean CTC over the feasible rows; returns `(loss, n_skipped)`.

    An infeasible row is *dropped and counted* rather than `zero_infinity`'d into the mean —
    counted because a silent fix would hide a frame-rate mismatch for the whole run.
    """
    tgt, tl = batch["targets"], batch["target_lengths"]
    lpt = lp.transpose(0, 1)  # (T, B, V): CTC is time-first
    if impl == "torch":
        per = F.ctc_loss(lpt, tgt, out_len, tl, blank=vocab.BLANK, reduction="none")
    else:
        per = ctc_loss(lpt, tgt, out_len, tl, blank=vocab.BLANK, reduction="none")
    ok = torch.isfinite(per)
    skipped = int((~ok).sum())
    if not ok.any():
        return lp.sum() * 0.0, skipped
    # Each row is divided by its transcript length (PyTorch's "mean"), EXCEPT a no-speech row,
    # whose length is 0. Clamping that to 1 makes a 16-second silent row one term of ~400
    # frames x 3.4 nats — measured as a step-0 loss of 621 and a gradient norm of 29,000 on
    # LibriSpeech — and the batch then learns "always blank" before anything else. Per frame
    # is the same scale as per letter, so an empty row is divided by its frame count instead.
    denom = torch.where(tl > 0, tl, out_len.clamp_min(1)).to(per.dtype)
    return (per[ok] / denom[ok]).mean(), skipped


@torch.no_grad()
def evaluate(model: Recognizer, val: Utterances, cfg: AsrRunConfig, device: str) -> dict:
    """Val CTC loss and greedy WER/CER on the first `eval_utts` validation utterances.

    The same utterances every time, in the same batches, so successive evals are comparable.
    """
    was = model.training
    model.eval()
    idx = list(range(min(cfg.data.eval_utts, len(val))))
    sub_batches = [b for b in val.buckets(cfg.data.max_batch_seconds) if b]
    keep = set(idx)
    pairs, total, n = [], 0.0, 0
    for b in sub_batches:
        b = [i for i in b if i in keep]
        if not b:
            continue
        batch = val.collate(b, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            lp, out_len = model(batch["wave"], batch["n_samples"])
        loss, _ = compute_loss(lp, out_len, batch, "torch")
        total += float(loss) * len(b)
        n += len(b)
        for u, ids in zip(batch["utts"], greedy_decode(lp, out_len, vocab.BLANK), strict=True):
            pairs.append((u.speaker, u.text, vocab.decode(ids)))
    model.train(was)
    s = score(pairs)
    sil = silence_check(model, device)
    return {
        "val_loss": total / max(n, 1),
        "val_wer": s["wer"],
        "val_cer": s["cer"],
        "val_words": s["words"],
        "silence_chars": sil["chars"],
        "examples": [{"ref": r, "hyp": h} for _, r, h in pairs[:3]],
    }


def save(model, cfg: AsrRunConfig, step: int, best: float, path: Path, **extra) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "asr": asdict(cfg.asr),
            "alphabet": vocab.ALPHABET,
            "config": config_to_dict(cfg),
            "step": step,
            "best_val": best,
            "best_metric": "wer",
            "stage": "asr",
            "sample_rate": cfg.asr.sample_rate,
            **extra,
        },
        path,
    )


def load_recognizer(path: str | Path, device: str = "cpu") -> tuple[Recognizer, dict]:
    """A trained recogniser and its checkpoint blob. Refuses anything that is not one."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("stage") != "asr":
        raise ValueError(f"{path} is a {blob.get('stage', 'language-model')!r} checkpoint, "
                         "not a recogniser (stage 'asr')")
    if blob.get("alphabet") != vocab.ALPHABET:
        # A different alphabet means every output id is a different letter. It would not fail;
        # it would transcribe everything confidently wrong.
        raise ValueError(f"{path} was trained with alphabet {blob.get('alphabet')!r}, "
                         f"this code uses {vocab.ALPHABET!r}")
    shape = dict(blob["asr"])
    # Checkpoints from before `block_norm` existed were all trained with the per-block norm.
    shape.setdefault("block_norm", True)
    model = Recognizer(AsrModelConfig(**shape)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.asr.train",
                                description="Train the Conformer-CTC recogniser (docs/23).")
    p.add_argument("config", help="configs/<run>.yaml")
    p.add_argument("-o", "--override", action="append", default=[], metavar="key=value")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    args = p.parse_args(argv)

    cfg = load_asr_config(args.config, args.override)
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sr = cfg.asr.sample_rate
    ok = feasibility(cfg.asr, sr)
    common = {"min_seconds": cfg.data.min_seconds, "max_seconds": cfg.data.max_seconds,
              "feasible": ok}
    hold_out = cfg.data.val is None
    trains = []
    for i, c in enumerate(cfg.data.train):
        split = "train" if (hold_out and i == 0) else "all"
        trains.append(Utterances(c, split=split, val_clips=cfg.data.val_clips, **common))
    if hold_out:
        val = Utterances(cfg.data.train[0], split="val", val_clips=cfg.data.val_clips, **common)
    else:
        val = Utterances(cfg.data.val, **common)
    for u in trains + [val]:
        if u.sr != sr:
            raise SystemExit(f"{u.corpus} is {u.sr} Hz but asr.sample_rate is {sr}. Re-pack it.")

    samplers = [BatchSampler(u, cfg.data.max_batch_seconds, seed=cfg.train.seed + i)
                for i, u in enumerate(trains)]
    mix = Mixture(samplers, cfg.train.seed, cfg.data.weights)

    model = Recognizer(cfg.asr).to(device)
    decay = [q for q in model.parameters() if q.dim() >= 2]
    plain = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.optim.weight_decay},
         {"params": plain, "weight_decay": 0.0}],
        lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2),
    )
    aug = None
    if cfg.augment.enabled:
        aug = {k: getattr(cfg.augment, k)
               for k in ("gain_db", "freq_masks", "freq_width", "time_masks", "time_ratio")}
    gen = torch.Generator().manual_seed(cfg.train.seed)
    noise_rng = np.random.default_rng(cfg.train.seed + 99)

    start_step, best = 0, float("inf")
    ckpt_last = out_dir / "ckpt_last.pt"
    resume = cfg.train.resume
    if resume == "auto":
        resume = str(ckpt_last) if ckpt_last.is_file() else None
    if resume:
        blob = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        if "sampler" in blob:
            mix.load_state(blob["sampler"])
        if "augment_rng" in blob:
            # `.cpu()`: the blob was loaded with map_location=device, so on a GPU run this
            # tensor comes back on the card, and a CPU generator refuses it. Every resume of
            # the first LibriSpeech run died here; the CPU-only resume test could not see it.
            gen.set_state(blob["augment_rng"].cpu())
        if "noise_rng" in blob:
            noise_rng.bit_generator.state = blob["noise_rng"]
        start_step = int(blob.get("step", -1)) + 1
        best = float(blob.get("best_val", float("inf")))
        print(f"resumed {resume} at step {start_step}, best val WER {best:.4f}")
    elif cfg.train.init:
        # Adaptation: another run's weights AND its feature statistics (they are buffers in
        # the state dict), so the features this model sees are scaled exactly as the base was
        # trained on. Re-measuring them on the new corpus would move every input under weights
        # that never saw the new scaling.
        blob = torch.load(cfg.train.init, map_location=device, weights_only=False)
        if blob.get("stage") != "asr":
            raise SystemExit(f"train.init {cfg.train.init} is not a recogniser checkpoint")
        # Shape mismatches fail in load_state_dict. These two do not -- they change what the
        # same tensors mean -- so they are checked by name. Dropout may differ on purpose.
        base = {**{"block_norm": True}, **dict(blob["asr"])}
        for k in ("normalise", "block_norm", "n_mels", "sample_rate"):
            if k in base and getattr(cfg.asr, k, base[k]) != base[k]:
                raise SystemExit(f"train.init: asr.{k} is {getattr(cfg.asr, k)!r} here but "
                                 f"{base[k]!r} in {cfg.train.init}")
        model.load_state_dict(blob["model"])
        print(f"init       weights from {cfg.train.init} (step {blob.get('step')}); "
              "fresh optimizer, step 0")
    elif cfg.asr.normalise == "global":
        # Measured once, on up to 200 training utterances spread across the corpus, and then
        # part of the checkpoint. A resumed run must NOT re-measure: the weights were trained
        # against these exact numbers.
        u0 = trains[0]
        pick = np.linspace(0, len(u0) - 1, min(200, len(u0))).astype(int)
        model.set_feature_stats([torch.from_numpy(u0.wave(u0.utts[i])) for i in pick])

    n_params = model.n_params()
    train_s = sum(u.seconds for u in trains)
    print(f"=== {cfg.name} ===")
    print(f"model      {cfg.asr.describe()}")
    print(f"params     {n_params / 1e6:.2f}M")
    for u in trains:
        print(f"train      {u.corpus}: {len(u):,} utterances, {u.seconds / 3600:.2f} h"
              + (f"  (dropped {dict(u.dropped)})" if u.dropped else ""))
    print(f"val        {val.corpus}: {len(val):,} utterances, {val.seconds / 60:.1f} min; "
          f"WER on the first {min(cfg.data.eval_utts, len(val))}")
    print(f"batch      {cfg.data.max_batch_seconds:g} s of padded audio per step, "
          f"{sum(len(s.batches) for s in samplers)} length buckets")
    print(f"ctc        {cfg.train.ctc_impl}    augment {'on' if aug else 'off'}    "
          f"normalise {cfg.asr.normalise}    no-speech rows {cfg.data.no_speech_ratio:.0%}    "
          f"device {device}")
    if not (resume or cfg.train.init):
        print("expect     empty transcripts at first: an untrained CTC model predicts the blank everywhere")

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
        stop_at=stop_at, stop_by=stop_by, params=n_params, objective="ctc", metric="wer",
        sample_rate=sr, train_hours=round(train_s / 3600, 3),
        batch_seconds=cfg.data.max_batch_seconds,
    )
    if start_step >= cfg.train.max_steps:
        print(f"\nnothing to do: {cfg.name} has already trained its {cfg.train.max_steps:,} steps.")
        log_session("session_end", reason="already_complete",
                    last_step=cfg.train.max_steps - 1, steps=0)
        return 0

    model.train()
    t0 = time.time()
    prev_log_step = start_step - 1
    audio_since = chars_since = skipped_since = no_speech_since = 0
    ema, why, step = None, None, start_step

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, base_lr=cfg.optim.lr, warmup_steps=cfg.optim.warmup_steps,
                    max_steps=cfg.train.max_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                    schedule=cfg.optim.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = mix.next(device)
        no_speech_since += replace_with_no_speech(batch, cfg.data.no_speech_ratio, noise_rng, sr)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            lp, out_len = model(batch["wave"], batch["n_samples"], augment=aug, generator=gen)
        loss, skipped = compute_loss(lp, out_len, batch, cfg.train.ctc_impl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()

        lv = loss.item()
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        audio_since += int(batch["n_samples"].sum())
        chars_since += int(batch["target_lengths"].sum())
        skipped_since += skipped

        if cfg.train.log_every and (step % cfg.train.log_every == 0 or step == cfg.train.max_steps - 1):
            dt = time.time() - t0
            n_steps = max(1, step - prev_log_step)
            sps = dt / n_steps
            rec = {
                "step": step, "loss": lv, "ema": ema, "lr": lr, "grad_norm": float(grad_norm),
                "s_per_step": sps,
                # Transcript characters per second, under the key the portal's throughput
                # chart reads. A character is this model's output token.
                "tok_per_sec": chars_since / max(dt, 1e-9),
                "audio_s_per_s": audio_since / sr / max(dt, 1e-9),
                "skipped": skipped_since, "no_speech": no_speech_since, "elapsed": time.time() - run_t0,
                "eta_s": (cfg.train.max_steps - step - 1) * sps, "time": time.time(),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:>6}  loss {lv:>7.4f} (ema {ema:>7.4f})  gnorm {float(grad_norm):>6.2f}  "
                  f"{rec['audio_s_per_s']:>6.0f} audio-s/s  {sps:.2f} s/step  lr {lr:.2e}"
                  + (f"  skipped {skipped_since}" if skipped_since else ""))
            t0, prev_log_step = time.time(), step
            audio_since = chars_since = skipped_since = no_speech_since = 0

        if cfg.train.eval_every and step > start_step and step % cfg.train.eval_every == 0:
            m = evaluate(model, val, cfg, device)
            improved = m["val_wer"] < best
            if improved:
                best = m["val_wer"]
                save(model, cfg, step, best, out_dir / "ckpt_best.pt")
            examples = m.pop("examples")
            logf.write(json.dumps({"step": step, "time": time.time(), **m}) + "\n")
            logf.flush()
            pm = wer_interval(m["val_wer"], m["val_words"])
            print(f"           val loss {m['val_loss']:.4f}  WER {m['val_wer'] * 100:.1f}% "
                  f"± {pm * 100:.1f}  CER {m['val_cer'] * 100:.1f}%  "
                  f"silence {m['silence_chars']} chars{'  *best*' if improved else ''}")
            for ex in examples[:2]:
                print(f"             ref: {ex['ref'][:90]}")
                print(f"             hyp: {ex['hyp'][:90] or '(nothing)'}")
            t0 = time.time()

        if cfg.train.ckpt_every and step > start_step and step % cfg.train.ckpt_every == 0:
            save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict(),
                 sampler=mix.state(), augment_rng=gen.get_state(),
                 noise_rng=noise_rng.bit_generator.state)
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

    save(model, cfg, step, best, ckpt_last, optimizer=opt.state_dict(),
         sampler=mix.state(), augment_rng=gen.get_state(),
                 noise_rng=noise_rng.bit_generator.state)
    if not (out_dir / "ckpt_best.pt").exists():
        save(model, cfg, step, best, out_dir / "ckpt_best.pt")
    reason = why or "max_steps"
    log_session("session_end", reason=reason, last_step=step, trained_to=step,
                steps=step - start_step + 1, elapsed=time.time() - run_t0, best_val=best)
    logf.close()
    stop_file.unlink(missing_ok=True)
    print(f"\nstopped: {reason}. last step {step}, best val WER "
          + (f"{best * 100:.1f}%" if best < float("inf") else "(not measured yet)"))
    print(f"  checkpoint {ckpt_last}")
    print(f"  measure    {sys.executable} -m aksharallm.asr eval {out_dir / 'ckpt_best.pt'}")
    report.write_quietly(out_dir, run=cfg.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
