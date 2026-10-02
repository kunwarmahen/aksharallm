"""The Dictation tab's back end: speak, and see what our own recogniser heard.

Phase 7 is "dictation that is still right on day two", and the tab is built in that order:

1. **Say something** — the browser records the microphone (or decodes a file you drop on
   it), sends 16-bit samples at whatever rate it has, and gets back what the recogniser
   wrote. It is the only tab whose input is *you*, which is the point: LibriSpeech is read
   audiobooks, and a recogniser's test set is the most forgiving speech it will ever hear.
2. **The five day-two checks**, as they come in. Silence first, because it is the one that is
   already measured on every eval (`asr/measure.py`).
3. **Every evaluation so far** — WER with its error bar and the worst speakers beside the
   mean, from `logs/asr/*.json`. Never the mean alone (day-two problem 2).

**Resampling is ours and is declared.** A browser microphone runs at 44.1 or 48 kHz; the model
hears 16. The conversion is `audio/io.resample` — the windowed-sinc resampler built and tested
for the audio phase — and the response says the rate it came in at, because a silent
conversion is exactly what trap 5 of the audio phase warns about.

Runs inline, like `vision` and `audio`: a few seconds of speech is one forward pass of a 20M
encoder. Device policy is the repo's standing one, borrowed from the Playground: the CPU
while a training run holds the card.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

#: Thirty seconds of speech is a long dictated paragraph and a few hundred milliseconds of
#: compute. Past that, a request is more likely a mistake than a sentence.
MAX_SECONDS = 30.0
#: The request body limit for this route only: 30 s at 48 kHz as base64 int16 is ~3.9 MB.
MAX_BODY = 8 * 1024 * 1024


class DictationError(RuntimeError):
    """Something to show as a message rather than a stack trace."""


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


class Dictation:
    def __init__(self, root: Path | None = None, device_for=None):
        self.root = Path(root) if root else Path.cwd()
        self._device_for = device_for
        self._cache: dict = {}

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        if torch.cuda.is_available():
            return "cuda", "the card is free"
        return "cpu", "no CUDA device"

    # ---- discovery --------------------------------------------------------------------

    def checkpoints(self) -> list[dict]:
        """Recognisers, identified by `stage == "asr"` rather than by a filename convention."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "asr":
                continue
            a = blob.get("asr") or {}
            out.append({
                "rel": str(path.relative_to(self.root)),
                "run": path.parent.name,
                "step": blob.get("step"),
                "best_wer": blob.get("best_val") if math.isfinite(blob.get("best_val") or math.inf) else None,
                "shape": f"d={a.get('d_model')} x{a.get('n_layers')}",
                "mtime": path.stat().st_mtime,
            })
        # Best checkpoints first, then newest: `ckpt_best` is what you meant to talk to.
        return sorted(out, key=lambda r: (not r["rel"].endswith("ckpt_best.pt"), -r["mtime"]))

    def results(self) -> list[dict]:
        """Every `asr eval` result, newest first, without the per-utterance examples."""
        out = []
        for p in sorted((self.root / "logs/asr").glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if d.get("kind") != "asr_eval":
                continue
            sp = d.get("speakers") or {}
            rates = sorted(v["wer"] for v in sp.values()) if sp else []
            out.append({
                "file": p.name, "run": d.get("run"), "step": d.get("step"),
                "corpus": Path(d.get("corpus", "")).name, "time": d.get("time"),
                "wer": d.get("wer"), "wer_pm": d.get("wer_pm"), "cer": d.get("cer"),
                "words": d.get("words"), "utts": d.get("utts"),
                "speakers": len(rates),
                "median_speaker": rates[len(rates) // 2] if rates else None,
                "worst": (d.get("worst_speakers") or [])[:3],
                "silence_chars": (d.get("silence") or {}).get("chars"),
            })
        return sorted(out, key=lambda r: r.get("time") or "", reverse=True)

    def runs(self) -> list[dict]:
        """Recogniser runs and where they have got to: step, latest val WER, silence chars."""
        out = []
        for cfg_path in sorted(self.root.glob("configs/*.yaml")):
            if not cfg_path.read_text().startswith("asr:") and "\nasr:" not in cfg_path.read_text():
                continue
            name = cfg_path.stem
            run_dir = self.root / "checkpoints" / name
            step = wer = silence = None
            log = run_dir / "train_log.jsonl"
            if log.is_file():
                for line in reversed(log.read_text().splitlines()[-600:]):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if wer is None and "val_wer" in rec:
                        wer, silence = rec["val_wer"], rec.get("silence_chars")
                    if step is None and "step" in rec:
                        step = rec["step"]
                    if step is not None and wer is not None:
                        break
            out.append({"name": name, "training": _alive(run_dir / "train.pid"),
                        "step": step, "val_wer": wer, "silence_chars": silence})
        return out

    def overview(self) -> dict:
        device, why = self.device()
        return {"checkpoints": self.checkpoints(), "results": self.results(), "runs": self.runs(),
                "device": device, "device_reason": why, "max_seconds": MAX_SECONDS}

    # ---- the model ----------------------------------------------------------------------

    def _model(self, rel: str):
        from ..asr.train import load_recognizer

        device, _ = self.device()
        key = f"{rel}@{device}"
        if self._cache.get("key") != key:
            path = (self.root / rel).resolve()
            if not str(path).startswith(str(self.root.resolve())):
                raise DictationError("checkpoint path escapes the repository")
            if not path.is_file():
                raise DictationError(f"no such checkpoint: {rel}")
            try:
                model, blob = load_recognizer(path, device)
            except ValueError as e:
                raise DictationError(str(e)) from e
            self._cache = {"key": key, "model": model, "step": blob.get("step")}
        return self._cache["model"], self._cache["step"], device

    def transcribe(self, checkpoint: str, pcm_b64: str, sample_rate: int) -> dict:
        """Base64 little-endian int16 mono at `sample_rate` -> what the recogniser wrote."""
        from ..asr.measure import transcribe
        from ..audio.io import resample

        if not checkpoint:
            raise DictationError("pick a recogniser first")
        try:
            raw = base64.b64decode(pcm_b64, validate=True)
        except (ValueError, TypeError) as e:
            raise DictationError("the audio did not arrive as base64 int16") from e
        if not 8_000 <= int(sample_rate) <= 192_000:
            raise DictationError(f"a sample rate of {sample_rate} Hz is not a microphone's")
        x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        seconds_in = len(x) / sample_rate
        if seconds_in < 0.2:
            raise DictationError("that was under 0.2 seconds of audio — hold the button while you speak")
        if seconds_in > MAX_SECONDS:
            raise DictationError(f"{seconds_in:.0f} s is past the {MAX_SECONDS:.0f} s this tab accepts")
        model, step, device = self._model(checkpoint)
        sr = model.cfg.sample_rate
        if sample_rate != sr:
            x = resample(x, int(sample_rate), sr).astype(np.float32)
        peak = float(np.abs(x).max()) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        t0 = time.time()
        text = transcribe(model, [x], device)[0]
        ms = (time.time() - t0) * 1000
        return {
            "text": text, "seconds": round(len(x) / sr, 2), "ms": round(ms, 1),
            "realtime": round(len(x) / sr / max(ms / 1000, 1e-9), 1),
            "rate_in": int(sample_rate), "rate_model": sr, "resampled": sample_rate != sr,
            "peak_dbfs": round(20 * math.log10(max(peak, 1e-9)), 1),
            "rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 1),
            "device": device, "step": step, "checkpoint": checkpoint,
        }

    def silence(self, checkpoint: str) -> dict:
        """Day-two check 1, live: what it writes on five clips with no speech in them."""
        from ..asr.measure import silence_check

        model, step, device = self._model(checkpoint)
        r = silence_check(model, device)
        return {"checkpoint": checkpoint, "step": step, "device": device, **r}
