"""The Audio tab's back end: hear the bitrate/quality trade, instead of reading about it.

Shape of this one, and why
--------------------------
The panel this tab exists for is the **bitrate ladder** — the same clip decoded from 1, 2, 4
and 8 codebooks, played side by side in a browser. Every other visualisation in this repo
asks you to interpret a chart. This one asks you to listen, and the trade it makes audible
is the same one [quantization](../../docs/11-quantization.md) makes silently in the weights.

Like `interp` and `diffusion`, everything runs **inline** rather than as a subprocess job: a
codec is a few million parameters and reconstructing three seconds of audio is one forward
pass. Unlike those two it does **not** use the Playground's engine, because a codec is not a
language model — it has its own loader and its own checkpoint family, and `load_codec`
refuses a text checkpoint by name rather than failing on mismatched keys.

Device policy is the repo's standing one: the CPU while a training run holds the card. A
codec forward on the CPU is a second or two for a short clip, which is fine for listening
and is why this tab needs no GPU reservation.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import base64
import io
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch

from ..audio.codec import load_codec
from ..audio.dataset import AudioDataset, Manifest
from ..audio.features import MelConfig, log_mel, magnitude, spectral_convergence
from ..audio.measure import codebook_usage, mcd, reconstruct

#: Ceiling on how much audio one request may reconstruct. Eight seconds through a codec on
#: the CPU is a couple of seconds of work; a browser must not be able to ask for a minute.
MAX_SECONDS = 8.0
#: The rungs of the ladder. Anything past the codec's own codebook count is dropped, not faked.
RUNGS = (1, 2, 4, 8)


class AudioError(RuntimeError):
    """Something the tab should show as a message rather than a stack trace."""


def wav_data_uri(samples: np.ndarray, sample_rate: int) -> str:
    """A `data:` URI holding 16-bit PCM, so `<audio src=...>` needs no extra route.

    Inlining the audio rather than serving it from a path is deliberate: a reconstruction is
    derived from a checkpoint and a clip and has no stable identity, so a URL for it would be
    a cache-invalidation problem in exchange for nothing. A few seconds of 16 kHz mono is
    ~250 kB of base64, which a loopback page carries without noticing.
    """
    ints = np.clip(np.round(np.asarray(samples, dtype=np.float32) * 32767.0), -32768, 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(ints.astype("<i2").tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def spectrogram_rows(wave_t: torch.Tensor, cfg: MelConfig, width: int = 160) -> dict:
    """A log-mel as a small integer grid, 0-100, for the browser to paint.

    Sent as numbers rather than as a PNG because the page already draws its own charts and
    because a grid can be re-coloured, hovered and compared without a round trip. Downsampled
    by **nearest neighbour**: averaging would be prettier and would also hide a single dead
    mel band, which is exactly the thing worth spotting.
    """
    m = log_mel(wave_t, cfg).float().numpy()
    if m.shape[1] > width:
        xs = np.linspace(0, m.shape[1] - 1, width).round().astype(int)
        m = m[:, xs]
    lo, hi = float(m.min()), float(m.max())
    scaled = ((m - lo) / max(hi - lo, 1e-9) * 100).round().astype(int)
    return {
        "bands": m.shape[0],
        "frames": m.shape[1],
        # Low frequencies last, so the browser can render rows top-down and get the
        # conventional picture with no reversing logic of its own.
        "rows": scaled[::-1].tolist(),
        "db_low": round(lo, 2),
        "db_high": round(hi, 2),
    }


class Audio:
    """Everything the Audio tab can ask for."""

    def __init__(self, root: Path | None = None, device_for=None):
        self.root = Path(root) if root else Path.cwd()
        #: Injected so this shares the Playground's "is a run training?" answer rather than
        #: forming a second opinion about who owns the card.
        self._device_for = device_for
        self._cache: dict[str, object] = {}

    # ---- discovery --------------------------------------------------------------------

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        if torch.cuda.is_available():
            return "cuda", "the GPU is free"
        return "cpu", "no CUDA device"

    def checkpoints(self) -> list[dict]:
        """Every codec checkpoint under `checkpoints/`, newest first.

        Identified by the `stage` field rather than by a filename convention, because a run
        directory can hold both families and a name is not a contract.
        """
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "codec":
                continue
            cfg = blob.get("codec") or {}
            books, size = cfg.get("n_codebooks", 0), cfg.get("codebook_size", 0)
            hop = math.prod(cfg.get("strides", [1]))
            fps = cfg.get("sample_rate", 16000) / max(hop, 1)
            out.append({
                "rel": str(path.relative_to(self.root)),
                "step": blob.get("step"),
                "best_val": blob.get("best_val"),
                "sample_rate": cfg.get("sample_rate"),
                "n_codebooks": books,
                "codebook_size": size,
                "frames_per_second": round(fps, 1),
                "kbps": round(fps * books * math.log2(max(size, 2)) / 1000, 2),
                "mtime": path.stat().st_mtime,
            })
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def corpora(self) -> list[dict]:
        """Packed corpora on disk — what there is to listen to."""
        out = []
        for man_path in sorted(self.root.glob("data/audio/*/manifest.json")):
            if not (man_path.parent / "audio.bin").is_file():
                continue  # a codes-only directory has no waveforms to play
            try:
                man = Manifest.load(man_path)
            except Exception:
                continue
            out.append({
                "rel": str(man_path.parent.relative_to(self.root)),
                "clips": man.n_clips,
                "hours": round(man.seconds / 3600, 3),
                "sample_rate": man.sample_rate,
            })
        return out

    def overview(self) -> dict:
        device, why = self.device()
        return {
            "checkpoints": self.checkpoints(),
            "corpora": self.corpora(),
            "device": device,
            "device_reason": why,
            "rungs": list(RUNGS),
            "max_seconds": MAX_SECONDS,
        }

    # ---- loading ----------------------------------------------------------------------

    def _codec(self, rel: str):
        device, _ = self.device()
        key = f"{rel}@{device}"
        if self._cache.get("key") != key:
            path = (self.root / rel).resolve()
            if not str(path).startswith(str(self.root.resolve())):
                raise AudioError("checkpoint path escapes the repository")
            if not path.is_file():
                raise AudioError(f"no such checkpoint: {rel}")
            try:
                model = load_codec(path, device)
            except ValueError as e:
                raise AudioError(str(e)) from e
            self._cache = {"key": key, "model": model}
        return self._cache["model"]

    def _clip(self, corpus: str, index: int, seconds: float, sample_rate: int):
        path = (self.root / corpus).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise AudioError("corpus path escapes the repository")
        try:
            ds = AudioDataset(path, window=1, split="val", val_clips=16)
        except Exception as e:
            raise AudioError(f"cannot read {corpus}: {e}") from e
        if ds.manifest.sample_rate != sample_rate:
            raise AudioError(
                f"{corpus} is {ds.manifest.sample_rate} Hz and the codec is {sample_rate} Hz. "
                "Re-pack the corpus — resampling here would leave the two disagreeing about "
                "what a frame is."
            )
        clips = ds.clips
        wave_t = ds.clip(clips[index % len(clips)])
        return wave_t[: int(min(seconds, MAX_SECONDS) * sample_rate)], len(clips)

    # ---- the ladder -------------------------------------------------------------------

    def ladder(self, checkpoint: str, corpus: str, index: int = 0,
               seconds: float = 4.0) -> dict:
        """The panel this tab exists for: one clip at every bitrate, with its numbers.

        Residual VQ makes the *prefix* of a code a valid code, so this needs no retraining
        and no second checkpoint — decode fewer codebooks and stop.
        """
        model = self._codec(checkpoint)
        sr = model.cfg.sample_rate
        clip, n_clips = self._clip(corpus, index, seconds, sr)
        cfg = MelConfig(sample_rate=sr)
        target = magnitude(clip, cfg)

        rows = [{
            "codebooks": None,
            "label": "original",
            "kbps": sr * 16 / 1000,
            "audio": wav_data_uri(clip.numpy(), sr),
            "spectrogram": spectrogram_rows(clip, cfg),
        }]
        for n in RUNGS:
            if n > model.cfg.n_codebooks:
                continue
            y = reconstruct(model, clip, n_codebooks=n).cpu()
            rows.append({
                "codebooks": n,
                "label": f"{n} codebook{'s' if n > 1 else ''}",
                "kbps": round(
                    model.cfg.frames_per_second * n * math.log2(model.cfg.codebook_size) / 1000,
                    2),
                "compression": round(
                    sr * 16 / (model.cfg.frames_per_second * n
                               * math.log2(model.cfg.codebook_size)), 1),
                "convergence": round(spectral_convergence(target, magnitude(y, cfg)), 4),
                "mcd_db": round(mcd(clip, y, cfg), 2),
                "audio": wav_data_uri(y.numpy(), sr),
                "spectrogram": spectrogram_rows(y, cfg),
            })
        device, why = self.device()
        return {
            "checkpoint": checkpoint, "corpus": corpus, "index": index,
            "n_clips": n_clips, "seconds": round(len(clip) / sr, 2),
            "rows": rows, "device": device, "device_reason": why,
            # Repeated on every response so the caveat travels with the numbers rather than
            # living in a tooltip somebody has to find.
            "caveat": "MCD and convergence are proxies, calibrated on constructed "
                      "distortions. Compare them across our own checkpoints, not against "
                      "a paper's.",
        }

    def tokens(self, checkpoint: str, corpus: str, index: int = 0,
               seconds: float = 2.0, frames: int = 40) -> dict:
        """The integers a clip becomes — the whole claim of the phase, as a grid."""
        model = self._codec(checkpoint)
        sr = model.cfg.sample_rate
        clip, _ = self._clip(corpus, index, seconds, sr)
        device, _ = self.device()
        codes = model.encode(clip.unsqueeze(0).to(device))[0].cpu()
        n_books, n_frames = codes.shape
        return {
            "n_codebooks": n_books,
            "n_frames": n_frames,
            "codebook_size": model.cfg.codebook_size,
            "frames_per_second": model.cfg.frames_per_second,
            "seconds": round(len(clip) / sr, 2),
            "positions": n_frames * n_books,
            "codes": codes[:, : min(frames, n_frames)].tolist(),
            "truncated": n_frames > frames,
        }

    def usage(self, checkpoint: str, corpus: str, clips: int = 6) -> dict:
        """Per-codebook usage — the collapse detector, and the thing to look at first."""
        model = self._codec(checkpoint)
        sr = model.cfg.sample_rate
        samples = []
        for i in range(clips):
            try:
                clip, n_clips = self._clip(corpus, i, MAX_SECONDS, sr)
            except AudioError:
                break
            samples.append(clip)
            if i + 1 >= n_clips:
                break
        if not samples:
            raise AudioError(f"no clips in {corpus}")
        rows = codebook_usage(model, samples)
        worst = min(rows, key=lambda r: r["usage"])
        return {
            "rows": [{**r, "perplexity": round(r["perplexity"], 1),
                      "usage": round(r["usage"], 4)} for r in rows],
            "clips": len(samples),
            "worst": worst["codebook"],
            "worst_usage": round(worst["usage"], 4),
            # Said here rather than in the page, so the terminal and the browser give the
            # same reading of the same numbers.
            "note": "A RISING curve is normal: later stages quantize a residual closer to "
                    "noise, and noise spreads evenly over a codebook. Collapse is a "
                    "perplexity of a few dozen out of a thousand ANYWHERE.",
        }

    def runs(self) -> list[dict]:
        """Audio training runs and where they are, for the tab's status strip."""
        out = []
        for cfg_path in sorted(self.root.glob("configs/*.yaml")):
            text = cfg_path.read_text()
            kind = "codec" if "\ncodec:" in text else ("audiolm" if "\naudiolm:" in text else None)
            if kind is None:
                continue
            name = cfg_path.stem
            run_dir = self.root / "checkpoints" / name
            live = (run_dir / "train.pid").is_file()
            step = None
            log = run_dir / "train_log.jsonl"
            if log.is_file():
                for line in reversed(log.read_text().splitlines()[-200:]):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "step" in rec:
                        step = rec["step"]
                        break
            out.append({"name": name, "kind": kind, "training": live, "step": step,
                        "samples": str((run_dir / "samples").relative_to(self.root))
                        if (run_dir / "samples").is_dir() else None})
        return out
