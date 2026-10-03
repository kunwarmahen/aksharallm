"""Real recorded noise, at known loudness: how far from a quiet room the recogniser still works.

LibriSpeech is people reading books into good microphones in quiet rooms — the most forgiving
speech a recogniser will ever hear. Dictation happens in kitchens, cafés and cars. This file
measures the distance between the two:

```mermaid
flowchart LR
    S["test-clean utterance"] --> M["mix at a chosen SNR<br/>20 · 10 · 5 · 0 dB"]
    N["real noise recording<br/>DEMAND: kitchen, café,<br/>street, car, office, home"] --> M
    M --> R["the recogniser"]
    R --> W["WER per environment × SNR"]
```

**The noise is real and the recogniser has never heard it.** DEMAND (Thiemann, Ito & Vincent,
2013; CC BY 4.0) is 16-channel recordings of 18 everyday places; one channel of six of them is
used. Training only ever saw the synthetic no-speech families in `asr/noise.py`, so this is a
test of generalisation, not of memory.

**SNR is set exactly, per utterance.** The noise is scaled so that speech power / noise power
equals the target — 20 dB is a quiet office, 10 dB a busy café, 0 dB noise as loud as the
voice. Speech power is measured over the utterance's *voiced* frames (frames within 30 dB of
its loudest), because LibriSpeech clips carry pauses, and averaging the pauses in would call a
clip quieter than it is and drown it in more noise than the label says.

**Deterministic.** The noise segment for an utterance is chosen by a seed derived from the
utterance index and the environment, so two runs, or two checkpoints, hear identical mixtures
and their difference is the model's.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np

DEMAND = "https://zenodo.org/api/records/1227121/files/{env}_16k.zip/content"
#: Places people dictate. DEMAND has eighteen; these six span the kinds of noise that matter
#: (steady machine hum, babble, transient clatter, road).
ENVIRONMENTS = {
    "DKITCHEN": "kitchen", "DLIVING": "living room", "OOFFICE": "office",
    "PCAFETER": "café", "STRAFFIC": "street", "TCAR": "car",
}
SNRS = (20.0, 10.0, 5.0, 0.0)
NOISE_DIR = Path("data/asr/noise/demand")


def fetch_demand(dest: str | Path = NOISE_DIR, envs=tuple(ENVIRONMENTS), progress=print) -> Path:
    """Download each environment's 16 kHz zip, keep channel 1 as `<ENV>.wav`, drop the zip."""
    from .data import download
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for env in envs:
        out = dest / f"{env}.wav"
        if out.is_file():
            progress(f"already have {out}")
            continue
        z = dest / f"{env}_16k.zip"
        download(DEMAND.format(env=env), z, progress=progress, note="~0.1 GB")
        with zipfile.ZipFile(z) as f:
            member = next(m for m in f.namelist() if m.endswith("ch01.wav"))
            out.write_bytes(f.read(member))
        z.unlink()
        progress(f"{env}: kept {member} -> {out}")
    (dest / "SOURCE.json").write_text(json.dumps({
        "dataset": "DEMAND (Diverse Environments Multichannel Acoustic Noise Database)",
        "url": "https://zenodo.org/records/1227121", "licence": "CC BY 4.0",
        "citation": "Thiemann, Ito & Vincent, 2013", "channel": "ch01",
        "environments": {e: ENVIRONMENTS.get(e, e) for e in envs},
        "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    return dest


def load_noise(dest: str | Path = NOISE_DIR) -> dict[str, np.ndarray]:
    from ..audio.io import read_wav
    out = {}
    for env in ENVIRONMENTS:
        p = Path(dest) / f"{env}.wav"
        if p.is_file():
            x, sr = read_wav(p)
            if sr != 16_000:
                raise ValueError(f"{p} is {sr} Hz; the 16 kHz DEMAND files are expected")
            x = np.asarray(x, dtype=np.float32)
            out[env] = x if x.ndim == 1 else x.mean(axis=-1)
    return out


def speech_power(x: np.ndarray, frame: int = 400, floor_db: float = 30.0) -> float:
    """Mean power over frames within `floor_db` of the loudest — the voiced part."""
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2)) + 1e-12
    p = (x[: n * frame].reshape(n, frame) ** 2).mean(1)
    keep = p >= p.max() * 10 ** (-floor_db / 10)
    return float(p[keep].mean()) + 1e-12


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Speech + a random stretch of `noise`, scaled so voiced speech power / noise power is
    exactly `snr_db`. If the sum would clip, both are scaled down together (the SNR holds)."""
    n = len(speech)
    if len(noise) < n:
        noise = np.tile(noise, n // len(noise) + 1)
    start = int(rng.integers(0, len(noise) - n + 1))
    seg = noise[start : start + n].astype(np.float64)
    gain = np.sqrt(speech_power(speech) / (np.mean(seg ** 2) + 1e-12) / 10 ** (snr_db / 10))
    y = speech.astype(np.float64) + gain * seg
    peak = np.abs(y).max()
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32)


def robustness(model, device: str, corpus, noise: dict[str, np.ndarray], snrs=SNRS,
               decoder=None, workers: int = 8, seed: int = 0) -> dict:
    """WER for clean audio, then for every environment × SNR, on the same utterances."""
    from .measure import log_probs, score
    from .__main__ import _decode

    utts = corpus.utts
    refs = [(u.speaker, u.text) for u in utts]
    clean = [corpus.wave(u) for u in utts]

    def wer_of(waves):
        lps = log_probs(model, waves, device)
        hyps = _decode(lps, "beam" if decoder else "greedy", decoder, workers)
        return score([(sp, r, h) for (sp, r), h in zip(refs, hyps, strict=True)])["wer"], hyps

    base, _ = wer_of(clean)
    table, examples = {}, {}
    for e_i, (env, nz) in enumerate(noise.items()):
        table[env] = {}
        for snr in snrs:
            waves = [mix(w, nz, snr, np.random.default_rng([seed, e_i, k])) for k, w in enumerate(clean)]
            w, hyps = wer_of(waves)
            table[env][f"{snr:g}"] = w
            if snr == min(snrs):
                examples[env] = {"ref": utts[0].text, "hyp": hyps[0]}
    by_snr = {f"{s:g}": float(np.mean([table[e][f"{s:g}"] for e in table])) for s in snrs}
    return {"clean": base, "table": table, "mean_by_snr": by_snr, "examples": examples,
            "environments": {e: ENVIRONMENTS.get(e, e) for e in noise}, "snrs": list(snrs),
            "utts": len(utts)}
