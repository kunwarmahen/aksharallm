"""Clips with nothing to transcribe — the training half of day-two problem 1.

A recogniser trained only on speech has learned, correctly, that *there is always something
to write*. Every training clip had words in it. So given four seconds of a fan, it writes
the most fan-like words it knows. The first smoke run here did exactly that on every one of
the five silence-check clips, digital silence included (`asr/measure.py`).

The fix that every production system makes is unglamorous: **show it clips with an empty
transcript.** CTC handles that natively — a target of length zero has exactly one path, blank
on every frame — so a no-speech clip costs nothing to add and teaches the one thing nothing
else in the corpus can.

**Training noise and test noise come from different generators, on purpose.** The silence
check in `measure.py` is five fixed, named clips. If the model trained on those same five, the
check would measure memory. So this file draws from *families* — coloured noise with a random
spectral tilt, random tone stacks, random click trains, mixtures, at random levels — and the
check's specific clips are never produced by it. They are still synthetic, and still
*related* (a hum is a tone stack); the honest test is real recorded noise, which is a later
download (MUSAN), and docs/23 says so.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import numpy as np


def coloured(rng: np.random.Generator, n: int) -> np.ndarray:
    """Noise with a power spectrum ~ f^tilt, tilt in [-2, 0.5]: from brown to slightly blue."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.arange(len(spec)), 1)
    spec *= f ** (rng.uniform(-2.0, 0.5) / 2.0)
    x = np.fft.irfft(spec, n)
    return x / max(np.abs(x).max(), 1e-9)


def tones(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """One to three fundamentals between 40 and 2,000 Hz, each with a few harmonics — hums,
    whines, a fridge, a monitor. Steady, which is what separates them from a voice."""
    t = np.arange(n) / sr
    x = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        f0 = float(np.exp(rng.uniform(np.log(40), np.log(2000))))
        for h in range(1, rng.integers(2, 6)):
            if f0 * h < sr / 2:
                x += rng.uniform(0.1, 1.0) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3))
    return x / max(np.abs(x).max(), 1e-9)


def clicks(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Short decaying bursts at random times — typing, a mouse, a desk being tapped."""
    x = np.zeros(n)
    k = max(1, int(rng.uniform(0.5, 8.0) * n / sr))
    burst = np.exp(-np.arange(int(0.004 * sr)) / (0.0008 * sr))
    for at in rng.integers(0, max(1, n - len(burst)), k):
        x[at : at + len(burst)] += burst * rng.standard_normal(len(burst)) * rng.uniform(0.3, 1.0)
    return x / max(np.abs(x).max(), 1e-9)


def no_speech(rng: np.random.Generator, n: int, sr: int = 16_000) -> np.ndarray:
    """One random no-speech clip of `n` samples, float32, at a random level.

    Level is drawn in dBFS from -70 (very nearly silent) to -15 (loud), so the model learns
    that *quiet does not mean speech* and also that *loud does not either*. One in ten clips
    is exact digital silence, which a real microphone almost never produces and a muted one
    produces all the time.
    """
    if rng.random() < 0.1:
        return np.zeros(n, dtype=np.float32)
    kinds = [lambda: coloured(rng, n), lambda: tones(rng, n, sr), lambda: clicks(rng, n, sr)]
    x = np.zeros(n)
    for i in rng.choice(3, size=rng.integers(1, 3), replace=False):
        x += kinds[i]() * rng.uniform(0.3, 1.0)
    x /= max(np.abs(x).max(), 1e-9)
    return (x * 10 ** (rng.uniform(-70.0, -15.0) / 20.0)).astype(np.float32)


def replace_with_no_speech(batch: dict, ratio: float, rng: np.random.Generator,
                           sr: int = 16_000) -> int:
    """Overwrite a random `ratio` of a batch's rows with no-speech clips and empty targets.

    In place, keeping each row's length — so a length bucket stays rectangular and a row's
    padding is unchanged. Returns how many rows were replaced.
    """
    if ratio <= 0:
        return 0
    B = batch["wave"].shape[0]
    rows = np.nonzero(rng.random(B) < ratio)[0]
    for r in rows:
        n = int(batch["n_samples"][r])
        x = no_speech(rng, n, sr)
        batch["wave"][r].zero_()
        batch["wave"][r, :n] = batch["wave"].new_tensor(x)
        batch["targets"][r].zero_()
        batch["target_lengths"][r] = 0
    return len(rows)
