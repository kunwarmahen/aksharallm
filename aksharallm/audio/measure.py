"""Numbers for a codec — including an honest answer to "is this any good?".

Audio quality has one real metric and it is a room full of people scoring clips (mean
opinion score). We do not have that, and every number below is a **proxy**, so each one is
labelled with what it can and cannot see. That labelling is the point of this file; trap 7
of the phase is that TTS cannot be scored honestly by a number, and the way to live with it
is to say so beside every number rather than to stop measuring.

| metric | what it sees | what it misses |
|---|---|---|
| **spectral convergence** | whether the loud parts of the spectrum are in the right place | phase, and therefore anything that sounds robotic while being spectrally correct |
| **MCD** (mel-cepstral distortion) | the *shape* of the spectral envelope — formants, i.e. which vowel | pitch, timing, and noise |
| **codebook usage** | whether the model is using the capacity it was given | nothing about quality directly, and it is the first thing to go wrong |
| **bitrate ladder** | the actual trade the codec makes, at each rung | it is a comparison, not an absolute |

**MCD is the one to quote**, because it is the standard number in the speech literature and
because it has a published interpretation: under ~4 dB is generally taken as very close,
6-8 dB as recognisably degraded. It is a distance between **cepstra** — the DCT of the
log-mel spectrum — and the reason for that DCT is worth understanding: it separates the
slowly-varying envelope (the vocal tract's shape, i.e. which sound is being made) from the
fast ripple (the pitch harmonics). Dropping coefficient 0 drops overall loudness, and keeping
1-13 keeps the envelope. So MCD asks "is it saying the same thing", not "is it the same
recording".

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math

import torch

from .codec import Codec
from .features import MelConfig, log_mel, magnitude, mel_filterbank, spectral_convergence

#: How many cepstral coefficients MCD compares. 13 is the convention, and coefficient 0 —
#: overall energy — is excluded, because a reconstruction 1 dB quiet is not a reconstruction
#: that says something different.
MCD_COEFFS = 13


def dct2(x: torch.Tensor, n_out: int) -> torch.Tensor:
    """Type-II DCT along the second-to-last axis, orthonormal. `(..., n, T) -> (..., n_out, T)`.

    Written out rather than pulled from a library because it is four lines and because the
    matrix *is* the explanation: row `k` is a cosine at frequency `k` across the mel axis, so
    coefficient `k` measures how much of the log-spectrum ripples at that rate.
    """
    n = x.shape[-2]
    k = torch.arange(n_out, device=x.device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
    basis = basis * math.sqrt(2.0 / n)
    if n_out > 0:
        basis[0] = basis[0] / math.sqrt(2.0)
    return torch.matmul(basis.to(x.dtype), x)


#: Everything more than this many dB below the loudest part of the clip is flattened to one
#: value before the DCT. See `cepstrum` — this constant is the whole reason MCD works here,
#: and it is calibrated rather than guessed: at 40 dB, additive noise 34 dB below the signal
#: scores 4.2 dB (the published "very close" band) and the metric rises monotonically with
#: noise. At 80 dB the same distortion scores 93, because the extra 40 dB of range is bands
#: nobody can hear, and MCD would be measuring them instead of the speech.
DYNAMIC_RANGE_DB = 40.0


def cepstrum(
    wave: torch.Tensor, cfg: MelConfig | None = None, *, dynamic_range: float = DYNAMIC_RANGE_DB
) -> torch.Tensor:
    """Mel-frequency cepstral coefficients 1..13. `(..., n) -> (..., 13, frames)`.

    **It does not use `features.log_mel`, and the difference is the point.** That function
    floors at an *absolute* `log_eps`, which is right for a training loss — it stops one
    silent frame owning the batch. It is wrong for a cepstral distance, because a mel band
    120 dB down is numerically far from another band 160 dB down and *perceptually identical
    to it*: both are silence. Take the DCT of that and inaudible differences dominate the
    coefficients.

    Measured here, on the synthetic corpus: adding noise at amplitude 0.001 — which is
    below the noise floor of any recording — scored **86 dB** MCD with an absolute floor and
    **0.23 dB** with this one. The metric was not measuring the signal at all.

    So the floor is *relative*: everything more than `dynamic_range` dB below the clip's
    loudest mel band is flattened to the same value, which is what `power_to_db(top_db=...)`
    does in every standard implementation.

    **Comparability caveat, stated because it would otherwise be assumed away.** Published
    MCD figures are usually computed from a *mel-generalized cepstrum* of a smoothed spectral
    envelope (SPTK/WORLD), not from MFCCs of a raw STFT as here. Ours is calibrated so the
    familiar interpretation bands hold on the distortions we can construct, and it is exactly
    reproducible run to run — but treat it as a number to compare **our** checkpoints with,
    not one to put beside a paper's.
    """
    cfg = cfg or MelConfig()
    mel = torch.matmul(mel_filterbank(cfg, device=wave.device), magnitude(wave, cfg))
    peak = mel.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    floor = peak * (10.0 ** (-dynamic_range / 20.0))
    return dct2(torch.log(torch.maximum(mel, floor)), MCD_COEFFS + 1)[..., 1:, :]


def mcd(a: torch.Tensor, b: torch.Tensor, cfg: MelConfig | None = None,
        *, top_db: float = 40.0) -> float:
    """Mel-cepstral distortion between two waveforms, in dB. Lower is better.

    `(10/ln 10)·√(2·Σ(c_a − c_b)²)`, averaged over frames. The constant converts a natural-log
    cepstral distance into decibels and the √2 comes from the symmetry of the DCT; both are
    conventional, and reporting a number computed any other way makes it incomparable with
    every published figure, which is the only reason to use this metric at all.

    **Silent frames are excluded**, because a cepstrum is the *shape* of a spectrum and
    silence has no shape. A corpus that is a third pauses would otherwise spend a third of
    this number measuring its own pauses. `top_db` keeps frames within 40 dB of the loudest
    frame **of the reference**, which is the usual convention. (The larger correction is in
    `cepstrum`'s relative floor — read that docstring, it is where the metric was rescued.)

    **It assumes the two signals are aligned frame for frame.** True for a codec, where the
    output is the input reconstructed. Not true for TTS, where the model may say the same
    words at a different rate — that needs dynamic time warping first, and this function does
    not do it. Handing it two unaligned utterances gives a large number that means nothing.
    """
    cfg = cfg or MelConfig()
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    ca, cb = cepstrum(a, cfg), cepstrum(b, cfg)
    d = ((ca - cb) ** 2).sum(dim=-2).sqrt()  # (..., frames)

    # Frame energy of the REFERENCE, in dB relative to its own peak.
    energy = log_mel(a, cfg).mean(dim=-2)  # (..., frames), already a log
    keep = energy >= (energy.amax(dim=-1, keepdim=True) - top_db / (10.0 / math.log(10.0)))
    if not bool(keep.any()):
        return float("nan")
    scale = (10.0 / math.log(10.0)) * math.sqrt(2.0)
    return float(scale * (d * keep).sum() / keep.sum())


@torch.no_grad()
def reconstruct(model: Codec, wave: torch.Tensor, n_codebooks: int | None = None) -> torch.Tensor:
    """Encode and decode one waveform `(n,)` or `(B, n)` at the given bitrate."""
    solo = wave.dim() == 1
    x = wave.unsqueeze(0) if solo else wave
    device = next(model.parameters()).device
    x = x.to(device)
    was = model.training
    model.eval()
    codes = model.encode(x)
    y = model.decode(codes, n_codebooks=n_codebooks)[..., : x.shape[-1]]
    model.train(was)
    return y.squeeze(0) if solo else y


@torch.no_grad()
def bitrate_ladder(
    model: Codec,
    clips: list[torch.Tensor],
    rungs: tuple[int, ...] = (1, 2, 4, 8),
    cfg: MelConfig | None = None,
) -> list[dict]:
    """The demo of the phase, as numbers: the same audio at four bitrates.

    Residual VQ makes the *prefix* of the code a valid code, so this needs no retraining and
    no second checkpoint — decode fewer codebooks and stop. The portal plays the same four
    files side by side, and hearing the trade is worth more than reading it.
    """
    cfg = cfg or MelConfig(sample_rate=model.cfg.sample_rate)
    rows = []
    for n in rungs:
        if n > model.cfg.n_codebooks:
            continue
        convs, mcds = [], []
        for clip in clips:
            y = reconstruct(model, clip, n_codebooks=n).cpu()
            x = clip[..., : y.shape[-1]].cpu()
            convs.append(spectral_convergence(magnitude(x, cfg), magnitude(y, cfg)))
            mcds.append(mcd(x, y, cfg))
        rows.append(
            {
                "codebooks": n,
                "kbps": model.cfg.frames_per_second * n * math.log2(model.cfg.codebook_size) / 1000,
                "convergence": sum(convs) / len(convs),
                "mcd_db": sum(mcds) / len(mcds),
                # The compression ratio against the 16-bit PCM it came from, which is the
                # comparison anyone actually cares about.
                "compression": (model.cfg.sample_rate * 16)
                / (model.cfg.frames_per_second * n * math.log2(model.cfg.codebook_size)),
            }
        )
    return rows


@torch.no_grad()
def codebook_usage(model: Codec, clips: list[torch.Tensor]) -> list[dict]:
    """How much of each codebook a real corpus actually reaches.

    The training log reports usage on one batch; this reports it over whatever you give it,
    which is the number to quote.

    **A curve that RISES with the index is normal, and it is worth knowing why**, because
    the intuitive guess is the opposite. Later codebooks carry less *energy* — that is the
    whole point of a residual — but they quantize a residual that is closer to noise, and
    noise is spread evenly over the codebook. So the first stage, which sees structured
    latents concentrated in a few regions, typically has the **lowest** perplexity of all.
    Measured on the synthetic corpus at 1,500 steps: 21% at stage 0 rising to 58% by stage 4.

    What is fatal is a perplexity of a few dozen out of a thousand **anywhere** — that is a
    collapsed stage, and reading the shape of the curve instead of the numbers would hide it.
    """
    device = next(model.parameters()).device
    size = model.cfg.codebook_size
    counts = torch.zeros(model.cfg.n_codebooks, size, dtype=torch.long)
    was = model.training
    model.eval()
    for clip in clips:
        codes = model.encode(clip.unsqueeze(0).to(device))[0].cpu()
        for i in range(codes.shape[0]):
            counts[i] += torch.bincount(codes[i], minlength=size)
    model.train(was)

    rows = []
    for i in range(model.cfg.n_codebooks):
        p = counts[i].float() / counts[i].sum().clamp_min(1)
        entropy = -(p * torch.log(p.clamp_min(1e-10))).sum()
        rows.append(
            {
                "codebook": i,
                "used": int((counts[i] > 0).sum()),
                "size": size,
                "perplexity": float(entropy.exp()),
                "usage": float(entropy.exp()) / size,
            }
        )
    return rows
