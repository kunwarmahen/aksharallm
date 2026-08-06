"""The config for a codec run — a YAML file, like every other run here.

`aksharallm/config.py`'s `Config` describes a *language model*: a vocabulary, a context
window, a token budget. A codec has none of those. It has a sample rate, a stack of strides
and a codebook, so it gets its own schema — but it is loaded by the same `load_into`, takes
the same `-o dotted.key=value` overrides, and obeys the same `train:` contract (out_dir,
max_steps, the STOP file, resume), so `scripts/stop.sh`, `scripts/sessions.py` and the
portal drive it without knowing it is not a transformer.

The one thing worth setting deliberately is `data.window_seconds`. It is the codec's
equivalent of `seq_len`, and it trades two things off: the multi-scale STFT loss needs a
window at least as long as its longest FFT (2,048 samples = 128 ms) to see the low
frequencies at all, and memory grows linearly with it. One second is comfortable on a 3090
at batch 16.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import OptimConfig, load_into
from .codec import CodecConfig


@dataclass
class AudioDataConfig:
    """Where the packed corpus is, and how much of it a step sees."""

    corpus: str = "data/audio/synth"
    window_seconds: float = 1.0
    #: Clips held out for validation, taken from the END of the manifest. Held out by clip
    #: and never by offset — two windows of one utterance are near-duplicates, so splitting
    #: inside a clip leaks the val set into training.
    val_clips: int = 16


@dataclass
class AudioTrainConfig:
    """The same field names as `TrainConfig`, so the shared tooling reads them unchanged.

    `seq_len` is absent on purpose (it lives in `data.window_seconds`) and `grad_accum` is
    absent because a codec batch is small and cheap; if that changes, add it here rather
    than reaching into the text trainer's config.
    """

    out_dir: str = "checkpoints/codec-synth"
    batch_size: int = 16
    max_steps: int = 20_000
    eval_every: int = 500
    eval_batches: int = 20
    #: Write a reconstruction WAV every N steps. **The single most useful setting in the
    #: file** — a codec's loss curve tells you almost nothing about whether it is intelligible
    #: yet, and a 3-second file you can play tells you immediately.
    sample_every: int = 1000
    sample_seconds: float = 3.0
    ckpt_every: int = 2000
    keep_last_n: int = 2
    log_every: int = 20
    compile: bool = False
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class LossConfig:
    """Weights on the three terms. See `codec.ReconstructionLoss` for what each one fixes."""

    scales: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    log_weight: float = 1.0
    convergence_weight: float = 1.0
    wave_weight: float = 0.1
    #: Multiplies the quantizer's commitment loss. The commitment weight *inside* the
    #: quantizer (`codec.commit`) is the published one; this is the outer dial, kept separate
    #: so an experiment can turn the whole VQ term off without changing the codec's shape and
    #: therefore without invalidating the checkpoint.
    vq_weight: float = 1.0


@dataclass
class CodecRunConfig:
    name: str = "codec-synth"
    codec: CodecConfig = field(default_factory=CodecConfig)
    data: AudioDataConfig = field(default_factory=AudioDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: AudioTrainConfig = field(default_factory=AudioTrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @property
    def window(self) -> int:
        """Window length in samples, rounded up to a whole number of codec frames.

        Rounding matters: a window that is not a multiple of `hop` is padded by the codec on
        every single step, so the model spends part of its training budget reconstructing
        our zero padding.
        """
        raw = int(self.data.window_seconds * self.codec.sample_rate)
        hop = self.codec.hop
        return max(hop, ((raw + hop - 1) // hop) * hop)


def load_codec_config(path, overrides: list[str] | None = None) -> CodecRunConfig:
    cfg = load_into(CodecRunConfig, path, overrides)
    # A window shorter than the longest analysis window means the largest STFT scale never
    # runs, silently — the loss quietly becomes a different loss. Refuse instead.
    longest = max(cfg.loss.scales)
    if cfg.window < longest:
        raise ValueError(
            f"data.window_seconds={cfg.data.window_seconds} is {cfg.window} samples, shorter "
            f"than the longest STFT scale ({longest}). Raise it, or drop that scale."
        )
    return cfg
