"""The config for a recogniser run — one YAML, the shared `load_into`, the shared `train:` contract.

Same arrangement as `audio/config.py`: a recogniser is not a language model (no token budget,
no context window), so it has its own schema, but it is loaded the same way, takes the same
`-o dotted.key=value` overrides, and obeys the same out_dir / STOP / resume rules — which is
what lets `scripts/stop.sh`, `scripts/sessions.py` and the portal drive it unchanged.

The top-level section is `asr:`, and that one word is how `scripts/audio.sh` and
`portal/runs.py` tell this run apart from a codec, an audio LM or a text model.

The knob to set deliberately is `data.max_batch_seconds` — the *padded* audio per step. It is
this run's batch size, and it is in seconds rather than utterances because utterances vary
30x in length and memory follows the seconds.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import OptimConfig, load_into
from .model import AsrModelConfig


@dataclass
class AsrDataConfig:
    #: One or more packed corpora (`audio.bin` + `manifest.json` + transcripts). Several are
    #: sampled in proportion to their hours.
    train: list[str] = field(default_factory=lambda: ["data/audio/synth"])
    #: A separate validation corpus (LibriSpeech dev-clean). Unset: hold out the last
    #: `val_clips` utterances of the first training corpus, by clip — never by offset.
    val: str | None = None
    val_clips: int = 20
    min_seconds: float = 0.5
    max_seconds: float = 16.0
    max_batch_seconds: float = 320.0
    #: How many validation utterances WER is measured on at every eval. Greedy decoding is
    #: cheap; the number is noisy below a few hundred.
    eval_utts: int = 300
    #: Share of each batch replaced by clips with **no speech and an empty transcript** —
    #: coloured noise, tones, clicks (`asr/noise.py`). Without them a model has never seen a
    #: clip where nothing was said, and writes something on every one (day-two problem 1).
    no_speech_ratio: float = 0.1


@dataclass
class AugmentConfig:
    """SpecAugment. LibriSpeech's "LD" policy is 2 x F=27 frequency masks and time masks up to
    5% of the utterance; we use fewer time masks because the run is shorter."""

    enabled: bool = True
    #: A random gain per utterance, uniform in dB over this range. With `normalise: global`
    #: this is what keeps the model from learning one microphone's level. `null` = off.
    gain_db: list[float] | None = field(default_factory=lambda: [-18.0, 6.0])
    freq_masks: int = 2
    freq_width: int = 27
    time_masks: int = 5
    time_ratio: float = 0.05


@dataclass
class AsrTrainConfig:
    out_dir: str = "checkpoints/asr-synth"
    max_steps: int = 2000
    eval_every: int = 250
    ckpt_every: int = 1000
    log_every: int = 20
    #: `ours` is `asr/ctc.py`; `torch` is `F.ctc_loss`. They agree to the last bit in float64
    #: (`tests/test_asr.py`), so this is a speed choice, exactly like `model.attn_impl` — and
    #: the default is `torch` for the same reason `attn_impl` defaults to `sdpa`. Measured on
    #: the 3090: ours 195 ms vs 1.3 ms fwd+bwd at (400 frames, 25 rows, 240 letters), because
    #: a Python loop over frames is kernel launches, not arithmetic. That made a whole step of
    #: the 20M Conformer 3.2x slower (475 vs 1,544 audio-s/s). `ours` is the reference.
    ctc_impl: str = "torch"
    seed: int = 1337
    resume: str | None = "auto"
    stop_after: int | None = None
    stop_at: int | None = None
    stop_after_s: float | None = None


@dataclass
class AsrRunConfig:
    name: str = "asr-synth"
    asr: AsrModelConfig = field(default_factory=AsrModelConfig)
    data: AsrDataConfig = field(default_factory=AsrDataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: AsrTrainConfig = field(default_factory=AsrTrainConfig)


def load_asr_config(path, overrides: list[str] | None = None) -> AsrRunConfig:
    cfg = load_into(AsrRunConfig, path, overrides)
    if isinstance(cfg.data.train, str):
        cfg.data.train = [cfg.data.train]
    if cfg.train.ctc_impl not in ("ours", "torch"):
        raise ValueError(f"train.ctc_impl must be 'ours' or 'torch', not {cfg.train.ctc_impl!r}")
    return cfg
