"""The ear — a Conformer encoder that turns a log-mel spectrogram into one letter-guess per 40 ms.

```mermaid
flowchart TB
    W["waveform (B, n)"] --> M["log-mel, 10 ms hop<br/>audio/features.py"]
    M --> N["per-utterance normalisation<br/>(valid frames only)"]
    N --> A["SpecAugment<br/>(training only)"]
    A --> S["conv subsampling 4x<br/>-> 25 frames/s"]
    S --> C["Conformer block x N<br/>½FFN · attention · conv · ½FFN"]
    C --> H["linear -> blank + alphabet"]
    H --> L["CTC  (asr/ctc.py)"]
```

**Why a Conformer.** Speech has two kinds of structure at once. A phoneme is *local* — 50 to
100 ms of formant movement, exactly what a convolution sees. A word's identity often is not:
whether a sound was "two" or "too" depends on the sentence. Attention gets the second and
convolutions get the first, and the Conformer interleaves them in every block — the macaron
shape: half a feed-forward, self-attention, a convolution module, the other half of the
feed-forward. That is the 2020 Gulati et al. design and it is still the default encoder for
CTC systems.

**What is reused rather than rewritten.** The log-mel is `audio/features.py`, unchanged. The
attention's positions are **RoPE from `model/rope.py`**, the same rotation the language model
uses — the original Conformer used Transformer-XL relative positions; RoPE is relative too,
costs nothing, and is already tested here. Attention itself is `F.scaled_dot_product_attention`
with a padding mask, **not causal**: the encoder hears the whole utterance before it commits
to a letter. (Streaming dictation would want chunked attention; that is a later piece.)

**Three things about padding, because a batch of utterances is mostly padding.** LibriSpeech
utterances run 1–35 s; a length-bucketed batch still pads every row to its longest.

1. **Normalisation uses valid frames only.** A mean over a padded row includes zeros that are
   not silence, they are nothing — and the result depends on which other utterances shared
   the batch. `normalise` masks them.
2. **No BatchNorm** (trap 1 in PLAN.md § Phase 7). The paper's conv module uses it; BatchNorm's
   statistics would average real frames with padding, and differently at training (batch
   stats) and inference (running stats) — a model that tests worse than it validated. LayerNorm
   is per frame and cannot see the padding.
3. **Padded frames are zeroed before every depthwise convolution.** A kernel of 31 frames
   reaches 15 frames past the end of an utterance; without the zeroing, whatever the previous
   layer left in the padding leaks into the last 600 ms of real speech.

`tests/test_asr.py` asserts the consequence directly: an utterance's output is the same alone
as it is padded into a batch beside a longer one.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.features import MelConfig, mel_filterbank, stft
from ..model.rope import build_cache
from ..model.transformer import apply_rope


@dataclass
class AsrModelConfig:
    """The encoder's shape. Saved into every checkpoint, so a `.pt` describes itself."""

    vocab_size: int = 29          # blank + space + a-z + apostrophe (asr/vocab.py)
    n_mels: int = 80
    sample_rate: int = 16_000
    n_fft: int = 400              # 25 ms — the speech-recognition standard window
    hop: int = 160                # 10 ms
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 31         # 31 subsampled frames = 1.24 s of context per conv
    subsample_channels: int = 256
    dropout: float = 0.1
    rope_theta: float = 10_000.0
    max_frames: int = 4096        # subsampled frames the RoPE cache covers: 164 s of audio
    #: `global` (default): one fixed mean/std per mel band, measured on the training corpus
    #: and saved in the checkpoint, so **loudness stays visible** — a clip of faint hiss still
    #: looks faint. `utterance`: every clip scaled to unit variance, which is the textbook
    #: choice and is what made the first model transcribe -50 dBFS hiss as "ee e eh" (see
    #: `normalise`). Gain augmentation (`augment.gain_db`) is what stops `global` from
    #: learning one microphone's level instead.
    normalise: str = "global"
    #: The paper's layout ends every block with a LayerNorm. **Measured to kill the input on
    #: LibriSpeech:** that norm pins the residual stream at |x| ~ 16, sub-layers learned to emit
    #: time-constant vectors of norm 100-500, and by block 4 two different utterances had
    #: identical hidden states — 11,737 steps at ~100% WER, the same transcript for every
    #: input. False (the default) is pre-norm: no per-block norm, one after subsampling and
    #: one before the head, the layout `model/transformer.py` uses. True exists only so the
    #: checkpoints trained before the fix still load (`load_recognizer` sets it for them).
    block_norm: bool = False

    @property
    def mel(self) -> MelConfig:
        return MelConfig(sample_rate=self.sample_rate, n_fft=self.n_fft, hop=self.hop,
                         n_mels=self.n_mels)

    @property
    def frames_per_second(self) -> float:
        """Encoder output rate — the CTC clock. 100 mel frames/s, subsampled 4x."""
        return self.sample_rate / self.hop / 4

    def describe(self) -> str:
        return (f"conformer d={self.d_model} x{self.n_layers} h={self.n_heads} "
                f"k={self.conv_kernel}, {self.n_mels} mels @ {1000 * self.hop / self.sample_rate:.0f} ms, "
                f"{self.frames_per_second:.0f} out-frames/s, vocab {self.vocab_size}")


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------


def mel_lengths(n_samples: torch.Tensor, hop: int) -> torch.Tensor:
    """Frames a centred STFT gives a clip of `n` samples: `1 + n // hop`."""
    return 1 + n_samples // hop


def normalise(feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Per-utterance, per-band mean/variance normalisation over **valid** frames only.

    **Measured, and the reason it is no longer the default.** The plan said per-utterance
    (trap 5: global stats are one recording chain's). The first smoke run then wrote text on
    *every* clip of the silence check — digital silence, -50 dBFS hiss, mains hum. In the log
    domain a gain is an additive offset, so subtracting each clip's own mean removes its
    loudness entirely, and dividing by its own spread scales faint hiss up to exactly the
    variance of speech. The model was never shown how loud anything was. Trap 5 is still real;
    it is answered with gain augmentation instead (`AugmentConfig.gain_db`), which teaches
    "level varies" without hiding "this is nearly silent".
    """
    B, _, T = feats.shape
    mask = (torch.arange(T, device=feats.device)[None, :] < lengths[:, None]).to(feats.dtype)
    m = mask[:, None, :]
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    mean = (feats * m).sum(-1, keepdim=True) / n
    var = (((feats - mean) * m) ** 2).sum(-1, keepdim=True) / n
    return (feats - mean) / torch.sqrt(var + 1e-5) * m


def spec_augment(feats: torch.Tensor, lengths: torch.Tensor, *, freq_masks: int, freq_width: int,
                 time_masks: int, time_ratio: float, generator: torch.Generator | None = None):
    """SpecAugment (Park et al., 2019): blank out random bands and random stretches of time.

    The single most effective regulariser for speech, and the reason is worth knowing: a
    model that may lose any 27 mel bands cannot rely on any one formant, and one that may
    lose any short stretch of time has to use context to fill it in. Time masks are sized
    **relative to each utterance's own length** (`time_ratio`), so a 2-second clip is not
    wiped out by a mask sized for a 20-second one. Applied after normalisation, so a masked
    region is exactly the mean — zero — rather than a value that depends on the gain.
    """
    B, F_, T = feats.shape
    out = feats.clone()
    dev = feats.device

    def rand(n):
        return torch.rand(n, device="cpu", generator=generator)

    for b in range(B):
        for _ in range(freq_masks):
            w = int(rand(1) * (freq_width + 1))
            f0 = int(rand(1) * max(1, F_ - w))
            out[b, f0 : f0 + w, :] = 0.0
        L = int(lengths[b])
        max_w = max(1, int(time_ratio * L))
        for _ in range(time_masks):
            w = int(rand(1) * (max_w + 1))
            t0 = int(rand(1) * max(1, L - w))
            out[b, :, t0 : t0 + w] = 0.0
    return out.to(dev)


# ---------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------


class Subsample(nn.Module):
    """Two stride-2 3x3 convolutions over (time, frequency): 100 frames/s -> 25.

    4x is the trade between CTC's needs and attention's cost. CTC needs at least one frame
    per letter (plus one per doubled letter); English speech runs ~15 characters a second, so
    25 frames/s leaves headroom. Attention is quadratic in frames, so 100 frames/s would cost
    16x as much per layer for no extra information.
    """

    def __init__(self, n_mels: int, channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2), nn.SiLU(),
        )
        f = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.out = nn.Linear(channels * f, d_model)

    @staticmethod
    def lengths(n: torch.Tensor) -> torch.Tensor:
        """Output frames for `n` input frames: each valid 3-wide stride-2 conv maps n -> (n-3)//2+1."""
        n = torch.div(n - 3, 2, rounding_mode="floor") + 1
        return torch.div(n - 3, 2, rounding_mode="floor") + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, T) -> (B, 1, T, F): time is the first spatial axis so the Linear sees bands.
        y = self.conv(x.transpose(1, 2).unsqueeze(1))
        B, C, T, F_ = y.shape
        return self.out(y.permute(0, 2, 1, 3).reshape(B, T, C * F_))


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, d * mult)
        self.down = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(self.drop(F.silu(self.up(self.norm(x))))))


class SelfAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE and a key-padding mask."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, "d_model must divide by n_heads"
        self.h, self.hd = n_heads, d // n_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, key_mask, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        # (B, 1, 1, T): every query may look at every *real* key, none of the padding.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_mask[:, None, None, :],
            dropout_p=self.p if self.training else 0.0,
        )
        return self.drop(self.out(y.transpose(1, 2).reshape(B, T, D)))


class ConvModule(nn.Module):
    """Pointwise -> GLU -> depthwise (the local part) -> LayerNorm -> SiLU -> pointwise.

    LayerNorm where the paper has BatchNorm — see the module docstring, point 2.
    """

    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "an odd kernel keeps frame t centred on frame t"
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.mid = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, frame_mask):
        y = F.glu(self.pw1(self.norm(x)), dim=-1)
        y = y * frame_mask[..., None]  # point 3: nothing in the padding reaches a real frame
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.pw2(F.silu(self.mid(y))))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: AsrModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn = SelfAttention(d, cfg.n_heads, cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.norm = nn.LayerNorm(d) if cfg.block_norm else nn.Identity()

    def forward(self, x, frame_mask, cos, sin):
        # Macaron: the feed-forward is split in two halves around attention and convolution,
        # each added at weight ½ — the paper's ablation found it beats one full FFN.
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, frame_mask, cos, sin)
        x = x + self.conv(x, frame_mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


# ---------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------


class Recognizer(nn.Module):
    """Waveform -> per-frame log-probabilities over blank + alphabet."""

    def __init__(self, cfg: AsrModelConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = Subsample(cfg.n_mels, cfg.subsample_channels, cfg.d_model)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        # Pre-norm needs the stream normalised once at each end: after subsampling (whose
        # output measured |x| = 584, which would otherwise swamp every block's contribution)
        # and before the head. The old layout normalised inside every block instead.
        self.in_norm = nn.Identity() if cfg.block_norm else nn.LayerNorm(cfg.d_model)
        self.out_norm = nn.Identity() if cfg.block_norm else nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        cos, sin = build_cache(cfg.d_model // cfg.n_heads, cfg.max_frames, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("fb", mel_filterbank(cfg.mel), persistent=False)
        # Persistent: these are measured on the training corpus (`set_feature_stats`) and are
        # as much a part of the model as its weights. Identity until measured.
        self.register_buffer("feat_mean", torch.zeros(cfg.n_mels))
        self.register_buffer("feat_std", torch.ones(cfg.n_mels))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def features(self, wave: torch.Tensor, n_samples: torch.Tensor):
        """`(B, n)` waveform -> normalised log-mel `(B, n_mels, T)` and its valid lengths.

        Under `no_grad` because nothing upstream of the mel is learned, and computed in float32
        whatever autocast says — a log of a bf16 magnitude near the floor is a coarse number.
        """
        cfg = self.cfg.mel
        with torch.autocast(device_type=wave.device.type, enabled=False):
            # The START is reflected, as `stft(center=True)` does (a hard edge to silence is a
            # click). The END is zero-padded: reflecting it would make a clip alone end in a
            # mirror image of itself while the same clip in a batch ends in the next row's
            # zero padding, so its last frame would depend on who it was batched with.
            half = cfg.n_fft // 2
            x = F.pad(wave.float().unsqueeze(1), (half, 0), mode="reflect").squeeze(1)
            x = F.pad(x, (0, half))
            mel = torch.matmul(self.fb, stft(x, cfg, center=False).abs())
            feats = torch.log(mel.clamp_min(cfg.log_eps))
        lengths = mel_lengths(n_samples, self.cfg.hop).clamp(max=feats.shape[-1])
        if self.cfg.normalise == "utterance":
            return normalise(feats, lengths), lengths
        mask = torch.arange(feats.shape[-1], device=feats.device)[None, :] < lengths[:, None]
        out = (feats - self.feat_mean[None, :, None]) / self.feat_std[None, :, None]
        return out * mask[:, None, :].to(out.dtype), lengths

    @torch.no_grad()
    def set_feature_stats(self, waves: list[torch.Tensor]) -> None:
        """Measure the per-band mean and std of the log-mel over some training audio.

        Only meaningful for `normalise: global`. Called once by the trainer on a fresh run; a
        resumed run loads them from the checkpoint like any other buffer.
        """
        cfg = self.cfg.mel
        acc, acc2, n = 0.0, 0.0, 0
        for w in waves:
            x = F.pad(w.float()[None, None].to(self.fb.device), (cfg.n_fft // 2, 0), mode="reflect")
            x = F.pad(x[:, 0], (0, cfg.n_fft // 2))
            f = torch.log(torch.matmul(self.fb, stft(x, cfg, center=False).abs()).clamp_min(cfg.log_eps))[0]
            acc = acc + f.sum(-1)
            acc2 = acc2 + (f ** 2).sum(-1)
            n += f.shape[-1]
        mean = acc / n
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(torch.sqrt((acc2 / n - mean ** 2).clamp_min(1e-4)))

    def encode(self, feats: torch.Tensor, lengths: torch.Tensor):
        """Normalised log-mel -> `(log_probs (B, T', V) float32, out_lengths (B,))`."""
        x = self.in_norm(self.subsample(feats))
        T = x.shape[1]
        if T > self.rope_cos.shape[0]:
            raise ValueError(
                f"{T} encoder frames is past the RoPE cache ({self.rope_cos.shape[0]} = "
                f"{self.rope_cos.shape[0] / self.cfg.frames_per_second:.0f} s). Raise "
                "asr.max_frames, or split the audio."
            )
        out_len = Subsample.lengths(lengths).clamp(min=0, max=T)
        mask = torch.arange(T, device=x.device)[None, :] < out_len[:, None]
        # A row with no valid frames at all would give attention nothing to attend to and a
        # softmax of all -inf; give it one frame so the maths stays finite. Its CTC loss is
        # infeasible anyway and the trainer drops it.
        mask[:, 0] = True
        x = x * mask[..., None]
        for blk in self.blocks:
            x = blk(x, mask, self.rope_cos, self.rope_sin)
        logits = self.head(self.out_norm(x)).float()
        return logits.log_softmax(-1), out_len

    def forward(self, wave: torch.Tensor, n_samples: torch.Tensor, augment: dict | None = None,
                generator: torch.Generator | None = None):
        augment = dict(augment or {})
        gain = augment.pop("gain_db", None)
        if gain and self.training:
            # A random level per utterance, in dB — the answer to trap 5 that keeps loudness
            # visible. Drawn on the CPU generator so a resumed run continues the same stream.
            lo, hi = gain
            g = lo + (hi - lo) * torch.rand(wave.shape[0], generator=generator)
            wave = wave * (10.0 ** (g / 20.0)).to(wave.device, wave.dtype)[:, None]
        feats, lengths = self.features(wave, n_samples)
        if augment and self.training:
            feats = spec_augment(feats, lengths, generator=generator, **augment)
        return self.encode(feats, lengths)


def config_dict(cfg: AsrModelConfig) -> dict:
    return asdict(cfg)


def seconds_to_frames(seconds: float, cfg: AsrModelConfig) -> int:
    """How many encoder frames `seconds` of audio becomes — for sizing `max_frames`."""
    mel = 1 + int(seconds * cfg.sample_rate) // cfg.hop
    return int(Subsample.lengths(torch.tensor(mel)))


__all__ = ["AsrModelConfig", "Recognizer", "normalise", "spec_augment", "mel_lengths",
           "seconds_to_frames", "config_dict"]
