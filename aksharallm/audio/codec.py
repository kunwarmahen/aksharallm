"""The codec — a convolutional autoencoder with a quantizer in the middle.

This is the only genuinely new training loop in the audio phase, and the whole phase depends
on it: everything downstream consumes its **integers**, so if the codec is bad the audio LM
is learning to predict noise fluently.

```mermaid
flowchart LR
    W["waveform<br/>16,000/s"] --> E["conv encoder<br/>stride 2·4·5·8 = 320"]
    E --> Z["latent<br/>128 floats, 50/s"]
    Z --> Q["residual VQ<br/>8 x 1024"]
    Q --> T["8 integers<br/>50 times a second"]
    T --> D["conv decoder<br/>transposed, 320x up"]
    D --> W2["waveform back"]
```

**The arithmetic that decides everything.** 16,000 samples a second, downsampled by 320, is
**50 frames a second**. Eight codebooks of 1,024 entries is 10 bits each, so 80 bits a frame
= **4,000 bits a second** — against 256,000 for the 16-bit PCM it came from, a 64x
compression. And 50 tokens/second × 8 is the sequence length the transformer downstairs
pays for: ten seconds of speech is 4,000 tokens, which is why the frame rate is the single
most consequential number in this file.

**Why the reconstruction loss is not on the waveform.** L2 between two waveforms is
*phase-sensitive*: shift a signal by one sample — inaudible, and at 16 kHz that is 60
microseconds — and the loss is enormous. Optimise it and the model spends its capacity
aligning phase nobody can hear. The loss that works is on **STFT magnitudes at several
window sizes**, which measures what the ear measures: which frequencies are present, at what
strength, at roughly what time. Short windows catch transients (the burst of a /t/), long
ones catch pitch. That is trap 2 of the phase, written down before it cost anything.

An adversarial term on top of this is what takes a codec from "clearly the same words" to
"hard to tell apart", and it is deliberately **not** here: a GAN that fails to converge is
indistinguishable from a codec that fails to converge, and debugging the pair together is
how a day becomes a week.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import MelConfig, mel_filterbank, stft
from .vq import ResidualVQ, VQStats


@dataclass
class CodecConfig:
    """Everything that changes the shape of a codec checkpoint.

    `strides` is the load-bearing one: its product is the downsample factor, and therefore
    the token rate, and therefore the context length every audio model above needs.
    """

    sample_rate: int = 16_000
    channels: int = 32  # width after the input conv; doubles at every stride
    strides: tuple[int, ...] = (2, 4, 5, 8)  # product 320 -> 50 frames/s at 16 kHz
    dim: int = 128  # the latent the quantizer sees
    n_codebooks: int = 8
    codebook_size: int = 1024
    n_residual: int = 2  # residual units per stage
    dilations: tuple[int, ...] = (1, 3)  # per residual unit, so the stack sees wider context
    commit: float = 0.25
    decay: float = 0.99
    restart_after: int = 200
    quantizer_dropout: bool = True

    @property
    def hop(self) -> int:
        return math.prod(self.strides)

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop

    @property
    def bits_per_second(self) -> float:
        return self.frames_per_second * self.n_codebooks * math.log2(self.codebook_size)

    def describe(self) -> str:
        return (
            f"{self.frames_per_second:.0f} frames/s x {self.n_codebooks} codebooks "
            f"x {self.codebook_size} = {self.bits_per_second / 1000:.1f} kbps"
        )


class Snake(nn.Module):
    """`x + sin²(αx)/α`, with a learned α per channel.

    An ELU or a ReLU has no reason to produce anything periodic, and speech is *made* of
    periodicity — a vowel is a pitch and its harmonics. Snake builds that bias in: the
    activation itself oscillates, so a decoder can generate a periodic signal without having
    to synthesise one out of piecewise-linear segments. It is the single cheapest quality
    win in a neural vocoder and costs one parameter per channel.

    `alpha` is stored in log space so it stays positive without a clamp, and so a gradient
    step is multiplicative — α wants to move over orders of magnitude between the layer that
    sees 16 kHz samples and the layer that sees 50 Hz frames.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp().reshape(1, -1, 1)
        return x + (torch.sin(alpha * x) ** 2) / (alpha + 1e-9)


class ResidualUnit(nn.Module):
    """`Snake -> dilated 3-wide conv -> Snake -> 1-wide conv`, added to the input.

    The 1-wide second conv is a per-position mixer, not a filter: the receptive field comes
    entirely from the dilated one. Stacking units with dilations 1 and 3 widens the field
    geometrically for a linear number of parameters, which is how a stack this shallow ends
    up seeing tens of milliseconds.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = dilation  # keeps the length, for kernel 3
        self.block = nn.Sequential(
            Snake(channels),
            nn.Conv1d(channels, channels, 3, dilation=dilation, padding=pad),
            Snake(channels),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Down(nn.Module):
    """One stride-`r` stage: residual units at the current rate, then a `2r`-wide conv.

    The padding is explicit and asymmetric rather than `padding=r//2`, because for an odd
    stride (we use 5) the symmetric form is off by a frame and the encoder and decoder then
    disagree about how long the signal is — a bug that shows up as a slowly growing offset
    over a long clip and as nothing at all over a short one.
    """

    def __init__(self, c_in: int, c_out: int, stride: int, dilations):
        super().__init__()
        self.units = nn.Sequential(*(ResidualUnit(c_in, d) for d in dilations))
        self.act = Snake(c_in)
        self.conv = nn.Conv1d(c_in, c_out, 2 * stride, stride=stride)
        self.left, self.right = math.ceil(stride / 2), stride // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.units(x))
        return self.conv(F.pad(x, (self.left, self.right)))


class Up(nn.Module):
    """The mirror: a `2r`-wide transposed conv, cropped back to exactly `r · T`."""

    def __init__(self, c_in: int, c_out: int, stride: int, dilations):
        super().__init__()
        self.act = Snake(c_in)
        self.conv = nn.ConvTranspose1d(c_in, c_out, 2 * stride, stride=stride)
        self.left, self.right = math.ceil(stride / 2), stride // 2
        self.units = nn.Sequential(*(ResidualUnit(c_out, d) for d in dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self.act(x))
        y = y[..., self.left : y.shape[-1] - self.right]
        return self.units(y)


class Codec(nn.Module):
    """Waveform in, integers out, waveform back.

    Shapes, once, because everything here depends on them:

    * waveform `(B, n_samples)`; `n_samples` must be a multiple of `cfg.hop` — `pad_to_hop`
      does that and hands back the original length to trim to afterwards.
    * latent `(B, dim, n_frames)` with `n_frames = n_samples / hop`.
    * codes `(B, n_codebooks, n_frames)` — the thing the transformer sees.
    """

    def __init__(self, cfg: CodecConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CodecConfig()

        widths = [cfg.channels * 2**i for i in range(len(cfg.strides) + 1)]
        self.pre = nn.Conv1d(1, widths[0], 7, padding=3)
        self.down = nn.ModuleList(
            Down(widths[i], widths[i + 1], s, cfg.dilations) for i, s in enumerate(cfg.strides)
        )
        self.to_latent = nn.Sequential(Snake(widths[-1]), nn.Conv1d(widths[-1], cfg.dim, 3, padding=1))

        self.quantizer = ResidualVQ(
            cfg.dim,
            cfg.n_codebooks,
            cfg.codebook_size,
            dropout=cfg.quantizer_dropout,
            decay=cfg.decay,
            commit=cfg.commit,
            restart_after=cfg.restart_after,
        )

        self.from_latent = nn.Conv1d(cfg.dim, widths[-1], 3, padding=1)
        self.up = nn.ModuleList(
            Up(widths[-i - 1], widths[-i - 2], s, cfg.dilations)
            for i, s in enumerate(reversed(cfg.strides))
        )
        self.post = nn.Sequential(Snake(widths[0]), nn.Conv1d(widths[0], 1, 7, padding=3))

    # -- shapes -------------------------------------------------------------------------

    def pad_to_hop(self, wave: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Right-pad to a whole number of frames. Returns the padded audio and the true length.

        Right-padding with zeros rather than reflecting: a frame boundary is not a signal
        edge, the pad is at most 20 ms, and reflecting would invent content the decoder is
        then scored on reproducing.
        """
        n = wave.shape[-1]
        rem = (-n) % self.cfg.hop
        return (F.pad(wave, (0, rem)) if rem else wave), n

    # -- the three directions -----------------------------------------------------------

    def latent(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.unsqueeze(1) if wave.dim() == 2 else wave
        x = self.pre(x)
        for stage in self.down:
            x = stage(x)
        return self.to_latent(x)

    @torch.no_grad()
    def encode(self, wave: torch.Tensor, *, n_codebooks: int | None = None) -> torch.Tensor:
        """Waveform `(B, n)` -> codes `(B, n_codebooks, frames)`. The tokenizer, basically."""
        padded, _ = self.pad_to_hop(wave)
        _, idx, _, _ = self.quantizer(self.latent(padded), n_active=n_codebooks)
        return idx if n_codebooks is None else idx[:, :n_codebooks]

    def synthesize(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        for stage in self.up:
            x = stage(x)
        # `tanh` bounds the output to [-1, 1], which is where a waveform lives. Without it
        # early training happily produces samples of ±40 and the STFT loss chases them.
        return torch.tanh(self.post(x)).squeeze(1)

    @torch.no_grad()
    def decode(self, idx: torch.Tensor, *, n_codebooks: int | None = None) -> torch.Tensor:
        """Codes `(B, N, frames)` -> waveform. `n_codebooks` is the bitrate dial."""
        return self.synthesize(self.quantizer.decode(idx, n_active=n_codebooks))

    def forward(self, wave: torch.Tensor, *, n_codebooks: int | None = None):
        """The training path. Returns `(reconstruction, codes, vq_loss, [VQStats])`."""
        padded, n = self.pad_to_hop(wave)
        z = self.latent(padded)
        q, idx, vq_loss, stats = self.quantizer(z, n_active=n_codebooks)
        return self.synthesize(q)[..., :n], idx, vq_loss, stats

    # -- reporting ----------------------------------------------------------------------

    def n_params(self) -> dict[str, int]:
        """Split out because "how big is the codec" has three different honest answers: the
        codebooks are parameters that no optimizer touches."""
        # The codebooks only — not the EMA bookkeeping beside them, which is not weights.
        book = sum(layer.codebook.numel() for layer in self.quantizer.layers)
        enc = sum(p.numel() for p in self.pre.parameters())
        enc += sum(p.numel() for m in self.down for p in m.parameters())
        enc += sum(p.numel() for p in self.to_latent.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "encoder": enc, "decoder": total - enc, "codebooks": book}


# ---------------------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------------------


@dataclass
class ReconstructionLoss:
    """Multi-scale spectral loss, plus a small time-domain term.

    Each scale contributes two pieces, and they do different jobs:

    * **spectral convergence** on the linear magnitudes — `‖|X| − |X̂|‖_F / ‖|X|‖_F` — is
      scale-free, so it is dominated by the *loud* parts of the spectrum. It fixes the
      formants.
    * **log-mel L1** weights every band equally on a log scale, so it is dominated by the
      *quiet* parts. It fixes the noise floor and the breath, which is most of what makes a
      reconstruction sound synthetic.

    Using only the first gives clean vowels over a dead, gated-sounding background; only the
    second gives a smooth background around mushy vowels. Both is the point.

    **Why the log term is on MEL bands and not on the FFT bins.** Measured here during the
    build: the magnitudes of one 2,048-point frame of a harmonic signal span **6e-6 to 211**
    — nine orders of magnitude — and more than half the bins sit at or below any sane floor.
    An L1 on their logs is then mostly a measurement of FFT numerical noise in bins that
    carry no energy: a one-sample circular shift, which is 60 microseconds and completely
    inaudible, scored **0.66** on the linear-bin version. A mel band sums dozens of bins, so
    it is never at the numerical floor — the fix is structural rather than a better-chosen
    epsilon, and it is what the ear does anyway.

    The waveform L1 term has weight 0.1 and is not doing the heavy lifting — see the module
    docstring for why. It is kept because at exactly zero the model has no incentive to get
    the overall gain right, and a codec that reconstructs a perfect spectrum 3 dB quiet is
    an annoying thing to discover at the end.
    """

    scales: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    log_weight: float = 1.0
    convergence_weight: float = 1.0
    wave_weight: float = 0.1
    eps: float = 1e-5
    _cfgs: list[MelConfig] = field(default_factory=list, repr=False)
    _banks: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # hop = n_fft/4 at every scale, so each is a COLA analysis in its own right. The
        # band count follows the window: a 128-point frame resolves 65 bins and asking it
        # for 80 mel bands would leave most of them empty, which is the bug this avoids.
        self._cfgs = [
            MelConfig(n_fft=n, hop=n // 4, n_mels=max(5, min(80, n // 16))) for n in self.scales
        ]

    def _bank(self, cfg: MelConfig, like: torch.Tensor) -> torch.Tensor:
        key = (cfg.n_fft, cfg.n_mels, like.device, like.dtype)
        if key not in self._banks:  # built once, then reused every step
            self._banks[key] = mel_filterbank(cfg, device=like.device, dtype=like.dtype)
        return self._banks[key]

    def __call__(self, target: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        parts: dict[str, float] = {}
        total = pred.new_zeros(())

        for cfg in self._cfgs:
            if target.shape[-1] < cfg.n_fft:
                continue  # a clip shorter than the window has no such scale
            a = stft(target, cfg).abs()
            b = stft(pred, cfg).abs()
            conv = torch.linalg.norm(a - b) / torch.linalg.norm(a).clamp_min(self.eps)

            fb = self._bank(cfg, a)
            logl1 = F.l1_loss(
                torch.log(torch.matmul(fb, b).clamp_min(self.eps)),
                torch.log(torch.matmul(fb, a).clamp_min(self.eps)),
            )
            total = total + self.convergence_weight * conv + self.log_weight * logl1
            parts[f"stft{cfg.n_fft}"] = float(conv)
            parts[f"mel{cfg.n_fft}"] = float(logl1)

        wave = F.l1_loss(pred, target)
        total = total + self.wave_weight * wave
        parts["wave_l1"] = float(wave)
        return total, parts


def load_codec(path, device: str = "cpu") -> Codec:
    """Rebuild a codec from a checkpoint written by `train_codec.save`.

    The shape comes out of the file, never out of a config the caller happens to have: a
    checkpoint whose `strides` differ from the config on disk would otherwise load with
    mismatched keys or, worse, load fine and run at the wrong frame rate.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if "codec" not in blob:
        raise ValueError(
            f"{path} is not a codec checkpoint (stage={blob.get('stage', 'unknown')!r}). "
            "Audio and text checkpoints are separate families — see docs/21."
        )
    cfg = CodecConfig(**{**blob["codec"], "strides": tuple(blob["codec"]["strides"]),
                         "dilations": tuple(blob["codec"]["dilations"])})
    model = Codec(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    model.step = int(blob.get("step", -1))  # for reporting only
    model.best_val = float(blob.get("best_val", float("nan")))
    return model


def codebook_report(stats: list[VQStats], size: int) -> dict:
    """Collapse the per-codebook diagnostics into the row a training log carries.

    **This is the number to watch from step one.** Codebook collapse is invisible in the
    reconstruction loss — it plateaus, which every loss curve does — and completely obvious
    here: `perplexity` is the *effective* codebook size, so 40 out of 1,024 means the model
    has quietly thrown away 96% of its capacity and is still improving smoothly.
    """
    if not stats:
        return {"used": 0, "perplexity": 0.0, "usage": 0.0, "dead": 0}
    return {
        "used": int(sum(s.used for s in stats) / len(stats)),
        "perplexity": sum(s.perplexity for s in stats) / len(stats),
        # Fraction of the codebook in effective use, averaged over stages. One number for a
        # chart; `per_book` keeps the detail, because the LAST codebook collapsing is normal
        # and the FIRST one collapsing is fatal.
        "usage": sum(s.perplexity for s in stats) / len(stats) / size,
        "dead": sum(s.dead for s in stats),
        "per_book": [round(s.perplexity, 1) for s in stats],
    }
