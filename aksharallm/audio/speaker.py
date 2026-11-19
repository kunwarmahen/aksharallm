"""Does it still sound like *that person*? A speaker embedding, from scratch.

Nothing in this repo could measure voice identity. MCD measures the spectral envelope, which
a codec could match while turning everyone into the same voice; WER measures the words. So
a multi-speaker codec or TTS had no number for the one thing it exists to get right.

The tool is a **speaker embedding**: a network that maps a clip to a vector, trained so two
clips of the same person point the same way and two people point apart. The similarity of a
reconstruction to its own original is then a voice-identity score, and it means something
only beside two reference numbers measured with the same model:

    same speaker, different sentence    what "the same voice" scores
    different speakers                  what "a different voice" scores

A reconstruction scoring near the first is the same voice; near the second, it is not.

```
waveform ─► log-mel (per-utterance normalised) ─► Conformer encoder (asr/model.py)
          ─► mean + std over time ─► Linear ─► unit vector e (192-d)

training:  cos(e, w_speaker) for ~1,170 LibriSpeech speakers, additive margin m, scale s
           logits = s·(cos − m·[target])  ->  cross-entropy        (AM-softmax)
measure:   EER on speakers it NEVER trained on (test-clean's 40)
```

* **The encoder is the recogniser's**, reused rather than rewritten: `Recognizer.encode(
  hidden=True)` already returns per-frame states, and its head is simply unused here.
* **Per-utterance normalisation**, the opposite of the recogniser's default. The recogniser
  needs loudness (to stay silent on faint hiss); a speaker model should *not* be able to
  tell two speakers apart by how loud their microphones were.
* **The margin is the point of AM-softmax.** Plain softmax only has to make the right
  speaker's score the largest; the margin `m` makes it win by a gap, which is what keeps
  unseen speakers apart once the classifier is thrown away.
* **EER on unseen speakers is the only number that matters.** Accuracy over training
  speakers measures a classifier, and the classifier is discarded. Equal error rate is the
  threshold where false accepts equal false rejects; chance is 50%.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..asr.model import AsrModelConfig, Recognizer
from .dataset import Manifest
from .transfer import speaker_of


@dataclass
class SpeakerModelConfig:
    encoder: AsrModelConfig = field(default_factory=lambda: AsrModelConfig(
        d_model=192, n_layers=4, n_heads=4, conv_kernel=15, subsample_channels=128,
        dropout=0.1, max_frames=1024, normalise="utterance"))
    emb_dim: int = 192
    margin: float = 0.2
    scale: float = 30.0

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = AsrModelConfig(**self.encoder)
        if self.encoder.normalise != "utterance":
            # See the module docstring: global stats let the model identify microphones.
            raise ValueError("a speaker encoder must use normalise: utterance")


class SpeakerEncoder(nn.Module):
    """Waveform `(B, n)` -> unit-length embeddings `(B, emb_dim)`."""

    def __init__(self, cfg: SpeakerModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Recognizer(cfg.encoder)
        d = cfg.encoder.d_model
        self.proj = nn.Linear(2 * d, cfg.emb_dim)

    def forward(self, wave: torch.Tensor, n_samples: torch.Tensor | None = None) -> torch.Tensor:
        if n_samples is None:
            n_samples = torch.full((wave.shape[0],), wave.shape[-1], dtype=torch.long,
                                   device=wave.device)
        feats, lengths = self.encoder.features(wave, n_samples)
        h, out_len = self.encoder.encode(feats, lengths, hidden=True)
        mask = (torch.arange(h.shape[1], device=h.device)[None] < out_len[:, None]).float()
        n = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean = (h * mask[..., None]).sum(1) / n
        var = (((h - mean[:, None]) ** 2) * mask[..., None]).sum(1) / n
        pooled = torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1)
        return F.normalize(self.proj(pooled.float()), dim=-1)


class AMSoftmax(nn.Module):
    """Additive-margin softmax over the training speakers (thrown away after training)."""

    def __init__(self, emb_dim: int, n_speakers: int, margin: float, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_speakers, emb_dim) * 0.01)
        self.margin, self.scale = margin, scale

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        cos = emb @ F.normalize(self.weight, dim=-1).T
        logits = self.scale * (cos - self.margin * F.one_hot(labels, cos.shape[1]))
        return F.cross_entropy(logits, labels), (cos.argmax(-1) == labels).float().mean()


class SpeakerCorpus:
    """Random crops of packed corpora, labelled by speaker (`<speaker>-<chapter>-<utt>`).

    Speakers are numbered across all the corpora given, in a fixed sorted order, so a resumed
    run's labels mean what they meant before. The last `val_clips` clips of each speaker are
    held out (by clip, never inside one) for the training-speaker accuracy check.
    """

    def __init__(self, corpora: list[str], crop: int, *, seed: int = 0, val_clips: int = 1,
                 split: str = "train"):
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.parts = []
        by_spk: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for ci, c in enumerate(corpora):
            d = Path(c)
            man = Manifest.load(d / "manifest.json")
            self.parts.append((np.memmap(d / "audio.bin", dtype="<i2", mode="r"), man))
            for i, name in enumerate(man.names):
                if man.offsets[i + 1] - man.offsets[i] >= crop:
                    by_spk[f"{d.name}/{speaker_of(name, d.name)}"].append((ci, i))
        self.speakers = sorted(by_spk)
        self.items = []
        for label, spk in enumerate(self.speakers):
            clips = by_spk[spk]
            keep = clips[-val_clips:] if split == "val" else clips[:-val_clips] or clips
            self.items += [(label, ci, i) for ci, i in keep]
        if not self.items:
            raise ValueError("no clip is long enough for one crop")

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    def _crop(self, ci: int, i: int, start: int | None = None) -> np.ndarray:
        samples, man = self.parts[ci]
        a, b = man.offsets[i], man.offsets[i + 1]
        s = a + (self.rng.integers(0, b - a - self.crop + 1) if start is None else start)
        return samples[s:s + self.crop].astype(np.float32) / 32768.0

    def batch(self, batch_size: int, device: str = "cpu"):
        picks = self.rng.integers(0, len(self.items), size=batch_size)
        waves = np.stack([self._crop(*self.items[p][1:]) for p in picks])
        labels = torch.tensor([self.items[p][0] for p in picks], dtype=torch.long)
        return torch.from_numpy(waves).to(device), labels.to(device)


def eer(scores: np.ndarray, same: np.ndarray) -> float:
    """Equal error rate: the threshold where false accepts equal false rejects."""
    order = np.argsort(-scores)
    same = same[order].astype(bool)
    pos, neg = same.sum(), (~same).sum()
    if not pos or not neg:
        raise ValueError("EER needs both same- and different-speaker pairs")
    fa = np.cumsum(~same) / neg            # accepting everything scored at or above here
    fr = 1 - np.cumsum(same) / pos
    k = int(np.argmin(np.abs(fa - fr)))
    return float((fa[k] + fr[k]) / 2)


@torch.no_grad()
def embed(model: SpeakerEncoder, waves: list[np.ndarray], device: str = "cpu",
          batch: int = 16) -> torch.Tensor:
    """Embeddings of whole clips of any length, one at a time per length bucket."""
    model.eval()
    out = []
    for w in waves:
        x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None].to(device)
        out.append(model(x)[0].float().cpu())
    return torch.stack(out)


def verification(model: SpeakerEncoder, clips: list[dict], device: str = "cpu") -> dict:
    """EER and the two reference similarities over every pair of `clips`
    (`{speaker, wave}`), on speakers the model never trained on."""
    e = embed(model, [c["wave"] for c in clips], device)
    spk = [c["speaker"] for c in clips]
    scores, same = [], []
    for i, j in itertools.combinations(range(len(clips)), 2):
        scores.append(float(e[i] @ e[j]))
        same.append(spk[i] == spk[j])
    scores, same = np.array(scores), np.array(same)
    return {"eer": eer(scores, same), "pairs": len(scores),
            "same_speaker": float(scores[same].mean()),
            "different_speaker": float(scores[~same].mean()),
            "speakers": len(set(spk)), "clips": len(clips)}


def save_speaker(model: SpeakerEncoder, path, **extra) -> None:
    torch.save({"stage": "speaker", "speaker": asdict(model.cfg),
                "model": model.state_dict(), **extra}, path)


def load_speaker(path, device: str = "cpu") -> SpeakerEncoder:
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("stage") != "speaker":
        raise ValueError(f"{path} is a {blob.get('stage')!r} checkpoint, not a speaker model")
    model = SpeakerEncoder(SpeakerModelConfig(**blob["speaker"])).to(device)
    model.load_state_dict(blob["model"])
    return model.eval()
