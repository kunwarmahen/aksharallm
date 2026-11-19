"""A content codebook: teach the codec's first codebook to carry the words.

Today codebook 0 carries whatever reduces the reconstruction error most, which is words and
voice together. A text-to-speech model over those codes then has to predict the voice from
text, which it cannot know. The split SpeechTokenizer and Mimi make instead: pull codebook 0
towards the states of an encoder that was trained to recognise speech, and so to ignore who
is speaking. The words then live in codebook 0, which text predicts well, and the voice in
codebooks 1-7, which is where a speaker id or a reference clip can act.

The teacher here is our own recogniser (`asr-libri460`, docs/23), frozen. Its encoder runs at
25 frames/s and 256 wide; codebook 0's output is 50 frames/s and 128 wide. So:

```
codebook 0 output (B, 128, 50/s) ──► Linear 128→256 ──┐
                                                       ├─► 1 − cos, per frame, mean
recogniser states (B, T', 256) ──► resample to 50/s ──┘        (× loss.content_weight)
```

* **Only a projection is added** (`ContentHead`, ~33k parameters). It is saved beside the
  codec in the checkpoint, never inside its state dict, so every reader of a codec checkpoint
  (encode, the audio LM, codec-transfer) is unaffected.
* **Cosine, not L2.** The recogniser's states are layer-normalised; their direction is what
  carries the phone, and an L2 term would mostly teach the projection their norm.
* **The teacher is not the judge's opponent.** The same recogniser judges the codec in
  `codec-transfer`, and a codec trained to please its judge is the trap named in docs/21. This
  loss shapes the *codes* to match the encoder's states, not the reconstruction to minimise
  its error, so it can make codebook 0 legible to a model without making the audio sound
  right to the judge. Read the WER rows of a content-trained codec with that in mind anyway;
  MCD and a second judge (Whisper) are the cross-checks.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentHead(nn.Module):
    """codebook 0's output -> the recogniser's width."""

    def __init__(self, dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, teacher_dim)

    def forward(self, q0: torch.Tensor) -> torch.Tensor:
        return self.proj(q0.transpose(1, 2))   # (B, frames, teacher_dim)


@torch.no_grad()
def teacher_states(recognizer, wave: torch.Tensor, frames: int) -> torch.Tensor:
    """The recogniser's last-block states for `wave`, resampled to `frames` per row.

    Every row of a codec batch is a full window, so no padding mask is needed. Linear
    interpolation in time is the resampling: 25 -> 50 frames/s puts one teacher frame
    between every two codec frames, and the recogniser's subsampling has already blurred
    each frame over ~40 ms.
    """
    n = torch.full((wave.shape[0],), wave.shape[-1], dtype=torch.long, device=wave.device)
    feats, lengths = recognizer.features(wave, n)
    h, _ = recognizer.encode(feats, lengths, hidden=True)
    h = F.interpolate(h.float().transpose(1, 2), size=frames, mode="linear",
                      align_corners=False)
    return h.transpose(1, 2)


def content_loss(head: ContentHead, q0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over frames of `1 - cos(head(q0), target)`: 0 when every frame points the
    recogniser's way, 1 when unrelated, 2 when opposite."""
    pred = head(q0.float())
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def load_teacher(path: str, device: str):
    """The frozen recogniser. Refuses a sample-rate mismatch rather than resampling."""
    from ..asr.train import load_recognizer

    model, _ = load_recognizer(path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
