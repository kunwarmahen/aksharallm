"""Vision — a second modality, and the cheapest proof of the same claim.

The audio phase needed a codec to turn a waveform into integers. An image needs nothing: it
is **already** a grid of numbers, so cutting it into patches produces a sequence directly.
What is left is a bridge — a two-layer MLP that puts vision vectors where the language
model's word vectors live — and that bridge is the whole of LLaVA.

```mermaid
flowchart LR
    I["image"] --> E["ViT<br/>one vector per patch"]
    E --> P["projector<br/>2-layer MLP"]
    P --> C["concatenate with<br/>text embeddings"]
    C --> L["the FROZEN language model"]
```

Everything expensive is already trained. The only new parameters are the tower, and the only
*required* new parameters are the projector — under a million of them.

Read with: docs/21-vision.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from .encoder import Projector, VisionConfig, VisionEncoder, VisionTower, patchify
from .image import ImageCaptions, ImageManifest, caption_of, render, synth_corpus
from .lm import VisionLanguageModel, caption, score_batch, score_caption

__all__ = [
    "VisionConfig", "VisionEncoder", "Projector", "VisionTower", "patchify",
    "ImageCaptions", "ImageManifest", "synth_corpus", "render", "caption_of",
    "VisionLanguageModel", "caption", "score_caption", "score_batch",
]
