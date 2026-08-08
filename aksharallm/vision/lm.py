"""Putting an image into a language model that has never seen one.

This file is short, and that is the result. The seam it uses — `inputs_embeds` — was added
for the audio phase and is unchanged here; the language model is imported, frozen, and told
nothing. What arrives at its first block is a sequence of vectors, and some of them happen
to have come from a picture.

```mermaid
flowchart LR
    I["image"] --> V["vision tower<br/>-> N vectors in d_model"]
    T["'a photo of'"] --> E["tok_emb<br/>-> M vectors in d_model"]
    V --> C["concatenate:<br/>image tokens, then text"]
    E --> C
    C --> L["the FROZEN language model"]
    L --> O["loss on the TEXT only"]
```

**The loss is on the text only**, which by now is the third time that sentence appears in
this repo: it is the SFT assistant-only mask (`docs/06`), it is how TTS and ASR work
(`docs/21`), and it is this. The image is the prompt; the caption is the response.

**Stage one trains the projector and nothing else.** The language model is frozen, the vision
encoder can be too, and what is learned is a two-layer map between two spaces that both
already exist. That is why a vision-language model is cheap to *bolt on* and expensive to
train from scratch — and it is why this fits in an evening on a 3090 when a codec took hours.

Two mistakes this is built to avoid
-------------------------------------
**Unfreezing by accident.** `requires_grad = False` on the parameters is not enough on its
own to make a claim of "frozen" true — an optimizer constructed over `model.parameters()`
would still hold them and weight decay would still move them. `trainable_parameters()` is
the single place that decides, and a test asserts the language model's weights are
bit-identical after a training step.

**Counting image tokens wrong.** The loss mask, the position of the first caption token, and
the generation prompt all depend on exactly how many vectors the image became. That number
is `VisionTower.n_image_tokens` and nothing recomputes it.

Read with: docs/22-vision.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/06-posttraining.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.transformer import Transformer
from .encoder import VisionConfig, VisionTower


class VisionLanguageModel(nn.Module):
    """A frozen language model with a vision tower bolted onto its input.

    `freeze_language_model=False` is stage two — fine-tuning the whole thing on instruction
    data — and is a different, much more expensive run. Stage one is the default.
    """

    def __init__(self, language_model: Transformer, vision: VisionConfig,
                 *, freeze_language_model: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.lm = language_model
        self.tower = VisionTower(vision, language_model.cfg.d_model)
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder

        if freeze_language_model:
            for p in self.lm.parameters():
                p.requires_grad_(False)
        if freeze_encoder:
            for p in self.tower.encoder.parameters():
                p.requires_grad_(False)

    # -- what is actually trained --------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        """**The single place that decides what an optimizer is given.**

        Building an optimizer over `model.parameters()` and relying on `requires_grad`
        alone leaves the frozen weights in the optimizer's state, where weight decay still
        moves them — a decay term does not consult the gradient. So the optimizer gets this
        list, and `test_vision.py` asserts the language model comes out bit-identical.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def n_params(self) -> dict[str, int]:
        counts = self.tower.n_params()
        counts["language_model"] = sum(p.numel() for p in self.lm.parameters())
        counts["trainable"] = sum(p.numel() for p in self.trainable_parameters())
        return counts

    @property
    def n_image_tokens(self) -> int:
        return self.tower.n_image_tokens

    # -- the forward pass ----------------------------------------------------------------

    def embed(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` and `(B, T)` -> `(B, n_image + T, d_model)`.

        Image first. It could go either way — a causal mask means only the *caption* needs
        to see the image, not the reverse — and image-first is the convention because it
        makes the caption a continuation, which is exactly what the language model already
        knows how to do.
        """
        vision = self.tower(images)
        if self.freeze_language_model:
            # The embedding lookup is a frozen weight too. No grad flows to it either way,
            # but saying so keeps the graph small on a 32k x 1024 table.
            with torch.no_grad():
                words = self.lm.tok_emb(text)
        else:
            words = self.lm.tok_emb(text)
        return torch.cat([vision, words], dim=1)

    def forward(self, images: torch.Tensor, text: torch.Tensor,
                targets: torch.Tensor | None = None):
        """Returns `(logits over the text positions, loss)`.

        `targets` is `(B, T)` aligned with `text`, `-100` where the loss must not look.
        """
        x = self.embed(images, text)
        hidden, _ = self.lm(None, inputs_embeds=x, return_hidden=True)
        # Only the text half has a next token to predict. The image positions predict the
        # first caption token, which is genuinely useful signal, so the slice starts one
        # position early rather than at the caption itself.
        n_img = self.n_image_tokens
        text_hidden = hidden[:, n_img - 1 :]
        logits = self.lm.lm_head(text_hidden)

        loss = None
        if targets is not None:
            # `logits[:, i]` predicts `text[:, i]` — the `n_img - 1` slice above IS the shift,
            # so `targets` must be the caption *unshifted* (with -100 on the BOS position).
            # Shifting again in the caller stacks two shifts and produces a model that emits
            # the token after next: perfect training loss, garbage generation.
            loss = F.cross_entropy(
                logits[:, : targets.shape[1]].reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


@torch.no_grad()
def caption(model: VisionLanguageModel, image: torch.Tensor, tokenizer, *,
            prompt: str = "", max_tokens: int = 24, temperature: float = 0.0,
            device: str = "cpu") -> str:
    """Greedy by default, because a caption of a shape corpus has a right answer.

    No KV cache. The sequence is under a hundred positions and re-encoding it each step is
    cheaper than the code to keep a cache correct across a prefix of image embeddings would
    be — the same call `audio/lm.py` makes, for the same reason.
    """
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    ids = tokenizer.encode(prompt, bos=True) if prompt else [tokenizer.bos_id]
    for _ in range(max_tokens):
        text = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(image, text)
        row = logits[0, -1].float()
        nxt = int(row.argmax()) if temperature <= 0 else int(
            torch.multinomial(torch.softmax(row / temperature, -1), 1)
        )
        if tokenizer.eos_id is not None and nxt == tokenizer.eos_id:
            break
        ids.append(nxt)
    return tokenizer.decode(ids[1:])


def score_caption(fact: dict, said: str) -> dict:
    """Did it get the count, the colour and the shape? Three booleans, not one number.

    **This is the whole reason the corpus is synthetic.** A caption model on real data can
    only be judged by reading its output; here the ground truth is a dict we chose, so the
    three attributes can be scored separately — and separately is the point, because a model
    that gets colour and shape right while never counting correctly is a specific, diagnosable
    failure that a single accuracy would average away.
    """
    from .image import NUMBERS

    said = said.lower()
    return {
        "count": NUMBERS[fact["count"] - 1] in said,
        "colour": fact["colour"] in said,
        "shape": fact["shape"] in said,
    }


def score_batch(pairs) -> dict:
    """`[(fact, said)]` -> per-attribute accuracy, plus the all-three rate."""
    rows = [score_caption(f, s) for f, s in pairs]
    if not rows:
        return {"n": 0}
    n = len(rows)
    out = {k: sum(r[k] for r in rows) / n for k in ("count", "colour", "shape")}
    out["all_three"] = sum(all(r.values()) for r in rows) / n
    out["n"] = n
    return out
