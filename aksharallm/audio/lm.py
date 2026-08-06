"""The audio language model — the same transformer, over the codec's integers.

This file is deliberately thin, and that is the claim of the whole phase: `model/
transformer.py` is imported unchanged and knows nothing about sound. What is added here is
only what the *shape* of an audio token demands.

```mermaid
flowchart LR
    subgraph in["one position"]
        E0["book 0 embed"] --> S["sum"]
        E1["book 1 embed"] --> S
        EN["... book 7"] --> S
    end
    S --> B["the SAME Transformer<br/>blocks, RoPE, GQA, KV cache"]
    B --> H["8 heads<br/>one per codebook"]
    H --> C["8 code distributions"]
```

**Why the embeddings are summed.** A position carries eight integers. Concatenating their
embeddings would make `d_model` depend on `n_codebooks`; summing keeps the body's width
fixed and is what every published codec LM does. It works because the tables are separate —
code 5 of book 0 and code 5 of book 3 are different vectors — so the sum is a genuine
composition rather than a collision.

**Why there are eight heads and not one.** The eight codebooks have eight unrelated
vocabularies. A single head over `8 × 1024` would let the model put mass on a book-3 code
while predicting book 0, which is not a possible answer.

**Three models in one class.** `text_vocab_size > 0` adds a text embedding table and a text
head, and then the same module is:

| use | sequence | loss on |
|---|---|---|
| audio LM | audio frames | audio |
| **TTS** | text, then audio frames | audio only |
| **ASR** | audio frames, then text | text only |

That is the SFT assistant-only loss mask, with audio in the assistant's role — the same idea
`train/sft.py` already implements, which is why neither of those two is a new training loop.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/03-model.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..model.transformer import Transformer
from .delay import special_ids, undelay, valid_mask


@dataclass
class AudioLMConfig:
    """The audio LM's shape. `codebook_size` and `n_codebooks` must match its codec exactly.

    They are recorded in the checkpoint and checked on load: an audio LM run against a
    different codec produces confident nonsense, because the integers mean different sounds.
    """

    codebook_size: int = 1024
    n_codebooks: int = 8
    #: Frames of audio the model can see. The real context is `max_frames + n_codebooks − 1`
    #: audio positions plus `max_text` text positions, and `model.max_seq_len` must cover it.
    max_frames: int = 500  # 10 s at 50 frames/s
    max_text: int = 0  # > 0 turns on the text path (TTS / ASR)
    text_vocab_size: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def audio_positions(self) -> int:
        return self.max_frames + self.n_codebooks - 1

    @property
    def vocab_per_book(self) -> int:
        """Real codes plus `[PAD]` and `[BOS]`. Only the real codes are ever *predicted*."""
        return self.codebook_size + 2


class AudioLM(nn.Module):
    def __init__(self, cfg: AudioLMConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id, self.bos_id = special_ids(cfg.codebook_size)

        body_cfg = ModelConfig(**{**cfg.model.__dict__})
        # The body's own `tok_emb` and `lm_head` are unused: this model has eight of each.
        # Sized to 1 rather than deleted, so `Transformer` stays exactly the class the rest
        # of the repo trains — no subclass, no monkey-patch, and `interp/` still works on it.
        # The cost is 2 x d_model dead parameters, which at d_model 512 is one kilobyte.
        body_cfg.vocab_size = 1
        body_cfg.tie_embeddings = False
        need = cfg.audio_positions + cfg.max_text
        if body_cfg.max_seq_len < need:
            raise ValueError(
                f"model.max_seq_len={body_cfg.max_seq_len} cannot hold {cfg.max_frames} frames "
                f"+ {cfg.n_codebooks - 1} of delay + {cfg.max_text} of text = {need} positions"
            )
        self.body = Transformer(body_cfg)

        d = body_cfg.d_model
        self.embeds = nn.ModuleList(
            nn.Embedding(cfg.vocab_per_book, d) for _ in range(cfg.n_codebooks)
        )
        self.heads = nn.ModuleList(
            nn.Linear(d, cfg.codebook_size, bias=False) for _ in range(cfg.n_codebooks)
        )
        self.text_emb = self.text_head = None
        if cfg.text_vocab_size:
            self.text_emb = nn.Embedding(cfg.text_vocab_size, d)
            self.text_head = nn.Linear(d, cfg.text_vocab_size, bias=False)

        for m in list(self.embeds) + list(self.heads):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if self.text_emb is not None:
            nn.init.normal_(self.text_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.text_head.weight, mean=0.0, std=0.02)

    # -- embedding ----------------------------------------------------------------------

    def embed_audio(self, delayed: torch.Tensor) -> torch.Tensor:
        """Delayed codes `(B, N, S)` -> `(B, S, d_model)`, by summing the N tables."""
        return sum(emb(delayed[:, k]) for k, emb in enumerate(self.embeds))

    def embed_text(self, text: torch.Tensor) -> torch.Tensor:
        if self.text_emb is None:
            raise ValueError("this audio LM has no text path (text_vocab_size = 0)")
        return self.text_emb(text)

    # -- the forward pass ---------------------------------------------------------------

    def forward(
        self,
        delayed: torch.Tensor,
        *,
        text: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        text_targets: torch.Tensor | None = None,
        text_first: bool = True,
        caches=None,
    ):
        """Returns `(logits, loss)`.

        `logits` is `(B, S, N, codebook_size)` — the audio positions only, whatever the text
        did. `targets` is the delayed code tensor `(B, N, S)` with `-100` wherever the loss
        must not look, which is every padding cell of the delay triangle.

        `text_first=True` is TTS (condition on text, generate audio); `False` is ASR. The
        only thing that changes is which side of the concatenation each block of embeddings
        goes, because a causal mask does the rest.
        """
        audio_x = self.embed_audio(delayed)
        n_audio = audio_x.shape[1]

        if text is None:
            x, audio_slice = audio_x, slice(0, n_audio)
        else:
            text_x = self.embed_text(text)
            if text_first:
                x = torch.cat([text_x, audio_x], dim=1)
                audio_slice = slice(text_x.shape[1], text_x.shape[1] + n_audio)
            else:
                x = torch.cat([audio_x, text_x], dim=1)
                audio_slice = slice(0, n_audio)

        hidden, _ = self.body(None, inputs_embeds=x, caches=caches, return_hidden=True)
        audio_h = hidden[:, audio_slice]
        # (B, S, N, V). Stacked rather than concatenated so the codebook axis stays a real
        # axis — every consumer of this indexes by codebook, including generation.
        logits = torch.stack([head(audio_h) for head in self.heads], dim=2)

        loss = None
        if targets is not None:
            # The shift is the ordinary next-token one: position s predicts position s+1.
            # It happens on the DELAYED sequence, which is what makes the pattern work —
            # book 1 at column s+1 is frame s, and book 0 at column s (already seen) is the
            # same frame. The dependency is carried by the context, not by the loss.
            pred = logits[:, :-1]  # (B, S-1, N, V)
            gold = targets[:, :, 1:].permute(0, 2, 1)  # (B, S-1, N)
            loss = F.cross_entropy(
                pred.reshape(-1, self.cfg.codebook_size).float(),
                gold.reshape(-1),
                ignore_index=-100,
            )
        if text_targets is not None:
            text_h = hidden[:, : text.shape[1]] if text_first else hidden[:, n_audio:]
            tl = F.cross_entropy(
                self.text_head(text_h)[:, :-1].reshape(-1, self.cfg.text_vocab_size).float(),
                text_targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = tl if loss is None else loss + tl
        return logits, loss

    # -- reporting ----------------------------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_targets(delayed: torch.Tensor, n_frames: int, pad_id: int) -> torch.Tensor:
    """Turn a delayed code tensor into training targets, with the pad triangle ignored.

    Kept beside the model rather than in the trainer because getting it wrong is silent:
    training on the padding teaches the model to predict a token that carries no information
    at exactly the two places where it has the least context to do it from, and the loss
    curve merely looks a little better than it should.
    """
    n_codebooks = delayed.shape[1]
    mask = valid_mask(n_codebooks, n_frames, device=delayed.device)
    return torch.where(mask.unsqueeze(0), delayed, torch.full_like(delayed, -100))


@torch.no_grad()
def generate(
    model: AudioLM,
    n_frames: int,
    *,
    text: torch.Tensor | None = None,
    prompt: torch.Tensor | None = None,
    temperature: float = 0.9,
    top_k: int = 250,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample `n_frames` of audio codes. Returns undelayed `(1, N, n_frames)`.

    No KV cache: the sequence is a few hundred positions and re-encoding it each step costs
    less than the code to keep a cache correct across a delay pattern would. `serve/` is
    where that would go if audio ever needed to be fast.

    **The delay triangle is placed, not predicted.** Whether a cell is padding is a property
    of the pattern and is known exactly, so sampling it would only give the model a chance
    to be wrong about arithmetic. Heads therefore emit real codes only, and the loop writes
    `pad_id` where the pattern says it belongs.
    """
    cfg = model.cfg
    n, total = cfg.n_codebooks, n_frames + cfg.n_codebooks - 1
    model.eval()

    seq = torch.full((1, n, 1), model.bos_id, dtype=torch.long, device=device)
    n_prompt = 0
    if prompt is not None:
        # A prompt continues real audio, so it arrives already delayed and its own triangle
        # is already correct. Only the frames after it are generated.
        n_prompt = prompt.shape[-1]
        seq = torch.cat([seq, prompt.to(device)], dim=-1)

    for s in range(n_prompt, total):
        logits, _ = model(seq, text=text)
        step = logits[0, -1]  # (N, codebook_size) — the prediction for the next column
        nxt = torch.full((1, n, 1), model.pad_id, dtype=torch.long, device=device)
        for k in range(n):
            if not (k <= s < k + n_frames):
                continue  # outside this codebook's window: the pattern says padding
            row = step[k].float() / max(temperature, 1e-6)
            if top_k:
                keep = min(top_k, row.numel())
                cut = torch.topk(row, keep).values[-1]
                row = row.masked_fill(row < cut, float("-inf"))
            probs = torch.softmax(row, dim=-1)
            nxt[0, k, 0] = torch.multinomial(probs, 1, generator=generator)
        seq = torch.cat([seq, nxt], dim=-1)

    return undelay(seq[:, :, 1:], n_frames=n_frames)  # drop the BOS column
