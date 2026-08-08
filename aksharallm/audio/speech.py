"""TTS and ASR — the same model, the same loss mask, the sequence written in two orders.

This is the last piece of the audio phase, and it is the shortest, because both directions
turn out to be one idea already implemented in `train/sft.py`: **put two things in one
sequence and take the loss on only one of them.**

```mermaid
flowchart LR
    subgraph tts["TTS — text_first = True"]
        T1["h e l l o"] --> A1["audio frames"]
        A1 --> L1["loss HERE only"]
    end
    subgraph asr["ASR — text_first = False"]
        A2["audio frames"] --> T2["h e l l o"]
        T2 --> L2["loss HERE only"]
    end
```

That is exactly the assistant-only mask: the prompt is context, the response is supervised,
and a causal mask does the rest. The only thing audio changes is *what* is on each side —
which is the claim the whole phase exists to demonstrate.

**ASR is the piece with a real number.** Word error rate is an established metric with a
published range anyone can check, and it needs no human. TTS has no such number, which is
trap 7 and is handled honestly here: we report mel-cepstral distortion against the reference
recording, **and our own ASR model's WER on our own TTS output**, labelled as
*intelligibility* — because that is what it measures. It is not quality, and a system that
scores well on it can still sound like a robot.

**The alignment trap, which MCD walks straight into.** `measure.mcd` compares two waveforms
frame by frame. A TTS model saying the right words at a slightly different rate is *correct*
and scores terribly. So `tts_report` reports MCD only for the case where alignment is
defensible (a reconstruction of a held-out utterance the model was conditioned on) and
leans on intelligibility otherwise. Reporting an unaligned MCD as if it meant something is
the mistake this docstring exists to prevent.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/06-posttraining.md.
"""

from __future__ import annotations

import torch

from .delay import delay, undelay
from .lm import AudioLM, make_targets
from .text import BOS, EOS, PAD, CharTokenizer, character_error_rate, word_error_rate


def pad_text(batch: list[list[int]], device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, L)` ids and targets, right-padded. Padding is `-100` in the targets.

    Right-padding needs no attention mask under a causal mask — the same argument the eval
    harness makes for its scoring batches — because a padded position can only be attended
    to by later padded positions, whose loss is ignored anyway.
    """
    width = max(len(b) for b in batch)
    ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b)] = torch.tensor(b)
        tgt[i, : len(b)] = torch.tensor(b)
    return ids.to(device), tgt.to(device)


def tts_batch(model: AudioLM, codes: torch.Tensor, texts: list[list[int]]):
    """Loss for text -> audio. Supervises the audio only; the text is context."""
    d = delay(codes, model.pad_id)
    ids, _ = pad_text(texts, codes.device)
    return model(d, text=ids, targets=make_targets(d, codes.shape[-1], model.pad_id),
                 text_first=True)


def asr_batch(model: AudioLM, codes: torch.Tensor, texts: list[list[int]]):
    """Loss for audio -> text. Supervises the text only; the audio is context."""
    d = delay(codes, model.pad_id)
    ids, tgt = pad_text(texts, codes.device)
    return model(d, text=ids, text_targets=tgt, text_first=False)


# ---------------------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def speak(
    model: AudioLM,
    text_ids: list[int],
    n_frames: int,
    *,
    temperature: float = 0.8,
    top_k: int = 100,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """TTS. Text ids in, undelayed codes `(1, N, n_frames)` out — feed them to `codec.decode`.

    Sampling rather than greedy, and that is not a stylistic choice: greedy TTS produces a
    flat monotone, because the model's single most likely continuation of any prosody is the
    average of all prosody. Temperature 0.8 is the usual compromise between "expressive" and
    "occasionally says something else entirely".
    """
    cfg = model.cfg
    n = cfg.n_codebooks
    model.eval()
    text = torch.tensor([text_ids], dtype=torch.long, device=device)
    seq = torch.full((1, n, 1), model.bos_id, dtype=torch.long, device=device)

    for s in range(n_frames + n - 1):
        logits, _ = model(seq, text=text, text_first=True)
        step = logits[0, -1]
        nxt = torch.full((1, n, 1), model.pad_id, dtype=torch.long, device=device)
        for k in range(n):
            if not (k <= s < k + n_frames):
                continue
            row = step[k].float() / max(temperature, 1e-6)
            if top_k:
                cut = torch.topk(row, min(top_k, row.numel())).values[-1]
                row = row.masked_fill(row < cut, float("-inf"))
            nxt[0, k, 0] = torch.multinomial(torch.softmax(row, -1), 1, generator=generator)
        seq = torch.cat([seq, nxt], dim=-1)
    return undelay(seq[:, :, 1:], n_frames=n_frames)


@torch.no_grad()
def transcribe(
    model: AudioLM,
    codes: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    max_chars: int = 200,
    device: str = "cpu",
) -> str:
    """ASR. Codes `(1, N, T)` in, a string out. Greedy, on purpose.

    Greedy here and sampled in `speak`, which looks inconsistent and is not: transcription
    has a right answer and sampling can only move away from it, while speech has no single
    right answer and greedy collapses to a monotone. The decoding strategy follows the task,
    not the codebase.
    """
    model.eval()
    d = delay(codes.to(device), model.pad_id)
    out = [BOS]
    for _ in range(max_chars):
        text = torch.tensor([out], dtype=torch.long, device=device)
        hidden, _ = model.body(
            None,
            inputs_embeds=torch.cat(
                [model.embed_audio(d), model.embed_text(text)], dim=1
            ),
            return_hidden=True,
        )
        nxt = int(model.text_head(hidden[:, -1]).argmax(-1))
        if nxt == EOS:
            break
        out.append(nxt)
    return tokenizer.decode(out[1:])


# ---------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------


def asr_report(pairs: list[tuple[str, str]]) -> dict:
    """`[(reference, hypothesis)]` -> WER and CER.

    Both, because a character-level model that misses one letter loses a whole word to WER,
    and quoting only WER would overstate how bad it is by a factor of the word length.
    """
    if not pairs:
        return {"n": 0, "wer": float("nan"), "cer": float("nan")}
    wers = [word_error_rate(r, h) for r, h in pairs]
    cers = [character_error_rate(r, h) for r, h in pairs]
    return {
        "n": len(pairs),
        "wer": sum(wers) / len(wers),
        "cer": sum(cers) / len(cers),
        # A hypothesis that never stops produces a WER over 1. Counting those separately is
        # what distinguishes "gets words wrong" from "will not shut up", which are different
        # problems with different fixes.
        "runaway": sum(1 for w in wers if w > 1.0),
    }


def intelligibility(asr_model, tokenizer, codec, tts_codes, reference: str, device="cpu") -> dict:
    """Our own ASR model's error rate on our own TTS output. **Labelled, not renamed.**

    This is the closest thing to an honest automatic score for TTS, and it is a measure of
    *intelligibility* — whether the words came out — not of quality. A synthesiser with a
    flat robotic monotone can score perfectly here. Mean opinion score needs humans, we do
    not have them, and the right response is to say so rather than to promote this number.

    It is also **self-referential**: a bad ASR model makes a good TTS model look bad. Report
    the ASR model's own WER on real speech beside it or the number cannot be interpreted.
    """
    hypothesis = transcribe(asr_model, tts_codes, tokenizer, device=device)
    return {
        "reference": reference,
        "heard": hypothesis,
        "wer": word_error_rate(reference, hypothesis),
        "cer": character_error_rate(reference, hypothesis),
        "measures": "intelligibility, not quality",
    }
