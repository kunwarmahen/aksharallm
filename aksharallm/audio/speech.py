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
import torch.nn.functional as F

from .delay import delay, undelay
from .lm import AudioLM
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


def utterance_loss(model: AudioLM, clips: list[torch.Tensor], texts: list[list[int]],
                   text_first: bool = True) -> torch.Tensor:
    """The loss for a batch of whole utterances of *different* lengths. TTS or ASR.

    `clips[i]` is undelayed codes `(N, T_i)`; `texts[i]` is that clip's character ids.

    **Each row is laid out exactly as inference lays it out, and padded only at its end:**

        TTS  [text_i][BOS + delayed audio_i][pad ...]     loss on the audio
        ASR  [delayed audio_i][text_i][pad ...]           loss on the text

    The obvious batch -- pad the texts to one width and the audio to another, then
    concatenate the two blocks -- puts `[PAD]`s *between* a short sentence and its audio, and
    starts the audio at a different position in every row. `speak` and `transcribe` never
    see those pads, so the model would learn a layout it is never asked to use. On the
    synthetic corpus every clip was the same length and nothing showed it; LJSpeech runs from
    one second to ten. Right-padding the combined row needs no attention mask under a causal
    mask, for the reason `pad_text` gives. `tests/test_audiolm.py` asserts a ragged batch
    scores exactly what its rows score one at a time.
    """
    from .delay import valid_mask

    n, pad, bos = model.cfg.n_codebooks, model.pad_id, model.bos_id
    dev = model.embeds[0].weight.device
    b = len(clips)
    frames = [int(c.shape[-1]) for c in clips]
    width = max(frames)
    codes = torch.full((b, n, width), pad, dtype=torch.long, device=dev)
    for i, c in enumerate(clips):
        codes[i, :, : frames[i]] = c.to(dev)
    d = delay(codes, pad)                                         # (B, N, W + N - 1)
    if text_first:  # TTS starts the audio from a BOS column, as `speak` does
        d = torch.cat([d.new_full((b, n, 1), bos), d], dim=-1)
    lead = 1 if text_first else 0
    a_len = [t + n - 1 + lead for t in frames]                    # audio positions per row

    tid, ttgt = pad_text(texts, dev)
    t_len = [len(t) for t in texts]
    audio_x, text_x = model.embed_audio(d), model.embed_text(tid)

    total = [a + t for a, t in zip(a_len, t_len)]
    x = audio_x.new_zeros((b, max(total), audio_x.shape[-1]))
    a_at, t_at = [], []                                           # where each part starts
    for i in range(b):
        first, second = (t_len[i], a_len[i]) if text_first else (a_len[i], t_len[i])
        a0 = first if text_first else 0
        t0 = 0 if text_first else first
        x[i, a0 : a0 + a_len[i]] = audio_x[i, : a_len[i]]
        x[i, t0 : t0 + t_len[i]] = text_x[i, : t_len[i]]
        a_at.append(a0)
        t_at.append(t0)
    hidden, _ = model.body(None, inputs_embeds=x, return_hidden=True)

    def gather(starts, width_):
        idx = torch.tensor(starts, device=dev).unsqueeze(1) + torch.arange(width_, device=dev)
        idx = idx.clamp(max=hidden.shape[1] - 1)    # past a row's end: targets are -100
        return hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

    if text_first:
        tgt = torch.full_like(d, -100)
        for i, t in enumerate(frames):
            s = t + n - 1
            m = valid_mask(n, t, device=dev)
            tgt[i, :, 1 : 1 + s] = torch.where(m, d[i, :, 1 : 1 + s],
                                               torch.full_like(d[i, :, 1 : 1 + s], -100))
        h = gather(a_at, d.shape[-1])
        logits = torch.stack([head(h) for head in model.heads], dim=2)   # (B, S, N, V)
        return F.cross_entropy(logits[:, :-1].reshape(-1, model.cfg.codebook_size).float(),
                               tgt[:, :, 1:].permute(0, 2, 1).reshape(-1), ignore_index=-100)
    h = gather(t_at, tid.shape[1])
    return F.cross_entropy(model.text_head(h)[:, :-1].reshape(-1, model.cfg.text_vocab_size).float(),
                           ttgt[:, 1:].reshape(-1), ignore_index=-100)


def tts_batch(model: AudioLM, codes: torch.Tensor, texts: list[list[int]]):
    """Loss for text -> audio. Supervises the audio only; the text is context."""
    return None, utterance_loss(model, list(codes), texts, text_first=True)


def asr_batch(model: AudioLM, codes: torch.Tensor, texts: list[list[int]]):
    """Loss for audio -> text. Supervises the text only; the audio is context."""
    return None, utterance_loss(model, list(codes), texts, text_first=False)


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


def plain_words(text: str) -> str:
    """What a recogniser can be compared against: lower case letters, apostrophes, spaces.
    LJSpeech's transcripts carry punctuation the recogniser never writes."""
    import re
    return " ".join(re.sub(r"[^a-z' ]+", " ", text.lower()).split())


def load_tts(path, device: str = "cpu"):
    """`(model, tokenizer, blob)` for an audio LM trained with `data.task: tts`."""
    from ..config import ModelConfig
    from .lm import AudioLMConfig

    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("stage") != "audiolm" or blob.get("task") != "tts":
        raise ValueError(f"{path} is not a TTS audio LM (stage={blob.get('stage')!r}, "
                         f"task={blob.get('task')!r})")
    shape = dict(blob["audiolm"])
    shape["model"] = ModelConfig(**shape["model"])
    model = AudioLM(AudioLMConfig(**shape)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, CharTokenizer.from_dict(blob["tokenizer"]), blob


@torch.no_grad()
def tts_ladder(model, tok, codec, utterances, originals, transcribe_fn, *, temperature=0.8,
               seed=0, device="cpu", keep=None) -> dict:
    """The same sentences three ways, judged by one independent recogniser.

        original   the recording itself            -> the judge's own error on this voice
        codec      the recording through the codec -> what the codec loses
        tts        the model speaking the text     -> what the TTS gets wrong

    Each row is only readable against the one above it. The TTS is given each sentence's
    **real duration** (`speak` needs a frame count, and this model has no stop token), which
    is an advantage a deployed system would not have, and the report says so.

    `utterances` is `[(codes (N, T), text ids)]`, `originals` the matching float waveforms,
    `transcribe_fn(list_of_waves) -> list_of_str`. `keep(j, name, wave)` is called for each
    waveform, so the caller can write them out to listen to.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    refs, waves = [], {"original": [], "codec": [], "tts": []}
    for j, ((codes, ids), orig) in enumerate(zip(utterances, originals)):
        refs.append(plain_words(tok.decode(ids[1:-1])))
        recon = codec.decode(codes.unsqueeze(0).to(device))[0].float().cpu().numpy()
        spoken = speak(model, ids, codes.shape[-1], temperature=temperature, device=device,
                       generator=gen)
        tts = codec.decode(spoken.to(device))[0].float().cpu().numpy()
        for name, w in (("original", orig), ("codec", recon), ("tts", tts)):
            waves[name].append(w)
            if keep:
                keep(j, name, w)
    rows = {}
    for name, ws in waves.items():
        hyps = [plain_words(h) for h in transcribe_fn(ws)]
        rows[name] = {**asr_report(list(zip(refs, hyps))),
                      "items": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)]}
    return {"n": len(refs), "temperature": temperature, "seed": seed,
            "duration": "the real recording's (given, not predicted)", "rows": rows}
