"""Autoregressive generation with a KV cache.

The naive way to generate is to re-run the whole prompt through the model for every new
token: O(T^2) work per token. The KV cache instead keeps every layer's keys and values
from previous positions, so each new token costs one forward pass over a *single*
position. That's the difference between ~2 tok/s and ~200 tok/s on this hardware.

    prefill:  feed the whole prompt once      -> cache holds T entries
    decode:   feed 1 token, read T from cache -> cache holds T+1 entries
              repeat

Two entry points, one loop. :func:`stream_generate` *is* the loop and yields each token as
it is sampled; :func:`generate` collects it into a list. Everything that watches a model
think — the CLI, the portal's Playground tab — consumes the generator, so there is one
sampling implementation to get right rather than one per caller.

Read with: docs/07-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F


def _filter_logits(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Restrict sampling to a sensible subset of the vocabulary.

    Without this, the long tail of ~50k near-zero-probability tokens collectively holds
    enough mass that you eventually sample a garbage token -- and since generation is
    autoregressive, one garbage token derails everything after it.

      top_k: keep the k most likely tokens.
      top_p (nucleus): keep the smallest set whose probabilities sum to p. Adapts to how
             confident the model is, which top_k can't do.
    """
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Drop everything past the nucleus, but always keep the top-1 token.
        remove = cumulative - probs > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    return logits


def fit_prompt(prompt_ids: list[int] | torch.Tensor, max_len: int,
               device: str = "cuda") -> torch.Tensor:
    """The prompt as a `(1, T)` tensor, trimmed to leave room for at least one new token.

    Shared by both entry points on purpose. When this trimming lived only inside the decode
    loop, :func:`generate` built its return value from the *original* prompt and could hand
    back a sequence longer than the context window it had just enforced — a discrepancy of
    exactly the tokens that were dropped.

    Trimming keeps the **end** of the prompt: the newest context is the part that matters,
    and a conversation that has outgrown the window should lose its oldest turns rather
    than the question being asked right now.
    """
    if isinstance(prompt_ids, torch.Tensor):
        idx = prompt_ids.to(device)
        if idx.dim() == 1:
            idx = idx[None]
    else:
        idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    if idx.size(1) >= max_len:
        idx = idx[:, -(max_len - 1):]
    return idx


def stream_generate(
    model,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    device: str = "cuda",
) -> Iterator[int]:
    """Yield generated token ids, one at a time, as they are sampled.

    The EOS token *is* yielded before the loop stops, so a caller that wants to know why
    generation ended can tell "it finished a thought" from "it hit the token budget".

    Abandoning the generator (a `break`, or the browser closing its connection) stops the
    model mid-sentence and frees the KV cache — which on a card that is simultaneously
    training a 300M model is the difference between a spare gigabyte and an out-of-memory
    error four days into a run.
    """
    model.eval()
    idx = fit_prompt(prompt_ids, model.cfg.max_seq_len, device=device)
    max_len = model.cfg.max_seq_len

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    caches = model.init_caches(idx.size(0), max_len, dtype=dtype, device=device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.startswith("cuda") else torch.autocast("cpu", enabled=False))

    out = idx[0].tolist()
    # The returned sequence can never exceed the context window, so cap the number of
    # new tokens by the room actually left after the prompt. Without this the loop emits
    # one token past capacity: it samples from the final valid position's logits but has
    # nowhere to cache the result.
    budget = min(max_new_tokens, max_len - idx.size(1))

    # `torch.no_grad()` is entered here rather than as a decorator: on a generator, a
    # decorator's context is a subtlety nobody should have to remember, and an autograd
    # graph accidentally kept alive across 256 decode steps is a memory leak that looks
    # like a slow OOM.
    with torch.no_grad(), ctx:
        # --- prefill: the whole prompt in one pass ---
        logits, _ = model(idx, caches=caches)

        for _ in range(budget):
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                # Divide the logit of already-used tokens. Negative logits must be
                # *multiplied* instead, or the penalty would increase their score.
                for t in set(out):
                    if logits[0, t] > 0:
                        logits[0, t] /= repetition_penalty
                    else:
                        logits[0, t] *= repetition_penalty

            if temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = _filter_logits(logits / temperature, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            tok = int(next_id.item())
            out.append(tok)
            yield tok
            if eos_id is not None and tok == eos_id:
                break
            if len(out) >= max_len:
                break  # context full; a sliding window would go here

            # --- decode: one token, reusing the cache ---
            logits, _ = model(next_id, caches=caches)


def generate(
    model,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    device: str = "cuda",
    stream_cb=None,
) -> list[int]:
    """Generate a continuation. Returns prompt + generated ids.

    The batteries-included form of :func:`stream_generate`, for callers that want the whole
    answer (evaluation, a one-shot CLI prompt) rather than to watch it arrive.
    """
    # The *trimmed* prompt, so the returned sequence can never exceed the context window —
    # `fit_prompt` is the same call the decode loop makes.
    out = fit_prompt(prompt_ids, model.cfg.max_seq_len, device=device)[0].tolist()
    for tok in stream_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            eos_id=eos_id, device=device):
        out.append(tok)
        if stream_cb is not None:
            stream_cb(tok)
    return out


class IncrementalDecoder:
    """Turns a stream of token ids into a stream of printable text.

    Two problems make this less trivial than `decode(one_token)`:

    1. **A BPE token is often only part of a character.** Every emoji and accented letter
       spans several tokens at the byte level, so tokens must be decoded *cumulatively* and
       only the new text emitted.
    2. **A buffer that ends mid-character decodes to a trailing U+FFFD.** Emitting it would
       print a replacement character and move on; the next decode would have the real
       character in that position, but the garbage is already on the screen. So a trailing
       U+FFFD is held back until the tokens that complete it arrive.

    :meth:`flush` releases anything still held at the end — for a genuinely invalid byte
    sequence, which a model that is only 4% trained will absolutely produce.
    """

    def __init__(self, tokenizer, skip_ids: set[int] | None = None):
        self.tok = tokenizer
        self.skip = skip_ids or set()
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        """Feed one token; return the text that is now safe to show (often "")."""
        if token_id in self.skip:
            return ""
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""            # incomplete character — wait for the rest of its bytes
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def flush(self) -> str:
        text = self.text
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    @property
    def text(self) -> str:
        return self.tok.decode(self.ids)
