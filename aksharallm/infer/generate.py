"""Autoregressive generation with a KV cache.

The naive way to generate is to re-run the whole prompt through the model for every new
token: O(T^2) work per token. The KV cache instead keeps every layer's keys and values
from previous positions, so each new token costs one forward pass over a *single*
position. That's the difference between ~2 tok/s and ~200 tok/s on this hardware.

    prefill:  feed the whole prompt once      -> cache holds T entries
    decode:   feed 1 token, read T from cache -> cache holds T+1 entries
              repeat
"""

from __future__ import annotations

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


@torch.no_grad()
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
    """Generate a continuation. Returns prompt + generated ids."""
    model.eval()
    if isinstance(prompt_ids, list):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        idx = prompt_ids.to(device)
        if idx.dim() == 1:
            idx = idx[None]

    max_len = model.cfg.max_seq_len
    if idx.size(1) >= max_len:
        idx = idx[:, -(max_len - 1):]  # leave room for at least one new token

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

    with ctx:
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
            if stream_cb is not None:
                stream_cb(tok)
            if eos_id is not None and tok == eos_id:
                break
            if len(out) >= max_len:
                break  # context full; a sliding window would go here

            # --- decode: one token, reusing the cache ---
            logits, _ = model(next_id, caches=caches)

    return out
