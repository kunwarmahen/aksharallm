"""The three ways this harness gets a number out of a model.

1. :func:`loglikelihood` — how surprised is the model by *this* continuation after *that*
   context? Nothing is sampled; the answer is a sum of log-probabilities and is therefore
   exactly reproducible. Every multiple-choice suite is built on it.
2. :func:`generate_until` — greedy decoding that stops at a string rather than a token
   count, because a base model does not stop on its own: asked a question it answers it and
   then writes the next question, and the answer has to be cut out of the continuation.
3. :func:`perplexity` — the familiar held-out loss, for continuity with the training curve.

Batching, and why the shapes are the way they are
-------------------------------------------------
Sequences are **right-padded**. Under a causal mask a position can only attend to positions
before it, so padding *after* the real tokens cannot influence any real position — no
attention mask is needed and none is built. (Left-padding would need one, and getting that
subtly wrong gives you a harness that scores every model a little too low and never tells
you.)

Batches are sized by a **token budget** rather than a row count. The logits tensor is
`(batch, tokens, 32768)`, which is where all the memory goes: at 2,048 tokens per batch it
is ~270 MB in float32 regardless of whether that is 32 short MMLU prompts or 2 long
HellaSwag ones. A fixed batch size would either waste the card on short items or run it out
of memory on long ones.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from ..infer.generate import IncrementalDecoder, stream_generate

#: Tokens per forward pass. Deliberately modest: this runs on the CPU whenever a training
#: run owns the card, and the logits tensor is the largest allocation in the process.
DEFAULT_BATCH_TOKENS = 2048

Progress = Callable[[int, int, str], None]


@dataclass
class Scored:
    """One (context, continuation) pair, measured."""

    logprob: float
    #: Tokens in the continuation — the divisor for per-token normalisation.
    n_tokens: int
    #: Characters in the continuation. The standard `acc_norm` divides by this rather than
    #: by token count, so that two models with different tokenizers can be compared.
    n_chars: int
    #: Whether the continuation is what greedy decoding would actually have produced.
    #: Free to compute here and the honest answer to "would it have *said* this?".
    greedy: bool


def _encode_pair(tok, context: str, continuation: str, max_len: int) -> tuple[list[int], int]:
    """Token ids for context+continuation, and how many of them are the continuation.

    The continuation is encoded **on its own** and appended, rather than encoding the
    joined string and counting backwards. Those are not the same thing: BPE merges across
    the boundary, so `encode(ctx + cont)[-len(encode(cont)):]` is off by a token whenever
    the join point falls inside a merge — which is most of the time, and shifts every score
    by one token's log-probability. Every choice in a question is affected equally, so it
    is invisible in the accuracy and wrong in the numbers.

    Over-long sequences are trimmed from the **left**, keeping the continuation whole: the
    thing being scored must never be truncated, or its score is of a different string.
    """
    ctx_ids = tok.encode(context, bos=True)
    cont_ids = tok.encode(continuation)
    if not cont_ids:                      # an empty choice would score 0 and always win
        cont_ids = [tok.eos_id]
    ids = ctx_ids + cont_ids
    if len(ids) > max_len:
        keep = max_len - len(cont_ids)
        if keep < 1:
            # A continuation longer than the whole window. Nothing sensible is left; keep
            # its tail so the comparison is at least between strings of the right length.
            cont_ids = cont_ids[-(max_len - 1):]
            keep = 1
        ids = ctx_ids[-keep:] + cont_ids if keep else cont_ids
        ids = ids[-max_len:]
    return ids, len(cont_ids)


def _batches(lengths: list[int], budget: int) -> Iterable[list[int]]:
    """Indices grouped so that `batch_size * longest_in_batch` stays under `budget`.

    Fed with indices sorted by length, so a batch of short items is large and a batch of
    long ones is small — which is the whole point of a token budget.
    """
    batch: list[int] = []
    longest = 0
    for idx, length in enumerate(lengths):
        nxt = max(longest, length)
        if batch and nxt * (len(batch) + 1) > budget:
            yield batch
            batch, longest = [idx], length
        else:
            batch.append(idx)
            longest = nxt
    if batch:
        yield batch


@torch.no_grad()
def loglikelihood(model, tok, pairs: list[tuple[str, str]], device: str = "cpu",
                  batch_tokens: int = DEFAULT_BATCH_TOKENS,
                  progress: Progress | None = None, label: str = "") -> list[Scored]:
    """Score every (context, continuation) pair. Order of the results matches `pairs`.

    Deterministic: no sampling, no dropout (the model is in eval mode), and the only
    floating-point non-determinism is the batching, which is why the per-token log-probs
    are summed in float32 rather than in the model's bf16.
    """
    model.eval()
    max_len = model.cfg.max_seq_len
    encoded = [_encode_pair(tok, ctx, cont, max_len) for ctx, cont in pairs]
    # Longest first: the first batch is then the one most likely to run out of memory, so a
    # budget that is too high fails in the first seconds rather than forty minutes in.
    order = sorted(range(len(pairs)), key=lambda i: -len(encoded[i][0]))
    lengths = [len(encoded[i][0]) for i in order]

    out: list[Scored | None] = [None] * len(pairs)
    done = 0
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.startswith("cuda") else torch.autocast("cpu", enabled=False))

    for group in _batches(lengths, max(batch_tokens, max(lengths, default=1))):
        rows = [order[g] for g in group]
        seqs = [encoded[r][0] for r in rows]
        conts = [encoded[r][1] for r in rows]
        width = max(len(s) for s in seqs) - 1          # inputs are the sequence minus its last token

        inp = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        tgt = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
        for b, (seq, n_cont) in enumerate(zip(seqs, conts)):
            length = len(seq)
            inp[b, : length - 1] = torch.tensor(seq[:-1], device=device)
            # Position i of the shifted arrays predicts seq[i+1], so the continuation's
            # first predicted position is (length - n_cont) - 1.
            start = length - n_cont - 1
            tgt[b, start: length - 1] = torch.tensor(seq[length - n_cont:], device=device)

        with amp:
            logits, _ = model(inp, full_logits=True)
        # reduction="none" gives the per-token negative log-probability; ignored positions
        # come back as exactly 0.0, so the sum over a row is the sum over its continuation.
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                              reduction="none", ignore_index=-100).view(len(rows), width)
        logprobs = -nll.sum(dim=1)
        # Would greedy decoding have produced this continuation? argmax at every position
        # the continuation occupies, compared with what is actually there.
        greedy_hit = ((logits.argmax(dim=-1) == tgt) | (tgt == -100)).all(dim=1)

        for b, row in enumerate(rows):
            _, cont = pairs[row]
            out[row] = Scored(logprob=float(logprobs[b].item()), n_tokens=conts[b],
                              n_chars=max(1, len(cont)), greedy=bool(greedy_hit[b].item()))
        done += len(rows)
        if progress:
            progress(done, len(pairs), label)
        del logits, nll
    return [s for s in out if s is not None]


def score_mc(model, tok, items, device: str = "cpu",
             batch_tokens: int = DEFAULT_BATCH_TOKENS,
             progress: Progress | None = None, label: str = "") -> dict:
    """Accuracy over a list of :class:`~aksharallm.eval.suites.MCItem`.

    Three numbers, and they disagree on purpose:

    * **acc** — argmax of the raw summed log-probability. Biased toward *short* answers,
      because every extra token can only make a sequence less likely.
    * **acc_norm** — the same, divided by the answer's length in characters. This is the
      headline number for HellaSwag, whose wrong endings are adversarially long, and the
      one most published figures quote.
    * **acc_greedy** — how often the correct answer is also what the model would have
      generated. Stricter, and near zero for a small model even when acc is respectable.

    `stderr` is the binomial standard error, which is the number that says whether a
    two-point move between checkpoints means anything. At n=500 it is about 2%.
    """
    pairs: list[tuple[str, str]] = []
    for item in items:
        for choice in item.choices:
            pairs.append((item.context, choice))

    scored = loglikelihood(model, tok, pairs, device=device, batch_tokens=batch_tokens,
                           progress=progress, label=label)

    correct = correct_norm = correct_greedy = 0
    groups: dict[str, list[int]] = {}
    per_item = []
    cursor = 0
    for item in items:
        n = len(item.choices)
        chunk = scored[cursor: cursor + n]
        cursor += n
        raw = [s.logprob for s in chunk]
        norm = [s.logprob / s.n_chars for s in chunk]
        pred = max(range(n), key=lambda i: raw[i])
        pred_norm = max(range(n), key=lambda i: norm[i])
        hit, hit_norm = int(pred == item.gold), int(pred_norm == item.gold)
        correct += hit
        correct_norm += hit_norm
        correct_greedy += int(chunk[item.gold].greedy)
        if item.group:
            groups.setdefault(item.group, []).append(hit_norm)
        per_item.append({"id": item.id, "gold": item.gold, "pred": pred,
                         "pred_norm": pred_norm, "correct": bool(hit_norm)})

    n_items = max(1, len(items))
    acc_norm = correct_norm / n_items
    result = {
        "n": len(items),
        "acc": correct / n_items,
        "acc_norm": acc_norm,
        "acc_greedy": correct_greedy / n_items,
        "stderr": math.sqrt(max(acc_norm * (1 - acc_norm), 1e-12) / n_items),
        "score": acc_norm,               # the one number the report table shows
        "items": per_item,
    }
    if groups:
        result["groups"] = {
            name: {"n": len(hits), "acc_norm": sum(hits) / len(hits)}
            for name, hits in sorted(groups.items())
        }
    return result


@torch.no_grad()
def generate_until(model, tok, prompt: str, stop: list[str] | None = None,
                   max_new_tokens: int = 256, device: str = "cpu") -> dict:
    """Greedy continuation, cut at the first stop string.

    Greedy (`temperature=0`) is not a style choice: a benchmark that samples gives a
    different score every time it runs, and the whole purpose of this harness is to compare
    a checkpoint against the same checkpoint's earlier self.

    The stop strings are checked against the decoded text rather than against token ids,
    because `"\\nQuestion:"` is not one token and which tokens it becomes depends on what
    precedes it.
    """
    ids = tok.encode(prompt, bos=True)
    decoder = IncrementalDecoder(tok)
    text = ""
    t0 = time.monotonic()
    n = 0
    for token in stream_generate(model, ids, max_new_tokens=max_new_tokens, temperature=0.0,
                                 top_k=None, top_p=None, eos_id=tok.eos_id, device=device):
        n += 1
        if token == tok.eos_id:
            break
        text += decoder.push(token)
        if stop and any(s in text for s in stop):
            break
    text += decoder.flush()
    cut = len(text)
    for s in stop or []:
        found = text.find(s)
        if found != -1:
            cut = min(cut, found)
    return {"text": text[:cut], "raw": text, "n_tokens": n,
            "seconds": time.monotonic() - t0}


@torch.no_grad()
def perplexity(model, bin_path: str, seq_len: int, n_batches: int = 200,
               batch_size: int = 8, device: str = "cpu",
               progress: Progress | None = None) -> dict:
    """Held-out loss and its exponential, over a tokenized `.bin`.

    The same measurement the trainer prints, so the harness and the loss curve agree. It
    is per *token*, which makes it meaningless across tokenizers: a model whose tokenizer
    splits text into more, easier pieces gets a better perplexity for free.
    """
    from ..data.loader import TokenDataset

    ds = TokenDataset(bin_path, seq_len, device)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    total_nll, total_tok, done = 0.0, 0, 0
    for x, y in ds.iter_eval_batches(batch_size, n_batches, seed=1234):
        with amp:
            logits, _ = model(x, full_logits=True)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.reshape(-1),
                              reduction="sum")
        total_nll += float(nll.item())
        total_tok += int(y.numel())
        done += 1
        if progress:
            progress(done, n_batches, "perplexity")
    if not total_tok:
        raise ValueError(f"{bin_path} produced no evaluation batches")
    mean = total_nll / total_tok
    return {"loss": mean, "perplexity": math.exp(mean), "tokens": total_tok,
            "score": math.exp(mean), "n": done}
