"""Generation by iterative unmasking — the forward process, run backwards.

An autoregressive model generates by *extending*: one token at a time, each one appended to
a prefix that never changes, which is what makes a KV cache possible. A diffusion model
generates by *revising*: it starts from a sequence that is entirely `[MASK]`, predicts every
masked position at once, commits the few it is most confident about, and repeats.

```
    [M][M][M][M][M][M][M][M]      step 0   all masked
    [M][M] cat [M][M][M][M] .     step 1   two positions committed
    The [M] cat [M][M] on  [M].   step 2
    The big cat sat on the mat.   step 8   nothing left to unmask
```

**There is no KV cache here, and there cannot be one.** A cache is the memo an
autoregressive model keeps because position *n*'s keys are settled the moment it is
generated. In a diffusion model every position may still change on the next step, so every
step re-encodes the whole sequence. That makes the cost `steps × T²` rather than the AR
model's `T` cached passes — and it is why `infer/generate.py`, `IncrementalDecoder` and
`KVCache` do not transfer to this file. In exchange, `steps` is a knob: 8 steps for a
64-token sequence is one eighth of the forward passes an AR model would need, at the cost
of committing several tokens per pass without seeing each other's choices.

**Which positions to commit** is the only real design decision, and the answer is
confidence. At each step the model produces a distribution over every masked position; the
probability it assigns to the token it picked is how sure it is. Committing the most
confident ones first means the easy, constrained positions (punctuation, the second half of
a word, a name already mentioned) get fixed early and then constrain everything else — the
sequence resolves from its skeleton outwards. `remask="random"` is offered for comparison
and is noticeably worse, which is the point of offering it.

**The schedule** is linear in the number left masked: after step *i* of *N*, `n * (N - i)/N`
positions are still masked. Nothing is ever un-committed, so this is equivalent to the
"remask the least confident" formulation in the papers, with the bookkeeping done once.

Read with: docs/20-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..infer.generate import _filter_logits


class DiffusionError(RuntimeError):
    """Asked to generate from something that is not a masked diffusion model."""


@dataclass
class DenoiseStep:
    """One denoising step, kept so the whole run can be replayed.

    This is the artifact the Diffusion tab animates. It is cheap — a few hundred ints per
    step — and it is the only honest way to show what the model did, because the finished
    text says nothing about the order the positions were decided in.
    """

    step: int
    #: The full sequence after this step, as token ids (mask id included where still masked).
    ids: list[int]
    #: Positions committed on this step, and how confident the model was about each.
    committed: list[int] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    #: How many positions are still masked once this step is done.
    remaining: int = 0

    def as_dict(self) -> dict:
        return {"step": self.step, "ids": self.ids, "committed": self.committed,
                "confidence": [round(c, 4) for c in self.confidence],
                "remaining": self.remaining}


def _require_diffusion(model) -> int:
    cfg = model.cfg
    if not cfg.is_diffusion:
        raise DiffusionError(
            "this checkpoint is an autoregressive model (causal attention, no mask token), "
            "so it has never seen a [MASK] and iterative unmasking would return noise. "
            "Generate from it with aksharallm.infer.generate instead.")
    return int(cfg.mask_token_id)


@torch.no_grad()
def diffusion_generate(
    model,
    *,
    length: int = 64,
    steps: int = 32,
    prefix: list[int] | None = None,
    suffix: list[int] | None = None,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    remask: str = "low_confidence",
    seed: int | None = None,
    device: str = "cpu",
    trace: bool = False,
) -> tuple[list[int], list[DenoiseStep]]:
    """Denoise a fresh sequence into text. Returns `(ids, trace)`.

    `prefix` and `suffix` are token ids that are given rather than generated — they are
    written into the sequence before the first step and are never masked, never predicted
    and never changed. A prefix alone is an ordinary prompt; a prefix *and* a suffix is
    infilling, which is the capability an autoregressive model does not have.

    `length` is how many positions between them the model fills. Unlike AR generation there
    is no way to stop early: the number of tokens is chosen before the first forward pass,
    because the sequence has to exist in order to be denoised. That is a real limitation and
    the reason published diffusion LMs decode in blocks.
    """
    mask_id = _require_diffusion(model)
    prefix = list(prefix or [])
    suffix = list(suffix or [])
    length = max(1, int(length))

    ctx = model.cfg.max_seq_len
    total = len(prefix) + length + len(suffix)
    if total > ctx:
        # Give up generated positions rather than truncating what the caller supplied: a
        # prompt with its head cut off is a different request.
        length = ctx - len(prefix) - len(suffix)
        if length < 1:
            raise DiffusionError(
                f"prefix + suffix are {len(prefix) + len(suffix)} tokens and the model's "
                f"context is {ctx}; there is no room left to generate into.")
        total = ctx

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros((1, total), dtype=torch.bool, device=device)
    if prefix:
        x[0, : len(prefix)] = torch.tensor(prefix, device=device)
        fixed[0, : len(prefix)] = True
    if suffix:
        x[0, total - len(suffix):] = torch.tensor(suffix, device=device)
        fixed[0, total - len(suffix):] = True

    n_gen = int((~fixed).sum().item())
    steps = max(1, min(int(steps), n_gen))          # more steps than positions is wasted work

    history: list[DenoiseStep] = []
    if trace:
        history.append(DenoiseStep(step=0, ids=x[0].tolist(), remaining=n_gen))

    for i in range(1, steps + 1):
        still = (x[0] == mask_id) & ~fixed[0]
        n_masked = int(still.sum().item())
        if n_masked == 0:
            break

        logits, _ = model(x, full_logits=True)       # (1, T, V) — every position, every step
        logits = logits[0].float()
        # The model must never write a [MASK] into its own output. It has seen the id in its
        # input on every training step, so it does assign it probability, and a committed
        # mask token is a position that can never be filled.
        logits[:, mask_id] = float("-inf")

        if temperature <= 0:
            probs = F.softmax(logits, dim=-1)
            conf, choice = probs.max(dim=-1)
        else:
            filtered = _filter_logits(logits / temperature, top_k, top_p)
            probs = F.softmax(filtered, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            # Confidence is the probability of the token actually chosen, not the maximum.
            # With sampling those differ, and using the maximum would rank a position the
            # model is sure about but that we sampled *against* as if it were settled.
            conf = probs.gather(-1, choice[:, None]).squeeze(-1)

        if remask == "random":
            conf = torch.rand(conf.shape, device=conf.device, generator=gen)
        elif remask != "low_confidence":
            raise DiffusionError(f"unknown remasking strategy {remask!r} "
                                 "(low_confidence or random)")

        # Linear schedule on the number still masked. `n_gen` rather than `n_masked` is the
        # base so the schedule does not stretch when a step commits more than its share.
        target_left = int(round(n_gen * (steps - i) / steps))
        commit_n = max(1, n_masked - target_left)
        commit_n = min(commit_n, n_masked)

        scores = torch.where(still, conf, torch.full_like(conf, float("-inf")))
        take = torch.topk(scores, commit_n).indices
        x[0, take] = choice[take]

        remaining = int(((x[0] == mask_id) & ~fixed[0]).sum().item())
        if trace:
            order = take.tolist()
            history.append(DenoiseStep(
                step=i, ids=x[0].tolist(), committed=order,
                confidence=[float(conf[p]) for p in order], remaining=remaining))

    return x[0].tolist(), history


def decode_with_masks(tok, ids, mask_id: int, placeholder: str = "▁") -> str:
    """Decode a partly-denoised sequence to text, drawing what is still masked.

    The tokenizer has never heard of the mask id — it is a row the *model* was given, one
    past the end of the vocabulary — so handing it to `decode` is at best a wrong word and
    at worst an exception. Runs of real tokens are decoded together (byte-level BPE only
    reassembles correctly in a batch) and anything past the vocabulary becomes a placeholder.
    """
    out: list[str] = []
    run: list[int] = []
    for i in ids:
        if i < tok.vocab_size:
            run.append(int(i))
            continue
        if run:
            out.append(tok.decode(run))
            run = []
        out.append(placeholder if i == mask_id else "")
    if run:
        out.append(tok.decode(run))
    return "".join(out)


def infill(model, prefix: list[int], suffix: list[int], length: int = 16, **kw):
    """Write `length` tokens between `prefix` and `suffix`. Returns `(middle_ids, trace)`.

    This is the headline capability: the *only* change from unconditional generation is that
    some of the fixed positions happen to sit at the end. An autoregressive model cannot do
    this without being retrained on a fill-in-the-middle objective, because it has no way to
    condition on text that comes after what it is writing.
    """
    ids, history = diffusion_generate(model, length=length, prefix=prefix, suffix=suffix,
                                      **kw)
    end = len(ids) - len(suffix) if suffix else len(ids)
    return ids[len(prefix):end], history
