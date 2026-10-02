"""CTC — how to train a model to transcribe audio without ever telling it *when* each letter was said.

A transcript says "hello". The audio says it over 0.6 seconds, which after subsampling is
fifteen encoder frames. Nobody labelled which frame is the `h`. **Connectionist Temporal
Classification** solves this by not choosing: the model emits one distribution per frame
over the alphabet *plus a blank*, and the loss sums the probability of **every** frame-level
path that collapses to the transcript.

```mermaid
flowchart LR
    P["frame labels<br/>h h _ e _ l l _ l o"] --> R["merge repeats<br/>h _ e _ l _ l o"]
    R --> B["drop blanks<br/>h e l l o"]
```

The blank is what lets a double letter survive: `l l` merges into one `l`, so "hello" needs
a blank between its two `l`s — `l _ l`. That is also why a transcript of `L` letters needs at
least `L + (number of repeated pairs)` frames, and why an utterance that violates it has
probability zero (trap 2 in PLAN.md § Phase 7: such a batch item is *skipped and counted*).

**Why this loss and not an attention decoder — the day-two argument, and where it stops.** A
sequence-to-sequence decoder writes text one token at a time and decides by itself when to
stop. Given silence it can and does write "thank you for watching", because that sentence
ends a great many videos in its training data — whole fluent sentences with no audio under
them. CTC cannot do *that*: it emits exactly one symbol per frame, tied to that frame, so it
can never write more than the audio has frames for, and never a sentence it is merely
reminded of.

**What CTC does not prevent, measured.** It will still label a frame of noise as a letter if
nothing in training said otherwise. The first smoke run wrote 44 characters across five clips
with no speech in them (digital silence included). Training on clips with *empty*
transcripts took that to **0** (`asr/noise.py`); changing the normalisation alone only halved
it. So CTC turns "invents sentences" into "mislabels frames", and the second is a data
problem with a data fix. docs/23 has the four-way table.

**The forward algorithm.** Interleave blanks into the label sequence, `_ h _ e _ l _ l _ o _`
(length `S = 2L + 1`), and let `alpha[t, s]` be the log-probability of all paths that have
consumed `t + 1` frames and are sitting on position `s`. A path at `s` came from `s` (stayed),
`s - 1` (advanced), or `s - 2` (skipped a blank) — the skip only being legal when it does not
jump between two identical letters, which would merge them. The loss is
`-log(alpha[T-1, S-1] + alpha[T-1, S-2])`: end on the last letter or on the trailing blank.

**The backward pass is not autograd.** Differentiating through a Python loop over 400 frames
would store every intermediate of every step. Instead the standard identity is used: run the
same recursion backwards (`beta`), and the posterior of being at `s` at time `t` is
`alpha + beta - emission - log p`. The gradient with respect to the log-probabilities is minus
that posterior summed over every position carrying the same letter. Memory is one `alpha`
table. `tests/test_asr.py` pins the loss **and** the gradient against `F.ctc_loss`, at
LibriSpeech lengths rather than toy ones (trap 3: log-space bugs only show up long).

**It is the reference, not the default.** On the 3090 it takes 195 ms where `F.ctc_loss` takes
1.3 ms (400 frames x 25 rows x 240 letters): the recursion is a Python loop over frames, so the
card spends its time launching a few thousand tiny kernels. A fused kernel is the fix, and is
the FlashAttention story again (`model/flash.py`). Until then `train.ctc_impl` defaults to
`torch`, and the bit-for-bit test is what makes swapping them safe.

Two choices that matter more than they look:

* **`NEG = -1e30`, not `-inf`.** A position nothing can reach has log-probability "minus
  infinity", and `logsumexp` of all `-inf` has a `nan` gradient. A large finite number
  behaves identically in every sum that matters and never produces a `nan` — the same choice
  `model/flash.py` makes for its running maximum, for the same reason.
* **Everything runs in float32**, whatever the model's dtype. The recursion adds hundreds of
  log-probabilities; in bf16 (eight bits of mantissa) the running sum stops resolving the
  per-frame differences long before the end of a 15-second utterance — gotcha 14's family.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import torch

#: "Unreachable", as a finite number. See the module docstring.
NEG = -1e30

#: A per-sample loss at or above this is an impossible alignment (too few frames for the
#: transcript). Real CTC losses are a few hundred nats at most.
INFEASIBLE = 1e29


def extend_targets(targets: torch.Tensor, target_lengths: torch.Tensor, blank: int):
    """`(B, L)` labels -> `(B, 2L+1)` with blanks interleaved, plus the skip-legal mask.

    `can_skip[b, s]` is True where a path may jump from `s - 2` straight to `s`: `s` holds a
    letter (odd index) and it differs from the letter at `s - 2`. Padding past a sample's own
    length is filled with blank and is never reachable, because `alpha` starts at `s <= 1`
    and advances at most two positions per frame — but the final read-out uses each sample's
    own `S_b`, so a padded position is never mistaken for an end state either.
    """
    B, L = targets.shape
    S = 2 * L + 1
    ext = torch.full((B, S), blank, dtype=torch.long, device=targets.device)
    # Letters at the odd positions, and blank (never a label) beyond each sample's length,
    # so padding cannot be read as a label. letters beyond each sample's length, so padding cannot be read as a label.
    pos = torch.arange(L, device=targets.device)
    valid = pos[None, :] < target_lengths[:, None]
    ext[:, 1::2] = torch.where(valid, targets, torch.full_like(targets, blank))
    can_skip = torch.zeros((B, S), dtype=torch.bool, device=targets.device)
    if S > 2:
        can_skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    return ext, can_skip


def _lse3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """`log(e^a + e^b + e^c)`, stable. Three terms is the whole recursion, so it is spelt out
    rather than stacked into a new tensor every frame."""
    m = torch.maximum(torch.maximum(a, b), c)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m))


def _shift(x: torch.Tensor, k: int, fill: float = NEG) -> torch.Tensor:
    """`x[..., s - k]` at position `s` (k > 0), or `x[..., s + |k|]` (k < 0); `fill` off the end."""
    out = torch.full_like(x, fill)
    if k > 0:
        out[..., k:] = x[..., :-k]
    else:
        out[..., :k] = x[..., -k:]
    return out


def ctc_forward(log_probs: torch.Tensor, ext: torch.Tensor, can_skip: torch.Tensor,
                input_lengths: torch.Tensor, ext_lengths: torch.Tensor):
    """`alpha` `(T, B, S)` and the per-sample log-likelihood `(B,)`.

    `log_probs` is `(T, B, C)`, float32. A sample's `alpha` stops changing after its own last
    frame (`t >= T_b`), so the read-out at `T_b - 1` is simply the final table's value.
    """
    T, B, _ = log_probs.shape
    S = ext.shape[1]
    emit = log_probs.gather(2, ext[None].expand(T, B, S))  # (T, B, S)
    alpha = torch.full((T, B, S), NEG, dtype=log_probs.dtype, device=log_probs.device)
    a = torch.full((B, S), NEG, dtype=log_probs.dtype, device=log_probs.device)
    a[:, 0] = emit[0, :, 0]
    if S > 1:
        a[:, 1] = emit[0, :, 1]
    alpha[0] = a
    for t in range(1, T):
        stay = a
        step = _shift(a, 1)
        skip = torch.where(can_skip, _shift(a, 2), torch.full_like(a, NEG))
        new = _lse3(stay, step, skip) + emit[t]
        live = (t < input_lengths)[:, None]
        a = torch.where(live, new, a)
        alpha[t] = a
    idx = torch.arange(B, device=ext.device)
    last = a[idx, ext_lengths - 1]
    prev = torch.where(ext_lengths >= 2, a[idx, (ext_lengths - 2).clamp_min(0)],
                       torch.full_like(last, NEG))
    m = torch.maximum(last, prev)
    ll = m + torch.log(torch.exp(last - m) + torch.exp(prev - m))
    return alpha, emit, ll


def ctc_backward_beta(emit: torch.Tensor, can_skip: torch.Tensor, input_lengths: torch.Tensor,
                      ext_lengths: torch.Tensor) -> torch.Tensor:
    """`beta` `(T, B, S)`: log-probability of finishing from `(t, s)`, *including* frame t.

    Each sample starts at its own last frame `T_b - 1`, on its own last two positions; before
    that frame (in reverse time) its rows stay unreachable.
    """
    T, B, S = emit.shape
    dev = emit.device
    beta = torch.full((T, B, S), NEG, dtype=emit.dtype, device=dev)
    s_idx = torch.arange(S, device=dev)[None, :]
    end = (s_idx == (ext_lengths - 1)[:, None]) | (s_idx == (ext_lengths - 2)[:, None])
    # A skip INTO s+2 from s is legal exactly when can_skip[s+2] is.
    skip_from = _shift(can_skip, -2, fill=False)
    b = torch.full((B, S), NEG, dtype=emit.dtype, device=dev)
    for t in range(T - 1, -1, -1):
        stay = b
        step = _shift(b, -1)
        skip = torch.where(skip_from, _shift(b, -2), torch.full_like(b, NEG))
        rec = _lse3(stay, step, skip) + emit[t]
        init = torch.where(end, emit[t], torch.full_like(emit[t], NEG))
        is_last = (t == input_lengths - 1)[:, None]
        before = (t < input_lengths - 1)[:, None]
        b = torch.where(is_last, init, torch.where(before, rec, torch.full_like(b, NEG)))
        beta[t] = b
    return beta


class _CTC(torch.autograd.Function):
    """Per-sample negative log-likelihood, with the alpha-beta gradient."""

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank):
        # float64 stays float64 (so a gradcheck is exact); anything narrower is promoted.
        lp = log_probs if log_probs.dtype == torch.float64 else log_probs.float()
        ext, can_skip = extend_targets(targets, target_lengths, blank)
        ext_lengths = 2 * target_lengths + 1
        alpha, emit, ll = ctc_forward(lp, ext, can_skip, input_lengths, ext_lengths)
        ctx.save_for_backward(lp, alpha, emit, ext, can_skip, input_lengths, ext_lengths, ll)
        ctx.in_dtype = log_probs.dtype
        loss = -ll
        # An impossible alignment reads as +inf, the same value F.ctc_loss reports, so that a
        # caller comparing the two sees the same thing — and so it cannot be averaged in as a
        # merely large number.
        return torch.where(loss >= INFEASIBLE, torch.full_like(loss, float("inf")), loss)

    @staticmethod
    def backward(ctx, grad_out):
        lp, alpha, emit, ext, can_skip, input_lengths, ext_lengths, ll = ctx.saved_tensors
        T, B, C = lp.shape
        beta = ctc_backward_beta(emit, can_skip, input_lengths, ext_lengths)
        # Posterior of each (t, s): alpha and beta both include frame t's emission, so it is
        # counted twice and subtracted once.
        log_post = alpha + beta - emit - ll[None, :, None]
        post = torch.exp(log_post.clamp(max=0.0))
        t_idx = torch.arange(T, device=lp.device)[:, None, None]
        post = post * (t_idx < input_lengths[None, :, None])
        grad = torch.zeros_like(lp)  # (T, B, C)
        grad.scatter_add_(2, ext[None].expand(T, B, ext.shape[1]), post)
        feasible = (-ll < INFEASIBLE).to(lp.dtype)
        g = -grad * (grad_out.float() * feasible)[None, :, None]
        g = torch.nan_to_num(g)
        return g.to(ctx.in_dtype), None, None, None, None


def ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor,
             target_lengths: torch.Tensor, *, blank: int = 0, reduction: str = "mean"):
    """Drop-in for `F.ctc_loss` with the same argument order and the same `reduction` meanings.

    * `log_probs` `(T, B, C)` — log-softmax over the vocabulary, time first (as `F.ctc_loss`).
    * `targets` `(B, L)` padded; `target_lengths` says how much of each row is real.
    * `reduction="mean"` divides each sample by its target length and then averages over the
      batch — PyTorch's convention, kept so the two are interchangeable in a test. `"sum"` and
      `"none"` mean what they say.

    Impossible samples come back as `inf`. The trainer drops them before reducing (and counts
    them); see `asr/train.py`.
    """
    per = _CTC.apply(log_probs, targets, input_lengths.long(), target_lengths.long(), blank)
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    if reduction == "mean":
        return (per / target_lengths.clamp_min(1).to(per.dtype)).mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Best label per frame, merge repeats, drop blanks. `log_probs` `(B, T, C)`.

    This is the decoder that ships first, and on a CTC model it is surprisingly close to
    beam search — CTC's per-frame distributions are very peaked once trained. Beam search
    earns its keep when there is something to *bias* it with (a personal dictionary, a
    language model), which is day-two problem 4 and lives in `asr/decode.py`.
    """
    best = log_probs.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, : int(lengths[b])].tolist()
        ids, prev = [], None
        for x in seq:
            if x != prev and x != blank:
                ids.append(x)
            prev = x
        out.append(ids)
    return out
