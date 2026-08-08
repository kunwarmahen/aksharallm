"""Making a model read further than it was trained to — RoPE scaling, from scratch.

The problem in one paragraph
----------------------------
Our 300M was trained with `max_seq_len: 1024`. Ask it about token 4,000 and it does not
merely get worse, it collapses: perplexity goes from single digits to hundreds within a few
hundred tokens of the edge. The reason is not that the model "ran out of memory" or "forgot"
— it is that RoPE turns a position into an *angle*, and past the trained window it is being
shown angles it has never seen. Every attention score becomes a question in a language the
model was never taught.

Recall what `build_rope_cache` does: channel pair *i* of every head is rotated by
`p · inv_freq[i]` radians, where

    inv_freq[i] = 1 / theta^(2i/D)

so early pairs spin fast (a full turn every few tokens) and late pairs spin slowly (less than
one turn across the whole context). Training with `max_seq_len = 1024` shows the model every
angle the *slow* pairs can produce in 1,024 steps — a fraction of one rotation. At position
4,000 those pairs are four times further round the circle than anything in training.

The three fixes, and they are all one line each
------------------------------------------------
Every method here changes **only `inv_freq`**. No weights, no architecture, nothing to
retrain in order to *try* it. They differ in which channels they slow down.

```
                 fast pairs (local detail)          slow pairs (global position)
none      ──────  unchanged  ─────────────────────────  unchanged, and off the map  ✗
linear    ──────  4x slower  ─────────────────────────  4x slower                   ~
ntk       ──────  ~unchanged ─────────────────────────  4x slower                   ✓
yarn      ──────  unchanged  ──── smooth ramp ────────  4x slower                   ✓✓
```

* **`linear`** — *position interpolation* (Chen et al. 2023). Divide every position by the
  factor: token 4,000 pretends to be token 1,000. Every angle is now one the model has seen,
  which is why it works at all. The cost is that it also squashes the fast pairs, and those
  were the ones encoding "the previous word" — so local resolution is thrown away to buy
  global range, and the model gets measurably worse at short contexts too.

* **`ntk`** — *NTK-aware scaling* (bloc97, 2023). Instead of dividing the positions, raise
  `theta`: `theta · factor^(D/(D-2))`. Because `inv_freq` is a *geometric* series in `theta`,
  changing the base tilts the whole ladder — the slowest pair ends up interpolated by
  almost exactly `factor` while the fastest pair is barely touched. One number, and it keeps
  the local resolution `linear` destroys. That exponent is chosen so the last pair lands
  exactly where linear interpolation would have put it.

* **`yarn`** — (Peng et al. 2023) does explicitly what NTK does as a side effect, per channel
  and with a smooth ramp. For each pair, ask how many full rotations it completes inside the
  *original* window: `r = original_len / wavelength`.
    - `r > beta_fast` (32): many rotations, the model has seen every angle. **Leave it alone.**
    - `r < beta_slow` (1): less than one rotation, the model has only ever seen a sliver of
      this circle. **Interpolate fully.**
    - between: ramp linearly between the two.
  Plus one extra piece nothing else has: as the context grows, attention entropy grows with
  it, so YaRN scales the attention logits by `(0.1·ln(factor) + 1)²`. We apply that by
  multiplying `cos`/`sin` — both Q and K get the factor, so their dot product gets its
  square, which is exactly the temperature the paper asks for and costs no extra code in
  the attention path.

* **`dynamic`** — NTK, but with the factor computed from the length actually being processed
  rather than fixed. At or below the original window the factor is 1 and the model is
  *bit-for-bit unscaled*, so short prompts pay nothing at all. This is the one to reach for
  when a checkpoint has to serve both.

What none of them do
--------------------
**They do not add information.** A model that never saw a fact 4,000 tokens back still has
to learn to *use* one; scaling only makes the positions legible. Published results extend by
4-8x without fine-tuning and further with a short one, and `docs/19-long-context.md` has our
own measured curves — including the part where the naive path falls off a cliff and where
each method stops helping.

Read with: docs/19-long-context.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/04-model.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: The methods `type:` accepts. "none" is the identity and stays the default everywhere.
METHODS = ("none", "linear", "ntk", "yarn", "dynamic")


@dataclass
class RopeScaling:
    """How to stretch RoPE past the window the weights were trained on.

    Lives under `model.rope_scaling:` in a config, and is carried inside the checkpoint, so
    an extended model reloads extended without anyone having to remember a flag.
    """

    #: One of :data:`METHODS`.
    type: str = "none"
    #: How much further than `original_max_seq_len` to reach. 4.0 = 1k -> 4k.
    factor: float = 1.0
    #: The window the weights were actually trained on. Left None it is derived as
    #: `max_seq_len / factor`, which is right whenever the config was written by
    #: `longctx extend` and worth stating explicitly when it was not.
    original_max_seq_len: int | None = None
    #: YaRN only. Channels completing more than this many rotations inside the original
    #: window are left untouched (the model has seen their whole circle).
    beta_fast: float = 32.0
    #: YaRN only. Channels completing fewer than this many are interpolated in full.
    beta_slow: float = 1.0
    #: YaRN only. A hand multiplier on the attention temperature; 1.0 is the paper's value.
    attn_factor: float = 1.0

    def __post_init__(self):
        if self.type not in METHODS:
            raise ValueError(f"rope_scaling.type must be one of {METHODS}, got {self.type!r}")
        if self.type != "none" and self.factor < 1.0:
            raise ValueError(f"rope_scaling.factor must be >= 1, got {self.factor}")
        if self.beta_fast <= self.beta_slow:
            raise ValueError("beta_fast must exceed beta_slow")

    @property
    def enabled(self) -> bool:
        return self.type != "none" and self.factor > 1.0

    def original_len(self, max_seq_len: int) -> int:
        """The window the weights were trained on."""
        if self.original_max_seq_len:
            return int(self.original_max_seq_len)
        return max(1, int(round(max_seq_len / max(self.factor, 1.0))))

    def describe(self, max_seq_len: int) -> str:
        """One line for a log, a report, or the portal."""
        if not self.enabled:
            return f"none (trained window {max_seq_len})"
        orig = self.original_len(max_seq_len)
        return (f"{self.type} x{self.factor:g} — trained on {orig}, "
                f"addressed up to {max_seq_len}")


def base_inv_freq(head_dim: int, theta: float, device=None) -> torch.Tensor:
    """The unscaled frequency ladder: `1 / theta^(2i/D)`, one entry per channel pair."""
    i = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (i / head_dim))


def _yarn_ramp(scaling: RopeScaling, head_dim: int, theta: float,
               original_len: int, device=None) -> torch.Tensor:
    """Per-channel blend weight: 1 = keep the original frequency, 0 = interpolate fully.

    The quantity being thresholded is **rotations completed inside the original window**,
    `original_len / wavelength`, which is the honest way to ask "has the model seen this
    channel's whole circle?" -- and is what makes YaRN per-channel rather than global.
    """
    inv = base_inv_freq(head_dim, theta, device)
    wavelength = 2 * math.pi / inv
    rotations = original_len / wavelength
    span = scaling.beta_fast - scaling.beta_slow
    return ((rotations - scaling.beta_slow) / span).clamp(0.0, 1.0)


def plan(head_dim: int, theta: float, scaling: RopeScaling | None,
         max_seq_len: int, seq_len: int | None = None,
         device=None) -> tuple[torch.Tensor, float]:
    """The whole module: `(inv_freq, mscale)` for these settings.

    `seq_len` is the length actually being processed and is only read by `dynamic`.
    `mscale` multiplies cos and sin — 1.0 for everything but YaRN.
    """
    if scaling is None or not scaling.enabled:
        return base_inv_freq(head_dim, theta, device), 1.0

    original = scaling.original_len(max_seq_len)
    factor = scaling.factor

    if scaling.type == "dynamic":
        # The factor the *current* input needs, never less than 1. A prompt inside the
        # original window is then handled by unscaled RoPE, bit for bit -- which is the
        # entire reason to choose this method over plain `ntk`.
        needed = (seq_len or max_seq_len) / original
        factor = max(1.0, needed)
        if factor <= 1.0:
            return base_inv_freq(head_dim, theta, device), 1.0

    if scaling.type == "linear":
        return base_inv_freq(head_dim, theta, device) / factor, 1.0

    if scaling.type in ("ntk", "dynamic"):
        # Raising the base tilts a geometric ladder: the slowest pair is interpolated by
        # ~factor, the fastest barely at all. The exponent D/(D-2) is what makes the last
        # pair land exactly where `linear` would have put it.
        adjusted = theta * factor ** (head_dim / (head_dim - 2))
        return base_inv_freq(head_dim, adjusted, device), 1.0

    # ---- yarn ---------------------------------------------------------------------
    inv = base_inv_freq(head_dim, theta, device)
    keep = _yarn_ramp(scaling, head_dim, theta, original, device)
    inv_freq = inv / factor * (1 - keep) + inv * keep
    # Attention temperature. Both q and k are multiplied by this, so the logits are scaled
    # by its square -- which is the `1/t` the paper puts inside the softmax.
    mscale = (0.1 * math.log(factor) + 1.0) * scaling.attn_factor
    return inv_freq, mscale


def build_cache(head_dim: int, max_seq_len: int, theta: float,
                scaling: RopeScaling | None = None, seq_len: int | None = None,
                device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)`, each `(max_seq_len, head_dim)` — the cache the model registers.

    This is `build_rope_cache` with a scaling step in front of it, and it is deliberately
    the *only* place the two meet: everything above works on `inv_freq`, everything below
    works on angles, and no caller has to know which method produced them.
    """
    inv_freq, mscale = plan(head_dim, theta, scaling, max_seq_len, seq_len, device)
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)                 # (T, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, D) -- matches rotate_half
    return emb.cos() * mscale, emb.sin() * mscale


def effective_window(max_seq_len: int, scaling: RopeScaling | None) -> int:
    """The window the *weights* know, as opposed to the one the cache can address."""
    if scaling is None or not scaling.enabled:
        return max_seq_len
    return scaling.original_len(max_seq_len)
