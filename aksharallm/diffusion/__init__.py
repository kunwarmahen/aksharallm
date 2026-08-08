"""Masked diffusion language modelling — the second paradigm in this repo.

Everything else here is **autoregressive**: predict token n+1 from tokens 1..n, left to
right, forever. A masked diffusion model drops that constraint. It is trained to *denoise*:
take a clean sequence, replace a random fraction of its tokens with `[MASK]`, and ask the
model to put them back — with every position able to see every other one. Generation runs
the corruption backwards: start from a row of masks and unmask a few positions at a time,
in whatever order the model is most confident about, until nothing is masked.

Two capabilities fall out of that which an autoregressive model structurally cannot offer:

* **infilling** — give it a prefix *and* a suffix and it writes the middle, because "the
  middle" is just the set of positions that are still masked;
* **parallel generation** — every masked position is predicted on every step, so the number
  of forward passes is a knob you turn, not the length of the text.

What it costs is data efficiency: the gradient only reaches the masked positions, where an
autoregressive model gets a prediction at every position of every sequence. Published
comparisons put the gap at roughly 3–16x the compute for equal quality. So this is never
the main model here — it is run at Phase-1 scale (13.8M, TinyStories) as a controlled
experiment against a dense baseline we already trust.

Read with: docs/20-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from .corrupt import Corruption, corrupt, diffusion_loss, sample_t
from .evaluate import elbo, loss_by_t
from .generate import DenoiseStep, diffusion_generate, infill

__all__ = [
    "Corruption", "corrupt", "diffusion_loss", "sample_t",
    "elbo", "loss_by_t",
    "DenoiseStep", "diffusion_generate", "infill",
]
