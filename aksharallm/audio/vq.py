"""Vector quantization — how a continuous vector becomes an integer, and back.

This is the hinge of the whole phase. The transformer downstairs consumes integers; the
encoder upstairs produces 128-dimensional floats fifty times a second. A vector quantizer
is the translation: keep a **codebook** of K learned vectors, and replace each input with
the index of the nearest one.

```mermaid
flowchart LR
    Z["encoder output<br/>z, 128 floats"] --> N["nearest codebook<br/>entry, by L2"]
    N --> I["index i<br/>0..1023"]
    I --> Q["codebook[i]<br/>= the quantized z"]
```

Three problems come with that, and this file is mostly their solutions.

**1. There is no gradient.** `argmin` has a derivative of zero almost everywhere, so the
encoder would never learn. The **straight-through estimator** cheats: forward, output the
codebook entry; backward, pretend the quantizer was the identity and hand the gradient
straight to the encoder. Written as `z + (q − z).detach()`, which is numerically `q` and
differentiates as `z`. Get this subtly wrong — detach the wrong side — and it still trains
smoothly and reconstructs noise, which is trap 3 of the phase.

**2. The codebook has to learn too**, and the straight-through path gives it nothing. Two
options: a loss term `‖sg(z) − q‖²` that pulls entries towards the vectors assigned to them,
or an **exponential moving average** of exactly the same quantity, which is the same k-means
step done without the optimizer. EMA is the default here because it is far less sensitive to
the learning rate — the codebook is not really a parameter being descended, it is a set of
cluster centroids being tracked.

**3. Codebook collapse is router collapse again.** A handful of entries win early, receive
all the assignments, and the rest are never selected — so a 1,024-entry codebook is quietly
a 40-entry one, reconstruction plateaus, **and the loss curve looks fine**. This is exactly
the failure `docs/15-moe.md` is built around, and it gets the same treatment: usage counted
every step and reported, plus **dead-code restart** — an entry nobody has chosen for a while
is reinitialised to a random encoder output from the current batch, which is a place we know
data actually lives.

**Residual VQ** is then one idea on top: quantize `z`, take what the codebook got *wrong*,
quantize that with a second codebook, and repeat. Each stage refines the last, so N
codebooks of 1,024 entries address `1024^N` points while storing only `N · 1024` vectors —
and, crucially, **the first codebook alone is already a usable approximation.** That is what
makes the bitrate a dial you can turn at decode time rather than a property of the
checkpoint, and it is the demo the portal's Audio tab is built around.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQStats:
    """What one quantizer did on one batch. All of it is diagnostics, none of it is loss.

    `used` and `perplexity` are the collapse detector. `perplexity` is `exp(H)` of the
    assignment distribution: it is the *effective* number of codes in play, so a codebook of
    1,024 entries with a perplexity of 40 is a 40-entry codebook whatever the shape says.
    """

    used: int  # distinct entries chosen in this batch
    perplexity: float  # exp(entropy of the assignment histogram)
    commitment: float  # ‖z − sg(q)‖², the encoder's half of the agreement
    dead: int  # entries restarted this step


class VectorQuantizer(nn.Module):
    """One codebook, with EMA updates and dead-code restart.

    Args:
        dim: width of the vectors being quantized.
        size: how many entries the codebook has (K).
        decay: EMA rate for the codebook. 0.99 is ~100 batches of memory; lower tracks the
            encoder faster and is noisier, higher is stabler and lags a moving encoder.
        commit: weight on the commitment term. It is the *encoder's* incentive to stay near
            the codebook rather than wander somewhere the codebook cannot follow.
        eps: Laplace smoothing on the cluster counts, so a code chosen zero times this batch
            does not divide by zero on its way to being restarted anyway.
        restart_after: an entry unchosen for this many steps is reinitialised. Set to 0 to
            turn restarts off and watch the collapse happen — that is a lesson, not a bug.
    """

    def __init__(
        self,
        dim: int,
        size: int = 1024,
        *,
        decay: float = 0.99,
        commit: float = 0.25,
        eps: float = 1e-5,
        restart_after: int = 200,
    ):
        super().__init__()
        self.dim, self.size = dim, size
        self.decay, self.commit, self.eps = decay, commit, eps
        self.restart_after = restart_after

        # The codebook is a BUFFER, not a Parameter: with EMA updates it is never touched by
        # the optimizer, and registering it as a Parameter would let weight decay quietly
        # shrink every entry towards the origin between updates.
        self.register_buffer("codebook", torch.randn(size, dim) * 0.02)
        self.register_buffer("cluster_size", torch.zeros(size))
        self.register_buffer("embed_avg", self.codebook.clone())
        #: Steps since each entry was last chosen. The restart trigger.
        self.register_buffer("idle", torch.zeros(size, dtype=torch.long))
        #: Whether the codebook has been seeded from real data yet. See `_seed`.
        self.register_buffer("seeded", torch.zeros((), dtype=torch.bool))

    # -- the lookup ---------------------------------------------------------------------

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """`(B, D, T)` -> indices `(B, T)`. The nearest entry to each vector, by L2.

        `‖z − c‖² = ‖z‖² − 2·z·c + ‖c‖²` and `‖z‖²` is the same for every candidate, so it
        drops out of the argmin. What is left is one matmul, which is the difference between
        a quantizer that runs at training speed and one that does not.
        """
        flat = z.transpose(1, 2).reshape(-1, self.dim)  # (B*T, D)
        dist = (self.codebook * self.codebook).sum(1) - 2.0 * flat @ self.codebook.t()
        return dist.argmin(dim=1).view(z.shape[0], z.shape[2])

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """Indices `(B, T)` -> vectors `(B, D, T)`. A plain embedding lookup."""
        return F.embedding(idx, self.codebook).transpose(1, 2)

    # -- training -----------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VQStats]:
        """Quantize `(B, D, T)`. Returns `(quantized, indices, commitment_loss, stats)`."""
        # **The quantizer always runs in float32, whatever autocast is doing around it.**
        # bf16 has eight bits of mantissa; an EMA with decay 0.99 adds one percent of a
        # centroid per step, and one percent of a bf16 number is below its own resolution —
        # so in bf16 the codebook would stop moving and nothing would say so. The
        # straight-through output is cast back, so the decoder still gets its fast dtype.
        dtype = z.dtype
        z = z.float()

        if self.training and not bool(self.seeded):
            self._seed(z)
        idx = self.encode(z)
        q = self.decode(idx)

        # The commitment term acts on the ENCODER only — `q` is detached — because the
        # codebook is moved by the EMA below, not by this gradient. Both halves of "meet in
        # the middle" exist; they are just implemented by different mechanisms.
        commitment = F.mse_loss(z, q.detach())

        dead = 0
        if self.training:
            dead = self._update(z, idx)

        # Straight-through: the value of `q`, the gradient of `z`. Reversing which side is
        # detached is the classic mistake here, and it trains smoothly while learning nothing.
        st = (z + (q - z).detach()).to(dtype)

        with torch.no_grad():
            counts = torch.bincount(idx.reshape(-1), minlength=self.size).float()
            p = counts / counts.sum().clamp_min(1)
            entropy = -(p * torch.log(p.clamp_min(1e-10))).sum()
            stats = VQStats(
                used=int((counts > 0).sum()),
                perplexity=float(entropy.exp()),
                commitment=float(commitment),
                dead=dead,
            )
        return st, idx, self.commit * commitment, stats

    @torch.no_grad()
    def _seed(self, z: torch.Tensor) -> None:
        """Initialise the codebook from the first batch the encoder produces.

        A codebook of `randn·0.02` against an encoder whose outputs happen to have a scale of
        3 is a codebook where *one* entry is nearest to everything — perplexity 1.0, every
        vector mapped to the same index, and the only way out is dead-code restart slowly
        rescuing 1,024 entries one batch at a time. Seeding from real data starts every
        entry somewhere the encoder actually goes.

        Sampled **with replacement** because a batch has 400 vectors and a codebook has
        1,024; the duplicates separate within a few steps because the EMA pulls each towards
        a different half of the vectors that chose it, and the restart mechanism collects
        whatever does not.
        """
        flat = z.transpose(1, 2).reshape(-1, self.dim).detach()
        pick = torch.randint(0, flat.shape[0], (self.size,), device=flat.device)
        chosen = flat[pick]
        # A little jitter, so identical duplicates are not exactly tied — an exact tie sends
        # every one of them to the same `argmin` forever.
        chosen = chosen + 0.01 * chosen.std() * torch.randn_like(chosen)
        self.codebook.copy_(chosen)
        self.embed_avg.copy_(chosen)
        self.cluster_size.fill_(1.0)
        self.seeded.fill_(True)

    @torch.no_grad()
    def _update(self, z: torch.Tensor, idx: torch.Tensor) -> int:
        """One EMA step of k-means, plus restarts. Returns how many entries were restarted."""
        flat = z.transpose(1, 2).reshape(-1, self.dim).detach()
        onehot = F.one_hot(idx.reshape(-1), self.size).to(flat.dtype)  # (N, K)

        counts = onehot.sum(0)  # how many vectors chose each entry
        totals = onehot.t() @ flat  # and their sum, per entry

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(totals, alpha=1 - self.decay)

        # Laplace smoothing keeps the denominator away from zero without changing the
        # centroid of a well-populated entry: the correction is O(eps) against a count of
        # hundreds, and O(1) against a count of zero, which is the only place it matters.
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.size * self.eps) * n
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

        self.idle = torch.where(counts > 0, torch.zeros_like(self.idle), self.idle + 1)
        if self.restart_after <= 0:
            return 0

        stale = self.idle >= self.restart_after
        n_stale = int(stale.sum())
        if n_stale:
            # Reinitialise from real encoder outputs, not from noise: the point of a restart
            # is to move the entry somewhere data actually is. Sampling WITH replacement is
            # deliberate — a batch can be smaller than the number of dead entries.
            pick = torch.randint(0, flat.shape[0], (n_stale,), device=flat.device)
            fresh = flat[pick]
            self.codebook[stale] = fresh
            self.embed_avg[stale] = fresh
            # Give a restarted entry a nonzero count, or the very next EMA step divides its
            # fresh position by ~eps and throws it to infinity.
            self.cluster_size[stale] = 1.0
            self.idle[stale] = 0
        return n_stale


class ResidualVQ(nn.Module):
    """N codebooks, each quantizing what the previous ones could not.

    ```
    r0 = z
    i1 = Q1(r0);  r1 = r0 − c1[i1]      <- the first codebook's error
    i2 = Q2(r1);  r2 = r1 − c2[i2]      <- the second's, and so on
    ẑ  = c1[i1] + c2[i2] + ... + cN[iN]
    ```

    Two properties fall out and both are used elsewhere in this phase:

    * **the prefix is a valid code.** Decoding only `i1..ik` gives a coarser but perfectly
      listenable reconstruction, so bitrate becomes a decode-time dial. `codec reconstruct
      --codebooks 1,2,4,8` is that dial, and hearing it is the best demo in the repo.
    * **the codebooks are ordered by importance.** The first carries the most energy, the
      last is nearly noise. The audio LM downstairs exploits that with a delay pattern —
      see `delay.py`.

    `dropout` implements *quantizer dropout*: during training, each batch uses a random
    number of codebooks. Without it the model is only ever optimised at the full bitrate and
    the low-bitrate prefixes, which are supposed to be usable, are not.
    """

    def __init__(
        self,
        dim: int,
        n_codebooks: int = 8,
        size: int = 1024,
        *,
        dropout: bool = True,
        **kw,
    ):
        super().__init__()
        self.n_codebooks, self.size, self.dropout = n_codebooks, size, dropout
        self.layers = nn.ModuleList(VectorQuantizer(dim, size, **kw) for _ in range(n_codebooks))

    @property
    def bits_per_frame(self) -> float:
        return self.n_codebooks * torch.log2(torch.tensor(float(self.size))).item()

    def forward(self, z: torch.Tensor, *, n_active: int | None = None):
        """Quantize `(B, D, T)` with the first `n_active` codebooks (default: all).

        Returns `(quantized, indices (B, N, T), commitment loss, [VQStats])`. Indices for
        inactive codebooks are **zero**, and the caller must not train an LM on them — see
        `n_active` handling in `codec.encode`.
        """
        if n_active is None:
            n_active = self.n_codebooks
            if self.training and self.dropout and self.n_codebooks > 1:
                # A different bitrate per batch, uniform over 1..N. Uniform rather than
                # weighted towards N: the coarse prefixes are the ones nothing else trains.
                n_active = int(torch.randint(1, self.n_codebooks + 1, (1,)).item())

        residual = z
        quantized = torch.zeros_like(z)
        idx = z.new_zeros((z.shape[0], self.n_codebooks, z.shape[2]), dtype=torch.long)
        loss = z.new_zeros(())
        stats: list[VQStats] = []

        for i, layer in enumerate(self.layers):
            if i >= n_active:
                break
            q, ids, commit, st = layer(residual)
            # The residual is taken against the STRAIGHT-THROUGH output, so the gradient
            # reaches the encoder through every stage rather than only the first.
            residual = residual - q
            quantized = quantized + q
            idx[:, i] = ids
            loss = loss + commit
            stats.append(st)

        return quantized, idx, loss / max(n_active, 1), stats

    def decode(self, idx: torch.Tensor, *, n_active: int | None = None) -> torch.Tensor:
        """Indices `(B, N, T)` -> the summed vectors `(B, D, T)`. The bitrate dial."""
        n_active = self.n_codebooks if n_active is None else n_active
        out = None
        for i, layer in enumerate(self.layers):
            if i >= n_active:
                break
            part = layer.decode(idx[:, i])
            out = part if out is None else out + part
        if out is None:
            raise ValueError("n_active must be at least 1")
        return out
