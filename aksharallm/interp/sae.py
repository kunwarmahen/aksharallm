"""A sparse autoencoder over the residual stream: pulling apart what a neuron will not.

The obvious way to ask what a model has learned is to look at single dimensions of its
residual stream. It does not work, and the reason has a name: **superposition**. A 1,024-wide
stream is asked to represent far more than 1,024 things, so features are stored as directions
that overlap — one dimension participates in dozens of unrelated concepts, and reading it
gives you a soup.

A sparse autoencoder attacks that directly. Project the stream up into a much *wider* space
and force the result to be almost all zeros:

    f = relu(W_enc (x - b_dec) + b_enc)          # 8x wider, and sparse
    x ~ W_dec f + b_dec                          # reconstruct from a handful of features
    loss = ||x - x_hat||^2  +  alpha * ||f||_1

The reconstruction term keeps the features faithful; the L1 term makes them few. What comes
out, when it works, is a dictionary: individual features that fire on one recognisable thing —
a Python keyword, the inside of a string literal, the second half of a proper noun — rather
than on everything at once.

Three details that decide whether it learns anything at all, all of them cheap and all of them
easy to leave out:

* **Unit-norm decoder columns.** Without this the model cheats: it shrinks `f` towards zero to
  pay less L1 and grows `W_dec` to compensate, so sparsity is met with no features. The columns
  are renormalised after every step.
* **Subtracting `b_dec` before encoding.** The residual stream has a large mean offset that has
  nothing to do with any feature, and every feature would otherwise spend capacity re-encoding
  it.
* **Dead features are counted, loudly.** A feature that has not fired in thousands of steps is
  wasted width, and a run where 90% are dead has learned a small dictionary badly while its
  loss curve looked fine — the same family of failure as MoE's router collapse, and reported
  the same way.

Trainable on this hardware in minutes: activations for a few million tokens at one layer, a
few thousand steps at batch 4,096.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capture import Capture, hooks_on


@dataclass
class SAEConfig:
    d_model: int
    n_features: int
    layer: int
    #: Weight on the L1 term. The one knob that matters: too low and every feature fires on
    #: everything, too high and they all die. 1e-3 to 1e-2 for a normalised stream.
    alpha: float = 3e-3
    lr: float = 1e-3
    steps: int = 2000
    batch: int = 4096
    run: str | None = None


class SAE(nn.Module):
    """Encoder, decoder, and the two constraints that make the result interpretable."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.d_model, cfg.n_features)
        self.decoder = nn.Linear(cfg.n_features, cfg.d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        # Initialise the decoder as the encoder's transpose: a reasonable starting dictionary,
        # and it makes the first few hundred steps far less likely to kill every feature.
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.normalise_decoder()

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        """Every dictionary direction has length 1, so a feature's *activation* is its
        strength. Without this, sparsity can be bought by rescaling rather than by using
        fewer features."""
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_dec))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        recon, f = self(x)
        mse = F.mse_loss(recon, x)
        l1 = f.abs().sum(dim=-1).mean()
        total = mse + self.cfg.alpha * l1
        with torch.no_grad():
            # Fraction of variance explained, which is the number to judge a run by: an SAE
            # that is sparse and reconstructs nothing has simply thrown the stream away.
            var = ((x - x.mean(0)) ** 2).sum()
            fvu = float(((x - recon) ** 2).sum() / var.clamp_min(1e-8))
            live = float((f > 0).any(dim=0).float().mean())
            l0 = float((f > 0).float().sum(dim=-1).mean())
        return total, {"mse": float(mse.detach()), "l1": float(l1.detach()), "l0": l0,
                       "fvu": fvu, "explained": 1.0 - fvu, "live_in_batch": live}


@torch.no_grad()
def collect_activations(model, batches, layer: int, device: str = "cuda",
                        limit: int | None = None) -> torch.Tensor:
    """Residual-stream activations at one layer, `(N, d_model)`, over a stream of token batches.

    Kept on the GPU and returned as one tensor: a few million activations at 1,024 wide in
    bf16 is a couple of gigabytes, which fits — and streaming them from disk per step would
    make the SAE's training loop slower than the model's.
    """
    model.eval()
    out = []
    total = 0
    for batch in batches:
        cap = Capture(tokens=[])
        with hooks_on(model, cap):
            model(batch.to(device))
        acts = cap.residual[layer]
        # Straight to the CPU. Half a million activations at 1,024 wide is 1.6 GB, and the
        # card is already holding the model they came from — the SAE's own training is small
        # enough that shipping one minibatch back per step costs nothing.
        out.append(acts.reshape(-1, acts.shape[-1]).float().cpu())
        total += out[-1].shape[0]
        del cap
        if limit and total >= limit:
            break
    return torch.cat(out)[:limit] if out else torch.empty(0)


def train_sae(acts: torch.Tensor, cfg: SAEConfig, device: str = "cuda",
              log_every: int = 200, echo=print) -> tuple[SAE, list[dict]]:
    """Fit the dictionary. Returns the model and the log, which is what the report reads.

    Dead features are checked against the *whole run*, not the batch: a feature that fires on
    one prompt in a thousand is rare and useful, while one that has not fired in ten thousand
    batches is width that was paid for and never used.
    """
    sae = SAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    fired = torch.zeros(cfg.n_features, dtype=torch.bool, device=device)
    history: list[dict] = []
    n = acts.shape[0]
    t0 = time.time()

    for step in range(cfg.steps):
        idx = torch.randint(0, n, (min(cfg.batch, n),), device=acts.device)  # acts may be CPU
        x = acts[idx].to(device)
        total, stats = sae.loss(x)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        sae.normalise_decoder()
        with torch.no_grad():
            fired |= (sae.encode(x) > 0).any(dim=0)
        if step % log_every == 0 or step == cfg.steps - 1:
            row = {"step": step, "loss": float(total), "alive": float(fired.float().mean()),
                   **stats, "elapsed": time.time() - t0}
            history.append(row)
            echo(f"  step {step:>5} | loss {row['loss']:.4f} | explains "
                 f"{row['explained'] * 100:5.1f}% | L0 {row['l0']:6.1f} | "
                 f"alive {row['alive'] * 100:5.1f}%")
    return sae, history


@torch.no_grad()
def feature_report(sae: SAE, acts: torch.Tensor, top: int = 20,
                   device: str = "cuda") -> dict:
    """Which features are worth looking at: how often each fires, and how strongly.

    A dictionary is not a result until you have looked at what its entries respond to, and the
    firing rate is how you choose which twenty of eight thousand to look at first.
    """
    counts = torch.zeros(sae.cfg.n_features, device=device)
    strength = torch.zeros(sae.cfg.n_features, device=device)
    seen = 0
    for start in range(0, acts.shape[0], 8192):
        x = acts[start:start + 8192].to(device)
        f = sae.encode(x)
        counts += (f > 0).float().sum(dim=0)
        strength += f.sum(dim=0)
        seen += x.shape[0]
    rate = counts / max(seen, 1)
    mean_when_on = strength / counts.clamp_min(1)
    order = torch.argsort(rate, descending=True)
    interesting = [i for i in order.tolist() if 1e-4 < rate[i] < 0.2][:top]
    return {
        "n_features": sae.cfg.n_features,
        "dead": int((counts == 0).sum()),
        "dead_fraction": float((counts == 0).float().mean()),
        "mean_rate": float(rate.mean()),
        "features": [{"id": int(i), "rate": float(rate[i]),
                      "mean_activation": float(mean_when_on[i])} for i in interesting],
    }


@torch.no_grad()
def top_activating(model, sae: SAE, feature: int, token_batches, layer: int,
                   decode, top: int = 10, device: str = "cuda") -> list[dict]:
    """The tokens that make one feature fire hardest — the only real evidence of what it means.

    Context matters as much as the token: a feature that fires on `"` inside a docstring and
    one that fires on `"` opening a string are different features, and the token alone cannot
    tell them apart. So a window either side comes back with it.
    """
    best: list[tuple[float, list[int], int]] = []
    for batch in token_batches:
        cap = Capture(tokens=[])
        with hooks_on(model, cap):
            model(batch.to(device))
        acts = cap.residual[layer]                       # (T, d) for the first row
        f = sae.encode(acts.float())[:, feature]
        rows = batch[0].tolist() if batch.dim() > 1 else batch.tolist()
        for pos, value in enumerate(f.tolist()):
            if value <= 0:
                continue
            best.append((value, rows, pos))
    best.sort(key=lambda r: -r[0])
    out = []
    for value, rows, pos in best[:top]:
        lo, hi = max(0, pos - 8), min(len(rows), pos + 4)
        out.append({
            "activation": value,
            "token": decode([rows[pos]]),
            "context": decode(rows[lo:pos]),
            "after": decode(rows[pos + 1:hi]),
        })
    return out


def save(sae: SAE, path: Path, history: list[dict], report: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": sae.state_dict(), "config": asdict(sae.cfg)}, path)
    path.with_suffix(".json").write_text(json.dumps(
        {"config": asdict(sae.cfg), "history": history, "report": report}, indent=1))
    return path


def load(path: Path, device: str = "cpu") -> SAE:
    blob = torch.load(path, map_location=device, weights_only=False)
    sae = SAE(SAEConfig(**blob["config"])).to(device)
    sae.load_state_dict(blob["state"])
    sae.eval()
    return sae
