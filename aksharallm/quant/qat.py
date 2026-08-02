"""QAT -- quantization-aware training: let the model learn to tolerate the rounding.

Everything else in this package is *post-training* quantization. The weights are fixed and
the only question is how cleverly to round them. RTN rounds naively, AWQ rescales first,
GPTQ compensates for the error it has already made -- but all three are working with a
model that has never been told it will be quantized.

QAT changes the question. Put the rounding *inside* the training loop: every forward pass
uses quantized weights, so the loss the optimiser sees is the quantized model's loss, and
gradient descent moves the weights somewhere that survives being rounded. The model stops
being a victim of quantization and starts accommodating it.

The straight-through estimator
------------------------------
There is an obvious problem: rounding has zero gradient almost everywhere and undefined
gradient at the steps. Backpropagating through it honestly gives zero everywhere, and
nothing trains.

The straight-through estimator is the standard dodge: use the quantized weight going
forward, and pretend the quantizer was the identity going backward.

    w_q = w + (quantize_dequantize(w) - w).detach()

Read it carefully -- the value equals `quantize_dequantize(w)` exactly, because the two
`w` terms cancel numerically. But `.detach()` hides the correction from autograd, so
d(w_q)/dw = 1. Forward is quantized; backward is as if it were not. It is not the true
gradient of anything, and it works remarkably well.

Why it beats the post-training methods, and what it costs
---------------------------------------------------------
GPTQ can only minimise the damage to a layer's *existing* output. QAT can move the weights
themselves to a nearby configuration that quantizes better -- a strictly larger search
space, so given enough steps it should win, and at 4 bits it generally does.

The cost is that it is training. It needs data, gradients, an optimiser and GPU hours, and
it needs the model it is fine-tuning to already exist. Which is exactly why it sits last in
this package: it is the most expensive option and the only one that cannot be run on a
checkpoint in thirty seconds.

Use it as a short fine-tune from an already-trained model (a few hundred steps at a low
learning rate), not as a from-scratch training mode. The weights only need to shuffle a
little to find a quantization-friendly basin.

The learning rate is the whole ball game, and the window is narrow. Measured on the 13.8M
model at int4 per-channel, 800 steps, against an RTN baseline of +0.156 perplexity:

    lr 1e-5   +0.155    recovers nothing; the weights barely move
    lr 5e-5   +0.096    best -- beats GPTQ's +0.100 on the same setting
    lr 2e-4   +0.126    too far; now losing pretraining faster than it gains

That non-monotonic shape is the thing to internalise. QAT is not "more training is
better" -- it is a search for a nearby basin, and a large step leaves the neighbourhood.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train import stopfile
from .convert import _set_module, linear_layers
from .qlinear import QuantLinear
from .qtensor import QuantScheme, fake_quantize, resolve_group_size


class QATLinear(nn.Module):
    """A Linear that quantizes its own weight on every forward pass.

    Unlike QuantLinear this still *holds a float weight*, and that weight is what trains.
    Nothing is smaller yet -- during QAT the model is temporarily larger than the float
    original. The payoff arrives at `convert_qat`, when the trained float weights are
    quantized for real, having been shaped by training to survive it.
    """

    def __init__(self, lin: nn.Linear, scheme: QuantScheme):
        super().__init__()
        if lin.bias is not None:
            raise ValueError("QATLinear supports bias=False layers only")
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.scheme = scheme
        self.group_size = resolve_group_size(scheme.group_size, lin.in_features)
        self.weight = lin.weight  # the same Parameter: the optimiser keeps training it

    def quantized_weight(self) -> torch.Tensor:
        w = self.weight
        # Fake-quant in fp32 whatever the autocast dtype: the scale is a ratio of extremes
        # and computing it in bf16 loses enough precision to make the rounding noisy.
        # scale_dtype=fp16 matches how QuantLinear stores scales. Without it, QAT trains
        # against fp32 scales and the model shifts slightly when converted -- small
        # (~1e-4 on the logits) but it means the thing you measured is not the thing you
        # shipped, which defeats the purpose of simulating quantization during training.
        wq = fake_quantize(w.float(), self.scheme, group_size=self.group_size,
                           scale_dtype=torch.float16).to(w.dtype)
        return w + (wq - w).detach()  # straight-through: value of wq, gradient of w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight())

    def extra_repr(self) -> str:
        g = "chan" if self.group_size == -1 else self.group_size
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.scheme.bits}, group={g} (fake-quant)")


def prepare_qat(model, scheme: QuantScheme, quantize_head: bool = False,
                skip: tuple[str, ...] = ()) -> list[str]:
    """Swap Linear -> QATLinear in place. Returns the names that were wrapped.

    The same lm_head rule as post-training quantization applies, and for the same reason:
    with tied embeddings, quantizing the head buys no bytes.
    """
    tied = bool(getattr(model.cfg, "tie_embeddings", False))
    wrapped = []
    for name, lin in linear_layers(model).items():
        if name.endswith("lm_head") and tied and not quantize_head:
            continue
        if any(s in name for s in skip):
            continue
        _set_module(model, name, QATLinear(lin, scheme))
        wrapped.append(name)
    return wrapped


def convert_qat(model, scheme: QuantScheme) -> int:
    """Turn the trained QATLinears into real packed QuantLinears. Returns how many.

    This is the step where the model actually gets smaller. Because training used exactly
    the same `fake_quantize` arithmetic, the converted model computes what the last
    training forward pass computed -- no surprise drop at the end.
    """
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, QATLinear):
            lin = nn.Linear(mod.in_features, mod.out_features, bias=False,
                            device=mod.weight.device, dtype=mod.weight.dtype)
            lin.weight.data.copy_(mod.weight.data)
            _set_module(model, name, QuantLinear.from_linear(lin, scheme))
            n += 1
    return n


@dataclass
class QATResult:
    steps: int
    loss_start: float
    loss_end: float
    ppl_before: float | None = None
    ppl_after: float | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"steps": self.steps, "loss_start": self.loss_start,
                "loss_end": self.loss_end, "ppl_before": self.ppl_before,
                "ppl_after": self.ppl_after, "seconds": self.seconds}


def train_qat(
    model,
    train_bin: str,
    seq_len: int,
    scheme: QuantScheme,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 5e-5,
    warmup: int = 20,
    device: str = "cuda",
    log_every: int = 20,
    log=print,
    stop_file: Path | None = None,
    stop_by: float | None = None,
) -> QATResult:
    """A short quantization-aware fine-tune.

    Deliberately small: a low learning rate and a few hundred steps. QAT from a trained
    checkpoint is a nudge, not a retrain -- push the learning rate up and you will undo
    the pretraining faster than you recover the quantization loss.

    `stop_file` and `stop_by` end it early on request (`aksharallm.train.stopfile`), leaving
    the model where the last step put it. That is safe here in a way it is not for every
    loop: QAT starts from a trained checkpoint and only nudges it, so stopping at step 300
    of 800 gives you a partly-recovered model, not a broken one -- and everything after this
    call (export, perplexity, saving) runs exactly as it would have.
    """
    from ..data.loader import TokenDataset

    ds = TokenDataset(train_bin, seq_len, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = torch.Generator(device="cpu").manual_seed(1234)
    npg = __import__("numpy").random.default_rng(1234)

    model.train()
    t0 = time.monotonic()
    first = last = float("nan")
    for step in range(1, steps + 1):
        # Linear warmup then cosine, same shape as the main trainer uses.
        if step <= warmup:
            cur = lr * step / max(1, warmup)
        else:
            frac = (step - warmup) / max(1, steps - warmup)
            cur = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gparam in opt.param_groups:
            gparam["lr"] = cur

        x, y = ds.get_batch(batch_size, npg)
        _logits, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        val = float(loss.detach())
        if step == 1:
            first = val
        last = val

        why = stopfile.reached(stopfile.read(stop_file) if stop_file else None, step)
        if why is None and stop_by is not None and time.time() >= stop_by:
            why = "reached the time budget for this QAT run"
        if log and (step % log_every == 0 or step == 1 or why):
            log(f"    qat step {step}/{steps}  loss {val:.4f}  lr {cur:.2e}")
        if why:
            if log:
                log(f"    [stop] {why} -- ending QAT at step {step} of {steps} and "
                    "exporting what it has")
            steps = step
            break

    model.eval()
    del rng
    return QATResult(steps=steps, loss_start=first, loss_end=last,
                     seconds=time.monotonic() - t0)
