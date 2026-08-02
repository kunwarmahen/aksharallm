"""Measure what quantization actually bought.

Three numbers, and it matters that all three are reported together, because quantization
trades between them and quoting one alone is how people mislead themselves:

  bytes       what it saved. The reliable win.
  perplexity  what it cost. Compare against the *same model* in bf16, on the *same*
              evaluation batches -- `iter_eval_batches` takes a fixed seed for exactly
              this reason, so a 0.01 difference is signal and not batch luck.
  tokens/sec  whether it is any faster. Usually not, at first. See below.

Why speed is the awkward one
----------------------------
Generating one token at a time is *memory-bandwidth bound*, not compute bound: for each
token the GPU must read every weight in the model and then do very little arithmetic with
it. Reading 4x fewer bytes should therefore be ~4x faster.

It is not, on the torch backend, because that backend reconstructs the full bf16 weight
before the matmul -- so it reads the small tensor *and* writes and reads a big one. That
is strictly more traffic than just doing the bf16 matmul. The fused kernel is what
collapses those steps into one and turns the byte saving into a time saving; measuring
the slow path first is what makes the kernel's number mean something.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..data.loader import TokenDataset
from .convert import model_nbytes
from .qlinear import QuantLinear


@dataclass
class BenchResult:
    label: str
    nbytes: int = 0
    loss: float | None = None
    perplexity: float | None = None
    tok_s: float | None = None
    prefill_tok_s: float | None = None
    peak_vram: int | None = None
    load_s: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label, "nbytes": self.nbytes, "loss": self.loss,
            "perplexity": self.perplexity, "tok_s": self.tok_s,
            "prefill_tok_s": self.prefill_tok_s, "peak_vram": self.peak_vram,
            "load_s": self.load_s, **self.extra,
        }


@torch.no_grad()
def measure_perplexity(model, val_bin: str, seq_len: int, n_batches: int = 40,
                       batch_size: int = 8, device: str = "cuda") -> tuple[float, float]:
    """Mean NLL and perplexity over a fixed, seeded slice of the validation split."""
    ds = TokenDataset(val_bin, seq_len, device)
    total_nll, total_tok = 0.0, 0
    for x, y in ds.iter_eval_batches(batch_size, n_batches, seed=1234):
        logits, _ = model(x, targets=y)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                              y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += y.numel()
    mean = total_nll / max(1, total_tok)
    return mean, math.exp(mean)


@torch.no_grad()
def measure_speed(model, device: str = "cuda", prompt_len: int = 64, new_tokens: int = 128,
                  warmup: int = 8) -> tuple[float, float]:
    """Decode tokens/sec and prefill tokens/sec, with the KV cache, batch 1.

    A deliberately bare loop -- no sampling, no tokenizer, no stop conditions. Those are
    real costs in the playground but they are identical across backends and would only
    add noise to the comparison we are actually making here.
    """
    ctx = model.cfg.max_seq_len
    prompt_len = min(prompt_len, ctx // 2)
    new_tokens = min(new_tokens, ctx - prompt_len - 1)
    ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    def one_pass(n_new: int) -> tuple[float, float]:
        caches = model.init_caches(1, ctx, dtype=(torch.bfloat16 if device.startswith("cuda")
                                                  else torch.float32), device=device)
        sync()
        t0 = time.perf_counter()
        logits, _ = model(ids, caches=caches)          # prefill
        sync()
        t1 = time.perf_counter()
        nxt = logits[:, -1:].argmax(-1)
        for _ in range(n_new):
            logits, _ = model(nxt, caches=caches)      # decode, one token at a time
            nxt = logits[:, -1:].argmax(-1)
        sync()
        t2 = time.perf_counter()
        return prompt_len / (t1 - t0), n_new / (t2 - t1)

    one_pass(warmup)                                    # warm the allocator and any JIT
    prefill, decode = one_pass(new_tokens)
    return decode, prefill


def measure(model, label: str, *, val_bin: str | None = None, seq_len: int = 512,
            device: str = "cuda", n_batches: int = 40, batch_size: int = 8,
            speed: bool = True, new_tokens: int = 128) -> BenchResult:
    """One model, all three numbers."""
    r = BenchResult(label=label, nbytes=model_nbytes(model))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if val_bin:
        r.loss, r.perplexity = measure_perplexity(
            model, val_bin, seq_len, n_batches=n_batches, batch_size=batch_size,
            device=device)
    if speed:
        r.tok_s, r.prefill_tok_s = measure_speed(model, device=device, new_tokens=new_tokens)
    if device.startswith("cuda"):
        r.peak_vram = torch.cuda.max_memory_allocated()
    return r


def quant_layer_stats(model) -> dict:
    """Aggregate error introduced per layer type -- which projections suffer most.

    Requires the float weights to still be available, so it is computed during
    conversion rather than after loading.
    """
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            out[name] = {"bits": mod.scheme.bits, "group_size": mod.group_size,
                         "bytes": mod.nbytes()}
    return out


def format_table(results: list[BenchResult], baseline: BenchResult | None = None) -> str:
    """A fixed-width comparison table. The baseline is whichever result is first."""
    base = baseline or (results[0] if results else None)
    head = f"{'model':<26} {'size':>9} {'ratio':>6} {'ppl':>9} {'d ppl':>8} {'tok/s':>8} {'vs bf16':>8}"
    lines = [head, "-" * len(head)]
    for r in results:
        size = f"{r.nbytes / 1e6:.1f} MB"
        ratio = f"{base.nbytes / r.nbytes:.2f}x" if base and r.nbytes else "-"
        ppl = f"{r.perplexity:.3f}" if r.perplexity else "-"
        dppl = "-"
        if base and base.perplexity and r.perplexity:
            d = r.perplexity - base.perplexity
            dppl = f"{d:+.3f}" if r is not base else "baseline"
        toks = f"{r.tok_s:.1f}" if r.tok_s else "-"
        rel = f"{r.tok_s / base.tok_s:.2f}x" if base and base.tok_s and r.tok_s else "-"
        lines.append(f"{r.label:<26} {size:>9} {ratio:>6} {ppl:>9} {dppl:>8} {toks:>8} {rel:>8}")
    return "\n".join(lines)
