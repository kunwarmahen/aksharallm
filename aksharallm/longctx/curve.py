"""Loss by *position* — the measurement that shows where a model's context actually ends.

Ordinary perplexity averages over every position in a window and reports one number. That
average is exactly the wrong shape for this question. A model extended from 1k to 4k can
have a perfectly respectable mean while being fluent for 1,000 tokens and producing noise
for the next 3,000 — the good part carries the average.

So: run one long window, keep the loss of every position *separately*, and bucket it.

    position   0-127   128-255   ...   896-1023 | 1024-1151   1152-1279   ...
    loss        3.4      3.1             3.0    |    9.8         14.6
                └──────── trained here ─────────┘ └──── past the edge ────┘

The shape is the whole result, and there are only three of them:

* **flat, or gently falling** — healthy. Loss normally *improves* with position, because
  later tokens have more context to condition on; a curve that keeps sloping down is a
  model genuinely using the extra room.
* **a cliff at the trained window** — the naive extension. Not a degradation, a collapse:
  RoPE is handing the model angles it has never seen, and every attention score becomes
  noise within a few hundred tokens.
* **a step up, then flat** — a working scaling method. It costs something everywhere (the
  positions have all been reinterpreted) and then holds, which is the trade the whole
  chapter is about.

The measurement is deliberately cheap and needs **no download and no fine-tune**: the
windows come from the run's own `val.bin`, so it works on any checkpoint the moment it
exists, and it is the first thing to run before believing any long-context claim.

Read with: docs/18-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def position_curve(model, bin_path: str, seq_len: int, bucket: int = 128,
                   n_windows: int = 16, batch_size: int = 1, device: str = "cpu",
                   seed: int = 1234, progress=None) -> dict:
    """Mean cross-entropy per position bucket over `seq_len`-token windows.

    Returns `{"buckets": [{start, end, loss, perplexity, tokens}, ...], "loss", ...}`.

    `batch_size` stays at 1 by default on purpose: this is run at lengths chosen to be
    uncomfortable, and the point of the exercise is defeated by an OOM.
    """
    from ..data.loader import TokenDataset

    ds = TokenDataset(bin_path, seq_len, device)
    rng = np.random.default_rng(seed)
    n_buckets = math.ceil(seq_len / bucket)
    nll = torch.zeros(n_buckets, dtype=torch.float64)
    count = torch.zeros(n_buckets, dtype=torch.float64)

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False))

    done = 0
    for _ in range(math.ceil(n_windows / batch_size)):
        x, y = ds.get_batch(batch_size, generator=rng)
        with amp:
            logits, _ = model(x, full_logits=True)
        # reduction="none" is the entire point — a mean here throws away the answer.
        per_tok = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none",
        ).view(y.shape).float().cpu()
        for b in range(n_buckets):
            lo, hi = b * bucket, min((b + 1) * bucket, seq_len)
            nll[b] += per_tok[:, lo:hi].sum().double()
            count[b] += per_tok[:, lo:hi].numel()
        done += batch_size
        if progress:
            progress(min(done, n_windows), n_windows, "position curve")

    buckets = []
    for b in range(n_buckets):
        mean = float(nll[b] / count[b])
        buckets.append({
            "start": b * bucket,
            "end": min((b + 1) * bucket, seq_len) - 1,
            "loss": mean,
            # A collapsed model produces losses around 10-15, and exp(15) is 3.2 million.
            # Capped so one broken bucket cannot flatten every other point on the chart
            # into the x-axis; the `loss` column is always the unclipped truth.
            "perplexity": min(math.exp(mean), 1e6),
            "tokens": int(count[b]),
        })
    total = float(nll.sum() / count.sum())
    return {
        "seq_len": seq_len, "bucket": bucket, "windows": done,
        "loss": total, "perplexity": min(math.exp(total), 1e6),
        "buckets": buckets,
        # The two numbers a human actually compares, pulled out so nothing has to
        # re-derive them: how much worse is the far end than the near end?
        "first_bucket_loss": buckets[0]["loss"],
        "last_bucket_loss": buckets[-1]["loss"],
    }


def cliff(curve: dict, trained_len: int, threshold: float = 1.0) -> dict | None:
    """The first bucket whose loss exceeds the in-window baseline by `threshold` nats.

    "In-window baseline" is the mean over buckets that sit **inside** the trained window,
    not the first bucket — position 0 has no context at all and is always the worst point
    on a healthy curve, so anchoring to it would report a cliff on every model.

    Returns None when the curve never breaks, which is the answer we are looking for.
    """
    inside = [b["loss"] for b in curve["buckets"] if b["end"] < trained_len]
    if not inside:
        return None
    baseline = sum(inside) / len(inside)
    for b in curve["buckets"]:
        if b["start"] >= trained_len and b["loss"] > baseline + threshold:
            return {"position": b["start"], "loss": b["loss"], "baseline": baseline,
                    "excess": b["loss"] - baseline}
    return None
