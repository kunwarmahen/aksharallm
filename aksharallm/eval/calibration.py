"""Is the model's confidence honest?

Every number in `docs/13` asks whether the model is *right*. This asks something different
and, for anything that has to be trusted rather than benchmarked, more useful: when it says
it is 80% sure, is it right 80% of the time?

A model can be accurate and badly calibrated (right often, but wildly overconfident about
the cases it gets wrong), or inaccurate and well calibrated (usually wrong, and it says so).
The second is far more useful in a system that can defer, retry, or ask a human. Accuracy
alone cannot tell them apart, which is the whole reason for this file.

```mermaid
flowchart LR
    L["logits at every<br/>position"] --> C["confidence =<br/>max softmax prob"]
    L --> A["correct? =<br/>argmax == target"]
    C --> B["bucket by confidence"]
    A --> B
    B --> R["reliability: for each bucket,<br/>mean confidence vs mean accuracy"]
    R --> E["ECE = weighted mean<br/>of the gaps"]
```

**Expected calibration error** is the average gap between confidence and accuracy, weighted
by how many predictions fall in each bucket. Zero is perfect. A number around 0.02 is good;
0.10 means the model is out by ten percentage points on average, which for a model whose
output feeds a decision is a lot.

Four things that make ECE lie, all handled here
------------------------------------------------
1. **The bin count changes the answer**, and there is no canonical choice. More bins measure
   finer structure and have noisier buckets. So `ece()` takes `n_bins`, `report()` computes
   it at three of them, and every number printed carries the count that produced it. An ECE
   quoted without its bin count is not a reproducible measurement.
2. **Equal-width bins are nearly empty at the top.** Next-token prediction over a 32k
   vocabulary puts most predictions below 0.5 confidence, so the interesting high-confidence
   buckets can hold a handful of samples and swing wildly. We therefore report **equal-mass**
   (quantile) bins beside equal-width ones — same data, same definition, bins chosen so each
   holds the same number of predictions.
3. **A degenerate model scores beautifully.** Something that always predicts the base rate
   with the base rate's confidence has an ECE near zero and is useless. ECE is a companion to
   accuracy, never a substitute, so `report()` returns accuracy and mean confidence beside
   it and the CLI prints all three.
4. **Padding and ignored positions must not be counted.** A position whose target is `-100`
   has no correct answer, and including it silently drags accuracy towards zero while
   confidence stays put — which looks exactly like overconfidence.

Temperature scaling
-------------------
The standard fix, and the reason to measure this at all: divide the logits by a single
scalar `T` before the softmax. `T > 1` flattens the distribution (less confident), `T < 1`
sharpens it. It **cannot change accuracy at all** — dividing by a positive constant does not
move the argmax — so it is free of the usual trade-off, and one number fitted on held-out
data typically removes most of the miscalibration. `fit_temperature` finds it by minimising
negative log likelihood, and `report()` shows ECE before and after.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

#: Bin counts to report at. Three, because the number changes the answer and quoting one is
#: how a measurement stops being reproducible.
BIN_COUNTS = (10, 15, 30)


@dataclass
class Bucket:
    """One row of a reliability table."""

    lo: float
    hi: float
    count: int
    confidence: float  # mean max-probability in this bucket
    accuracy: float  # fraction where argmax == target

    @property
    def gap(self) -> float:
        """Positive means **over**confident: it claimed more than it delivered."""
        return self.confidence - self.accuracy


@dataclass
class Calibration:
    """Everything a calibration report needs, for one temperature."""

    n: int
    accuracy: float
    confidence: float
    ece: dict[int, float] = field(default_factory=dict)  # equal-width, by bin count
    ece_massed: dict[int, float] = field(default_factory=dict)  # equal-mass, by bin count
    mce: float = 0.0
    buckets: list[Bucket] = field(default_factory=list)  # at BIN_COUNTS[0], equal-width
    temperature: float = 1.0

    @property
    def overconfident(self) -> bool:
        return self.confidence > self.accuracy

    def as_dict(self) -> dict:
        return {
            "n": self.n, "accuracy": self.accuracy, "confidence": self.confidence,
            "temperature": self.temperature, "mce": self.mce,
            "ece": {str(k): v for k, v in self.ece.items()},
            "ece_equal_mass": {str(k): v for k, v in self.ece_massed.items()},
            "buckets": [b.__dict__ | {"gap": b.gap} for b in self.buckets],
        }


# ---------------------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------------------


def buckets_equal_width(conf: torch.Tensor, correct: torch.Tensor, n_bins: int) -> list[Bucket]:
    """`n_bins` bins of equal *width* across [0, 1] — the textbook definition."""
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Right-closed except for the first bin, so a confidence of exactly 1.0 lands in the
        # last bucket rather than falling off the end.
        sel = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(sel.sum())
        if n == 0:
            out.append(Bucket(lo, hi, 0, 0.0, 0.0))
            continue
        out.append(Bucket(lo, hi, n, float(conf[sel].mean()), float(correct[sel].float().mean())))
    return out


def buckets_equal_mass(conf: torch.Tensor, correct: torch.Tensor, n_bins: int) -> list[Bucket]:
    """`n_bins` bins each holding the same *number* of predictions.

    The reason this exists: over a 32k vocabulary almost every prediction is below 0.5
    confidence, so equal-width bins put nearly everything in the first two and leave the
    high-confidence buckets — the ones that matter for trusting an answer — holding a
    handful of samples whose accuracy is noise.
    """
    order = torch.argsort(conf)
    conf, correct = conf[order], correct[order]
    out = []
    edges = [round(i * len(conf) / n_bins) for i in range(n_bins + 1)]
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        c, k = conf[a:b], correct[a:b]
        out.append(Bucket(float(c[0]), float(c[-1]), b - a,
                          float(c.mean()), float(k.float().mean())))
    return out


def ece_from(buckets: list[Bucket], n: int) -> tuple[float, float]:
    """`(ECE, MCE)` — the count-weighted mean gap, and the worst gap in any non-empty bin.

    MCE is reported because the mean hides the shape: a model that is perfectly calibrated
    everywhere except at very high confidence has a small ECE and is exactly the model you
    should not trust when it sounds certain.
    """
    if n == 0:
        return float("nan"), float("nan")
    ece = sum(b.count * abs(b.gap) for b in buckets) / n
    mce = max((abs(b.gap) for b in buckets if b.count), default=0.0)
    return ece, mce


def calibrate(logits: torch.Tensor, targets: torch.Tensor, *, temperature: float = 1.0,
              ignore_index: int = -100) -> Calibration:
    """Measure calibration from logits `(N, V)` and targets `(N,)`.

    `temperature` divides the logits before the softmax. It cannot change accuracy — dividing
    by a positive constant leaves the argmax alone — which is asserted by a test, because it
    is the property that makes temperature scaling free.
    """
    logits = logits.reshape(-1, logits.shape[-1]).float()
    targets = targets.reshape(-1)
    keep = targets != ignore_index
    logits, targets = logits[keep], targets[keep]

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred == targets
    n = int(conf.numel())

    cal = Calibration(
        n=n,
        accuracy=float(correct.float().mean()) if n else float("nan"),
        confidence=float(conf.mean()) if n else float("nan"),
        temperature=temperature,
    )
    if n == 0:
        return cal
    for bins in BIN_COUNTS:
        wide = buckets_equal_width(conf, correct, bins)
        cal.ece[bins], mce = ece_from(wide, n)
        cal.ece_massed[bins], _ = ece_from(buckets_equal_mass(conf, correct, bins), n)
        if bins == BIN_COUNTS[0]:
            cal.buckets, cal.mce = wide, mce
    return cal


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *,
                    ignore_index: int = -100, steps: int = 100) -> float:
    """The single scalar that minimises NLL. Returns `T`; 1.0 means already calibrated.

    **NLL-optimal is not ECE-optimal, and on an already-calibrated model this can make ECE
    slightly worse.** Measured on the 13.8M TinyStories model, which starts at ECE 0.0135:
    the fit returns T = 0.986 and ECE moves to 0.0149. That is not a bug in either number —
    they are different objectives, and when the gap being corrected is smaller than the noise
    in the estimate, fitting to one of them chases the noise. Fit ECE directly if ECE is what
    you care about; the reason this one minimises NLL is that NLL is what the model was
    trained on, so `T` is interpretable as "how far off was the training objective".


    Optimised in **log space** so `T` cannot go negative or hit zero without a clamp, which
    is the kind of guard that silently stops being needed and then silently is again. LBFGS
    is the usual choice and Adam over 100 steps on one parameter reaches the same place.

    **Fit this on data the model will not be scored on.** A temperature fitted and evaluated
    on the same tokens reports the calibration of a model that has seen the answers — which
    is a smaller number and a meaningless one. `report()` splits its batches in half.
    """
    logits = logits.reshape(-1, logits.shape[-1]).float()
    targets = targets.reshape(-1)
    keep = targets != ignore_index
    logits, targets = logits[keep], targets[keep]
    if logits.numel() == 0:
        return 1.0

    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        opt.step()
    return float(log_t.detach().exp())


# ---------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------


#: How many positions to keep logits for. The binding constraint in this file is memory, not
#: time — see `collect`.
MAX_POSITIONS = 20_000


@torch.no_grad()
def collect(model, dataset, batches: int, batch_size: int, *,
            max_positions: int = MAX_POSITIONS,
            seed: int = 0, progress=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over `batches` batches and return `(logits, targets)` on the CPU.

    **The binding constraint here is memory, and it is worth stating.** Calibration needs the
    *whole* distribution at each position, not just the top-1 probability, because
    `fit_temperature` has to recompute `logsumexp(z/T)` at every candidate temperature. So
    the natural thing — keep every position — is `(batches · B · T, vocab)`, which for 20
    batches of 8×1024 over a 32k vocabulary is **21 GB in float32**.

    Two bounds instead. Positions are **randomly subsampled** to `max_positions`, and kept in
    **float16**: 20,000 × 32,768 × 2 bytes is 1.3 GB, and a histogram over confidences does
    not need more precision than that. The subsample is uniform over positions, so it is an
    unbiased sample of exactly the distribution the model is asked to predict — and it is
    *seeded*, so two runs of this on one checkpoint give the same number.

    Batches are moved to the CPU as they are produced, so the GPU never holds more than one.
    """
    was = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    per_batch = max(1, max_positions // max(batches, 1))

    all_logits, all_targets = [], []
    for done in range(batches):
        # This loop is the whole wall-clock cost, so it is what a progress bar has to watch.
        if progress:
            progress(done, batches, "collecting logits")
        x, y = dataset.get_batch(batch_size)
        logits, _ = model(x, full_logits=True)
        flat = logits.reshape(-1, logits.shape[-1])
        flat_y = y.reshape(-1)
        if flat.shape[0] > per_batch:
            pick = torch.randperm(flat.shape[0], generator=rng)[:per_batch]
            flat, flat_y = flat[pick.to(flat.device)], flat_y[pick.to(flat_y.device)]
        all_logits.append(flat.to(torch.float16).cpu())
        all_targets.append(flat_y.cpu())
    if progress:
        progress(batches, batches, "collecting logits")
    model.train(was)
    return torch.cat(all_logits), torch.cat(all_targets)


def report(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calibration before and after temperature scaling, with the fit held out.

    The split is the point: the temperature is fitted on the first half and *scored on the
    second*. Fitting and scoring on the same tokens reports a number for a model that has
    seen the answers.
    """
    n = logits.shape[0]
    half = n // 2
    fit_logits, fit_targets = logits[:half], targets[:half]
    test_logits, test_targets = logits[half:], targets[half:]

    before = calibrate(test_logits, test_targets)
    temperature = fit_temperature(fit_logits, fit_targets)
    after = calibrate(test_logits, test_targets, temperature=temperature)

    return {
        "n_total": int(n),
        "n_fit": int(half),
        "n_scored": int(n - half),
        "temperature": temperature,
        "before": before.as_dict(),
        "after": after.as_dict(),
        # Stated rather than left to be inferred, because "T > 1" is not self-explanatory.
        "reading": (
            f"temperature {temperature:.3f} — the model was "
            f"{'OVER' if temperature > 1.0 else 'UNDER'}confident, and its logits are "
            f"{'flattened' if temperature > 1.0 else 'sharpened'} by that factor"
            if abs(temperature - 1.0) > 0.01 else
            "temperature ~1.0 — the model was already about as calibrated as one scalar can make it"
        ),
        "caveat": (
            "ECE is a companion to accuracy, never a substitute: a model that always "
            "predicts the base rate with the base rate's confidence scores near zero and "
            "is useless. Every ECE here carries the bin count that produced it, because "
            "the count changes the answer."
        ),
    }


def format_report(res: dict) -> str:
    """The terminal rendering, shared with the CLI so both say the same thing."""
    b, a = res["before"], res["after"]
    lines = [
        f"scored on {res['n_scored']:,} predictions "
        f"(temperature fitted on a held-out {res['n_fit']:,})",
        "",
        f"{'':<22}{'accuracy':>10}{'confidence':>12}{'gap':>9}",
        f"{'as trained':<22}{b['accuracy']:>10.4f}{b['confidence']:>12.4f}"
        f"{b['confidence'] - b['accuracy']:>+9.4f}",
        f"{'T = ' + format(res['temperature'], '.3f'):<22}{a['accuracy']:>10.4f}"
        f"{a['confidence']:>12.4f}{a['confidence'] - a['accuracy']:>+9.4f}",
        "",
        f"{'bins':>6}{'ECE':>10}{'ECE (T)':>10}{'equal-mass':>12}{'equal-mass (T)':>16}",
    ]
    for bins in BIN_COUNTS:
        k = str(bins)
        lines.append(
            f"{bins:>6}{b['ece'][k]:>10.4f}{a['ece'][k]:>10.4f}"
            f"{b['ece_equal_mass'][k]:>12.4f}{a['ece_equal_mass'][k]:>16.4f}"
        )
    lines += [
        "",
        f"worst bucket gap (MCE, {BIN_COUNTS[0]} bins): {b['mce']:.4f} -> {a['mce']:.4f}",
        "",
        "reliability, as trained (equal width):",
        f"  {'range':>14}{'count':>10}{'conf':>9}{'acc':>9}{'gap':>9}",
    ]
    for bucket in b["buckets"]:
        if not bucket["count"]:
            continue
        lines.append(
            f"  {bucket['lo']:>6.2f}-{bucket['hi']:<7.2f}{bucket['count']:>10,}"
            f"{bucket['confidence']:>9.3f}{bucket['accuracy']:>9.3f}{bucket['gap']:>+9.3f}"
        )
    lines += ["", res["reading"], "", res["caveat"]]
    return "\n".join(lines)


def perplexity(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Cross-entropy as a perplexity, so a calibration run also reports the familiar number.

    Cheap, and it is the cross-check: if this disagrees with the run's own recorded val loss,
    the calibration numbers are being computed on something other than what was trained.
    """
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                         targets.reshape(-1), ignore_index=ignore_index)
    return math.exp(min(float(ce), 20.0))
