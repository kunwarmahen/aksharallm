"""One validation number is hiding two.

`configs/small-code.yaml` trains on 85% FineWeb-Edu and 15% Python, and reports a single
val loss. That average is a blend of two very different distributions, and it can move for
reasons it will not tell you about — code perplexity improving while prose stalls looks
exactly like slow steady progress. Worse, the mix is 85/15, so the *prose* number is almost
the whole average and the Python half of the model's job is nearly invisible in it.

This splits it.

Where the boundaries come from
-------------------------------
`prepare_blend` writes `val.bin` by **concatenating** one part per source, in the order the
sources were given, each capped at `val_tokens x weight`:

```mermaid
flowchart LR
    A["fineweb-edu<br/>weight 0.85<br/>tokens 0 .. 8,499,999"] --> C["val.bin"]
    B["codeparrot-python<br/>weight 0.15<br/>tokens 8,500,000 .. 9,999,999"] --> C
```

Nothing records those offsets, so they are *derived* from the weights — and derived numbers
that nobody checks are how a report becomes confidently wrong. So `spans()` derives them and
`verify_spans()` reads the actual tokens either side of each boundary and asks whether the
content agrees. A boundary that does not verify is reported as unverified rather than used,
because a prose/Python split that has the split in the wrong place is worse than no split at
all: it produces two plausible numbers that are both averages of the same mixture.

`prepare_blend` now also writes `val.manifest.json` alongside the bin, so future blends do
not have to derive anything. The derivation stays for the blends that predate it.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/02-data.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: The filename `prepare_blend` writes beside `val.bin`.
MANIFEST = "val.manifest.json"

#: Tokens sampled either side of a boundary when verifying it.
PROBE_TOKENS = 4_000


@dataclass
class Span:
    name: str
    start: int
    end: int          # exclusive
    weight: float | None = None
    verified: bool | None = None      # None = not checked
    note: str = ""

    @property
    def tokens(self) -> int:
        return self.end - self.start


def manifest_path(val_bin: str | Path) -> Path:
    return Path(val_bin).with_name(MANIFEST)


def load_manifest(val_bin: str | Path) -> list[Span] | None:
    """The authoritative spans, if the blend that wrote this recorded them."""
    path = manifest_path(val_bin)
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return [Span(s["name"], int(s["start"]), int(s["end"]), s.get("weight"),
                 verified=True, note="from val.manifest.json") for s in blob.get("spans", [])]


def derive_spans(val_bin: str | Path, sources: list[dict]) -> list[Span]:
    """Reconstruct the layout from `data.train_sources` weights.

    Exact whenever each part reached its cap, which is the normal case — a source only falls
    short if its stream ran out. The last span absorbs any rounding so the spans always tile
    the file exactly rather than leaving a token nobody owns.
    """
    total = Path(val_bin).stat().st_size // 2
    weights = [float(s.get("weight", 0.0)) for s in sources]
    names = [Path(str(s.get("bin", f"source{i}"))).stem
             for i, s in enumerate(sources)]
    if not weights or sum(weights) <= 0:
        return [Span("all", 0, total, 1.0)]

    spans, pos = [], 0
    for i, (name, w) in enumerate(zip(names, weights)):
        end = total if i == len(weights) - 1 else min(total, pos + int(total * w))
        spans.append(Span(name, pos, end, w, note="derived from train_sources weights"))
        pos = end
    return [s for s in spans if s.tokens > 0]


#: Cheap, explicit signals that a decoded window is source code rather than prose. Deliberately
#: crude: this is a *check* on an arithmetic derivation, not a classifier anyone should trust
#: on its own, and something subtle would be harder to argue with when it disagrees.
_CODE_MARKERS = ("def ", "import ", "class ", "return ", "self.", "()", "{", "};", "//",
                 "#include", "</", "=>", "):", "];")


def looks_like_code(text: str) -> float:
    """Fraction of code markers present. Above ~0.25 is code, below ~0.1 is prose."""
    return sum(m in text for m in _CODE_MARKERS) / len(_CODE_MARKERS)


def verify_spans(val_bin: str | Path, spans: list[Span], tok,
                 probe: int = PROBE_TOKENS) -> list[Span]:
    """Read each span's middle and ask whether the content matches what the name claims.

    Only decides for spans whose name says something checkable — a source called
    `codeparrot-python` should read as code and one called `fineweb-edu` should not. A name
    this cannot form an opinion about is left `verified=None`, which the report prints as
    "unverified" rather than as a pass.
    """
    data = np.memmap(Path(val_bin), dtype=np.uint16, mode="r")
    out = []
    for span in spans:
        mid = (span.start + span.end) // 2
        chunk = np.asarray(data[mid: min(mid + probe, span.end)])
        text = tok.decode([int(x) for x in chunk]) if chunk.size else ""
        score = looks_like_code(text)
        name = span.name.lower()
        expect_code = any(k in name for k in ("code", "python", "stack", "parrot"))
        expect_prose = any(k in name for k in ("fineweb", "web", "wiki", "book", "story"))
        if expect_code:
            span.verified, span.note = score >= 0.25, f"code markers {score:.0%} (expected code)"
        elif expect_prose:
            span.verified, span.note = score <= 0.15, f"code markers {score:.0%} (expected prose)"
        else:
            span.verified, span.note = None, f"code markers {score:.0%} (no expectation)"
        out.append(span)
    return out


def spans_for(val_bin: str | Path, sources: list[dict] | None, tok=None) -> list[Span]:
    """The spans to measure: the manifest if there is one, otherwise derived and checked."""
    known = load_manifest(val_bin)
    if known:
        return known
    if not sources:
        total = Path(val_bin).stat().st_size // 2
        return [Span("all", 0, total, 1.0, note="no train_sources; nothing to split")]
    spans = derive_spans(val_bin, sources)
    return verify_spans(val_bin, spans, tok) if tok is not None else spans


@torch.no_grad()
def per_domain_loss(model, val_bin: str | Path, spans: list[Span], seq_len: int,
                    batches: int = 20, batch_size: int = 4, device: str = "cpu",
                    seed: int = 1234, progress=None) -> list[dict]:
    """Held-out loss inside each span, sampled the same way for every one of them.

    Same seed, same number of batches and the same window length per span, so the numbers
    are comparable with each other. They are **not** comparable with the trainer's val loss,
    which averages over the whole file — a fact worth stating wherever these are printed.
    """
    data = np.memmap(Path(val_bin), dtype=np.uint16, mode="r")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False))
    out = []
    for si, span in enumerate(spans):
        room = span.tokens - seq_len - 1
        if room <= 0:
            out.append({"name": span.name, "tokens": span.tokens, "loss": None,
                        "note": "too short to sample"})
            continue
        rng = np.random.default_rng(seed)
        total_nll, total_tok = 0.0, 0
        for b in range(batches):
            starts = span.start + rng.integers(0, room, size=batch_size)
            xs = np.stack([np.asarray(data[s: s + seq_len]) for s in starts]).astype(np.int64)
            ys = np.stack([np.asarray(data[s + 1: s + 1 + seq_len]) for s in starts]).astype(np.int64)
            x = torch.from_numpy(xs).to(device)
            y = torch.from_numpy(ys).to(device)
            with amp:
                logits, _ = model(x, full_logits=True)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_tok += int(y.numel())
            if progress:
                progress(si * batches + b + 1, len(spans) * batches, "per-domain loss")
        mean = total_nll / total_tok
        out.append({
            "name": span.name, "tokens": span.tokens, "weight": span.weight,
            "start": span.start, "end": span.end,
            "loss": mean, "perplexity": math.exp(min(mean, 20)),
            "verified": span.verified, "note": span.note,
            "sampled_tokens": total_tok,
        })
    return out


def blended(rows: list[dict]) -> float | None:
    """The weight-blended loss, for comparison with the single number the trainer prints.

    If this disagrees badly with the run's own val loss, the spans are wrong — which is the
    cheapest end-to-end check available, and the reason it is computed at all.
    """
    usable = [r for r in rows if r.get("loss") is not None and r.get("weight")]
    if not usable:
        return None
    total_w = sum(r["weight"] for r in usable)
    return sum(r["loss"] * r["weight"] for r in usable) / total_w if total_w else None
