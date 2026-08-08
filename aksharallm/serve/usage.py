"""What the server actually served, so its electricity can be divided by something.

`portal/cost.py` already answers "what did this training run cost" by integrating the GPU's
power draw over the run's wall-clock time. Serving needs one more number to be answerable in
the same way: **how many tokens came out**. Watt-hours alone say what a server *cost*; only
watt-hours per million tokens say whether it was worth running.

So every completed request appends one line to `logs/serve/usage.jsonl`:

```json
{"t0": 1786052.1, "t1": 1786054.7, "prompt": 41, "completion": 128, "run": "small-code"}
```

Append-only, one line per request, flushed immediately — the same contract as
`train_log.jsonl`, for the same reason: a server killed with `kill -9` should still be able
to account for what it did before it died.

Three decisions that are the whole design
------------------------------------------
**1. Prompt tokens and completion tokens are counted separately, and never added up.**
Prefill runs the whole prompt through the model in one batched pass; decode runs the model
once per token produced. They differ by orders of magnitude in cost per token, so a single
"tokens served" figure is an average over two different things whose mix changes with every
request. Providers price them separately for exactly this reason, and so do we.

**2. The headline is per million *completion* tokens.** That is the number that is comparable
to a price list, and it is the number that changes when you improve the decode path —
speculative decoding, continuous batching, a bigger batch. Prefill cost per token barely
moves.

**3. Time is recorded, not just counts.** A server holding the card at idle still burns
power, and the interesting question is usually *"should I leave this up?"* — which needs the
split between energy spent generating and energy spent waiting. `t0`/`t1` per request give
`cost.py` the busy intervals to subtract.

Read with: docs/17-serving.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/10-running-and-watching.md.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

#: Where the record lives. Under `logs/serve/` beside the pid file and the log, so a server's
#: whole footprint is one directory.
USAGE_LOG = Path("logs/serve/usage.jsonl")

#: Stop the file growing without bound on a server left up for weeks. At ~110 bytes a
#: request this is a few hundred thousand requests, and the ledger only ever reads a window.
MAX_BYTES = 32 * 1024 * 1024


@dataclass
class Request:
    """One served request. `t0` is when generation started, not when the socket opened."""

    t0: float
    t1: float
    prompt: int
    completion: int
    run: str = ""
    stream: bool = False

    @property
    def seconds(self) -> float:
        return max(self.t1 - self.t0, 0.0)

    def as_dict(self) -> dict:
        return {"t0": round(self.t0, 3), "t1": round(self.t1, 3), "prompt": self.prompt,
                "completion": self.completion, "run": self.run, "stream": self.stream}


class UsageLog:
    """Append-only request accounting, safe to call from the server's worker threads."""

    def __init__(self, path: Path | str = USAGE_LOG, max_bytes: int = MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes
        # The HTTP server is threaded, so two requests can finish in the same microsecond.
        # A lock around a line-buffered append is cheaper than the alternative of one
        # writer thread, and a torn line here would corrupt the only record of the work.
        self._lock = threading.Lock()

    def record(self, req: Request) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._trim()
                with open(self.path, "a") as f:
                    f.write(json.dumps(req.as_dict()) + "\n")
                    f.flush()
        except OSError:
            # Accounting must never be what fails a request. The whole point of the server
            # is to serve; a lost line costs a fraction of a percent of one report.
            pass

    def _trim(self) -> None:
        """Keep the newest half when the file gets large. Called under the lock."""
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                lines = self.path.read_text().splitlines()
                keep = lines[len(lines) // 2 :]
                self.path.write_text("\n".join(keep) + "\n")
        except OSError:
            pass


def load(path: Path | str = USAGE_LOG, since: float | None = None,
         until: float | None = None) -> list[Request]:
    """Every recorded request, optionally within a window. Bad lines are skipped, not fatal."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[Request] = []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            req = Request(float(d["t0"]), float(d["t1"]), int(d["prompt"]),
                          int(d["completion"]), str(d.get("run", "")),
                          bool(d.get("stream", False)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # a half-written last line after a kill -9
        # A request that *overlaps* the window counts: a 40-second generation straddling a
        # boundary did real work on both sides, and dropping it would under-count exactly
        # the slow requests that matter most.
        if since is not None and req.t1 < since:
            continue
        if until is not None and req.t0 > until:
            continue
        out.append(req)
    return out


def busy_intervals(requests: list[Request]) -> list[tuple[float, float]]:
    """Merged `[t0, t1)` spans during which at least one request was generating.

    Merged rather than summed, because the server **batches**: thirty concurrent requests
    over one ten-second window are ten seconds of card time, not three hundred. Summing the
    per-request durations is the mistake that makes a busy server look like it ran for
    longer than the day contains.
    """
    if not requests:
        return []
    spans = sorted((r.t0, r.t1) for r in requests if r.t1 > r.t0)
    if not spans:
        return []
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def summarise(requests: list[Request]) -> dict:
    """Counts and rates, with no energy in them — `cost.py` adds that."""
    busy = busy_intervals(requests)
    busy_s = sum(b - a for a, b in busy)
    completion = sum(r.completion for r in requests)
    prompt = sum(r.prompt for r in requests)
    return {
        "requests": len(requests),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "busy_seconds": busy_s,
        # Tokens per second of *card* time, so batching shows up here as a win. Per-request
        # throughput is a different (and much lower) number, and the two get confused.
        "completion_tok_per_s": completion / busy_s if busy_s > 0 else None,
        "first": min((r.t0 for r in requests), default=None),
        "last": max((r.t1 for r in requests), default=None),
    }


def now() -> float:
    return time.time()
