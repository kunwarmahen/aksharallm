"""Every generation, kept — with the training state of the model that produced it.

The decision this file implements: **do not archive checkpoints, archive their output.**

Keeping a copy of the weights every few thousand steps would let you re-run an old model,
and would cost 1.2 GB a time for a run with 40,000 steps in it. But the question people
actually ask is not "can I re-run step 5,000", it is *"is this getting better?"* — and that
question is answered by two pieces of text sitting next to each other, each labelled with
the step and the loss of the model that wrote it. That costs about a kilobyte.

So every generation appends one JSON line to `logs/playground.jsonl` carrying the prompt,
the output, the sampling parameters, and — the part that makes it worth keeping — the
checkpoint's provenance: run, step, best validation loss, smoothed training loss, tokens
seen. `ckpt_last.pt` is overwritten every 500 steps and the file is gone; the record of
what it said at step 7,000 with a val loss of 2.89 is permanent.

:meth:`History.compare` is what that buys you: the same probe, every time it has been run,
oldest step first. Two lines of a table and you can see grammar arrive.

Format and conventions follow `checkpoints/<run>/train_log.jsonl` — one JSON object per
line, appended, never rewritten in place, unparseable lines skipped on read — so the same
habits (and `tail -f`) work on both.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from .checkpoints import repo_root

#: Fields never worth keeping at full length in a log that is read as a table. The output
#: itself is kept whole — it is the thing being recorded.
MAX_PROMPT = 8_000
MAX_OUTPUT = 20_000


class History:
    """The playground's transcript, on disk.

    Trimming happens on write and is deliberately blunt: once the file exceeds `max_records`
    by a quarter it is rewritten with the newest `max_records` lines. Rewriting on every
    append would be silly, and a file that grows without limit is how `logs/` quietly
    becomes the biggest thing in the repo.
    """

    def __init__(self, root: Path | str | None = None, max_records: int = 2000):
        self.root = Path(root).resolve() if root else repo_root()
        self.path = self.root / "logs" / "playground.jsonl"
        self.max_records = max(50, int(max_records))
        self._lock = threading.Lock()

    # ---- writing -----------------------------------------------------------------------
    def append(self, record: dict) -> dict:
        """Add one generation. Returns the record as stored (with its id and timestamps)."""
        now = time.time()
        stored = {
            "id": f"{int(now * 1000):x}-{os.getpid():x}",
            "time": now,
            "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            **record,
        }
        for key, limit in (("prompt", MAX_PROMPT), ("output", MAX_OUTPUT),
                           ("rendered", MAX_PROMPT)):
            value = stored.get(key)
            if isinstance(value, str) and len(value) > limit:
                stored[key] = value[:limit] + f"\n… {len(value) - limit} more characters"

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append: a portal killed mid-write loses at most the line it was
            # writing, and `load` already skips a truncated final line.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(stored, default=str) + "\n")
            self._trim_if_needed()
        return stored

    def _trim_if_needed(self):
        try:
            if not self.path.exists():
                return
            lines = self.path.read_text(errors="replace").splitlines()
            if len(lines) <= self.max_records * 1.25:
                return
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-self.max_records:]) + "\n")
            tmp.replace(self.path)      # atomic, same as the checkpoint writer
        except OSError:
            pass                        # a log that cannot be trimmed must not fail a request

    # ---- reading -----------------------------------------------------------------------
    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                # truncated final line from a kill -9
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def recent(self, limit: int = 50, *, run: str | None = None, mode: str | None = None,
               probe: str | None = None) -> list[dict]:
        """The newest records first, optionally narrowed."""
        records = self.load()
        if run:
            records = [r for r in records if r.get("run") == run]
        if mode:
            records = [r for r in records if r.get("mode") == mode]
        if probe:
            records = [r for r in records if r.get("probe") == probe]
        return list(reversed(records))[:max(1, int(limit))]

    def get(self, record_id: str) -> dict | None:
        for rec in reversed(self.load()):
            if rec.get("id") == record_id:
                return rec
        return None

    def compare(self, probe: str, run: str | None = None) -> dict:
        """One fixed prompt, every time it has been asked, oldest step first.

        This is the whole point of the log. Run the `fluency` probe at step 2,000 and again
        at step 20,000 and this returns both, labelled — which is how you tell a model that
        is learning from a sample that got lucky.
        """
        rows = [r for r in self.load()
                if r.get("probe") == probe and (not run or r.get("run") == run)]
        rows.sort(key=lambda r: ((r.get("step") if r.get("step") is not None else -1),
                                 r.get("time", 0)))
        return {
            "probe": probe,
            "run": run,
            "count": len(rows),
            "runs": sorted({r.get("run") for r in rows if r.get("run")}),
            "rows": [{
                "id": r.get("id"), "iso": r.get("iso"), "run": r.get("run"),
                "checkpoint": r.get("checkpoint"), "step": r.get("step"),
                "best_val": r.get("best_val"), "train_loss": r.get("train_loss"),
                "tokens_seen": r.get("tokens_seen"), "stage": r.get("stage"),
                "device": r.get("device"), "output": r.get("output"),
                "test": r.get("test"), "sampling": r.get("sampling"),
            } for r in rows],
        }

    def probes_seen(self) -> list[dict]:
        """Which fixed prompts have history, and how much — for the compare picker."""
        counts: dict[str, dict] = {}
        for rec in self.load():
            probe = rec.get("probe")
            if not probe:
                continue
            entry = counts.setdefault(probe, {"probe": probe, "count": 0, "runs": set(),
                                              "steps": []})
            entry["count"] += 1
            if rec.get("run"):
                entry["runs"].add(rec["run"])
            if rec.get("step") is not None:
                entry["steps"].append(rec["step"])
        out = []
        for entry in counts.values():
            steps = sorted(entry.pop("steps"))
            out.append({**entry, "runs": sorted(entry["runs"]),
                        "first_step": steps[0] if steps else None,
                        "last_step": steps[-1] if steps else None})
        return sorted(out, key=lambda e: -e["count"])

    def stats(self) -> dict:
        records = self.load()
        return {
            "count": len(records),
            "path": str(self.path.relative_to(self.root))
            if self.path.is_relative_to(self.root) else str(self.path),
            "size": self.path.stat().st_size if self.path.exists() else 0,
            "max_records": self.max_records,
            "oldest": records[0].get("iso") if records else None,
            "newest": records[-1].get("iso") if records else None,
        }


def record_from(stats: dict, *, mode: str, prompt: str, output: str,
                probe: str | None = None, task: str | None = None,
                test: dict | None = None, system: str | None = None) -> dict:
    """Flatten a `Engine.stream` "done" payload into one history row.

    Provenance is flattened to the top level rather than nested, so the file can be read
    with `jq -r '[.step, .probe, .output] | @tsv'` without knowing this module exists.
    """
    prov = dict(stats.get("provenance") or {})
    return {
        "mode": mode,
        "probe": probe,
        "task": task,
        "prompt": prompt,
        "system": system,
        "output": output,
        "device": stats.get("device"),
        "tokens": stats.get("tokens"),
        "tok_per_s": stats.get("tok_per_s"),
        "elapsed_s": stats.get("elapsed_s"),
        "finish": stats.get("finish"),
        "prompt_tokens": stats.get("prompt_tokens"),
        "truncated_tokens": stats.get("truncated_tokens"),
        "sampling": stats.get("params"),
        "test": test,
        # --- the model that said it ---
        "run": prov.get("run"),
        "checkpoint": prov.get("checkpoint"),
        "stage": prov.get("stage"),
        "step": prov.get("step"),
        "best_val": prov.get("best_val"),
        "train_loss": prov.get("train_loss"),
        "tokens_seen": prov.get("tokens_seen"),
        "max_steps": prov.get("max_steps"),
        "params": prov.get("params"),
    }
