"""The Context tab's back end: how far this model can read, and what would extend it.

Shape of this one, and why it differs from its neighbours
----------------------------------------------------------
`quantize`, `finetune`, `evals` and `synth` shell out to a CLI and stream a log, because
those jobs run for minutes to hours. `interp` runs inline on the resident model, because a
logit lens is one forward pass.

Long context sits between the two, and it is split accordingly:

* **`plan`** is pure arithmetic — it answers "what would extending this checkpoint change?"
  without loading a single weight. That is the panel a layman should meet first: the whole
  trade-off is legible before anything is spent.
* **`curve`, `sweep` and `needle`** are real forward passes over long windows, so they run
  as a **background job**, one at a time, exactly like the Quantize tab — and with the same
  device policy, because a 4,096-token forward holds half a gigabyte of logits and the
  training run that owns the card has about three.

Results are the JSON files `python -m aksharallm.longctx` already writes into
`logs/longctx/`, so the tab and the terminal read the same measurements and neither is the
authority. Nothing here is recomputed for display.

Read with: docs/19-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

from pathlib import Path

from ..infer.checkpoints import InferError
from ..longctx.extend import default_out_name, describe, plan_extension
from ..model.rope import METHODS

#: Where the CLI leaves its JSON. The tab reads this directory and nothing else.
RESULTS = "logs/longctx"

#: One job at a time, same contract as the Quantize tab.
PID_FILE = "logs/longctx/job.pid"
LOG_FILE = "logs/longctx/job.log"

#: Lengths offered in the UI. Powers of two so the "x the trained window" label is exact.
LENGTHS = (512, 1024, 2048, 4096, 8192)


class LongContext:
    """Everything the Context tab can ask for."""

    def __init__(self, store, runs=None, root: Path | None = None):
        self.store = store
        self.runs = runs
        self.root = Path(root) if root else Path.cwd()

    # ---- the free part: arithmetic, no weights ------------------------------------------

    def overview(self, ckpt_id: str | None = None) -> dict:
        """Checkpoints, their current window, and whether a job is running."""
        cks = [c.as_dict() for c in self.store.list()]
        out = {
            "checkpoints": cks,
            "methods": [m for m in METHODS],
            "lengths": list(LENGTHS),
            "job": self.status(),
            "results": self.results(),
            "training": self._training(),
        }
        if ckpt_id:
            out["current"] = self.describe_checkpoint(ckpt_id)
        return out

    def describe_checkpoint(self, ckpt_id: str) -> dict:
        """What this checkpoint's context looks like right now — from the metadata only.

        `CheckpointStore` reads a `.pt` header with `mmap=True`, so this costs milliseconds
        and never loads weights. It is what lets the tab render the moment it opens, even
        while a training run owns the card.
        """
        info = self.store.get(ckpt_id)
        if info is None or info.error:
            raise InferError(f"cannot read {ckpt_id!r}: {info.error if info else 'no such checkpoint'}")
        scaling = info.rope_scaling or {"type": "none", "factor": 1.0}
        return {
            "checkpoint": ckpt_id,
            "trained_window": info.trained_window,
            "addressable": info.max_seq_len,
            "scaling": scaling,
            "extended": scaling.get("type") not in (None, "none"),
            "window": info.attn_window,
            "sinks": info.attn_sinks,
        }

    def plan(self, ckpt_id: str, method: str, factor: float) -> dict:
        """What `extend` *would* do. Pure, instant, and safe to call on every keystroke."""
        info = self.store.get(ckpt_id)
        if info is None or info.max_seq_len is None:
            raise InferError(f"{ckpt_id!r} does not record a context window")
        # `plan_extension` only ever reads these three keys, and `Checkpoint` already has
        # them from the header — so planning stays free rather than opening a 3.6 GB file.
        before = {"max_seq_len": info.max_seq_len, "rope_scaling": info.rope_scaling,
                  "attn_window": info.attn_window, "attn_sinks": info.attn_sinks}
        after = plan_extension(before, method, float(factor))
        return {
            "before": {"max_seq_len": before.get("max_seq_len"),
                       "rope_scaling": before.get("rope_scaling")},
            "after": {"max_seq_len": after.get("max_seq_len"),
                      "rope_scaling": after.get("rope_scaling")},
            "changes": describe(before, after),
            "weights_change": False,   # stated explicitly; it is the surprising part
            "advice": self._advice(method, float(factor)),
            # Named from `default_out_name`, the same function the CLI writes with — a
            # confirmation dialog quoting a path that turns out not to be the one used is
            # worse than no dialog at all.
            "out_name": (None if method == "none"
                         else default_out_name(info.path, method, float(factor)).name),
            "exists": (False if method == "none" else
                       default_out_name(info.path, method, float(factor)).exists()),
        }

    @staticmethod
    def _advice(method: str, factor: float) -> str:
        """One sentence, from our own measurements. The layman's version of the sweep table."""
        if method == "none":
            return "No scaling — the model falls off a cliff one token past its trained window."
        if method == "linear":
            return ("Simple, and it damages short contexts badly: on our 300M it took the "
                    "in-window loss from 2.356 to 3.035 at 2x, and to 4.379 at 4x.")
        if method == "ntk":
            return ("Best value up to about 2x — on our 300M, doubling the context cost "
                    "0.009 nats in-window (2.356 → 2.365). Effectively free."
                    if factor <= 2 else
                    "Past ~3.5x NTK's tilt runs out and grows a cliff of its own — on our "
                    "300M at 4x, at position 3,584 (loss 2.895 against YaRN's 2.464). "
                    "Try YaRN.")
        if method == "yarn":
            return ("The one that holds up furthest. Our 300M extended 4x this way scores "
                    "92.5% on the needle test at 4,096 tokens — four times the window its "
                    "weights were trained on, with nothing retrained.")
        return ("Recomputes the factor from each input's length, so prompts inside the "
                "original window are bit-for-bit unscaled. Stateful — read doc 18 first.")

    # ---- the measured part: a background job ---------------------------------------------

    def start(self, kind: str, ckpt_id: str, **opts) -> dict:
        """Launch a measurement detached. One job at a time.

        `flash` is the odd one out: it benchmarks the *kernel* rather than a checkpoint, so
        it runs `aksharallm.model.flash` instead and takes no model at all. It lives on this
        tab because it answers the other half of the same question — reading further costs
        attention, and attention is quadratic in how far you read.
        """
        if kind not in ("curve", "sweep", "needle", "flash", "extend"):
            raise InferError(f"unknown measurement {kind!r}")
        if self.status()["running"]:
            raise InferError("a measurement is already running — stop it or wait")

        if kind == "flash":
            if self._training():
                raise InferError(
                    "the kernel benchmark needs the GPU, and a run is training on it. "
                    "This is the one measurement here with no CPU fallback — the whole "
                    "point of it is what the card does.")
            cmd = [sys.executable, "-m", "aksharallm.model.flash",
                   "--seqlens", "512", "1024", "2048", "4096"]
            return self._spawn(cmd, kind, "cuda", "the card is free")

        if kind == "extend":
            # Writing a 3.6 GB copy is disk work; the device flag is irrelevant and `cpu`
            # keeps it away from a card someone may be using. No `--out`: the CLI names the
            # file from `default_out_name`, so no path ever arrives from a browser.
            method = str(opts.get("method") or "yarn")
            if method == "none":
                raise InferError("'none' removes a scaling rather than writing an extension")
            cmd = [sys.executable, "-m", "aksharallm.longctx", "extend", ckpt_id,
                   "--device", "cpu", "--quiet",
                   "--method", method, "--factor", str(opts.get("factor") or 4.0)]
            return self._spawn(cmd, kind, "cpu", "writing a checkpoint is disk work")

        cmd = [sys.executable, "-m", "aksharallm.longctx", kind, ckpt_id, "--quiet"]
        # The device decision is made here rather than offered as a checkbox: a long
        # forward pass beside a training run is how you lose a six-day run, and the tab
        # says which one it picked and why rather than silently choosing.
        device = "cpu" if self._training() else "cuda"
        cmd += ["--device", device]
        if device == "cuda":
            cmd.append("--force-gpu")   # we already checked; skip the CLI's own guard

        for flag, key in (("--len", "length"), ("--factor", "factor"),
                          ("--windows", "windows"), ("--bucket", "bucket"),
                          ("--trials", "trials"), ("--window", "window"),
                          ("--sinks", "sinks")):
            if opts.get(key) is not None:
                cmd += [flag, str(opts[key])]
        if opts.get("methods"):
            cmd += ["--methods", *[str(m) for m in opts["methods"]]]
        if opts.get("lengths"):
            cmd += ["--lengths", *[str(n) for n in opts["lengths"]]]

        return self._spawn(cmd, kind, device,
                           "a run is training, so this is on the CPU and will be slow"
                           if device == "cpu" else "the card is free")

    def _spawn(self, cmd: list[str], kind: str, device: str, why: str) -> dict:
        log = self.root / LOG_FILE
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        handle.write(f"$ {' '.join(cmd)}\n\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=self.root, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        (self.root / PID_FILE).write_text(str(proc.pid))
        return {"running": True, "pid": proc.pid, "kind": kind, "device": device,
                "why_device": why}

    def stop(self) -> dict:
        st = self.status()
        if not st["running"]:
            return {"running": False, "stopped": False}
        try:
            os.kill(st["pid"], signal.SIGTERM)
        except OSError:
            pass
        return {"running": False, "stopped": True}

    def status(self) -> dict:
        pid_file = self.root / PID_FILE
        log = self.root / LOG_FILE
        tail = log.read_text()[-4000:] if log.exists() else ""
        if not pid_file.exists():
            return {"running": False, "pid": None, "log": tail}
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            return {"running": False, "pid": None, "log": tail}
        return {"running": True, "pid": pid, "log": tail}

    # ---- results ---------------------------------------------------------------------------

    def results(self, limit: int = 30) -> list[dict]:
        """Every measurement on disk, newest first. Headline numbers only — the full curve
        is fetched by `result()` when a row is opened, because a sweep's JSON carries every
        bucket of every method and there is no reason to send four of them to draw a list."""
        folder = self.root / RESULTS
        if not folder.is_dir():
            return []
        rows = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:limit]:
            try:
                blob = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(self._headline(path, blob))
        return rows

    def result(self, name: str) -> dict:
        """One measurement in full. `name` is a bare filename — never a path."""
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise InferError("not a result name")
        path = self.root / RESULTS / name
        if not path.is_file():
            raise InferError(f"no result {name!r}")
        return json.loads(path.read_text())

    @staticmethod
    def _headline(path: Path, blob: dict) -> dict:
        row = {"name": path.name, "when": int(path.stat().st_mtime),
               "checkpoint": blob.get("checkpoint")}
        if "rows" in blob:            # a sweep
            row["kind"] = "sweep"
            row["seq_len"] = blob.get("seq_len")
            row["trained"] = blob.get("trained")
            row["methods"] = [
                {"method": r["method"], "loss": r["curve"]["loss"],
                 "cliff": (r.get("cliff") or {}).get("position")}
                for r in blob["rows"]]
        elif "grid" in blob:          # a needle sweep
            row["kind"] = "needle"
            row["accuracy"] = blob.get("accuracy")
            row["chance"] = blob.get("chance")
            row["lengths"] = blob.get("lengths")
        else:
            row["kind"] = "curve"
            row["loss"] = blob.get("loss")
        return row

    # ---- device policy ------------------------------------------------------------------

    def _training(self) -> str | None:
        """The name of a run currently holding the card, or None."""
        folder = self.root / "checkpoints"
        if not folder.is_dir():
            return None
        for pid_file in folder.glob("*/train.pid"):
            try:
                os.kill(int(pid_file.read_text().strip()), 0)
                return pid_file.parent.name
            except (ValueError, OSError):
                continue
        return None
