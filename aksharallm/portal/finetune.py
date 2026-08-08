"""Fine-tuning jobs, for the portal — the Finetune tab.

Same contract as the Quantize tab: this is a **view over the CLI, not a second
implementation**. Pressing Run shells out to

    python -m aksharallm.train.sft --base <ckpt> --data-dir <dir> --qlora ...

which is exactly what you would type. A job started in the browser and one started in a
terminal write the same adapter to the same place, and either can stop the other.

Why this tab exists at all
--------------------------
LoRA is the first thing in this project where the *interesting* number is a memory
budget rather than a loss curve, and a memory budget is much easier to understand as a
table you can look at before committing an hour of GPU time than as a paragraph. So the
tab leads with `budget()` — what full fine-tuning, LoRA and QLoRA would each cost on the
checkpoint you picked — and only then offers to run one. You can learn the whole
trade-off without training anything.

The three explanations it shows (`WHY`, `METHODS`, `TARGET_BLURBS`) are deliberately in
this file rather than in the JavaScript: they are the same sentences the CLI's `--help`
and `docs/12-lora.md` use, and having one copy means the tab cannot drift from the docs.

Sharing the GPU with a training run
-----------------------------------
A fine-tune wants the card, and so does the trainer. Unlike the playground — where the
model is small and the fallback is a slow tab — a fine-tuning job allocates optimiser
state and activations, and the downside of getting it wrong is the *pretraining run*
dying overnight. So when a run is live the default device becomes the CPU and the panel
says why. `device: cuda` overrides it, which is the right choice when nothing is training
(and the panel says that too).

The irony is worth stating in the UI and does get stated: QLoRA is precisely the
technique that makes fine-tuning fit alongside something else, and this panel still
refuses to do it by default. That is a policy about *this* machine having one card and one
irreplaceable 40,000-step run on it, not a claim about the technique.

Read with: docs/12-lora.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..infer.checkpoints import AdapterStore, CheckpointStore, InferError
from ..lora.inject import PRESET_BLURBS, PRESETS
from ..train import stopfile
from .runs import (MAX_STOP_SECONDS, RunError, _alive, _cmdline, _read_int, repo_root)

#: The one-paragraph answer to "why would I do this?", shown at the top of the tab.
WHY = (
    "Fine-tuning normally means training every weight — and paying for an optimiser state "
    "twice the size of the model. LoRA freezes the model and trains a small low-rank "
    "correction beside it (about 1% as many numbers), so the result is an adapter file of "
    "a few MB instead of a new copy of the model. QLoRA goes further and holds the frozen "
    "model in 4 bits, which is safe precisely because it is frozen and never receives a "
    "gradient. One base model plus several adapters is how you get a chat model and a "
    "Python model without storing two of everything."
)

#: method id -> (label, blurb, extra CLI flags)
METHODS = {
    "full": ("Full fine-tune",
             "Train every weight. The baseline everything else is measured against, and "
             "the only one that needs no explanation — it is also the one that does not "
             "fit once the model grows.", []),
    "lora": ("LoRA",
             "Freeze the weights, train a low-rank correction beside them. ~1% of the "
             "parameters, and the output is an adapter file you can swap.", ["--lora"]),
    "qlora": ("QLoRA",
              "LoRA, plus the frozen base is held in 4-bit NF4. Same adapter, a fraction "
              "of the memory — this is the one that makes a big model fine-tunable on one "
              "card.", ["--qlora", "--qlora-double-quant"]),
}

#: The ranks worth offering, with what each is for.
RANKS = (
    (4, "4 — smallest adapter; enough for a change of tone or format"),
    (8, "8 — the default. Good for teaching a model to follow instructions"),
    (16, "16 — more capacity for a real skill (a new language, a domain)"),
    (32, "32 — rarely better than 16 at this model size; try it and see"),
)

TARGET_BLURBS = {k: PRESET_BLURBS[k] for k in PRESETS}

_JOB_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_.-]+$")

#: Where prepared SFT data is expected to live, and what to run if it is not there.
DATA_HINT = ("python -m aksharallm.data.prepare_sft synthetic "
             "--tokenizer <tokenizer.json> --out-dir data/sft-synthetic --seq-len 256")


class FinetuneJobs:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()
        self.store = CheckpointStore(self.root)
        self.adapters = AdapterStore(self.root)

    # ---- paths ------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.root / "logs" / "finetune"

    @property
    def pid_file(self) -> Path:
        return self.dir / "finetune.pid"

    @property
    def current_file(self) -> Path:
        return self.dir / "current.json"

    def log_path(self, job: str) -> Path:
        return self.dir / f"{job}.log"

    @property
    def stop_file(self) -> Path:
        """Where a bounded stop for the running job is queued.

        Here rather than in the adapter's output directory, which is the base model's run
        directory: a file called STOP in *there* is the pretrainer's, and one fine-tune
        stopping a six-day pretraining run is not a mistake worth leaving available.
        """
        return self.dir / "STOP"

    # ---- what can be fine-tuned -------------------------------------------------------
    def checkpoints(self) -> list[dict]:
        """Every checkpoint, with whether it is already quantized noted.

        An already-quantized checkpoint is a perfectly good QLoRA base — better, in fact,
        because it was quantized with GPTQ or AWQ rather than the plain rounding `--qlora`
        does on the fly. So unlike the Quantize tab, nothing is excluded here.
        """
        out = []
        for ck in self.store.list():
            quantized = bool(re.search(r"-(rtn|gptq|awq|qat)-(int[48]|nf4)-", ck.name))
            out.append({
                "id": ck.rel, "run": ck.run, "name": ck.name, "rel": ck.rel,
                "size": ck.size, "step": ck.step, "best_val": ck.best_val,
                "params": ck.params, "stage": ck.stage, "quantized": quantized,
                "tokenizer": ck.tokenizer, "tokenizer_ok": ck.tokenizer_ok,
                "error": ck.error,
            })
        return out

    def datasets(self) -> list[dict]:
        """Prepared SFT datasets under `data/`, found by their file layout.

        `prepare_sft` writes four .npy files into one directory; a directory holding
        `train_tokens.npy` is therefore an SFT dataset and nothing else is.
        """
        out = []
        base = self.root / "data"
        if not base.is_dir():
            return out
        for d in sorted(base.iterdir()):
            tok = d / "train_tokens.npy"
            if not (d.is_dir() and tok.is_file()):
                continue
            try:
                import numpy as np

                arr = np.load(tok, mmap_mode="r")
                shape = list(arr.shape)
            except Exception:  # noqa: BLE001
                shape = []
            out.append({
                "id": f"data/{d.name}", "name": d.name,
                "blocks": shape[0] if shape else None,
                "seq_len": shape[1] if len(shape) > 1 else None,
                "bytes": sum(f.stat().st_size for f in d.glob("*.npy")),
            })
        return out

    # ---- the budget table -------------------------------------------------------------
    def budget(self, ckpt_id: str, ranks=(8, 16), targets: str = "all-linear") -> dict:
        """What each strategy would cost on this checkpoint, without training anything.

        Built from the real shapes, not a formula on the parameter count, so a config with
        tied embeddings or an unusual d_ff gives the number it will actually give.
        """
        from ..config import ModelConfig
        from ..lora.inject import LoRAConfig, apply_lora
        from ..lora.setup import describe_memory
        from ..model.transformer import Transformer
        from ..quant.convert import quantize_model
        from ..quant.qtensor import QuantScheme

        import torch

        info = self.store.get(ckpt_id)
        ck = torch.load(info.path, map_location="cpu", weights_only=False, mmap=True)
        mcfg = ModelConfig(**ck["model_config"])
        del ck

        rows = []
        n = sum(p.numel() for p in Transformer(mcfg).parameters())
        rows.append({"strategy": "full", "label": "Full fine-tune", "r": None,
                     "trainable_params": n, "weight_bytes": n * 4, "grad_bytes": n * 4,
                     "optimizer_bytes": n * 8, "total_bytes": n * 16})
        for kind in ("lora", "qlora"):
            for r in ranks:
                m = Transformer(mcfg)
                if kind == "qlora":
                    quantize_model(m, QuantScheme(bits=4, group_size=64, dtype="nf4",
                                                  double_quant=True, method="rtn"))
                apply_lora(m, LoRAConfig(r=r, targets=targets))
                mem = describe_memory(m)
                rows.append({
                    "strategy": kind, "label": f"{METHODS[kind][0]} r={r}", "r": r,
                    "trainable_params": mem["trainable_params"],
                    "weight_bytes": mem["frozen_bytes"] + mem["trainable_bytes"],
                    "grad_bytes": mem["grad_bytes"],
                    "optimizer_bytes": mem["optimizer_bytes"],
                    "total_bytes": mem["total_bytes"]})
                del m
        best = min(rows, key=lambda r: r["total_bytes"])
        return {
            "checkpoint": ckpt_id, "params": n, "targets": targets, "rows": rows,
            "headline": (
                f"{rows[0]['total_bytes'] / 1e6:,.0f} MB to fine-tune every weight, "
                f"{best['total_bytes'] / 1e6:,.0f} MB with {best['label']} — "
                f"{rows[0]['total_bytes'] / max(best['total_bytes'], 1):.0f}x less."),
            "note": ("Weights, gradients and optimiser state only. Activations depend on "
                     "the batch size and are the one part LoRA does not shrink: the "
                     "forward pass still runs through every layer at full width."),
        }

    # ---- process state ----------------------------------------------------------------
    def _pid(self) -> int | None:
        pid = _read_int(self.pid_file)
        if pid and _alive(pid) and "aksharallm.train.sft" in _cmdline(pid):
            return pid
        return None

    def _current(self) -> dict:
        try:
            return json.loads(self.current_file.read_text())
        except (OSError, ValueError):
            return {}

    def training(self) -> list[str]:
        """Pretraining runs that are live — a fine-tune would share the card with them."""
        live = []
        for run in self.store.dirs():
            pid = _read_int(run / "train.pid")
            if pid and _alive(pid):
                live.append(run.name)
        return live

    def plan_device(self, requested: str | None = None) -> dict:
        busy = self.training()
        if requested in ("cuda", "cpu"):
            return {"device": requested, "training": busy, "forced": True,
                    "reason": f"you chose {requested}" + (
                        f"; note {', '.join(busy)} is training" if busy and requested == "cuda"
                        else "")}
        if busy:
            return {"device": "cpu", "training": busy, "forced": False,
                    "reason": f"{', '.join(busy)} is training — fine-tuning on the CPU so "
                              "the run cannot be taken down. Slow, but it will not cost you "
                              "a week of pretraining. (Yes: QLoRA is the technique that "
                              "would fit alongside it. This machine has one card and one "
                              "irreplaceable run, so the default stays cautious.)"}
        return {"device": "cuda", "training": [], "forced": False,
                "reason": "nothing is training, so the GPU is free"}

    def status(self, tail: int = 200) -> dict:
        pid = self._pid()
        cur = self._current()
        running = pid is not None
        log = []
        if cur.get("job"):
            log = self._tail(self.log_path(cur["job"]), tail)
        if not running and cur and cur.get("state") == "running":
            # The process is gone but nobody wrote the ending: decide from the artifacts.
            done = cur.get("out") and (self.root / cur["out"]).exists()
            cur = {**cur, "state": "done" if done else "failed"}
        return {
            "running": running,
            "pid": pid,
            "current": cur or None,
            "stop": self.stop_request() if running else None,
            "can_bound": running,
            "log": log,
            "device": self.plan_device(),
            "why": WHY,
            "methods": [{"id": k, "label": v[0], "blurb": v[1]} for k, v in METHODS.items()],
            "ranks": [{"value": r, "label": lbl} for r, lbl in RANKS],
            "targets": [{"id": k, "layers": ", ".join(v), "blurb": TARGET_BLURBS[k]}
                        for k, v in PRESETS.items()],
            "datasets": self.datasets(),
            "data_hint": DATA_HINT,
            "adapters": [a.as_dict() for a in self.adapters.list()],
        }

    @staticmethod
    def _tail(path: Path, lines: int) -> list[str]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    # ---- actions ----------------------------------------------------------------------
    def start(self, spec: dict) -> dict:
        if self._pid():
            raise RunError("a fine-tuning job is already running — stop it first.")

        ref = str(spec.get("checkpoint") or "").strip()
        if not ref:
            raise RunError("pick a checkpoint to fine-tune")
        try:
            info = self.store.get(self.store.identify(ref))
        except (InferError, Exception) as exc:  # noqa: BLE001
            raise RunError(f"unknown checkpoint {ref!r}: {exc}")
        if info.error:
            raise RunError(f"{info.rel} cannot be loaded: {info.error}")
        if not info.tokenizer:
            raise RunError(
                f"{info.rel} does not record which tokenizer it was trained with, so an "
                "adapter trained on it could not be decoded safely.")

        method = str(spec.get("method") or "qlora")
        if method not in METHODS:
            raise RunError(f"unknown method: {method}")

        data_dir = str(spec.get("data_dir") or "").strip()
        if not data_dir:
            raise RunError("pick a prepared SFT dataset")
        known = {d["id"] for d in self.datasets()}
        if data_dir not in known:
            raise RunError(f"{data_dir} is not a prepared SFT dataset. "
                           f"Known: {', '.join(sorted(known)) or 'none'}. Make one with:  "
                           f"{DATA_HINT}")

        r = int(spec.get("r") or 8)
        if r < 1 or r > 256:
            raise RunError("rank must be between 1 and 256")
        targets = str(spec.get("targets") or "all-linear")
        if targets not in PRESETS:
            raise RunError(f"unknown target preset: {targets}")
        epochs = max(1, min(int(spec.get("epochs") or 2), 20))
        plan = self.plan_device(spec.get("device"))

        self.dir.mkdir(parents=True, exist_ok=True)
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{info.run}-{method}"
        if not _JOB_RE.match(job):
            raise RunError(f"bad job name: {job}")

        # The adapter lands beside its base, in the base's own run directory, so
        # `--list-adapters` and the Playground picker find it without configuration.
        out_dir = self.root / "checkpoints" / info.run
        tok = info.tokenizer or ""
        # A STOP left over from the last job would end this one at step 0 — the same trap
        # phase2.sh clears before a launch, for the same reason.
        self.stop_file.unlink(missing_ok=True)
        cmd = [sys.executable, "-u", "-m", "aksharallm.train.sft",
               "--base", str(info.path), "--data-dir", str(self.root / data_dir),
               "--tokenizer", tok, "--out-dir", str(out_dir),
               "--stop-file", str(self.stop_file),
               "--epochs", str(epochs), "--device", plan["device"]]
        cmd += METHODS[method][2]
        if method != "full":
            cmd += ["--lora-r", str(r), "--lora-targets", targets]
        if spec.get("lr"):
            cmd += ["--lr", str(float(spec["lr"]))]
        if spec.get("batch_size"):
            cmd += ["--batch-size", str(int(spec["batch_size"]))]

        log = self.log_path(job)
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                cmd, cwd=self.root, env={**os.environ}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid_file.write_text(str(proc.pid))
        suffix = ".lora.pt" if method != "full" else ".pt"
        current = {
            "job": job, "state": "running", "pid": proc.pid,
            "checkpoint": info.rel, "method": method, "r": r if method != "full" else None,
            "targets": targets if method != "full" else None,
            "data_dir": data_dir, "epochs": epochs,
            "device": plan["device"], "device_reason": plan["reason"],
            "started": time.time(),
            "out": f"checkpoints/{info.run}/sft_best{suffix}",
            "cmd": " ".join(cmd[2:]),   # drop the interpreter path; keep it readable
        }
        self.current_file.write_text(json.dumps(current))
        return {"ok": True, "action": "start", **current}

    def stop(self, mode: str = "now", steps: int | None = None,
             seconds: int | None = None) -> dict:
        """Stop the running fine-tune now, or bound it in steps or wall-clock.

        `now` is a SIGTERM, which `aksharallm.train.sft` catches: it finishes the step,
        evaluates, and saves `sft_last`/`sft_best`, so stopping still leaves a usable
        adapter. `at`/`in` write the same STOP file the trainer polls and return at once.
        `cancel` removes it.
        """
        if mode not in ("now", "at", "in", "cancel"):
            raise RunError(f"unknown stop mode: {mode!r}")
        pid = self._pid()
        if not pid:
            raise RunError("no fine-tuning job is running.")

        if mode == "cancel":
            if not self.stop_request():
                raise RunError("no stop is queued for this fine-tune.")
            self.stop_file.unlink(missing_ok=True)
            return {"ok": True, "action": "stop:cancel", "pid": pid,
                    "note": "queued stop withdrawn; the fine-tune runs to its last epoch."}

        if mode in ("at", "in"):
            if mode == "at":
                if not steps or steps < 1:
                    raise RunError("a bounded stop needs a positive step number.")
                request = stopfile.StopRequest(step=int(steps))
                note = f"queued: finish step {steps}, then evaluate, save and exit."
            else:
                if not seconds or seconds < 1:
                    raise RunError("a timed stop needs a duration of at least one second.")
                if seconds > MAX_STOP_SECONDS:
                    raise RunError(f"a timed stop is capped at {MAX_STOP_SECONDS // 3600} "
                                   "hours — bound it by steps instead.")
                request = stopfile.StopRequest(deadline=time.time() + int(seconds))
                note = (f"queued: {stopfile.fmt_left(seconds)} more, then evaluate, save "
                        "and exit.")
            stopfile.write(self.stop_file, request)
            return {"ok": True, "action": f"stop:{mode}", "pid": pid, "note": note}

        os.kill(pid, 15)
        cur = self._current()
        if cur:
            self.current_file.write_text(json.dumps({**cur, "state": "stopped"}))
        return {"ok": True, "action": "stop", "pid": pid,
                "note": "finishing the step in flight, then saving the adapter and exiting."}

    def stop_request(self) -> dict | None:
        """The stop queued for the running job, in the same shape the run panel uses."""
        req = stopfile.read(self.stop_file)
        if req is None:
            return None
        return {"target": req.step, "deadline": req.deadline, "now": req.now,
                "label": req.describe()}
