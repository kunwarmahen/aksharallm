"""The post-training pipeline, for the portal: SFT -> DPO / GRPO, with dependency gating.

The pretraining `RunStore` can't model these — they have no `configs/<run>.yaml`, they write
`sft_log.jsonl`/`dpo_log.jsonl`/`grpo_log.jsonl` instead of `train_log.jsonl`, and they have
*prerequisites* (you can't align a model you haven't fine-tuned). So this is a small parallel
reader, and — like everything else in the portal — it never trains anything itself: it shells
out to `scripts/stage.sh` (start) and `scripts/stop.sh` (stop), the same scripts you'd run by
hand. The dependency gate is enforced *in the script too*, so a stage can't start without its
prerequisite whether it's launched from here or a terminal.

    base (checkpoints/<base>/ckpt_best.pt)
        └─ SFT  (checkpoints/<base>-sft/sft_best.pt)
              ├─ DPO   (checkpoints/<base>-dpo/dpo_best.pt)
              └─ GRPO  (checkpoints/<base>-grpo/grpo_best.pt)

Read with: docs/09-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .runs import RUN_NAME_RE, RunError, _alive, _cmdline, _read_int, repo_root

#: stage -> (best-checkpoint filename, log filename, human blurb)
STAGES = {
    "sft":  ("sft_best.pt",  "sft_log.jsonl",  "supervised fine-tune: base → follows instructions"),
    "dpo":  ("dpo_best.pt",  "dpo_log.jsonl",  "preference tuning: sharpen with chosen/rejected pairs"),
    "grpo": ("grpo_best.pt", "grpo_log.jsonl", "RL on the code sandbox: reward = tests pass"),
}


class Pipeline:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()

    # ---- paths -----------------------------------------------------------------------
    def base_ckpt(self, base: str) -> Path:
        return self.root / "checkpoints" / base / "ckpt_best.pt"

    def stage_run(self, base: str, stage: str) -> str:
        return f"{base}-{stage}"

    def stage_dir(self, base: str, stage: str) -> Path:
        return self.root / "checkpoints" / self.stage_run(base, stage)

    def stage_ckpt(self, base: str, stage: str) -> Path:
        return self.stage_dir(base, stage) / STAGES[stage][0]

    # ---- gating: the one rule ---------------------------------------------------------
    def prerequisite(self, base: str, stage: str) -> tuple[Path, str]:
        """(file that must exist, how to make it) for a stage to be startable."""
        if stage == "sft":
            return self.base_ckpt(base), f"train a base model first (start '{base}')"
        # dpo and grpo both hang off the SFT checkpoint
        return self.stage_ckpt(base, "sft"), f"run SFT first (start '{base} · sft')"

    # ---- process state ----------------------------------------------------------------
    def _pid(self, base: str, stage: str) -> int | None:
        pid = _read_int(self.stage_dir(base, stage) / "train.pid")
        if pid and _alive(pid):
            args = _cmdline(pid)
            if f"aksharallm.train.{stage}" in args:
                return pid
        return None

    def _last(self, base: str, stage: str) -> dict:
        """The most recent logged line (step + the stage's headline metric)."""
        log = self.stage_dir(base, stage) / STAGES[stage][1]
        if not log.exists():
            return {}
        import json
        last = {}
        try:
            for line in log.read_text().splitlines():
                line = line.strip()
                if line:
                    last = json.loads(line)
        except (OSError, ValueError):
            return {}
        return last

    # ---- status -----------------------------------------------------------------------
    def stage_status(self, base: str, stage: str) -> dict:
        pid = self._pid(base, stage)
        done = self.stage_ckpt(base, stage).exists()
        prereq, how = self.prerequisite(base, stage)
        prereq_ok = prereq.exists()
        running = pid is not None

        if running:
            phase, can_start, reason = "running", False, None
        elif not prereq_ok:
            phase = "blocked"
            can_start = False
            reason = f"needs {prereq.relative_to(self.root)} — {how}"
        else:
            phase = "done" if done else "ready"
            can_start = True
            reason = None

        last = self._last(base, stage)
        # each stage's headline number: reward for GRPO, val loss for SFT/DPO
        metric = ({"key": "reward", "value": last.get("reward")} if stage == "grpo"
                  else {"key": "val_loss", "value": last.get("val_loss")})
        return {
            "stage": stage,
            "run": self.stage_run(base, stage),
            "blurb": STAGES[stage][2],
            "phase": phase,          # blocked | ready | running | done
            "can_start": can_start,
            "can_stop": running,
            "reason": reason,        # why it can't start (shown as the disabled tooltip)
            "done": done,
            "pid": pid,
            "step": last.get("step"),
            "metric": metric,
            "ckpt": str(self.stage_ckpt(base, stage).relative_to(self.root)) if done else None,
            "log": f"train_{self.stage_run(base, stage)}.log",
        }

    def status(self, base: str) -> dict:
        if not RUN_NAME_RE.match(base or ""):
            raise RunError(f"invalid base run: {base!r}")
        return {
            "base": base,
            "base_ready": self.base_ckpt(base).exists(),
            "stages": [self.stage_status(base, s) for s in STAGES],
        }

    # ---- actions (shell out to the scripts) -------------------------------------------
    def start(self, base: str, stage: str) -> dict:
        if stage not in STAGES:
            raise RunError(f"unknown stage: {stage}")
        st = self.stage_status(base, stage)
        if st["phase"] == "running":
            raise RunError(f"'{st['run']}' is already running (pid {st['pid']}).")
        if not st["can_start"]:
            raise RunError(st["reason"] or f"cannot start {stage}")

        script = self.root / "scripts" / "stage.sh"
        if not script.exists():
            raise RunError(f"missing launcher: {script}")
        log = self.root / "logs" / self.stage_run(base, stage)
        log.mkdir(parents=True, exist_ok=True)
        launch_log = log / "portal_launch.log"
        with open(launch_log, "wb") as fh:
            proc = subprocess.Popen(
                ["bash", str(script), stage, base],
                cwd=self.root, env={**os.environ}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        return {"ok": True, "action": "start", "stage": stage, "run": self.stage_run(base, stage),
                "pid": proc.pid, "note": f"stage.sh {stage} {base}: prepares data if needed, "
                                         "then launches the trainer (watch the log)."}

    def stop(self, base: str, stage: str) -> dict:
        st = self.stage_status(base, stage)
        if st["phase"] != "running":
            raise RunError(f"'{st['run']}' is not running.")
        script = self.root / "scripts" / "stop.sh"
        subprocess.run(["bash", str(script), self.stage_run(base, stage)],
                       cwd=self.root, timeout=30, capture_output=True)
        return {"ok": True, "action": "stop", "run": self.stage_run(base, stage)}
