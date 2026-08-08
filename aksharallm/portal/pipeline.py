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

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..train import runlog
from .runs import (RUN_NAME_RE, RunError, RunStore, _alive, _cmdline, _read_int,
                   _read_meta, repo_root)

#: stage -> (best-checkpoint filename, log filename, human blurb)
STAGES = {
    "sft":  ("sft_best.pt",  "sft_log.jsonl",  "supervised fine-tune: base → follows instructions"),
    "dpo":  ("dpo_best.pt",  "dpo_log.jsonl",  "preference tuning: sharpen with chosen/rejected pairs"),
    "grpo": ("grpo_best.pt", "grpo_log.jsonl", "RL on the code sandbox: reward = tests pass"),
}

#: What the panel has to say beyond "press Start", because the question a person actually
#: arrives with is *which of these two*, and the cards alone cannot answer it: DPO and GRPO
#: sit side by side, are gated on the same checkpoint, and look interchangeable.
#:
#: `choose` is the one-line decision rule; `metric` names the number to watch once it is
#: running, and `watch_for` the failure that number shows. Kept here rather than in the
#: browser so the API answers the question too -- docs/06-posttraining.md § "Choosing between
#: them" is the long version and must not drift from these lines. Plain text only:
#: the browser inserts these through `escHtml`, so `*emphasis*` renders as asterisks.
GUIDANCE: dict[str, dict[str, str]] = {
    "sft": {
        "choose": "Always. Every route to a model you can talk to goes through here.",
        "metric": "val loss",
        "watch_for": "Not comparable with the base run's val loss — different data, and the "
                     "loss counts assistant tokens only.",
    },
    "dpo": {
        "choose": "Pick DPO when no program can tell whether an answer is right — tone, "
                  "length, helpfulness, refusals. It learns from pairs someone ranked.",
        "metric": "acc — 50% → 65–80%",
        "watch_for": "Past 90% is overfitting the preference set; the model starts hedging "
                     "or rambling. Stop early.",
    },
    "grpo": {
        "choose": "Pick GRPO when a program CAN tell — do the tests pass, is the number "
                  "right. The sandbox computes the reward, so there is nothing to download.",
        "metric": "reward, solved%",
        "watch_for": "Reward flat at zero means no completion ever passed: the task is "
                     "beyond the model, so improve SFT rather than the learning rate.",
    },
}

#: A stage that must build a dataset before it can train, and the file that proves it already
#: has one. `scripts/stage.sh` runs the same check and prepares the data when it is missing --
#: which is a *download*, minutes before step 1, and the reason a stage can sit in pre-flight
#: for a long time. SFT's data is prepared on this machine; DPO's is not.
STAGE_DATA: dict[str, tuple[str, str, str]] = {
    # stage: (file that proves it exists, default recipe, what preparing it costs)
    "sft": ("data/sft/train_tokens.npy", "smoltalk",
            "downloads and tokenizes SmolTalk first"),
    "dpo": ("data/dpo/train_chosen_tokens.npy", "ultrafeedback",
            "downloads and tokenizes UltraFeedback (~61k pairs) first"),
}


class Pipeline:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()
        # For `start(fresh=True)`, which archives the previous attempt with exactly the same
        # rename a base run's "Start fresh…" uses. One implementation of "set this aside".
        self.store = RunStore(self.root)

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

    def _crash(self, base: str, stage: str) -> str | None:
        """The error a dead stage left behind, or None if it did not die.

        `train.pid` is written by the launcher and removed by `scripts/stop.sh`, so a pid
        file naming a process that is gone means the trainer exited on its own without
        being asked to. If it had finished it would have left a checkpoint; the caller
        checks that first. What is left is a crash, and the useful thing to show is the
        last line it printed — for the OOM this catches, that line is the whole diagnosis.
        """
        rdir = self.stage_dir(base, stage)
        if _read_int(rdir / "train.pid") is None:
            return None
        log = _read_meta(rdir / "run.meta").get("log")
        if not log:
            return "the trainer exited during startup (no log recorded)"
        try:
            lines = [ln.strip() for ln in
                     (self.root / log).read_text(errors="replace").splitlines() if ln.strip()]
        except OSError:
            return f"the trainer exited during startup; see {log}"
        return lines[-1] if lines else f"the trainer exited without writing to {log}"

    def _last(self, base: str, stage: str) -> dict:
        """The newest reading worth putting on the card, via the shared log reader.

        This used to take the literal last line of the file, which is wrong the moment a run
        ends: the last line is a `session_end` record, which carries no `step` and no metric.
        So a stage that had just been stopped showed **no step and no number at all** — the
        panel went blank exactly when you were looking at it to find out where it got to.

        `runlog.latest` already solves this for the dashboard (it filters to real step
        records and reads backwards), and using it also gets `max_steps` and `trained_to`,
        which is what `_finished` needs below.
        """
        log = self.stage_dir(base, stage) / STAGES[stage][1]
        if not log.exists():
            return {}
        try:
            return runlog.latest(runlog.load_records(log))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _finished(last: dict) -> bool | None:
        """Did the run reach its budget, as opposed to merely producing a checkpoint?

        None when the log cannot say. The distinction is the whole of the bug below: every
        trainer here writes its `<stage>_best.pt` as soon as anything improves, so a run
        stopped at step 16 of 500 has one — and "a checkpoint exists" was being read as
        "this stage is done".
        """
        trained, budget = last.get("trained_to"), last.get("max_steps")
        if trained is None or not budget:
            return None
        return trained + 1 >= budget

    def _data(self, stage: str) -> dict | None:
        """Whether this stage's dataset is already on disk, and what making it would cost.

        The panel needs this *before* the button is pressed. `scripts/stage.sh dpo` prepares
        `data/dpo/` when it is missing, which means a download — so on this machine, where
        SFT's data exists and DPO's does not, the same-looking Start button means "begin
        training" on one card and "begin a download, then train" on the other.
        """
        spec = STAGE_DATA.get(stage)
        if spec is None:
            return {"needed": False}
        path, recipe, cost = spec
        return {"needed": True, "ready": (self.root / path).exists(),
                "path": path, "recipe": recipe, "cost": cost}

    def _preparing(self, base: str, stage: str) -> str | None:
        """The pre-flight step `scripts/stage.sh` is on, or None if it is not running.

        Read from `launch.pid` + `launch.meta`, the files the script itself writes, through
        the same `RunStore.launcher` the dashboard uses for a base run — so a stage launched
        from a terminal reports its pre-flight here too, and there is one implementation of
        "what is the launcher doing" rather than two that can disagree.
        """
        live = self.store.launcher(self.stage_run(base, stage))
        return (live.get("stage") or "starting") if live else None

    # ---- status -----------------------------------------------------------------------
    def stage_status(self, base: str, stage: str) -> dict:
        pid = self._pid(base, stage)
        done = self.stage_ckpt(base, stage).exists()
        prereq, how = self.prerequisite(base, stage)
        prereq_ok = prereq.exists()
        running = pid is not None
        preparing = None if running else self._preparing(base, stage)
        last = self._last(base, stage)
        finished = self._finished(last)

        if running:
            phase, can_start, reason = "running", False, None
        elif preparing is not None:
            # The launcher is alive and no trainer exists yet. Before `stage.sh` was in
            # `LAUNCH_SCRIPTS` this fell through to "ready", so a DPO launch spent its whole
            # dataset download looking like a button that had done nothing — with Start
            # still enabled, inviting a second launch on top of the first.
            phase, can_start = "preparing", False
            data = self._data(stage)
            reason = (f"{preparing}: {data['cost']}" if preparing == "data" and data
                      and data.get("needed") else f"pre-flight ({preparing})")
        elif not prereq_ok:
            phase = "blocked"
            can_start = False
            reason = f"needs {prereq.relative_to(self.root)} — {how}"
        elif done and finished is False:
            # A checkpoint exists but the budget was never reached: this run was *stopped*,
            # and the only sane offer is to continue it. Reading "a checkpoint exists" as
            # "this stage is finished" is how a GRPO run stopped at step 16 of 500 came back
            # offering "Start fresh…" — which archives the run and restarts at zero. The
            # trainers all write `<stage>_best.pt` the first time anything improves, so
            # *every* interrupted run has one from its first few steps.
            phase, can_start, reason = "stopped", True, None
        elif done:
            phase, can_start, reason = "done", True, None
        elif (err := self._crash(base, stage)) is not None:
            # It ran and it is not running now, and it produced no checkpoint. Saying
            # "ready" here is what made the first SFT attempt look like nothing happened:
            # the button went orange for five seconds and came back, and the traceback sat
            # in the log unread. Start stays enabled — the fix is usually to press it again
            # with a knob changed.
            phase, can_start, reason = "failed", True, err
        else:
            phase, can_start, reason = "ready", True, None

        # each stage's headline number: reward for GRPO, val loss for SFT/DPO
        metric = ({"key": "reward", "value": last.get("reward")} if stage == "grpo"
                  else {"key": "val_loss", "value": last.get("val_loss")})
        return {
            "stage": stage,
            "run": self.stage_run(base, stage),
            "blurb": STAGES[stage][2],
            # blocked | preparing | ready | running | done | failed
            "phase": phase,
            "can_start": can_start,
            "can_stop": running,
            # why it can't start (the disabled tooltip), or — when it failed — what killed it
            "reason": reason,
            "done": done,
            # `done` only means a checkpoint exists. `finished` is whether the budget was
            # reached, and None when the log cannot say — the two are what separate "Resume"
            # from "Start fresh…", and conflating them archives a run someone meant to continue.
            "finished": finished,
            "step_of": last.get("max_steps"),
            "pid": pid,
            "step": last.get("step"),
            "metric": metric,
            "ckpt": str(self.stage_ckpt(base, stage).relative_to(self.root)) if done else None,
            "log": f"train_{self.stage_run(base, stage)}.log",
            # --- what the panel needs in order to be more than three buttons ---
            "guidance": GUIDANCE[stage],
            "data": self._data(stage),
            # Both alignment stages read the SFT checkpoint and neither reads the other, so
            # they are a choice rather than a sequence. Saying so on the card is the whole
            # point: side by side and identically gated, they look interchangeable.
            "alternative": {"dpo": "grpo", "grpo": "dpo"}.get(stage),
            "starts_from": str(prereq.relative_to(self.root)),
            "writes": str(self.stage_ckpt(base, stage).relative_to(self.root)),
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
    def start(self, base: str, stage: str, fresh: bool = False) -> dict:
        """Start a stage. `fresh` sets the previous attempt aside first and begins at step 0.

        Without `fresh`, starting a *finished* stage is a no-op that looks like a bug:
        `stage.sh` passes `--resume auto`, the trainer loads `<stage>_last.pt`, sees the last
        epoch is already done, trains nothing and re-saves several GB. "Re-run" that trains
        zero steps is the wrong answer to the button's own label.

        `fresh` is the same move the dashboard's "Start fresh…" makes for a base run, and it
        uses the same `RunStore.archive`: a *rename* of `checkpoints/<run>` and
        `logs/<run>` to `<run>.<timestamp>`, so a 7 GB fine-tune is set aside instantly,
        nothing is copied and nothing is deleted. The archive keeps its checkpoints, its log
        and its report, and shows up in the run picker under the timestamped name — read-only,
        because no launcher knows it.
        """
        if stage not in STAGES:
            raise RunError(f"unknown stage: {stage}")
        st = self.stage_status(base, stage)
        if st["phase"] == "running":
            raise RunError(f"'{st['run']}' is already running (pid {st['pid']}).")
        if not st["can_start"]:
            raise RunError(st["reason"] or f"cannot start {stage}")

        archived_as = None
        if fresh and self.stage_dir(base, stage).is_dir():
            # Archive BEFORE launching, so the new run opens on an empty directory and
            # `--resume auto` finds nothing to resume — which is what makes it start at 0.
            archived_as = self.store.archive(self.stage_run(base, stage))["archive"]

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
        note = (f"stage.sh {stage} {base}: prepares data if needed, then launches the "
                "trainer (watch the log).")
        if archived_as:
            note = (f"set the previous run aside as '{archived_as}' (checkpoints, log and "
                    f"report all kept), then started {stage} from step 0. " + note)
        return {"ok": True, "action": "start", "stage": stage, "run": self.stage_run(base, stage),
                "pid": proc.pid, "archived": archived_as, "note": note}

    def stop(self, base: str, stage: str) -> dict:
        st = self.stage_status(base, stage)
        if st["phase"] != "running":
            raise RunError(f"'{st['run']}' is not running.")
        script = self.root / "scripts" / "stop.sh"
        subprocess.run(["bash", str(script), self.stage_run(base, stage)],
                       cwd=self.root, timeout=30, capture_output=True)
        return {"ok": True, "action": "stop", "run": self.stage_run(base, stage)}
