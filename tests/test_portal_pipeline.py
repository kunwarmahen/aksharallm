"""The post-training pipeline's gating state machine: base -> SFT -> DPO/GRPO.

Pure filesystem logic (which checkpoint exists gates which stage), so it's fast and doesn't
launch anything. The scripts enforce the same gate independently; this pins the portal's
view of it.
"""

from pathlib import Path

import pytest

from aksharallm.portal.pipeline import Pipeline
from aksharallm.portal.runs import RunError


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def stages(pl: Pipeline, base: str) -> dict:
    return {s["stage"]: s for s in pl.status(base)["stages"]}


def test_nothing_trained_everything_blocked(tmp_path):
    pl = Pipeline(tmp_path)
    st = stages(pl, "small-code")
    assert st["sft"]["phase"] == "blocked"
    assert st["dpo"]["phase"] == "blocked"
    assert st["grpo"]["phase"] == "blocked"
    assert not any(st[s]["can_start"] for s in st)
    # the reason points at the missing prerequisite
    assert "ckpt_best.pt" in st["sft"]["reason"]
    assert "sft" in st["grpo"]["reason"].lower()


def test_base_unlocks_only_sft(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    pl = Pipeline(tmp_path)
    st = stages(pl, "small-code")
    assert st["sft"]["phase"] == "ready" and st["sft"]["can_start"]
    # DPO and GRPO still need SFT
    assert st["dpo"]["phase"] == "blocked" and not st["dpo"]["can_start"]
    assert st["grpo"]["phase"] == "blocked" and not st["grpo"]["can_start"]


def test_sft_unlocks_dpo_and_grpo(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    pl = Pipeline(tmp_path)
    st = stages(pl, "small-code")
    assert st["sft"]["phase"] == "done"          # its best checkpoint exists
    assert st["dpo"]["phase"] == "ready" and st["dpo"]["can_start"]
    assert st["grpo"]["phase"] == "ready" and st["grpo"]["can_start"]


def test_done_when_stage_checkpoint_exists(tmp_path):
    for f in ("small-code/ckpt_best.pt", "small-code-sft/sft_best.pt",
              "small-code-grpo/grpo_best.pt"):
        _touch(tmp_path / "checkpoints" / f)
    st = stages(Pipeline(tmp_path), "small-code")
    assert st["grpo"]["phase"] == "done" and st["grpo"]["done"]
    assert st["grpo"]["can_start"]  # done but re-runnable


def test_cannot_start_a_blocked_stage(tmp_path):
    pl = Pipeline(tmp_path)
    with pytest.raises(RunError) as e:
        pl.start("small-code", "grpo")
    assert "sft" in str(e.value).lower()


def test_cannot_stop_what_isnt_running(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    with pytest.raises(RunError):
        Pipeline(tmp_path).stop("small-code", "grpo")


def test_invalid_base_name_rejected(tmp_path):
    with pytest.raises(RunError):
        Pipeline(tmp_path).status("../etc")


def test_grpo_headline_metric_is_reward(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    d = tmp_path / "checkpoints" / "small-code-grpo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "grpo_log.jsonl").write_text('{"step": 5, "reward": 0.42}\n')
    st = stages(Pipeline(tmp_path), "small-code")
    assert st["grpo"]["metric"]["key"] == "reward"
    assert st["grpo"]["metric"]["value"] == 0.42
    assert st["sft"]["metric"]["key"] == "val_loss"


def _crashed_sft(tmp_path, last_line: str = "torch.OutOfMemoryError: CUDA out of memory."):
    """An SFT run that started, died, and left no checkpoint — the state the first real SFT
    attempt was actually in while the portal showed 'ready'."""
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    d = tmp_path / "checkpoints" / "small-code-sft"
    d.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "logs" / "small-code-sft" / "sft_20260807-203006.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"started 2026-08-07 20:30:13\nTraceback (most recent call last):\n{last_line}\n")
    # a pid that is gone: `scripts/stop.sh` removes this file, so its survival means a crash
    (d / "train.pid").write_text("999999999")
    (d / "run.meta").write_text(f"pid     999999999\nlog     {log.relative_to(tmp_path)}\n")
    return d


def test_a_crashed_stage_reports_failed_not_ready(tmp_path):
    _crashed_sft(tmp_path)
    st = stages(Pipeline(tmp_path), "small-code")["sft"]
    assert st["phase"] == "failed"
    assert "out of memory" in st["reason"].lower()   # the last log line, verbatim
    assert st["can_start"]                           # you fix a knob and press it again
    assert not st["can_stop"]


def test_a_crashed_stage_still_blocks_what_depends_on_it(tmp_path):
    _crashed_sft(tmp_path)
    st = stages(Pipeline(tmp_path), "small-code")
    assert st["dpo"]["phase"] == "blocked" and not st["dpo"]["can_start"]
    assert st["grpo"]["phase"] == "blocked" and not st["grpo"]["can_start"]


def test_a_finished_stage_is_done_even_with_a_stale_pid(tmp_path):
    """Order matters: a checkpoint means it finished, whatever the pid file says."""
    d = _crashed_sft(tmp_path)
    _touch(d / "sft_best.pt")
    assert stages(Pipeline(tmp_path), "small-code")["sft"]["phase"] == "done"


def test_a_stage_that_never_ran_is_still_ready(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    st = stages(Pipeline(tmp_path), "small-code")["sft"]
    assert st["phase"] == "ready" and st["reason"] is None


def test_start_fresh_archives_the_previous_stage_and_keeps_everything(tmp_path):
    """"Start again" on a finished stage must not mean "overwrite the model you just
    trained", and must not mean "train zero steps" either — which is what a plain restart
    does, because `--resume auto` sees the last epoch is already done."""
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    d = tmp_path / "checkpoints" / "small-code-sft"
    _touch(d / "sft_best.pt")
    (d / "sft_log.jsonl").write_text('{"step": 5, "loss": 1.0, "time": 1.0, "elapsed": 1.0}\n')
    _touch(d / "report.md")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "stage.sh").write_text("#!/bin/sh\nexit 0\n")

    res = Pipeline(tmp_path).start("small-code", "sft", fresh=True)

    assert res["archived"], "nothing was archived"
    archive = tmp_path / "checkpoints" / res["archived"]
    # everything survives, under the timestamped name
    assert (archive / "sft_best.pt").exists() and (archive / "report.md").exists()
    assert (archive / "sft_log.jsonl").exists()
    # ...and the stage restarts from an empty directory, so `--resume auto` finds nothing
    assert not (tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt").exists()
    assert res["archived"] in Pipeline(tmp_path).store.runs(), "the archive must stay visible"


def test_a_plain_start_does_not_archive(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "stage.sh").write_text("#!/bin/sh\nexit 0\n")
    assert Pipeline(tmp_path).start("small-code", "sft")["archived"] is None


# --- pre-flight: the window between pressing Start and the trainer existing ----------------
# `scripts/stage.sh dpo` downloads and tokenizes UltraFeedback before it launches anything.
# That is minutes with no pid, no checkpoint and no crash — the exact state that used to fall
# through to "ready", so the button looked like it had done nothing and the natural response
# was to press it again, on top of the download already running.

def _preflighting(tmp_path, stage: str, step: str = "data"):
    """A live `stage.sh` in pre-flight, publishing the files the real script publishes."""
    import subprocess

    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    script = scripts / "stage.sh"          # the name is what `LAUNCH_SCRIPTS` matches on
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    d = tmp_path / "checkpoints" / f"small-code-{stage}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "launch.pid").write_text(str(proc.pid))
    (d / "launch.meta").write_text(f"pid     {proc.pid}\nstage   {step}\n")
    return proc


def test_a_stage_downloading_its_dataset_reports_preparing_not_ready(tmp_path):
    proc = _preflighting(tmp_path, "dpo")
    try:
        st = stages(Pipeline(tmp_path), "small-code")["dpo"]
        assert st["phase"] == "preparing", "a live pre-flight must not read as 'ready'"
        assert not st["can_start"], "Start stays disabled or it invites a second launch"
        assert not st["can_stop"], "there is no trainer to stop yet"
        # and it says what it is doing, so waiting is obviously the right move
        assert "UltraFeedback" in st["reason"]
    finally:
        proc.kill()
        proc.wait()


def test_pre_flight_stops_reading_as_preparing_once_the_launcher_is_gone(tmp_path):
    proc = _preflighting(tmp_path, "dpo")
    proc.kill()
    proc.wait()
    # a stale launch.pid must not pin the card at "preparing" forever
    assert stages(Pipeline(tmp_path), "small-code")["dpo"]["phase"] == "ready"


def test_the_panel_says_which_dataset_is_missing_before_you_press_start(tmp_path):
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    st = stages(Pipeline(tmp_path), "small-code")

    # DPO needs a dataset and this tree has none, so Start means "download, then train"
    assert st["dpo"]["data"] == {"needed": True, "ready": False,
                                 "path": "data/dpo/train_chosen_tokens.npy",
                                 "recipe": "ultrafeedback",
                                 "cost": st["dpo"]["data"]["cost"]}
    # GRPO never needs one — the sandbox computes the reward
    assert st["grpo"]["data"]["needed"] is False

    _touch(tmp_path / "data" / "dpo" / "train_chosen_tokens.npy")
    assert stages(Pipeline(tmp_path), "small-code")["dpo"]["data"]["ready"] is True


def test_dpo_and_grpo_are_named_as_alternatives_to_each_other(tmp_path):
    """Side by side and gated on the same checkpoint, they look like a sequence. They are
    not: neither reads the other's output, and the card has to say so."""
    _touch(tmp_path / "checkpoints" / "small-code" / "ckpt_best.pt")
    _touch(tmp_path / "checkpoints" / "small-code-sft" / "sft_best.pt")
    st = stages(Pipeline(tmp_path), "small-code")

    assert st["dpo"]["alternative"] == "grpo"
    assert st["grpo"]["alternative"] == "dpo"
    assert st["sft"]["alternative"] is None          # SFT is not optional
    # both read the same file, which is what makes them alternatives rather than a chain
    assert st["dpo"]["starts_from"] == st["grpo"]["starts_from"] == \
        "checkpoints/small-code-sft/sft_best.pt"
    assert st["dpo"]["writes"] != st["grpo"]["writes"]
    # and each carries the rule for picking it
    assert "no program can tell" in st["dpo"]["guidance"]["choose"]
    assert "program CAN tell" in st["grpo"]["guidance"]["choose"]
