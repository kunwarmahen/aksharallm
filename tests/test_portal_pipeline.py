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
