"""Which processes the portal recognises as ours — and what happens when it does not.

Two hardcoded strings caused two real incidents on one evening, and they are the same
mistake twice:

* `trainer_pid` validated a `train.pid` with `"aksharallm.train"`, which excludes every
  trainer outside `aksharallm/train/`. A **live** codec run reported `idle` with no Stop
  button while its log tail advanced — and `scripts/stop.sh` went further and deleted the
  pid file of the live process as "stale", removing the only handle anything had on it;
* `launcher` checked for `"phase2.sh"`, so a run pre-flighting under `experiment.sh` or
  `audio.sh` reported `idle` for the whole pre-flight, **with Start still enabled** — which
  invites a second launch on top of the first.

Both are now tuples, and this file is what keeps them honest: every entry must be
recognised, and a process that is genuinely not ours must still be refused.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from aksharallm.portal import runs as runs_mod
from aksharallm.portal.runs import LAUNCH_SCRIPTS, TRAINERS, RunStore

CONFIG = "model:\n  vocab_size: 8192\n"


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "demo.yaml").write_text(CONFIG)
    (tmp_path / "checkpoints" / "demo").mkdir(parents=True)
    (tmp_path / "checkpoints" / "demo" / "train_log.jsonl").write_text("")
    return tmp_path


def sleeper(*args):
    """A live process whose command line we control, so `_cmdline` has something to match."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *args])


def wait_for_cmdline(pid: int, needle: str, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if needle in runs_mod._cmdline(pid):
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------------------
# trainers
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", TRAINERS)
def test_every_trainer_in_the_list_is_recognised(repo, module):
    """A trainer missing from `TRAINERS` reports its run as idle **while it is training**."""
    proc = sleeper(module, "configs/demo.yaml")
    try:
        assert wait_for_cmdline(proc.pid, module)
        (repo / "checkpoints" / "demo" / "train.pid").write_text(f"{proc.pid}\n")
        assert RunStore(repo).trainer_pid("demo") == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_the_audio_trainer_is_the_one_that_caused_the_incident(repo):
    """Named separately because the general test above would still pass if someone removed
    exactly this entry and the parametrisation shrank with it."""
    assert "aksharallm.audio.train_codec" in TRAINERS


def test_a_process_that_is_not_ours_is_still_refused(repo):
    proc = sleeper("something.else.entirely")
    try:
        assert wait_for_cmdline(proc.pid, "something.else")
        (repo / "checkpoints" / "demo" / "train.pid").write_text(f"{proc.pid}\n")
        assert RunStore(repo).trainer_pid("demo") is None
    finally:
        proc.kill()
        proc.wait()


def test_the_smoke_test_is_never_mistaken_for_the_run(repo):
    """`phase2.sh` runs the identical command with `-o train.out_dir=/tmp/aksharallm_smoke`,
    and aiming a stop at it means writing a STOP file the real trainer never reads."""
    proc = sleeper("aksharallm.train.pretrain", "configs/demo.yaml",
                   "-o", "train.out_dir=/tmp/aksharallm_smoke")
    try:
        assert wait_for_cmdline(proc.pid, "aksharallm_smoke")
        (repo / "checkpoints" / "demo" / "train.pid").write_text(f"{proc.pid}\n")
        assert RunStore(repo).trainer_pid("demo") is None
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------------------
# launchers
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", LAUNCH_SCRIPTS)
def test_every_launcher_is_recognised_during_pre_flight(repo, script):
    """A launcher missing from `LAUNCH_SCRIPTS` reports its run as idle for the whole
    pre-flight, with Start still enabled — inviting a second launch on top of the first."""
    proc = sleeper(f"scripts/{script}", "demo")
    try:
        assert wait_for_cmdline(proc.pid, script)
        d = repo / "checkpoints" / "demo"
        (d / "launch.pid").write_text(f"{proc.pid}\n")
        (d / "launch.meta").write_text("pid     1\nstage   smoke\nstarted now\n")
        info = RunStore(repo).launcher("demo")
        assert info is not None and info["pid"] == proc.pid
        assert info["stage"] == "smoke"
    finally:
        proc.kill()
        proc.wait()


def test_a_run_pre_flighting_cannot_be_started_again(repo):
    """The consequence, asserted where it is felt: Start is disabled and Stop is offered,
    because aborting a pre-flight is a real thing to want."""
    proc = sleeper("scripts/audio.sh", "codec-lj")
    try:
        assert wait_for_cmdline(proc.pid, "audio.sh")
        d = repo / "checkpoints" / "demo"
        (d / "launch.pid").write_text(f"{proc.pid}\n")
        (d / "launch.meta").write_text("stage   preflight\n")
        summary = RunStore(repo).summary("demo")
        assert summary["phase"] == runs_mod.PHASE_LAUNCHING
        assert not summary["can_start"]
        assert summary["can_stop"]
    finally:
        proc.kill()
        proc.wait()


def test_a_dead_launcher_is_not_a_pre_flight(repo):
    d = repo / "checkpoints" / "demo"
    (d / "launch.pid").write_text("999999999\n")
    (d / "launch.meta").write_text("stage   smoke\n")
    assert RunStore(repo).launcher("demo") is None


# ---------------------------------------------------------------------------------------
# what counts as a run at all
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("section", ["model", "codec", "audiolm", "vision"])
def test_every_kind_of_run_config_is_listed(repo, section):
    (repo / "configs" / f"a-{section}.yaml").write_text(f"{section}:\n  x: 1\n")
    assert f"a-{section}" in RunStore(repo).runs()


def test_a_settings_file_is_not_a_run(repo):
    """`portal.yaml` configures the code explainer. A phantom run in the picker with no log
    and no launcher is the kind of thing you waste an evening on."""
    (repo / "configs" / "portal.yaml").write_text("explain:\n  model: gemma\n")
    assert "portal" not in RunStore(repo).runs()


def test_audio_runs_are_startable_from_the_portal(repo):
    """They were listed but not startable, which is a picker entry that does nothing."""
    for name in ("codec-synth", "codec-lj", "audiolm-synth"):
        script, args, _ = runs_mod.launcher_for(name)
        assert script == "scripts/audio.sh" and args == [name]


# ---------------------------------------------------------------------------------------
# what the charts need
# ---------------------------------------------------------------------------------------


def test_every_trainer_logs_the_keys_the_charts_read():
    """The portal's throughput chart reads `tok_per_sec` and nothing else.

    The codec trainer logged only `audio_s_per_s` — its own natural unit, and a better one
    for a codec — so its throughput chart was **empty**, which reads as "the run is not
    producing anything" rather than "nobody wrote this field". Every trainer now emits the
    shared key as well as whatever else it wants to say.

    A source check rather than a run: starting five trainers in a unit test would cost more
    than it proves, and what went wrong was a missing *name*.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    trainers = [
        "aksharallm/train/pretrain.py",
        "aksharallm/audio/train_codec.py",
        "aksharallm/audio/train_lm.py",
        "aksharallm/vision/train.py",
    ]
    missing = [f for f in trainers if "tok_per_sec" not in (root / f).read_text()]
    assert not missing, f"these trainers would draw an empty throughput chart: {missing}"


def test_the_series_keys_are_what_the_trainers_write():
    """If `SERIES_KEYS` and the trainers ever disagree, the chart is silently empty."""
    from aksharallm.train.runlog import SERIES_KEYS

    assert "tok_per_sec" in SERIES_KEYS and "loss" in SERIES_KEYS
