"""Tests for the run-log reader and the web portal.

Nothing here starts a trainer or touches a GPU: the portal is a view over files, so a
temporary repo with a hand-written `train_log.jsonl` exercises all of it. The one thing
these tests guard hardest is that the portal cannot be *tricked* into running something —
run names are whitelisted, writes need the guard header, and a run with no launcher can
never be started.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from aksharallm.portal import runs as runs_mod
from aksharallm.portal.runs import RunError, RunStore
from aksharallm.portal.server import serve
from aksharallm.train import runlog

# A miniature version of a real log: an unmarked pre-history session, then two bracketed
# ones, an eval, and a truncated final line of the sort a `kill -9` leaves behind.
LOG_LINES = [
    '{"step": 0, "loss": 10.6, "ema": 10.6, "lr": 6e-07, "grad_norm": 9.2, '
    '"tok_per_sec": 18767.0, "mfu": 0.5}',
    '{"step": 50, "loss": 8.3, "ema": 8.5, "lr": 3e-05, "grad_norm": 3.1, '
    '"tok_per_sec": 26858.0, "mfu": 0.71}',
    '{"event": "session_start", "time": 1785000000.0, "iso": "2026-07-25 09:35:41", '
    '"run": "demo", "pid": 4242, "start_step": 100, "max_steps": 400, "stop_at": null, '
    '"tokens_per_step": 245760}',
    '{"step": 100, "loss": 7.0, "ema": 7.2, "lr": 6e-05, "grad_norm": 1.6, '
    '"tok_per_sec": 26852.0, "mfu": 0.72, "time": 1785000100.0, "s_per_step": 9.1, '
    '"elapsed": 100.0, "eta_s": 2700.0}',
    '{"step": 150, "val_loss": 6.25}',
    '{"step": 150, "loss": 6.4, "ema": 6.6, "lr": 9e-05, "grad_norm": 1.2, '
    '"tok_per_sec": 27000.0, "mfu": 0.72, "time": 1785000560.0, "s_per_step": 9.2, '
    '"elapsed": 560.0, "eta_s": 2300.0}',
    '{"event": "session_end", "time": 1785000570.0, "iso": "2026-07-25 09:45:00", '
    '"run": "demo", "reason": "STOP file asked for step 150", "last_step": 150, '
    '"steps": 51, "elapsed": 570.0}',
    '{"event": "session_start", "time": 1785100000.0, "iso": "2026-07-26 09:17:31", '
    '"run": "demo", "pid": 5353, "start_step": 151, "max_steps": 400, "stop_at": 200, '
    '"tokens_per_step": 245760}',
    '{"step": 200, "loss": 5.9, "ema": 6.0, "lr": 0.0001, "grad_norm": 0.9, '
    '"tok_per_sec": 25000.0, "mfu": 0.67, "time": 1785100500.0, "s_per_step": 9.5, '
    '"elapsed": 500.0, "eta_s": 0.0}',
    '{"step": 200, "val_loss": 5.51}',
    '{"step": 250, "loss": 5.4, "ema": 5.',  # truncated by a kill -9
]

CONFIG = """
name: demo
model:
  vocab_size: 512
  d_model: 64
  n_layers: 2
  n_heads: 4
  max_seq_len: 128
optim:
  lr: 3.0e-4
  grad_clip: 1.0
train:
  out_dir: checkpoints/demo
  batch_size: 2
  grad_accum: 4
  seq_len: 128
  max_steps: 400
"""


@pytest.fixture
def repo(tmp_path):
    """A temporary repo laid out the way the real one is."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "demo.yaml").write_text(CONFIG)
    (tmp_path / "checkpoints" / "demo").mkdir(parents=True)
    (tmp_path / "checkpoints" / "demo" / "train_log.jsonl").write_text("\n".join(LOG_LINES))
    (tmp_path / "checkpoints" / "demo" / "ckpt_last.pt").write_bytes(b"not really a checkpoint")
    (tmp_path / "logs" / "demo").mkdir(parents=True)
    (tmp_path / "logs" / "demo" / "train_20260726-091731.log").write_text(
        "\n".join(f"line {i}" for i in range(500)))
    # Stub launcher/stopper: they record how the portal called them, which is the contract
    # that matters — the portal must drive the real scripts, not reimplement them.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "phase2.sh").write_text(
        '#!/usr/bin/env bash\necho "args: $*"\necho "STOP_AFTER=${STOP_AFTER:-}"\n'
        'echo "SKIP_SMOKE=${SKIP_SMOKE:-}"\necho "LAUNCH_LOG=${LAUNCH_LOG:-}"\n')
    (scripts / "stop.sh").write_text('#!/usr/bin/env bash\necho "args: $*"\n')
    for s in scripts.iterdir():
        s.chmod(0o755)
    return tmp_path


def spawn(*extra_args):
    """A live process whose command line we control, for pid-identification tests."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *extra_args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for(predicate, timeout=10.0):
    """Detached subprocesses write their logs asynchronously; give them a moment."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def store(repo):
    return RunStore(repo)


# ---- the log reader --------------------------------------------------------------------

def test_truncated_last_line_is_skipped_not_fatal(repo):
    records = runlog.load_records(repo / "checkpoints" / "demo" / "train_log.jsonl")
    assert len(records) == len(LOG_LINES) - 1
    assert records[-1]["val_loss"] == 5.51


def test_sessions_split_on_markers_and_on_a_backwards_step(repo):
    sessions = runlog.load_sessions(repo / "checkpoints" / "demo" / "train_log.jsonl")
    rows = runlog.summarise_sessions(sessions)
    assert [r["index"] for r in rows] == [1, 2, 3]
    # The first has no start marker: it is pre-history, split off because step 100 < ... no,
    # because a session_start arrived. Either way it must not be merged into the second.
    assert rows[0]["unmarked"] and rows[0]["first_step"] == 0 and rows[0]["last_step"] == 50
    assert rows[1]["pid"] == 4242 and rows[1]["ended"].startswith("STOP file")
    assert rows[1]["best_val"] == 6.25
    assert rows[2]["open"] is True and rows[2]["ended"] is None


def test_latest_reads_backwards_for_the_things_that_persist(repo):
    last = runlog.latest(runlog.load_records(repo / "checkpoints" / "demo" / "train_log.jsonl"))
    assert last["step"] == 200
    assert last["ema"] == 6.0
    assert last["best_val"] == 5.51          # the minimum, not merely the newest
    assert last["max_steps"] == 400          # from the newest session_start
    assert last["tokens_per_step"] == 245760
    assert last["n_sessions"] == 2


def test_series_are_columnar_and_downsample_keeping_the_last_point():
    records = [{"step": i, "loss": float(i), "ema": float(i), "tok_per_sec": 1.0}
               for i in range(1000)]
    full = runlog.series(records, max_points=0)
    assert full["step"][:3] == [0, 1, 2] and len(full["loss"]) == 1000
    small = runlog.series(records, max_points=100)
    assert len(small["step"]) <= 102
    assert small["step"][-1] == 999, "the newest reading is the one being watched"


def test_fmt_dur_matches_the_trainers_form():
    assert runlog.fmt_dur(45.2) == "45.2s"
    assert runlog.fmt_dur(750) == "12m30s"
    assert runlog.fmt_dur(3600 * 6 + 300) == "6h05m"
    assert runlog.fmt_dur(3600 * 76) == "3d04h"
    assert runlog.fmt_dur(None) == "?"


# ---- the run store ---------------------------------------------------------------------

def test_runs_are_discovered_from_configs_and_checkpoints(store, repo):
    (repo / "checkpoints" / "orphan").mkdir()
    (repo / "checkpoints" / "orphan" / "train_log.jsonl").write_text("")
    assert store.runs() == ["demo", "orphan"]


@pytest.mark.parametrize("bad", ["../../etc", "demo;rm -rf /", ".hidden", "", "a b"])
def test_run_names_are_whitelisted(store, bad):
    with pytest.raises(RunError):
        store.check(bad)


def test_status_of_an_idle_run(store):
    s = store.status("demo")
    assert s["phase"] == "idle"
    assert s["pid"] is None and s["can_stop"] is False
    assert s["step"] == 200 and s["max_steps"] == 400
    assert s["progress"] == pytest.approx(201 / 400)
    assert s["tokens_seen"] == 245760 * 201
    assert s["series"]["step"] == [0, 50, 100, 150, 200]
    assert s["config"]["grad_clip"] == 1.0
    assert [c["name"] for c in s["checkpoints"]] == ["ckpt_last.pt"]
    # 'demo' has no entry in LAUNCHERS, so the portal must not offer to start it.
    assert s["can_start"] is False and "no launcher" in s["start_hint"]


def test_starting_a_run_with_no_launcher_is_refused(store):
    with pytest.raises(RunError, match="no launcher"):
        store.start("demo")


def test_stopping_a_run_that_is_not_training_is_refused(store):
    with pytest.raises(RunError, match="not training"):
        store.stop("demo", "now")
    with pytest.raises(RunError, match="no stop is queued"):
        store.stop("demo", "cancel")
    with pytest.raises(RunError, match="unknown stop mode"):
        store.stop("demo", "obliterate")


def test_a_queued_stop_is_reported(store, repo):
    (repo / "checkpoints" / "demo" / "STOP").write_text("350\n")
    s = store.status("demo")
    assert s["stop"] == {"target": 350, "now": False}
    (repo / "checkpoints" / "demo" / "STOP").write_text("")
    assert store.status("demo")["stop"]["now"] is True


def test_log_tail_reads_only_the_end(store):
    tail = store.log_tail("demo", lines=10)
    assert tail["lines"] == [f"line {i}" for i in range(490, 500)]
    assert tail["file"].startswith("train_")
    with pytest.raises(RunError, match="no such log"):
        store.log_tail("demo", name="../../../etc/passwd")


# ---- identifying the right process -----------------------------------------------------

def test_the_smoke_test_is_never_mistaken_for_the_run(store, repo):
    """The 50-step smoke test runs the identical command line with a throwaway out_dir.

    Aiming a stop at it writes a STOP file it never reads, while the UI reports a pid that
    is about to vanish — which is exactly what happened before train.pid became the
    trainer's own, per-out_dir claim.
    """
    smoke = spawn("-m", "aksharallm.train.pretrain", "configs/demo.yaml",
                  "-o", "train.out_dir=/tmp/aksharallm_smoke", "-o", "train.max_steps=50")
    try:
        (repo / "checkpoints" / "demo" / "train.pid").write_text(f"{smoke.pid}\n")
        assert store.trainer_pid("demo") is None
        assert store.status("demo")["phase"] == "idle"
    finally:
        smoke.kill()
        smoke.wait()


def test_a_real_trainers_pid_file_is_trusted(store, repo):
    real = spawn("-m", "aksharallm.train.pretrain", "configs/demo.yaml")
    try:
        (repo / "checkpoints" / "demo" / "train.pid").write_text(f"{real.pid}\n")
        assert store.trainer_pid("demo") == real.pid
        s = store.status("demo")
        assert s["phase"] == "training" and s["can_stop"] and s["can_bound"]
    finally:
        real.kill()
        real.wait()


def test_a_preflight_started_anywhere_shows_as_launching(store, repo):
    """phase2.sh publishes launch.pid + launch.meta, so a launch from a terminal is visible
    to the portal and a launch from the portal is visible to scripts/stop.sh."""
    launcher = spawn("scripts/phase2.sh")
    try:
        rdir = repo / "checkpoints" / "demo"
        (rdir / "launch.pid").write_text(f"{launcher.pid}\n")
        (rdir / "launch.meta").write_text(
            f"pid     {launcher.pid}\nstage   smoke\nstarted 2026-07-26 12:19:00\n"
            "config  configs/demo.yaml\nlog     logs/demo/launch_x.log\n")
        s = store.status("demo")
        assert s["phase"] == "launching"
        assert s["launcher"]["stage"] == "smoke"
        assert s["launcher"]["log"] == "logs/demo/launch_x.log"
        # Stoppable (that aborts the launch), but not boundable: there is no step yet.
        assert s["can_stop"] is True and s["can_bound"] is False and s["can_start"] is False
        with pytest.raises(RunError, match="still in pre-flight"):
            store.stop("demo", "after", 500)
    finally:
        launcher.kill()
        launcher.wait()


def test_a_launch_that_is_starting_the_trainer_is_not_aborted(store, repo):
    """At stage 'launching' the trainer is seconds old and is still the launcher's child —
    signalling then could take the real run down with it."""
    launcher = spawn("scripts/phase2.sh")
    try:
        rdir = repo / "checkpoints" / "demo"
        (rdir / "launch.pid").write_text(f"{launcher.pid}\n")
        (rdir / "launch.meta").write_text(f"pid {launcher.pid}\nstage   launching\n")
        with pytest.raises(RunError, match="few seconds"):
            store.stop("demo", "now")
    finally:
        launcher.kill()
        launcher.wait()


def test_meta_files_are_parsed_as_the_shell_writes_them(repo):
    (repo / "checkpoints" / "demo" / "run.meta").write_text(
        "pid     4242\nstarted 2026-07-26 09:17:23\nconfig  configs/demo.yaml\n")
    meta = runs_mod._read_meta(repo / "checkpoints" / "demo" / "run.meta")
    assert meta == {"pid": "4242", "started": "2026-07-26 09:17:23",
                    "config": "configs/demo.yaml"}


# ---- the portal drives the scripts, it does not reimplement them ------------------------

def test_start_runs_phase2_with_the_env_it_promises(store, repo, monkeypatch):
    monkeypatch.setitem(runs_mod.LAUNCHERS, "demo", {})
    res = store.start("demo", stop_after=750, skip_smoke=True)
    log = repo / res["log"]
    assert wait_for(lambda: log.exists() and "LAUNCH_LOG=" in log.read_text())
    text = log.read_text()
    assert "STOP_AFTER=750" in text
    assert "SKIP_SMOKE=1" in text
    assert f"LAUNCH_LOG={res['log']}" in text, "phase2.sh records the log stop.sh should show"


@pytest.mark.parametrize("mode,steps,expected", [
    ("now", None, "args: demo"),
    ("after", 500, "--after 500"),
    ("at", 9000, "--at 9000"),
    ("cancel", None, "--cancel"),
])
def test_stop_shells_out_to_stop_sh_with_the_matching_flags(
        store, repo, mode, steps, expected):
    rdir = repo / "checkpoints" / "demo"
    if mode == "cancel":
        (rdir / "STOP").write_text("9000\n")
    real = spawn("-m", "aksharallm.train.pretrain", "configs/demo.yaml")
    try:
        (rdir / "train.pid").write_text(f"{real.pid}\n")
        res = store.stop("demo", mode, steps)
        log = repo / res["log"]
        assert wait_for(lambda: log.exists() and log.read_text().strip())
        assert expected in log.read_text()
    finally:
        real.kill()
        real.wait()


# ---- the HTTP layer --------------------------------------------------------------------

@pytest.fixture
def server(repo):
    httpd = serve(repo, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def get(url, **kw):
    with urllib.request.urlopen(urllib.request.Request(url, **kw), timeout=10) as res:
        return res.status, res.read()


def test_page_and_assets_are_served(server):
    status, body = get(server + "/")
    assert status == 200 and b"aksharallm" in body
    for asset in ("/static/app.js", "/static/style.css"):
        status, body = get(server + asset)
        assert status == 200 and body


def test_api_lists_runs_and_reports_one(server):
    _, body = get(server + "/api/runs")
    assert [r["run"] for r in json.loads(body)["runs"]] == ["demo"]
    _, body = get(server + "/api/run/demo")
    assert json.loads(body)["step"] == 200
    _, body = get(server + "/api/run/demo/log?lines=3")
    assert json.loads(body)["lines"] == ["line 497", "line 498", "line 499"]


def test_unknown_run_and_path_are_rejected(server):
    for path in ("/api/run/nope", "/api/run/..%2F..%2Fetc"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server + path)
        assert exc.value.code in (400, 404)


def test_writes_require_the_guard_header(server):
    """A page you visit elsewhere must not be able to stop your training run."""
    req = urllib.request.Request(server + "/api/run/demo/stop", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server + "/api/run/demo/stop", data=b"{}", method="POST")
    assert exc.value.code == 403

    req.add_header("X-Portal", "1")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 409  # now it gets as far as "that run is not training"


def test_static_serving_has_no_traversal(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server + "/static/..%2F..%2Fserver.py")
    assert exc.value.code == 404


# ---- the schedule ------------------------------------------------------------------------

from datetime import datetime, timedelta  # noqa: E402

from aksharallm.portal.schedule import Rule, Schedule, Scheduler, parse_days  # noqa: E402

MON, TUE, SAT, SUN = 0, 1, 5, 6


def at(day: int, hh: int, mm: int = 0) -> datetime:
    """A datetime `day` days after a known Monday (2026-07-27), so weekdays are readable."""
    return datetime(2026, 7, 27, hh, mm) + timedelta(days=day)


def test_days_are_parsed_the_way_people_write_them():
    assert parse_days("daily") == [0, 1, 2, 3, 4, 5, 6]
    assert parse_days("mon-fri") == [0, 1, 2, 3, 4]
    assert parse_days("sat,sun") == [5, 6]
    assert parse_days("mon wed fri") == [0, 2, 4]
    assert parse_days("fri-mon") == [0, 4, 5, 6], "a range may wrap past Sunday"
    assert parse_days(None) == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(RunError):
        parse_days("smorsday")


def test_a_rule_is_validated_when_it_is_made():
    with pytest.raises(RunError, match="HH:MM"):
        Rule(run="demo", action="start", at="25:00")
    with pytest.raises(RunError, match="start' or 'stop"):
        Rule(run="demo", action="obliterate", at="09:00")
    with pytest.raises(RunError, match="Monday to Sunday"):
        Rule(run="demo", action="stop", at="09:00", days=[9])


def test_next_fire_finds_the_right_day():
    rule = Rule(run="demo", action="start", at="22:00", days=[0, 1, 2, 3, 4])
    assert rule.next_fire(at(MON, 21, 0)) == at(MON, 22, 0)        # later today
    assert rule.next_fire(at(MON, 22, 30)) == at(TUE, 22, 0)       # already gone
    assert rule.next_fire(at(SAT, 12, 0)) == at(MON + 7, 22, 0)    # skips the weekend


def test_a_missed_fire_stays_missed():
    """Waking the machine at 07:00 must not trigger the 22:00 start."""
    rule = Rule(run="demo", action="start", at="22:00")
    assert rule.due(at(MON, 22, 0)) == at(MON, 22, 0)
    assert rule.due(at(MON, 22, 5)) == at(MON, 22, 0), "a few minutes late still counts"
    assert rule.due(at(TUE, 7, 0)) is None, "nine hours late does not"


def test_an_occurrence_fires_once():
    rule = Rule(run="demo", action="stop", at="06:30")
    occurrence = rule.due(at(MON, 6, 31))
    assert occurrence is not None
    rule.last_fired = occurrence.isoformat(timespec="seconds")
    assert rule.due(at(MON, 6, 32)) is None, "same occurrence, already handled"
    assert rule.due(at(TUE, 6, 30)) == at(TUE, 6, 30), "tomorrow's is a new occurrence"


def test_a_window_over_midnight_shifts_the_stop_to_the_next_day(repo):
    sched = Schedule(repo, repo / "schedule.json")
    start, stop = sched.add_window("demo", "22:00", "06:30", parse_days("mon-fri"))
    assert start.days == [0, 1, 2, 3, 4]
    assert stop.days == [1, 2, 3, 4, 5], "Mon-Fri nights end Tue-Sat mornings"
    # A window inside one day keeps both on the same days.
    s2, e2 = sched.add_window("demo", "13:00", "17:30", parse_days("sat,sun"))
    assert s2.days == e2.days == [5, 6]


def test_the_schedule_round_trips_through_the_file(repo):
    sched = Schedule(repo, repo / "schedule.json")
    sched.add(Rule(run="demo", action="start", at="09:00", days=[0], stop_after=250))
    again = Schedule(repo, repo / "schedule.json")
    assert len(again.rules) == 1
    assert again.rules[0].stop_after == 250 and again.rules[0].at == "09:00"
    # A rule that no longer parses is dropped, not fatal — the file is hand-editable.
    (repo / "schedule.json").write_text(
        '{"enabled": true, "rules": [{"run": "demo", "action": "start", "at": "nope"},'
        ' {"run": "demo", "action": "stop", "at": "06:30"}]}')
    assert [r.at for r in Schedule(repo, repo / "schedule.json").rules] == ["06:30"]


def test_firing_is_idempotent_and_never_raises(store, repo, monkeypatch):
    """A start when it is already training, or a stop when nothing runs, is a no-op — the
    schedule's intent already holds, and an unattended loop must not die on it."""
    monkeypatch.setitem(runs_mod.LAUNCHERS, "demo", {})
    sched = Schedule(repo, repo / "schedule.json")
    scheduler = Scheduler(store, sched, tick=0.01)

    stop_rule = Rule(run="demo", action="stop", at="06:30")
    sched.add(stop_rule)
    result = scheduler.fire(stop_rule, at(MON, 6, 30))
    assert "skipped" in result and "not training" in result
    assert stop_rule.last_fired is not None, "a skip still counts as handled"
    assert "skipped" in scheduler.recent()[-1]


def test_the_master_switch_stops_everything_firing(store, repo):
    sched = Schedule(repo, repo / "schedule.json")
    sched.add(Rule(run="demo", action="stop", at="06:30"))
    sched.enabled = False
    sched.save()
    scheduler = Scheduler(store, Schedule(repo, repo / "schedule.json"), tick=0.01)
    fired = scheduler.check(datetime.now().replace(hour=6, minute=30, second=0))
    assert fired == []


def test_only_one_scheduler_holds_the_clock(store, repo):
    first = Scheduler(store, Schedule(repo, repo / "schedule.json"))
    assert first.lock() is True
    assert first.holder() is None, "our own pid is not a rival"
    other = spawn("-m", "aksharallm.portal")
    try:
        (repo / "logs" / "scheduler.pid").write_text(f"{other.pid}\n")
        second = Scheduler(store, Schedule(repo, repo / "schedule.json"))
        assert second.holder() == other.pid
        assert second.lock() is False
    finally:
        other.kill()
        other.wait()


def test_schedule_api_add_list_and_remove(server, repo):
    body = json.dumps({"run": "demo", "start_at": "22:00", "stop_at": "06:30",
                       "days": "mon-fri"}).encode()
    req = urllib.request.Request(server + "/api/schedule/window", data=body, method="POST")
    req.add_header("X-Portal", "1")
    # 'demo' has no launcher, so a scheduled start is refused rather than silently useless.
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 409

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(runs_mod.LAUNCHERS, "demo", {})
        req = urllib.request.Request(server + "/api/schedule/window", data=body, method="POST")
        req.add_header("X-Portal", "1")
        with urllib.request.urlopen(req, timeout=10) as res:
            added = json.loads(res.read())
        assert len(added["rules"]) == 2 and "crosses midnight" in added["note"]

    _, listing = get(server + "/api/schedule")
    data = json.loads(listing)
    assert len(data["rules"]) == 2 and data["enabled"] is True
    assert all(r["next_fire"] for r in data["rules"])

    rid = data["rules"][0]["id"]
    req = urllib.request.Request(server + "/api/schedule/remove",
                                 data=json.dumps({"id": rid}).encode(), method="POST")
    req.add_header("X-Portal", "1")
    with urllib.request.urlopen(req, timeout=10) as res:
        assert json.loads(res.read())["ok"] is True
    _, listing = get(server + "/api/schedule")
    assert len(json.loads(listing)["rules"]) == 1


# ---- gpu telemetry -----------------------------------------------------------------------

from aksharallm.portal import gpu as gpumod  # noqa: E402

FAKE_STATIC = [["0", "NVIDIA GeForce RTX 3090", "24576", "390.00"]]
FAKE_LIVE = [["0", "98", "19140", "71", "310.50"]]


@pytest.fixture
def fake_smi(monkeypatch):
    """nvidia-smi, faked — the tests must pass on a machine with no GPU and give the same
    answer on one with four."""
    def runner(fields, timeout=5.0):
        return FAKE_STATIC if "name" in fields else FAKE_LIVE
    monkeypatch.setattr(gpumod, "_run_smi", runner)
    return runner


def test_devices_and_a_sample_are_parsed(fake_smi, store):
    assert gpumod.devices() == [{"index": 0, "name": "NVIDIA GeForce RTX 3090",
                                 "mem_total": 24576.0, "power_limit": 390.0}]
    rec = gpumod.sample(store)
    assert rec["gpus"][0] == {"index": 0, "util": 98.0, "mem_used": 19140.0,
                              "temp": 71.0, "power": 310.5}
    assert "run" not in rec, "nothing is training in the fixture repo"


def test_unreported_fields_are_missing_not_zero(monkeypatch, store):
    """nvidia-smi prints [N/A] for what a card doesn't report. Averaging that as 0 W would
    quietly halve the reported power draw."""
    monkeypatch.setattr(gpumod, "_run_smi",
                        lambda fields, timeout=5.0: [["0", "50", "1024", "40", "[N/A]"]])
    assert gpumod.sample(store)["gpus"][0]["power"] is None


def test_no_gpu_is_reported_honestly(monkeypatch, store):
    monkeypatch.setattr(gpumod, "_run_smi", lambda fields, timeout=5.0: None)
    snap = gpumod.snapshot(store)
    assert snap["available"] is False and "no NVIDIA GPU" in snap["reason"]
    assert gpumod.Sampler(store).start() is False


def test_samples_are_written_read_back_and_windowed(fake_smi, store, repo):
    sampler = gpumod.Sampler(store)
    for _ in range(3):
        assert sampler.tick() is not None
    # An old sample must fall outside a short window.
    with open(sampler.path, "a") as fh:
        fh.write(json.dumps({"time": time.time() - 7200,
                             "gpus": [{"index": 0, "util": 1.0}]}) + "\n")
    assert len(gpumod.read_records(sampler.path, window_s=60)) == 3
    assert len(gpumod.read_records(sampler.path, window_s=None)) == 4


def test_series_are_bucket_averaged_down_to_max_points():
    now = time.time()
    records = [{"time": now + i, "gpus": [{"index": 0, "util": float(i % 10),
                                           "mem_used": 1000.0, "temp": 50.0,
                                           "power": 100.0}]} for i in range(1000)]
    s = gpumod.series(records, max_points=50)
    assert len(s["time"]) == 50
    assert 4.0 <= sum(s["util"]) / 50 <= 5.0, "the mean survives downsampling"


def test_training_spans_split_on_a_gap():
    """A portal restart must leave a gap in the band, not one continuous lie across the
    hours nothing was watching."""
    t = 1_000_000.0
    records = (
        [{"time": t + i * 5, "run": "small-code", "gpus": []} for i in range(5)]
        + [{"time": t + 3600 + i * 5, "run": "small-code", "gpus": []} for i in range(5)]
        + [{"time": t + 7200, "gpus": []}]
    )
    spans = gpumod.training_spans(records)
    assert len(spans) == 2
    assert spans[0]["start"] == t and spans[0]["end"] == t + 20
    assert all(s["run"] == "small-code" for s in spans)


def test_summary_splits_training_from_idle():
    def rec(util, power, run=None):
        r = {"time": time.time(),
             "gpus": [{"index": 0, "util": util, "mem_used": 19000.0, "temp": 70.0,
                       "power": power}]}
        if run:
            r["run"] = run
        return r
    records = [rec(98, 310, "small-code"), rec(96, 300, "small-code"), rec(2, 25)]
    summary = gpumod.summarise(records)
    assert summary["training"]["samples"] == 2
    assert summary["training"]["util"] == 97.0 and summary["training"]["power"] == 305.0
    assert summary["idle"]["samples"] == 1 and summary["idle"]["util"] == 2.0
    assert summary["training"]["temp_max"] == 70.0


def test_gpu_api_serves_the_panel(server, repo, monkeypatch):
    monkeypatch.setattr(gpumod, "_run_smi",
                        lambda fields, timeout=5.0:
                        FAKE_STATIC if "name" in fields else FAKE_LIVE)
    store = RunStore(repo)
    sampler = gpumod.Sampler(store)
    sampler.tick()
    _, body = get(server + "/api/gpu?window=3600")
    data = json.loads(body)
    assert data["available"] is True
    assert data["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert data["current"]["util"] == 98.0
    assert data["series"]["util"] == [98.0]
    assert data["summary"]["idle"]["samples"] == 1


def _launch_portal(repo):
    """A real portal process serving on a free port, rooted at the test repo."""
    return subprocess.Popen(
        [sys.executable, "-m", "aksharallm.portal", "--port", "0",
         "--root", str(repo), "--no-schedule", "--no-gpu"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_a_second_portal_does_not_steal_the_pid_file(repo):
    """A scratch portal on another port must leave the real one's pid file alone.

    It used to take the file over and then, being a well-behaved process, delete it on the
    way out — after which `scripts/portal.sh --status|--stop|--restart` all answered "portal
    is not running" about a portal that was still serving pages and still holding the
    scheduler lock. Found the hard way, on a portal that had been up for sixteen hours.
    """
    pid_file = repo / "logs" / "portal.pid"
    first = _launch_portal(repo)
    second = None
    try:
        wait_for(lambda: pid_file.exists() and pid_file.read_text().strip().isdigit(),
                 timeout=20)
        assert pid_file.read_text().strip() == str(first.pid)

        second = _launch_portal(repo)
        time.sleep(4)                      # long enough to have written it, had it wanted to
        assert pid_file.read_text().strip() == str(first.pid), \
            "the second portal overwrote a live portal's pid file"

        second.terminate()
        second.wait(timeout=15)
        second = None
        assert pid_file.exists(), "the second portal deleted a pid file it did not write"
        assert pid_file.read_text().strip() == str(first.pid)
    finally:
        for proc in (second, first):
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=15)

    # The owner still cleans up after itself.
    wait_for(lambda: not pid_file.exists(), timeout=10)
    assert not pid_file.exists()
