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
