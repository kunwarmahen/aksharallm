"""The World and Tools tabs and the Audio tab's TTS panel (portal/world.py, tooluse.py, tts.py,
jobs.py), and the comparisons they share with the CLI (audio/ttsreports.py, tools/agent.py)."""

from __future__ import annotations

import json
import sys
import time

import pytest

from aksharallm.audio import ttsreports
from aksharallm.portal.jobs import CliJobs, JobError
from aksharallm.tools import agent


def _tts_report(refs, hyps, own=None, speakers=None):
    items = [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)]
    rep = {"n": len(refs), "rows": {"original": {"wer": 0.1, "items": items},
                                    "codec": {"wer": 0.1, "items": items},
                                    "tts": {"wer": 0.5, "runaway": 0, "items": items}}}
    if own is not None:
        rep["voice"] = {"tts_own": sum(own) / len(own), "tts_other": float("nan"),
                        "codec_own": 0.44, "same_speaker": 0.6, "own_closer": 0, "pairs": 0,
                        "p": float("nan"),
                        "per_item": [{"speaker": s, "tts_own": o} for s, o in zip(speakers, own)]}
    return rep


def test_a_one_speaker_report_reads_as_strict_json(tmp_path):
    """The author's reports have no "different speaker", so tts_other is NaN. Python writes
    NaN; a browser's JSON.parse rejects the whole response, and the panel was empty."""
    rep = _tts_report(["a b"], ["a b"], own=[0.2], speakers=["me"])
    p = tmp_path / "tts-x-step1-own.json"
    p.write_text(json.dumps(rep))
    s = ttsreports.summary(p, json.loads(p.read_text()))
    json.dumps(s, allow_nan=False)                     # raises on any NaN left in it
    assert s["voice"]["tts_other"] is None


def test_tts_reports_are_compared_per_sentence_and_only_on_the_same_sentences():
    refs = ["one two three", "four five six", "seven eight nine"]
    a = _tts_report(refs, ["one two three", "four x six", "x x x"], own=[0.1, 0.2, 0.3],
                    speakers="abc")
    b = _tts_report(refs, ["one x three", "four five six", "seven eight nine"],
                    own=[0.2, 0.1, 0.4], speakers="abc")
    c = ttsreports.compare(a, b)
    assert (c["wer"]["b_better"], c["wer"]["b_worse"]) == (2, 1)
    assert (c["voice"]["b_closer"], c["voice"]["b_further"]) == (2, 1)
    other = _tts_report(["different"] * 3, ["different"] * 3)
    with pytest.raises(ValueError, match="different sentences"):
        ttsreports.compare(a, other)


def test_tool_reports_are_paired_only_on_the_same_problems():
    rows = lambda qs, tool: [{"kind": "k", "question": q, "tool": t, "direct": False}  # noqa: E731
                             for q, t in zip(qs, tool)]
    a = {"rows": rows("pqr", [False, False, True])}
    b = {"rows": rows("pqr", [True, True, False])}
    c = agent.compare(a, b)
    assert c["paired"]["with the tool"] == {"b_only": 2, "a_only": 1,
                                            "p": agent._sign(2, 1)}
    with pytest.raises(ValueError, match="not on the same problems"):
        agent.compare(a, {"rows": rows("xyz", [True] * 3)})


def _wait(jobs, s=20):
    end = time.time() + s
    while time.time() < end:
        st = jobs.status()
        if not st["running"]:
            return st
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_a_job_is_judged_by_its_exit_code(tmp_path):
    """done on exit 0, failed on anything else: never by whether an output file appeared."""
    ok = CliJobs(tmp_path, "jobs-ok", "platform")
    ok.start(["terse"], "print the platform")
    st = _wait(ok)
    assert st["current"]["state"] == "done" and st["current"]["rc"] == 0
    assert st["current"]["command"] == "python -m platform terse"

    bad = CliJobs(tmp_path, "jobs-bad", "aksharallm.world")
    bad.start(["no-such-command"], "fail on purpose")
    st = _wait(bad)
    assert st["current"]["state"] == "failed" and st["current"]["rc"] == 2


def test_one_job_at_a_time(tmp_path):
    """A live job (recorded pid, matching command line) holds its folder."""
    import subprocess
    jobs = CliJobs(tmp_path, "jobs", "platform")
    jobs.dir
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    from aksharallm.portal.runs import exec_cmdline
    (tmp_path / "jobs" / "current.json").write_text(json.dumps(
        {"job": "x", "state": "running", "pid": proc.pid,
         "cmdline": exec_cmdline([sys.executable, "-c", "import time; time.sleep(5)"])}))
    try:
        with pytest.raises(JobError, match="already running"):
            jobs.start(["anything"], "second")
    finally:
        proc.kill()


def test_the_world_playout_counts_exactly_the_cells_it_got_wrong():
    from tests.test_world import CopyLastFrame
    from aksharallm.world.evaluate import play
    """Open loop, so only step 1 (predicted from a real frame) can be checked against the
    outcome: a copying model is wrong exactly where something moved. Later steps feed its own
    frames back, and an early error carries forward."""
    seen = set()
    for seed in range(12):
        steps = play(CopyLastFrame(), horizon=4, seed=seed)
        assert steps[0]["true"] == steps[0]["model"]
        for st in steps[1:]:
            assert st["wrong"] == sum(a != b for a, b in zip(st["true"], st["model"]))
        first = steps[1]
        seen.add(first["outcome"])
        if first["outcome"] in ("blocked", "stayed"):
            assert first["wrong"] == 0
        else:
            assert first["wrong"] >= 2                    # the agent left one cell, entered one
    assert {"moved", "blocked"} <= seen


def test_speak_refuses_a_speaker_that_is_not_an_id(tmp_path, monkeypatch):
    from aksharallm.portal.tts import Tts
    t = Tts(tmp_path)
    monkeypatch.setattr(t, "checkpoints", lambda: [{"rel": "c.pt", "multi": True}])
    started = []
    monkeypatch.setattr(t.jobs, "start", lambda argv, label: started.append(argv) or {})
    with pytest.raises(JobError, match="LibriSpeech id"):
        t.start({"kind": "speak", "checkpoint": "c.pt", "text": "hi", "speaker": "1; rm -rf ~"})
    t.start({"kind": "speak", "checkpoint": "c.pt", "text": "hello  there", "speaker": "1089"})
    argv = started[-1]
    assert argv[:3] == ["speak", "c.pt", "hello there"] and argv[-2:] == ["--speaker-id", "1089"]
    with pytest.raises(JobError, match="not a TTS checkpoint"):
        t.start({"kind": "speak", "checkpoint": "other.pt", "text": "hi"})


def test_a_hybrid_checkpoint_says_so_wherever_it_is_listed():
    """It decides whether the model has a KV cache (no drafting, no serving), so the run
    list, the checkpoint picker and the report must not show it as a plain transformer."""
    from aksharallm.infer.checkpoints import _arch_str
    base = {"d_model": 384, "n_layers": 6, "n_heads": 6, "n_kv_heads": 6, "max_seq_len": 512}
    assert "hybrid" not in _arch_str(base)
    assert _arch_str({**base, "full_attn_every": 3}).endswith("hybrid: DeltaNet in 4/6")


def test_the_world_tab_reads_a_keys_result_and_both_datasets(tmp_path):
    """A keys evaluation carries `calibration` and `cells: None`; the overview must pass it
    through as strict JSON, and list every prepared world, not only data/world."""
    from aksharallm.portal.world import World
    (tmp_path / "logs/world").mkdir(parents=True)
    nf = {"n": 1, "exact": 0.5, "stderr": 0.1, "cells": None, "copy_baseline": 0.3,
          "by_outcome": {"moved": {"n": 1, "exact": 0.5, "cells": None, "copy_baseline": 0.0,
                                   "kl": 0.2, "off": 0.0, "p_mode_true": 0.9,
                                   "p_mode_model": float("nan")}}}
    (tmp_path / "logs/world/world-keys-step9.json").write_text(json.dumps(
        {"game": "keys", "next_frame": nf, "rollout": {"exact_by_step": [1.0]},
         "calibration": {"kl": 0.2, "off": 0.0, "doors": {}}}))
    for d, game in (("world", None), ("world-keys", "keys")):
        (tmp_path / "data" / d).mkdir(parents=True)
        meta = {"tokens": {"train": 1}, "episodes": {"train": 1}, "steps": 2, "outcomes": {}}
        if game:
            meta["game"] = game
        (tmp_path / "data" / d / "meta.json").write_text(json.dumps(meta))
    ov = World(tmp_path, device_for=lambda: ("cpu", "test")).overview()
    json.dumps(ov, allow_nan=False)
    assert [r["game"] for r in ov["results"]] == ["keys"]
    assert ov["results"][0]["calibration"]["kl"] == 0.2
    assert {(x["dir"], x["game"]) for x in ov["data"]} == {("data/world", "sokoban"),
                                                         ("data/world-keys", "keys")}


def test_the_tools_jobs_take_a_problem_set_and_refuse_an_unknown_one(tmp_path, monkeypatch):
    from aksharallm.portal.tooluse import ToolUse
    t = ToolUse(tmp_path)
    monkeypatch.setattr(t, "checkpoints", lambda: [{"rel": "c.pt"}])
    started = []
    monkeypatch.setattr(t.jobs, "start", lambda argv, label: started.append(argv) or {})
    t.start({"kind": "data", "set": "v2"})
    t.start({"kind": "eval", "checkpoint": "c.pt", "set": "v2", "label": "x"})
    assert started[0] == ["data", "--set", "v2"]
    assert started[1][:4] == ["eval", "c.pt", "--set", "v2"]
    with pytest.raises(JobError, match="problem set"):
        t.start({"kind": "data", "set": "v3; rm -rf ~"})


def test_the_world_jobs_offer_the_blind_agent_and_refuse_anything_else(tmp_path, monkeypatch):
    from aksharallm.portal.world import World
    w = World(tmp_path, device_for=lambda: ("cpu", "test"))
    monkeypatch.setattr(w, "checkpoints", lambda: [{"rel": "c.pt", "game": "keys"}])
    started = []
    monkeypatch.setattr(w.jobs, "start", lambda argv, label: started.append(argv) or {})
    w.start({"kind": "prepare", "game": "keys-blind", "episodes": 1000})
    w.start({"kind": "eval", "checkpoint": "c.pt", "policy": "blind"})
    w.start({"kind": "eval", "checkpoint": "c.pt"})
    assert started[0] == ["prepare", "--game", "keys", "--policy", "blind", "--episodes", "1000"]
    assert started[1][:4] == ["eval", "c.pt", "--policy", "blind"]
    assert "--policy" not in started[2]
    with pytest.raises(JobError):
        w.start({"kind": "eval", "checkpoint": "c.pt", "policy": "blind; rm -rf ~"})
    with pytest.raises(JobError):
        w.start({"kind": "prepare", "game": "keys-x"})
