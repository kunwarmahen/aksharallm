"""Tests for the energy ledger and what a run cost.

Nothing here touches a GPU. Cost is arithmetic over a list of samples, so the samples are
hand-written — which is the only way to test the cases that matter, because they are all
cases where the *real* record has a hole in it: the portal was down, the card stopped
reporting watts, the telemetry file was trimmed.

The property worth protecting above all: a total may be incomplete, but it must never be
silently wrong. Every path here either measures energy or refuses to, and says which.
"""

from __future__ import annotations

import json
import time

import pytest

from aksharallm.portal import cost as costmod
from aksharallm.portal.cost import CostConfig, Ledger, integrate

T0 = 1_785_000_000.0


def rec(t, power, *, run=None, job=None, index=0, util=98.0):
    r = {"time": t, "gpus": [{"index": index, "util": util, "mem_used": 19000.0,
                              "temp": 70.0, "power": power}]}
    if run:
        r["run"] = run
    if job:
        r["job"] = job
    return r


def steady(n, power, *, start=T0, step=5.0, **kw):
    return [rec(start + i * step, power, **kw) for i in range(n)]


# ---- the integral ------------------------------------------------------------------------

def test_energy_is_watt_seconds_not_a_sample_count():
    """An hour at 360 W is 360 Wh. If this ever drifts, every number in the panel is a
    plausible-looking lie."""
    records = steady(721, 360.0, run="small-code")     # 720 intervals x 5s = 3600s
    out = integrate(records)
    assert out["seconds"] == pytest.approx(3600.0)
    assert out["wh"] == pytest.approx(360.0, rel=1e-9)
    assert out["entries"][0]["label"] == "small-code"


def test_a_ramp_is_trapezoidal_not_held_flat():
    """Power ramps between samples. Holding either endpoint flat is wrong by half the step
    — visible on a card that goes 30 W -> 350 W in one interval."""
    records = [rec(T0, 30.0, run="r"), rec(T0 + 5, 350.0, run="r")]
    assert integrate(records)["wh"] == pytest.approx((30 + 350) / 2 * 5 / 3600)


def test_a_gap_is_uncovered_not_bridged():
    """The portal was down for an hour. That hour has no reading, and inventing one from the
    samples either side would bill an hour of full load that may never have happened."""
    records = steady(3, 300.0, run="r") + steady(3, 300.0, start=T0 + 3600, run="r")
    out = integrate(records)
    assert out["seconds"] == pytest.approx(20.0), "only the four 5s intervals count"
    assert out["uncovered_s"] == pytest.approx(3590.0)
    assert out["wh"] == pytest.approx(300.0 * 20 / 3600)


def test_missing_power_contributes_no_energy_and_no_time():
    """nvidia-smi prints [N/A] on cards that don't report watts. A confident zero-watt bill
    is the worst possible answer."""
    records = [rec(T0, None, run="r"), rec(T0 + 5, None, run="r"), rec(T0 + 10, 300.0, run="r")]
    out = integrate(records)
    assert out["wh"] == 0.0 and out["seconds"] == 0.0


def test_work_is_attributed_to_run_job_and_idle_separately():
    records = (steady(3, 300.0, run="small-code") + steady(3, 200.0, start=T0 + 15, job="eval")
               + steady(3, 25.0, start=T0 + 30))
    by = {e["label"]: e for e in integrate(records)["entries"]}
    assert by["small-code"]["kind"] == "training"
    assert by["eval"]["kind"] == "job"
    assert by[None]["kind"] == "idle"
    assert by["small-code"]["wh"] > by["eval"]["wh"] > by[None]["wh"]


def test_a_second_card_is_not_billed_as_the_first():
    records = [{"time": T0 + i * 5,
                "gpus": [{"index": 0, "power": 300.0}, {"index": 1, "power": 50.0}]}
               for i in range(3)]
    assert integrate(records, index=0)["wh"] == pytest.approx(300.0 * 10 / 3600)
    assert integrate(records, index=1)["wh"] == pytest.approx(50.0 * 10 / 3600)


# ---- the ledger --------------------------------------------------------------------------

def test_the_ledger_survives_the_telemetry_being_trimmed(tmp_path):
    """The whole reason this file exists: logs/gpu.jsonl is a rolling buffer, so the total
    cannot be a scan of it. Fold everything, delete the samples, keep the energy."""
    led = Ledger(tmp_path / "energy.jsonl")
    for r in steady(721, 360.0, run="small-code"):
        led.fold(r)
    led.close()
    total = sum(e["wh"] for e in led.entries())
    assert total == pytest.approx(360.0, rel=1e-3)
    assert len(led.entries()) == 6, "an hour is six ten-minute buckets"


def test_buckets_split_on_a_change_of_run(tmp_path):
    led = Ledger(tmp_path / "energy.jsonl")
    for r in steady(60, 300.0, run="a") + steady(60, 300.0, start=T0 + 300, run="b"):
        led.fold(r)
    led.close()
    labels = {e["label"] for e in led.entries()}
    assert labels == {"a", "b"}
    assert all(e["seconds"] > 0 for e in led.entries())


def test_a_hole_closes_the_bucket_rather_than_spanning_it(tmp_path):
    """A bucket that spans a two-hour outage would claim ten minutes of measured time it
    never had."""
    led = Ledger(tmp_path / "energy.jsonl")
    for r in steady(3, 300.0, run="r") + steady(3, 300.0, start=T0 + 7200, run="r"):
        led.fold(r)
    led.close()
    entries = led.entries()
    assert len(entries) == 2
    assert all(e["seconds"] == pytest.approx(10.0) for e in entries)


def test_the_open_bucket_is_visible_before_it_is_written(tmp_path):
    """Ten minutes of invisible energy would make the panel look stalled."""
    led = Ledger(tmp_path / "energy.jsonl")
    for r in steady(13, 300.0, run="r"):
        led.fold(r)
    assert not (tmp_path / "energy.jsonl").exists()
    assert sum(e["wh"] for e in led.entries()) > 0
    assert sum(e["wh"] for e in led.entries(include_open=False)) == 0


def test_a_torn_line_costs_one_bucket_not_the_file(tmp_path):
    path = tmp_path / "energy.jsonl"
    good = {"start": T0, "seconds": 600.0, "label": "r", "kind": "training", "wh": 50.0}
    path.write_text(json.dumps(good) + "\n" + '{"start": 1785, "wh": 1.0, "sec')
    assert Ledger(path).entries() == [good]


def test_entries_are_cached_until_the_file_changes(tmp_path):
    path = tmp_path / "energy.jsonl"
    path.write_text(json.dumps({"start": T0, "seconds": 600.0, "wh": 50.0,
                                "label": "r", "kind": "training"}) + "\n")
    led = Ledger(path)
    assert led.entries() is led.entries(), "the panel polls this every couple of seconds"


# ---- the rate ----------------------------------------------------------------------------

def _cfg(**kw):
    cfg = CostConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_no_rate_means_no_money_not_free():
    priced = _cfg().price(1000.0, 3600.0)
    assert priced["money"] is None and priced["wall_wh"] == 1000.0


def test_a_kwh_rate_prices_the_card_only_by_default():
    priced = _cfg(per_kwh=0.30).price(1000.0, 3600.0)
    assert priced["money"] == pytest.approx(0.30)
    assert priced["host_wh"] == 0.0


def test_host_watts_and_psu_loss_are_what_the_meter_would_show():
    """The card is not the machine. 1 kWh of GPU plus 100 W of everything else for an hour,
    through a 90% PSU, is (1000 + 100) / 0.9 = 1222 Wh at the wall."""
    priced = _cfg(per_kwh=0.30, host_watts=100.0, psu_efficiency=0.9).price(1000.0, 3600.0)
    assert priced["wall_wh"] == pytest.approx(1100 / 0.9)
    assert priced["money"] == pytest.approx(1100 / 0.9 / 1000 * 0.30)


def test_the_two_rates_are_added_because_they_answer_different_questions():
    priced = _cfg(per_kwh=0.30, per_hour=1.20).price(1000.0, 3600.0)
    assert priced["energy"] == pytest.approx(0.30)
    assert priced["rental"] == pytest.approx(1.20)
    assert priced["money"] == pytest.approx(1.50)


def test_a_nonsense_psu_efficiency_is_refused_not_divided_by(tmp_path):
    cfg = CostConfig(path=tmp_path / "portal.yaml")
    (tmp_path / "portal.yaml").write_text("cost:\n  per_kwh: 0.3\n  psu_efficiency: 0\n")
    cfg.reload()
    assert cfg.psu_efficiency == 1.0 and "psu_efficiency" in (cfg.note or "")


def test_the_rate_is_read_from_yaml_and_overridden_by_the_environment(tmp_path, monkeypatch):
    (tmp_path / "portal.yaml").write_text(
        "cost:\n  currency: '₹'\n  per_kwh: 8.0\n  host_watts: 100\n")
    cfg = CostConfig(path=tmp_path / "portal.yaml").reload()
    assert (cfg.currency, cfg.per_kwh, cfg.host_watts) == ("₹", 8.0, 100.0)
    assert cfg.configured is True
    monkeypatch.setenv("AKSHARALLM_COST_PER_KWH", "12.5")
    assert cfg.reload().per_kwh == 12.5


def test_broken_yaml_does_not_take_the_panel_down(tmp_path):
    (tmp_path / "portal.yaml").write_text("cost: [this is not a mapping\n")
    cfg = CostConfig(path=tmp_path / "portal.yaml").reload()
    assert cfg.configured is False and cfg.note


# ---- the report --------------------------------------------------------------------------

def test_report_totals_each_run_and_the_whole_machine(tmp_path):
    led = Ledger(tmp_path / "energy.jsonl")
    now = time.time()
    for r in (steady(121, 360.0, start=now - 600, run="small-code")
              + steady(61, 60.0, start=now - 300, job="eval")):
        led.fold(r)
    led.close()
    rep = costmod.report(led, _cfg(per_kwh=0.30), now=now)
    runs = {r["label"]: r for r in rep["runs"]}
    assert set(runs) == {"small-code", "eval"}
    assert runs["small-code"]["kind"] == "training"
    assert runs["small-code"]["money"] > runs["eval"]["money"] > 0
    assert rep["total"]["wh"] == pytest.approx(sum(r["wh"] for r in rep["runs"]))
    assert rep["today"]["wh"] == pytest.approx(rep["total"]["wh"])
    assert rep["configured"] is True and rep["hint"] is None


def test_report_says_what_is_missing_when_no_rate_is_set(tmp_path):
    led = Ledger(tmp_path / "energy.jsonl")
    for r in steady(13, 300.0, run="r"):
        led.fold(r)
    rep = costmod.report(led, _cfg())
    assert rep["configured"] is False
    assert "per_kwh" in rep["hint"] and rep["total"]["money"] is None
    assert rep["total"]["wh"] > 0, "energy is still measured without a price"
    assert "card only" in rep["basis"]


def test_coverage_and_tokens_come_from_the_runs_own_log(tmp_path, monkeypatch):
    """A run trained from a terminal all weekend is only *partly* recorded. Reporting 12%
    coverage is the difference between an honest number and an understated bill."""
    from aksharallm.portal.runs import RunStore

    root = tmp_path
    (root / "configs").mkdir()
    (root / "checkpoints" / "demo").mkdir(parents=True)
    (root / "configs" / "demo.yaml").write_text("model:\n  n_layer: 2\n")
    (root / "checkpoints" / "demo" / "train_log.jsonl").write_text("\n".join([
        json.dumps({"event": "session_start", "time": T0, "run": "demo", "pid": 1,
                    "start_step": 0, "max_steps": 100, "tokens_per_step": 1000}),
        json.dumps({"step": 99, "loss": 2.0, "time": T0 + 100, "tokens_per_step": 1000}),
        json.dumps({"event": "session_end", "time": T0 + 1000, "run": "demo",
                    "last_step": 99, "elapsed": 1000.0}),
    ]) + "\n")

    led = Ledger(root / "logs" / "energy.jsonl")
    for r in steady(101, 360.0, run="demo"):     # 500s of a 1000s run
        led.fold(r)
    led.close()
    rep = costmod.report(led, _cfg(per_kwh=0.30), store=RunStore(root))
    demo = rep["runs"][0]
    assert demo["label"] == "demo"
    assert demo["coverage"] == pytest.approx(0.5, abs=0.01)
    assert demo["tokens"] == 100_000
    # Half the run was recorded, so the whole run cost about twice the measured amount...
    assert demo["estimated_money"] == pytest.approx(demo["money"] * 2, rel=0.02)
    # ...and cost-per-token must use the tokens of the measured half, not all of them.
    # Against all 100k tokens the figure would be half this, and look precise while being
    # wrong by exactly the part of the run nobody watched.
    assert demo["per_mtoken"] == pytest.approx(demo["money"] / 0.05, rel=0.02)
