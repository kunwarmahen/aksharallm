"""Serving accounting: what was served, and what share of the electricity produced nothing.

Three arithmetic mistakes would each make a cost-per-token number confidently wrong, and all
three are the kind nobody checks because the result still looks like a plausible price:

* **summing per-request durations instead of merging them.** The server batches, so thirty
  concurrent requests over ten seconds are ten seconds of card time. Summing gives 300 and
  makes a busy server look like it ran longer than the day contains;
* **adding prompt and completion tokens together.** Prefill and decode differ by orders of
  magnitude per token and their mix changes with every request, so the sum is an average
  over two different things;
* **charging idle energy to the tokens.** A server sitting loaded and unused still draws
  power. Folding that into the rate makes an under-used server look expensive *per token*
  when what is actually true is that it was mostly not serving — a different problem with a
  different fix.
"""

from __future__ import annotations

import json

import pytest

from aksharallm.portal.cost import CostConfig, Ledger, serving_report
from aksharallm.serve.usage import Request, UsageLog, busy_intervals, load, summarise


def req(t0, t1, prompt=10, completion=20, **kw) -> Request:
    return Request(t0, t1, prompt, completion, **kw)


# ---------------------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------------------


def test_a_request_round_trips_through_the_file(tmp_path):
    log = UsageLog(tmp_path / "usage.jsonl")
    log.record(req(100.0, 102.5, prompt=41, completion=128, run="small-code"))
    got = load(tmp_path / "usage.jsonl")
    assert len(got) == 1
    assert got[0].prompt == 41 and got[0].completion == 128 and got[0].run == "small-code"
    assert got[0].seconds == pytest.approx(2.5)


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path):
    """A server killed with `kill -9` mid-write leaves one truncated line. Losing the whole
    record because of it would be worse than losing the line."""
    p = tmp_path / "usage.jsonl"
    log = UsageLog(p)
    log.record(req(1.0, 2.0))
    with open(p, "a") as f:
        f.write('{"t0": 3.0, "t1":')
    assert len(load(p)) == 1


def test_accounting_never_fails_a_request(tmp_path):
    """The point of the server is to serve. A lost line costs a fraction of one report."""
    log = UsageLog(tmp_path / "nope" / "deep")  # a directory where a file should be
    log.path = tmp_path  # writing to a directory raises IsADirectoryError
    log.record(req(1.0, 2.0))  # must not raise


def test_a_missing_file_is_an_empty_record(tmp_path):
    assert load(tmp_path / "never-written.jsonl") == []


def test_the_window_keeps_requests_that_straddle_it(tmp_path):
    """A forty-second generation crossing the boundary did real work on both sides, and
    dropping it would under-count exactly the slow requests that matter most."""
    p = tmp_path / "u.jsonl"
    log = UsageLog(p)
    log.record(req(90.0, 110.0))  # straddles
    log.record(req(10.0, 20.0))  # entirely before
    got = load(p, since=100.0)
    assert len(got) == 1 and got[0].t0 == 90.0


def test_the_file_is_trimmed_rather_than_growing_forever(tmp_path):
    p = tmp_path / "u.jsonl"
    log = UsageLog(p, max_bytes=400)
    for i in range(40):
        log.record(req(float(i), float(i) + 0.5))
    assert p.stat().st_size <= 1200
    assert load(p), "trimming must keep the newest records, not all of them"


# ---------------------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------------------


def test_concurrent_requests_are_merged_not_summed():
    """**The one that matters.** The server batches: thirty requests over one ten-second
    window are ten seconds of card time, not three hundred."""
    concurrent = [req(0.0, 10.0) for _ in range(30)]
    assert busy_intervals(concurrent) == [(0.0, 10.0)]
    assert summarise(concurrent)["busy_seconds"] == pytest.approx(10.0)


def test_overlapping_spans_merge_and_gaps_do_not():
    spans = busy_intervals([req(0.0, 5.0), req(3.0, 8.0), req(20.0, 22.0)])
    assert spans == [(0.0, 8.0), (20.0, 22.0)]


def test_zero_length_requests_do_not_create_spans():
    assert busy_intervals([req(5.0, 5.0)]) == []


def test_prompt_and_completion_are_never_added_together():
    """Prefill and decode differ by orders of magnitude per token, so their sum is an
    average over two different things whose mix changes with every request."""
    s = summarise([req(0.0, 1.0, prompt=500, completion=10)])
    assert s["prompt_tokens"] == 500 and s["completion_tokens"] == 10
    assert "total_tokens" not in s


def test_throughput_is_over_card_time_not_wall_time():
    """Batching is a win and has to show up as one: eight concurrent requests producing 80
    tokens in one second is 80 tok/s of card time, not 10."""
    s = summarise([req(0.0, 1.0, completion=10) for _ in range(8)])
    assert s["completion_tok_per_s"] == pytest.approx(80.0)


def test_an_empty_summary_has_no_rate():
    s = summarise([])
    assert s["requests"] == 0 and s["completion_tok_per_s"] is None


# ---------------------------------------------------------------------------------------
# the money
# ---------------------------------------------------------------------------------------


def gpu_samples(spans, interval: float = 5.0) -> list[dict]:
    """Raw sampler records — `[(t0, t1, watts, label)]` — in the shape `gpu.py` writes.

    Built from the *real* sample shape and folded through the *real* `Ledger`, rather than
    writing ledger rows by hand. That distinction is not pedantry: the first version of these
    tests constructed entries directly, so it could not see that a bucket's `start` is a
    ten-minute boundary while its `seconds` is coverage scattered *inside* it. The report
    read `[start, start + seconds)` as a contiguous span and reported **zero** busy energy
    for a server that had been flat out — and every test passed.
    """
    out = []
    for t0, t1, watts, label in spans:
        t = t0
        while t < t1:
            rec = {"time": t, "gpus": [{"index": 0, "util": 50.0, "mem_used": 3000.0,
                                        "temp": 50.0, "power": watts}]}
            if label:
                rec["job"] = label
            out.append(rec)
            t += interval
    return out


def real_ledger(tmp_path, spans) -> Ledger:
    """A ledger filled the way the portal fills it: by folding sampler records."""
    led = Ledger(tmp_path / "energy.jsonl")
    led.backfill(gpu_samples(spans))
    led.close()
    return Ledger(tmp_path / "energy.jsonl")


def test_busy_energy_is_found_when_the_traffic_is_late_in_a_bucket(tmp_path):
    """**The regression.** Buckets are ten minutes wide. A server that was idle for the first
    eight minutes and flat out for the last two has `seconds` covering the whole bucket and
    busy spans only at the end — and reading `[start, start + seconds)` as the window happens
    to work, while `[start, start + covered)` on a *partly* covered bucket does not.

    Measured before the fix: a real server generating for 2m44s reported `0 Wh generating,
    100% of the server's energy produced nothing`.
    """
    base = 1_800_000_000.0  # exactly on a 600 s bucket boundary
    # Alive for the whole bucket, but only sampled (and only busy) in the last two minutes.
    usage = tmp_path / "usage.jsonl"
    log = UsageLog(usage)
    log.record(req(base + 480, base + 600, prompt=10, completion=60_000))
    ledger = real_ledger(tmp_path, [(base + 480, base + 600, 300.0, "serve")])

    rep = serving_report(ledger, CostConfig(), usage_path=usage)
    assert rep["busy_wh"] > 0, "the traffic was in the back of the bucket and vanished"
    assert rep["idle_share"] is not None and rep["idle_share"] < 0.2


def test_idle_energy_is_reported_beside_the_rate_not_inside_it(tmp_path):
    """A server loaded for an hour that generated for six minutes spent 90% of its
    electricity producing nothing. That is the number that decides whether to keep it up,
    and folding it into the per-token rate would hide it as "tokens are expensive"."""
    base = 1_800_000_000.0
    usage = tmp_path / "usage.jsonl"
    log = UsageLog(usage)
    log.record(req(base, base + 360, prompt=100, completion=100_000))  # 6 min generating
    # An hour alive at 1,000 W; six minutes of it generating.
    ledger = real_ledger(tmp_path, [(base, base + 3600, 1000.0, "serve")])

    rep = serving_report(ledger, CostConfig(), usage_path=usage)
    # Six minutes of a measured hour at 1,000 W is ~100 Wh generating and ~900 Wh not.
    assert rep["busy_wh"] == pytest.approx(100.0, rel=0.1)
    assert rep["idle_wh"] == pytest.approx(900.0, rel=0.1)
    assert rep["idle_share"] == pytest.approx(0.9, rel=0.1)
    # The rate uses the busy energy only: ~100 Wh for 100k tokens = ~1,000 Wh per million.
    assert rep["wh_per_million_completion"] == pytest.approx(1000.0, rel=0.1)


def test_the_rate_is_per_million_completion_tokens(tmp_path):
    usage = tmp_path / "usage.jsonl"
    base = 1_800_000_000.0
    # Alive and busy for the same 100 seconds at 1,800 W = 50 Wh.
    UsageLog(usage).record(req(base, base + 100, prompt=1_000_000, completion=500_000))
    ledger = real_ledger(tmp_path, [(base, base + 100, 1800.0, "serve")])
    rep = serving_report(ledger, CostConfig(), usage_path=usage)
    # ~50 Wh bought 500k completion tokens -> ~100 Wh per million. The million PROMPT tokens
    # do not appear in the rate at all, which is the point.
    assert rep["wh_per_million_completion"] == pytest.approx(100.0, rel=0.15)


def test_no_requests_means_no_rate_rather_than_zero(tmp_path):
    """Zero would read as "tokens are free", which is the opposite of the truth."""
    base = 1_800_000_000.0
    ledger = real_ledger(tmp_path, [(base, base + 3600, 1000.0, "serve")])
    rep = serving_report(ledger, CostConfig(), usage_path=tmp_path / "absent.jsonl")
    assert rep["requests"] == 0
    assert rep["wh_per_million_completion"] is None
    assert rep["money_per_million_completion"] is None


def test_an_unpriced_ledger_still_reports_energy(tmp_path):
    """The same rule the rest of the ledger obeys: without a rate, say the watt-hours rather
    than printing a price of zero."""
    usage = tmp_path / "usage.jsonl"
    base = 1_800_000_000.0
    UsageLog(usage).record(req(base, base + 100, completion=1000))
    rep = serving_report(real_ledger(tmp_path, [(base, base + 100, 180.0, "serve")]),
                         CostConfig(), usage_path=usage)
    assert not rep["configured"]
    assert rep["wh_per_million_completion"] is not None
    assert rep["money_per_million_completion"] is None


def test_only_the_servers_energy_is_counted(tmp_path):
    """A training run's kilowatt-hours must not be billed to the tokens a server produced."""
    usage = tmp_path / "usage.jsonl"
    base = 1_800_000_000.0
    UsageLog(usage).record(req(base, base + 100, completion=1000))
    led = Ledger(tmp_path / "energy.jsonl")
    led.backfill(gpu_samples([(base, base + 100, 100.0, "serve")]))
    led.close()
    # A training run in the same window, drawing 30x as much.
    with open(tmp_path / "energy.jsonl", "a") as f:
        f.write(json.dumps({"start": base, "seconds": 100.0, "wh": 9000.0,
                            "label": "small-code", "kind": "training", "index": 0}) + "\n")
    rep = serving_report(Ledger(tmp_path / "energy.jsonl"), CostConfig(), usage_path=usage)
    assert rep["total"]["wh"] < 100, "a training run's kWh was billed to the served tokens"


def test_the_caveat_travels_with_the_numbers(tmp_path):
    rep = serving_report(real_ledger(tmp_path, []), CostConfig(),
                         usage_path=tmp_path / "none.jsonl")
    assert "COMPLETION" in rep["caveat"] and "idle" in rep["caveat"]
