"""Calibration: is the model's confidence honest, and does the measurement of that lie?

A calibration number is easy to compute and easy to compute wrongly, and every wrong version
looks plausible. So the tests here are mostly *constructions with a known answer*: a model
that is perfectly calibrated by construction must score zero, one that is overconfident by a
known amount must score that amount, and the property that makes temperature scaling free —
that it cannot change accuracy — is asserted rather than assumed.

The one that matters most is `test_a_degenerate_model_scores_perfectly`. ECE has a failure
mode that no amount of care in the implementation fixes: a model that always predicts the
base rate with the base rate's confidence is perfectly calibrated and completely useless. If
that test ever starts failing, the metric has been quietly redefined into something else.
"""

from __future__ import annotations

import math

import pytest
import torch

from aksharallm.eval.calibration import (
    BIN_COUNTS,
    buckets_equal_mass,
    buckets_equal_width,
    calibrate,
    ece_from,
    fit_temperature,
    perplexity,
    report,
)


def logits_for(probs: torch.Tensor) -> torch.Tensor:
    """Logits whose softmax is exactly `probs`, so a test can specify the distribution."""
    return torch.log(probs.clamp_min(1e-12))


def confident_model(n: int, confidence: float, accuracy: float, vocab: int = 4, seed: int = 0):
    """`n` predictions that claim `confidence` and are right `accuracy` of the time.

    The construction: the top class always carries `confidence`, the rest share what is left,
    and the target is the top class for exactly `accuracy·n` of them. Both numbers are then
    known exactly, so the gap — and therefore the ECE — is known exactly.
    """
    g = torch.Generator().manual_seed(seed)
    probs = torch.full((n, vocab), (1.0 - confidence) / (vocab - 1))
    probs[:, 0] = confidence
    targets = torch.zeros(n, dtype=torch.long)
    n_wrong = round(n * (1.0 - accuracy))
    wrong = torch.randperm(n, generator=g)[:n_wrong]
    targets[wrong] = 1  # a class that is not the argmax
    return logits_for(probs), targets


# ---------------------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------------------


def test_a_perfectly_calibrated_model_scores_zero():
    logits, targets = confident_model(2000, confidence=0.8, accuracy=0.8)
    cal = calibrate(logits, targets)
    assert cal.accuracy == pytest.approx(0.8, abs=0.01)
    assert cal.confidence == pytest.approx(0.8, abs=0.01)
    for bins in BIN_COUNTS:
        assert cal.ece[bins] < 0.02


def test_a_known_gap_is_reported_as_that_gap():
    """Claims 0.9, is right 0.6 of the time. ECE must be 0.3, not 0.3-ish."""
    logits, targets = confident_model(2000, confidence=0.9, accuracy=0.6)
    cal = calibrate(logits, targets)
    assert cal.ece[10] == pytest.approx(0.30, abs=0.02)
    assert cal.overconfident


def test_underconfidence_is_reported_too():
    """The sign is information: a model that is right more often than it claims is a
    different problem from one that is right less often, and both are miscalibration."""
    logits, targets = confident_model(2000, confidence=0.5, accuracy=0.9)
    cal = calibrate(logits, targets)
    assert not cal.overconfident
    assert cal.ece[10] == pytest.approx(0.40, abs=0.02)


def test_a_degenerate_model_scores_perfectly():
    """**The caveat that has to travel with every ECE.** A model that always predicts the
    base rate with the base rate's confidence is perfectly calibrated and useless. ECE is a
    companion to accuracy, never a substitute — and this test is what makes that concrete."""
    logits, targets = confident_model(2000, confidence=0.25, accuracy=0.25, vocab=4)
    cal = calibrate(logits, targets)
    assert cal.ece[10] < 0.02, "the degenerate model is calibrated, as it should be"
    assert cal.accuracy < 0.3, "...and useless, which ECE cannot see"


def test_mce_catches_a_single_bad_bucket():
    """The mean hides the shape. A model calibrated everywhere except where it sounds
    certain has a small ECE and is exactly the one not to trust when it sounds certain."""
    good_l, good_t = confident_model(1900, confidence=0.5, accuracy=0.5, seed=1)
    bad_l, bad_t = confident_model(100, confidence=0.95, accuracy=0.2, seed=2)
    cal = calibrate(torch.cat([good_l, bad_l]), torch.cat([good_t, bad_t]))
    assert cal.ece[10] < 0.06, "one small bucket barely moves the mean"
    assert cal.mce > 0.5, "...and MCE sees it immediately"


def test_ignored_targets_are_not_counted():
    """A position with no correct answer drags accuracy down while confidence stays put,
    which looks exactly like overconfidence."""
    logits, targets = confident_model(500, confidence=0.9, accuracy=0.9)
    padded_l = torch.cat([logits, logits[:500]])
    padded_t = torch.cat([targets, torch.full((500,), -100)])
    assert calibrate(padded_l, padded_t).n == 500
    assert calibrate(padded_l, padded_t).accuracy == pytest.approx(
        calibrate(logits, targets).accuracy)


def test_an_empty_input_does_not_crash():
    cal = calibrate(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
    assert cal.n == 0 and math.isnan(cal.accuracy)


# ---------------------------------------------------------------------------------------
# the bins
# ---------------------------------------------------------------------------------------


def test_equal_width_bins_span_the_whole_range_and_keep_the_endpoint():
    """A confidence of exactly 1.0 must land in the last bucket, not fall off the end."""
    conf = torch.tensor([0.0, 0.5, 1.0])
    correct = torch.tensor([True, True, True])
    bins = buckets_equal_width(conf, correct, 10)
    assert sum(b.count for b in bins) == 3
    assert bins[-1].count == 1


def test_equal_mass_bins_hold_equal_numbers():
    """The reason they exist: over a big vocabulary almost everything is low-confidence, so
    equal-width bins leave the interesting high-confidence buckets holding noise."""
    conf = torch.rand(1000)
    correct = torch.rand(1000) < 0.5
    bins = buckets_equal_mass(conf, correct, 10)
    counts = [b.count for b in bins]
    assert max(counts) - min(counts) <= 1


def test_the_bin_count_changes_the_answer_which_is_why_it_is_reported():
    """Not a bug — a property. An ECE quoted without its bin count is not reproducible."""
    torch.manual_seed(0)
    logits = torch.randn(3000, 8)
    targets = torch.randint(0, 8, (3000,))
    cal = calibrate(logits, targets)
    assert len({round(v, 4) for v in cal.ece.values()}) > 1


def test_ece_of_no_samples_is_not_a_number():
    ece, mce = ece_from([], 0)
    assert math.isnan(ece) and math.isnan(mce)


# ---------------------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------------------


def test_temperature_cannot_change_accuracy():
    """The property that makes temperature scaling free of the usual trade-off: dividing by
    a positive constant does not move the argmax. If this ever fails, the fix is not free."""
    torch.manual_seed(0)
    logits = torch.randn(500, 10)
    targets = torch.randint(0, 10, (500,))
    base = calibrate(logits, targets).accuracy
    for t in (0.5, 1.0, 2.0, 5.0):
        assert calibrate(logits, targets, temperature=t).accuracy == pytest.approx(base)


def test_temperature_moves_confidence_in_the_right_direction():
    torch.manual_seed(0)
    logits = torch.randn(500, 10) * 3
    targets = torch.randint(0, 10, (500,))
    cool = calibrate(logits, targets, temperature=3.0).confidence
    warm = calibrate(logits, targets, temperature=0.5).confidence
    assert cool < calibrate(logits, targets).confidence < warm


def test_fitting_recovers_a_temperature_that_was_applied():
    """Take a calibrated model, deliberately sharpen it by 2x, and check the fit undoes it."""
    torch.manual_seed(0)
    vocab = 20
    logits = torch.randn(4000, vocab) * 1.5
    targets = torch.distributions.Categorical(logits=logits).sample()
    fitted = fit_temperature(logits * 2.0, targets)  # sharpened -> should want T ~ 2
    assert 1.5 < fitted < 2.6, fitted


def test_an_already_calibrated_model_gets_a_temperature_near_one():
    torch.manual_seed(0)
    logits = torch.randn(4000, 20)
    targets = torch.distributions.Categorical(logits=logits).sample()
    assert fit_temperature(logits, targets) == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------


def test_the_report_fits_and_scores_on_different_halves():
    """A temperature fitted and evaluated on the same tokens reports the calibration of a
    model that has seen the answers — a smaller number, and a meaningless one."""
    torch.manual_seed(0)
    logits = torch.randn(2000, 10)
    targets = torch.randint(0, 10, (2000,))
    res = report(logits, targets)
    assert res["n_fit"] == 1000 and res["n_scored"] == 1000
    assert res["n_fit"] + res["n_scored"] == res["n_total"]


def test_the_report_says_which_way_the_temperature_went():
    torch.manual_seed(0)
    logits = torch.randn(2000, 10) * 4
    targets = torch.distributions.Categorical(logits=logits / 4).sample()
    res = report(logits, targets)
    assert "OVERconfident" in res["reading"] or "already" in res["reading"]
    assert "companion to accuracy" in res["caveat"]


def test_perplexity_is_the_familiar_cross_check():
    """If this disagrees with the run's own recorded val loss, the calibration numbers are
    being computed on something other than what the model trained on."""
    vocab = 8
    logits = torch.zeros(100, vocab)  # uniform
    targets = torch.randint(0, vocab, (100,))
    assert perplexity(logits, targets) == pytest.approx(vocab, rel=0.01)
