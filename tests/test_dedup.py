"""MinHash and LSH: an estimator, pinned against the thing it estimates.

The whole point of MinHash is that a cheap number approximates an expensive one, so the
tests are mostly *both* numbers side by side. `estimated_jaccard` against `jaccard`, on
documents whose true similarity is constructed rather than measured.

Two failure modes matter more than the arithmetic:

* **a deduplicator that finds nothing looks exactly like a clean corpus**, and is the more
  comfortable answer — the same trap `tests/test_contamination.py` leads with. So this file
  leads with a planted duplicate that must be found;
* **LSH's misses are invisible.** A similar pair that shares no band is never compared, so
  it does not show up as a near-miss. `detection_probability` has to be right, because it is
  the only thing that turns that miss rate into a number anyone can quote.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aksharallm.data.dedup import (
    PRIME,
    LSHParams,
    MinHashIndex,
    clusters,
    document_spans,
    estimated_jaccard,
    hash_family,
    jaccard,
    report,
    scan_bin,
    shingle_hashes,
    signature,
)

EOS = 0


def doc(n: int, seed: int) -> np.ndarray:
    """A pseudo-document. Ids start at 1 so no token is accidentally EOS."""
    return np.random.default_rng(seed).integers(1, 32_000, n, dtype=np.uint16)


def edited(base: np.ndarray, every: int, seed: int = 99) -> np.ndarray:
    """`base` with one token changed every `every` positions — a near-duplicate whose true
    similarity is set by construction rather than discovered."""
    out = base.copy()
    rng = np.random.default_rng(seed)
    out[::every] = rng.integers(1, 32_000, out[::every].size, dtype=np.uint16)
    return out


# ---------------------------------------------------------------------------------------
# the positive control
# ---------------------------------------------------------------------------------------


def test_a_planted_duplicate_is_found():
    """**Leads the file on purpose.** A deduplicator that reports nothing because it is
    broken looks exactly like a clean corpus, and that is the answer nobody investigates."""
    index = MinHashIndex()
    a = doc(600, 1)
    index.add(a)
    index.add(doc(600, 2))
    index.add(a.copy())  # the plant
    found = index.duplicates()
    assert found, "the planted duplicate was not found"
    assert (0, 2) in [(x, y) for x, y, _ in found]


def test_unrelated_documents_are_not_duplicates():
    index = MinHashIndex()
    for i in range(6):
        index.add(doc(600, i))
    assert index.duplicates() == []


# ---------------------------------------------------------------------------------------
# the estimator against the truth
# ---------------------------------------------------------------------------------------


def test_an_exact_copy_scores_one_both_ways():
    a = doc(500, 3)
    hf = hash_family(128, seed=0)
    sa, sb = signature(shingle_hashes(a), *hf), signature(shingle_hashes(a.copy()), *hf)
    assert estimated_jaccard(sa, sb) == 1.0
    assert jaccard(shingle_hashes(a), shingle_hashes(a)) == 1.0


def test_the_estimate_tracks_the_truth_within_its_own_error():
    """MinHash's standard error is `sqrt(t(1-t)/P)`. The estimate has to land inside a few
    of those, or the hash family is not behaving like a random permutation."""
    params = LSHParams()
    hf = hash_family(params.permutations, seed=0)
    base = doc(2000, 4)
    for every in (10, 25, 60, 200):
        other = edited(base, every)
        ha, hb = shingle_hashes(base), shingle_hashes(other)
        truth = jaccard(ha, hb)
        est = estimated_jaccard(signature(ha, *hf), signature(hb, *hf))
        tolerance = 4 * params.standard_error(truth) + 0.02
        assert abs(est - truth) < tolerance, (every, truth, est, tolerance)


def test_unrelated_documents_estimate_near_zero():
    hf = hash_family(128, seed=0)
    a = signature(shingle_hashes(doc(1500, 5)), *hf)
    b = signature(shingle_hashes(doc(1500, 6)), *hf)
    assert estimated_jaccard(a, b) < 0.05


def test_more_editing_means_less_similarity():
    """Monotonicity. An estimator that is accurate on average but not ordered would still
    pass the tolerance test above and be useless for ranking."""
    hf = hash_family(128, seed=0)
    base = doc(2000, 7)
    sims = [estimated_jaccard(signature(shingle_hashes(base), *hf),
                              signature(shingle_hashes(edited(base, every)), *hf))
            for every in (200, 60, 25, 10)]
    assert sims == sorted(sims, reverse=True), sims


def test_an_empty_document_matches_nothing():
    """It must not collide with everything. `PRIME` is above any real minimum, so no band
    of an empty document can share a bucket with a real one."""
    hf = hash_family(16, seed=0)
    empty = signature(np.empty(0, dtype=np.uint64), *hf)
    assert (empty == PRIME).all()
    real = signature(shingle_hashes(doc(400, 8)), *hf)
    assert estimated_jaccard(empty, real) == 0.0


def test_the_hash_family_is_seeded():
    """A dedup report that changes when rerun is not a measurement."""
    a1, b1 = hash_family(32, seed=5)
    a2, b2 = hash_family(32, seed=5)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, hash_family(32, seed=6)[0])


# ---------------------------------------------------------------------------------------
# the S-curve
# ---------------------------------------------------------------------------------------


def test_the_detection_curve_is_a_curve_and_not_a_step():
    """LSH is probabilistic. Pretending the threshold is sharp is how a miss rate becomes
    invisible: below the knee is *unlikely*, not impossible."""
    p = LSHParams(bands=16, rows=8)
    assert p.detection_probability(0.5) < 0.2
    assert p.detection_probability(0.9) > 0.99
    assert 0 < p.detection_probability(p.threshold) < 1


def test_the_threshold_matches_the_knee():
    p = LSHParams(bands=16, rows=8)
    assert p.threshold == pytest.approx((1 / 16) ** (1 / 8))
    # The rule of thumb should put the knee where the curve is genuinely bending.
    assert 0.3 < p.detection_probability(p.threshold) < 0.9


def test_more_bands_lowers_the_threshold():
    """The knob, stated as the thing it does: more bands means more candidates and a lower
    knee, at the cost of more pairs to check."""
    assert LSHParams(bands=32, rows=4).threshold < LSHParams(bands=8, rows=16).threshold


def test_the_standard_error_shrinks_with_more_permutations():
    assert LSHParams(bands=16, rows=16).standard_error(0.8) < \
        LSHParams(bands=8, rows=8).standard_error(0.8)


# ---------------------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------------------


def test_documents_are_split_on_eos():
    stream = np.array([1, 2, 3, EOS, 4, 5, 6, 7, EOS], dtype=np.uint16)
    assert document_spans(stream, EOS, min_tokens=3) == [(0, 3), (4, 8)]


def test_short_fragments_are_dropped():
    """A six-token document shares shingles with everything and would fill a duplicate
    report with noise."""
    stream = np.array([1, 2, EOS, 3, 4, 5, 6, 7, 8, EOS], dtype=np.uint16)
    assert document_spans(stream, EOS, min_tokens=5) == [(3, 9)]


def test_long_documents_are_truncated_and_that_is_reported():
    spans = document_spans(np.array([1] * 100 + [EOS], dtype=np.uint16), EOS,
                           min_tokens=1, max_tokens=20)
    assert spans == [(0, 20)]


def test_a_stream_with_no_eos_is_refused():
    """It would be treated as one enormous document and every number would be nonsense."""
    with pytest.raises(ValueError, match="no EOS"):
        document_spans(np.arange(1, 50, dtype=np.uint16), EOS)


# ---------------------------------------------------------------------------------------
# clustering and the report
# ---------------------------------------------------------------------------------------


def test_transitive_duplicates_become_one_cluster():
    """A~B and B~C puts all three together. Transitivity is an *assumption*, and the
    alternative — one representative per pair — removes far more than it should."""
    assert clusters([(0, 1, 0.9), (1, 2, 0.9)], 4) == [[0, 1, 2]]


def test_a_cluster_of_ten_removes_nine():
    """One of them is the original, and keeping it is the entire point."""
    index = MinHashIndex()
    a = doc(500, 11)
    for _ in range(10):
        index.add(a.copy())
    rep = report(index, index.duplicates(), 10, 5000)
    assert rep["clusters"] == 1
    assert rep["duplicate_documents"] == 9


def test_the_token_share_is_reported_not_just_the_document_share():
    """Documents are not equal. Dropping 200,000 stubs of 40 tokens changes almost nothing;
    dropping 3,000 duplicated articles changes the corpus measurably."""
    index = MinHashIndex()
    long_doc = doc(4000, 12)
    index.add(long_doc)
    index.add(long_doc.copy())
    index.add(doc(300, 13))
    rep = report(index, index.duplicates(), 3, 8300)
    assert rep["duplicate_documents"] == 1
    assert rep["duplicate_token_share"] > rep["duplicate_document_share"]


def test_the_caveat_and_the_curve_travel_with_the_numbers():
    index = MinHashIndex()
    index.add(doc(400, 14))
    rep = report(index, [], 1, 400)
    assert "ESTIMATES" in rep["caveat"] and "false negatives" in rep["caveat"]
    assert rep["curve"] and rep["standard_error"] > 0


def test_a_huge_bucket_does_not_explode_into_pairs():
    """5,000 copies of one licence header is 12 million pairs. Linking each to the first is
    enough to put them all in one cluster, and is linear."""
    index = MinHashIndex()
    a = doc(400, 15)
    for _ in range(200):
        index.add(a.copy())
    pairs = index.candidate_pairs()
    assert len(pairs) < 400, len(pairs)
    assert len(clusters([(x, y, 1.0) for x, y in pairs], 200)[0]) == 200


# ---------------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------------


def test_a_bin_file_scans_end_to_end(tmp_path):
    """The real entry point, over a real `.bin`, with duplicates planted at a known rate."""
    rng = np.random.default_rng(0)
    parts = []
    original = doc(400, 21)
    for i in range(30):
        body = original.copy() if i % 5 == 0 else rng.integers(1, 32_000, 400, dtype=np.uint16)
        parts.append(np.concatenate([body, np.array([EOS], dtype=np.uint16)]))
    path = tmp_path / "corpus.bin"
    np.concatenate(parts).astype("<u2").tofile(path)

    rep = scan_bin(path, eos_id=EOS, min_doc_tokens=50)
    assert rep["documents"] == 30
    # Six identical documents = one cluster, five removable.
    assert rep["clusters"] == 1
    assert rep["duplicate_documents"] == 5
    assert 0.1 < rep["duplicate_token_share"] < 0.25


def test_the_sample_flag_is_recorded(tmp_path):
    """A number from a sample and a number from a full pass are different claims."""
    parts = [np.concatenate([doc(200, i), np.array([EOS], dtype=np.uint16)]) for i in range(20)]
    path = tmp_path / "c.bin"
    np.concatenate(parts).astype("<u2").tofile(path)
    assert scan_bin(path, eos_id=EOS, limit=5, min_doc_tokens=50)["sampled"] is True
    assert scan_bin(path, eos_id=EOS, min_doc_tokens=50)["sampled"] is False


# ---------------------------------------------------------------------------------------
# the report has to survive the terminal
# ---------------------------------------------------------------------------------------

def _corpus(tmp_path, n: int = 20):
    parts = [np.concatenate([doc(200, i), np.array([EOS], dtype=np.uint16)]) for i in range(n)]
    path = tmp_path / "tinystories.bin"
    np.concatenate(parts).astype("<u2").tofile(path)
    return path


def test_a_scan_is_written_without_being_asked(tmp_path, monkeypatch):
    """It used to write JSON only when handed `--out`. The portal passed `--out` and so had
    a card; the same command typed by hand printed a table and kept nothing — so a dedup
    number could not be compared with one taken at another offset, which is the only honest
    way to read one, and the portal never saw a terminal scan at all.

    The name has to match what the portal globs for (`dedup-*.json`), or it is written and
    still invisible.
    """
    from aksharallm.data.dedup import main
    from aksharallm.portal.evals import EvalJobs

    # The destination is `report.results_dir()`, which resolves from the package rather than
    # from the shell's cwd — deliberately, so a scan run from anywhere lands where the portal
    # reads. Redirected here so the test does not write into the real logs/eval/.
    monkeypatch.setattr("aksharallm.eval.report.results_dir",
                        lambda root=None: tmp_path / "logs" / "eval")
    path = _corpus(tmp_path)
    assert main([str(path), "--limit", "10"]) == 0

    written = list((tmp_path / "logs" / "eval").glob("dedup-*.json"))
    assert len(written) == 1, "a scan run from a terminal left nothing behind"
    assert written[0].name.startswith("dedup-tinystories-")

    latest = EvalJobs(tmp_path).dedup()["latest"]
    assert latest["documents"] == 10
    assert latest["source"].endswith("tinystories.bin")


def test_out_still_chooses_the_destination(tmp_path):
    from aksharallm.data.dedup import main

    dest = tmp_path / "elsewhere" / "report.json"
    assert main([str(_corpus(tmp_path)), "--limit", "10", "--out", str(dest)]) == 0
    assert json.loads(dest.read_text())["documents"] == 10


def test_no_write_prints_only(tmp_path, monkeypatch):
    from aksharallm.data.dedup import main

    monkeypatch.setattr("aksharallm.eval.report.results_dir",
                        lambda root=None: tmp_path / "logs" / "eval")
    assert main([str(_corpus(tmp_path)), "--limit", "10", "--no-write"]) == 0
    assert not (tmp_path / "logs" / "eval").exists()
