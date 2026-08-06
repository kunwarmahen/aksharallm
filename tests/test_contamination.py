"""Tests for the contamination check and the per-domain loss split.

The first thing a contamination checker has to prove is that it can find something. A
scanner that reports 0% because it is broken looks exactly like a clean corpus, and it is
the more comfortable of the two answers — so the positive control here is the point of the
file, not a formality: plant a known item in a fake corpus and require it to be found.

The rest of it defends the specific ways an overlap check goes quietly wrong:

  * an n-gram straddling a chunk boundary (loses hits, reports too clean),
  * items shorter than n (cannot be checked, must not be counted as clean),
  * question-vs-answer conflation (the whole distinction the report exists to draw),
  * a hash collision reported as a real hit (`verify` drops it).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aksharallm.eval import contamination as con
from aksharallm.eval import domains as dom


class Tok:
    """A whitespace tokenizer with a stable vocabulary, so token ids are reproducible."""

    def encode(self, text, bos=False):
        return [(abs(hash(w)) % 30000) + 100 for w in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class Item:
    def __init__(self, iid, context, choices, gold):
        self.id, self.context, self.choices, self.gold = iid, context, choices, gold


def corpus(*token_runs, filler=5_000, seed=0):
    """A fake `.bin`: random filler with the given token runs planted inside it."""
    rng = np.random.default_rng(seed)
    parts = [rng.integers(100, 30100, size=filler, dtype=np.uint16)]
    for run in token_runs:
        parts.append(np.asarray(run, dtype=np.uint16))
        parts.append(rng.integers(100, 30100, size=filler, dtype=np.uint16))
    return np.concatenate(parts)


def write_bin(tmp_path, tokens, name="train.bin"):
    path = tmp_path / name
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    return path


LONG = ("the mitochondrion is the powerhouse of the cell and it converts chemical energy "
        "stored in nutrients into adenosine triphosphate for cellular work")


# ---- the hash --------------------------------------------------------------------------

def test_the_hash_is_position_independent_and_content_dependent():
    a = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    ha = con.ngram_hashes(a, 3)
    assert ha.size == 3
    # The same 3-gram anywhere hashes the same...
    b = np.array([9, 9, 1, 2, 3], dtype=np.uint16)
    assert con.ngram_hashes(b, 3)[-1] == ha[0]
    # ...and a different one does not.
    assert ha[0] != ha[1]


def test_a_sequence_shorter_than_n_produces_nothing():
    assert con.ngram_hashes(np.array([1, 2], dtype=np.uint16), 13).size == 0


# ---- the positive control: it must be able to find something ---------------------------

def test_a_planted_item_is_found(tmp_path):
    """The test the whole module stands on. If this fails, every 0% is meaningless."""
    tok = Tok()
    item = Item("q1", LONG, ["wrong", "right answer here"], 1)
    texts = con.item_texts("toy", [item])
    probe = con.build_probe(texts, tok, n=13)

    planted = tok.encode(f"{LONG} right answer here")
    path = write_bin(tmp_path, corpus(planted))
    hits = con.scan_bin(path, probe)

    assert hits, "the planted item was not found — the scanner is broken"
    assert any(k.endswith("answered") for k in hits)


def test_a_corpus_without_the_item_is_clean(tmp_path):
    """The negative control. Both directions have to work or neither number means
    anything."""
    tok = Tok()
    probe = con.build_probe(con.item_texts("toy", [Item("q1", LONG, ["a", "b"], 1)]), tok, 13)
    path = write_bin(tmp_path, corpus())          # filler only
    assert con.scan_bin(path, probe) == {}


def test_an_ngram_straddling_a_chunk_boundary_is_still_found(tmp_path):
    """Chunks overlap by n-1 for exactly this. Without the overlap one window per chunk is
    lost, which makes the report quietly optimistic — the wrong direction to be wrong in."""
    tok = Tok()
    probe = con.build_probe(con.item_texts("toy", [Item("q1", LONG, ["a", "b"], 1)]), tok, 13)
    planted = tok.encode(LONG)
    # Put the planted run so that it sits across a chunk edge.
    chunk = 1000
    lead = chunk - len(planted) // 2
    rng = np.random.default_rng(1)
    tokens = np.concatenate([
        rng.integers(100, 30100, size=lead, dtype=np.uint16),
        np.asarray(planted, dtype=np.uint16),
        rng.integers(100, 30100, size=2000, dtype=np.uint16)])
    path = write_bin(tmp_path, tokens)
    assert con.scan_bin(path, probe, chunk=chunk), "lost across the chunk boundary"


# ---- question vs answer ------------------------------------------------------------------

def test_the_question_and_the_answer_are_tracked_separately(tmp_path):
    """A corpus holding the *question* is common and mostly harmless. One holding the
    question with its answer attached is what makes a score meaningless. Collapsing the two
    throws away the only part a reader needs."""
    tok = Tok()
    item = Item("q1", LONG, ["wrong one", "right answer here"], 1)
    probe = con.build_probe(con.item_texts("toy", [item]), tok, 13, keep_tokens=True)

    path = write_bin(tmp_path, corpus(tok.encode(LONG)))    # question only
    hits = con.scan_bin(path, probe)
    parts = {k.split("\t")[2] for k in hits}
    assert "question" in parts
    out = con.summarise(hits, probe, 13)
    by_part = {p: v for p, v in out["suites"][0]["parts"].items()}
    assert by_part["question"]["dirty"] == 1
    assert by_part["answered"]["dirty"] == 0, "the answer never appeared in this corpus"


def test_items_too_short_to_check_are_not_counted_as_clean():
    """`checkable` exists so a suite of one-line questions cannot report 0% dirty when the
    truth is 0% *checked*."""
    tok = Tok()
    probe = con.build_probe(con.item_texts("toy", [Item("q1", "too short", ["a", "b"], 0)]),
                            tok, 13)
    out = con.summarise({}, probe, 13)
    part = out["suites"][0]["parts"]["question"]
    assert part["checkable"] == 0 and part["too_short"] == 1
    assert part["rate"] is None, "a rate over zero checkable items is not 0%, it is unknown"


def test_verify_drops_a_hash_collision(tmp_path):
    """A 64-bit match is *probably* an n-gram match. Verification re-reads the tokens at the
    position the scan recorded, so a collision cannot become a finding somebody quotes."""
    tok = Tok()
    item = Item("q1", LONG, ["a", "b"], 1)
    probe = con.build_probe(con.item_texts("toy", [item]), tok, 13, keep_tokens=True)
    clean = write_bin(tmp_path, corpus())
    key = next(k for k in probe.sizes if probe.sizes[k])
    # A hit pointing at filler: the tokens there are not the item's, so it must not survive.
    assert con.verify({key: 1}, probe, {key: (str(clean), 10)}, 13) == {}


def test_verify_keeps_a_real_hit_and_costs_one_seek(tmp_path):
    """The other direction, and the reason `scan_bin` records positions at all: a genuine
    hit is confirmed by reading `n` tokens, not by another pass over the corpus."""
    tok = Tok()
    item = Item("q1", LONG, ["a", "b"], 1)
    probe = con.build_probe(con.item_texts("toy", [item]), tok, 13, keep_tokens=True)
    path = write_bin(tmp_path, corpus(tok.encode(LONG)))
    where: dict[str, tuple[str, int]] = {}
    hits = con.scan_bin(path, probe, where=where)
    assert hits and where, "the scan must record where each hit came from"
    assert con.verify(hits, probe, where, 13) == hits


# ---- the clean score ---------------------------------------------------------------------

def test_the_clean_score_drops_contaminated_items():
    result = {"score": 0.75, "items": [{"id": "a", "correct": True},
                                       {"id": "b", "correct": True},
                                       {"id": "c", "correct": True},
                                       {"id": "d", "correct": False}]}
    out = con.clean_score(result, {"a", "b"})
    assert out["dropped"] == 2 and out["kept"] == 2
    assert out["clean"] == pytest.approx(0.5)


def test_the_clean_score_is_unavailable_without_per_item_verdicts():
    """`--no-items` produces a result this cannot re-score, and saying so beats inventing
    a number."""
    assert con.clean_score({"score": 0.6}, {"a"}) is None


def test_every_item_contaminated_is_reported_rather_than_divided_by_zero():
    out = con.clean_score({"score": 1.0, "items": [{"id": "a", "correct": True}]}, {"a"})
    assert out["clean"] is None and "every item" in out["note"]


# ---- per-domain loss ----------------------------------------------------------------------

def test_spans_derived_from_weights_tile_the_file_exactly(tmp_path):
    val = write_bin(tmp_path, np.zeros(1000, dtype=np.uint16), "val.bin")
    spans = dom.derive_spans(val, [{"bin": "a.bin", "weight": 0.85},
                                   {"bin": "b.bin", "weight": 0.15}])
    assert [(s.start, s.end) for s in spans] == [(0, 850), (850, 1000)]
    assert sum(s.tokens for s in spans) == 1000, "the spans must tile the file"


def test_the_last_span_absorbs_the_rounding(tmp_path):
    """Weights that do not divide the file evenly must not leave a token nobody owns."""
    val = write_bin(tmp_path, np.zeros(1001, dtype=np.uint16), "val.bin")
    spans = dom.derive_spans(val, [{"bin": "a", "weight": 1 / 3},
                                   {"bin": "b", "weight": 1 / 3},
                                   {"bin": "c", "weight": 1 / 3}])
    assert spans[-1].end == 1001 and sum(s.tokens for s in spans) == 1001


def test_a_manifest_beats_the_derivation(tmp_path):
    val = write_bin(tmp_path, np.zeros(1000, dtype=np.uint16), "val.bin")
    dom.manifest_path(val).write_text(json.dumps(
        {"spans": [{"name": "prose", "start": 0, "end": 700, "weight": 0.7},
                   {"name": "code", "start": 700, "end": 1000, "weight": 0.3}]}))
    spans = dom.spans_for(val, [{"bin": "a", "weight": 0.85}], tok=None)
    assert [(s.name, s.end) for s in spans] == [("prose", 700), ("code", 1000)]
    assert all(s.verified for s in spans)


def test_a_boundary_in_the_wrong_place_is_reported_not_used():
    """A split with the split in the wrong place is worse than no split: two plausible
    numbers that are both averages of the same mixture."""
    assert dom.looks_like_code("def f(x):\n    import os\n    return self.x") > 0.25
    assert dom.looks_like_code("The Independent Jane, for all the love and romance") < 0.15


def test_verification_marks_a_mismatched_span(tmp_path):
    class RealTok:
        def decode(self, ids):
            return "def f(): import os; return self.x" if ids and ids[0] > 500 else "plain prose here"

    tokens = np.concatenate([np.full(600, 10, dtype=np.uint16),
                             np.full(600, 900, dtype=np.uint16)])
    val = write_bin(tmp_path, tokens, "val.bin")
    # Names deliberately the wrong way round: prose first is right, code first is not.
    good = dom.verify_spans(val, [dom.Span("fineweb-edu", 0, 600),
                                  dom.Span("codeparrot-python", 600, 1200)], RealTok())
    assert [s.verified for s in good] == [True, True]
    bad = dom.verify_spans(val, [dom.Span("codeparrot-python", 0, 600),
                                 dom.Span("fineweb-edu", 600, 1200)], RealTok())
    assert [s.verified for s in bad] == [False, False]


def test_an_unrecognised_source_name_is_unverified_not_passed(tmp_path):
    class T:
        def decode(self, ids):
            return "some text"

    val = write_bin(tmp_path, np.zeros(1000, dtype=np.uint16), "val.bin")
    spans = dom.verify_spans(val, [dom.Span("mystery-corpus", 0, 1000)], T())
    assert spans[0].verified is None, "no opinion must not read as a pass"


def test_the_blend_of_the_parts_is_computable():
    rows = [{"name": "a", "loss": 2.0, "weight": 0.85},
            {"name": "b", "loss": 1.0, "weight": 0.15}]
    assert dom.blended(rows) == pytest.approx(2.0 * 0.85 + 1.0 * 0.15)


def test_the_rescore_drops_answer_leaks_only(tmp_path):
    """`dirty_ids` feeds the clean score. Dropping every item whose *question* appears in a
    web crawl would discard most of a public benchmark and then report a confident number
    computed on the remainder — which is worse than not checking at all."""
    tok = Tok()
    items = [Item("q1", LONG, ["wrong one", "right answer here"], 1),
             Item("q2", LONG.replace("mitochondrion", "chloroplast"), ["no", "yes indeed"], 1)]
    probe = con.build_probe(con.item_texts("toy", items), tok, 13)

    # A corpus with q1's question AND answer, but only q2's question.
    path = write_bin(tmp_path, corpus(tok.encode(f"{LONG} right answer here"),
                                      tok.encode(items[1].context)))
    out = con.summarise(con.scan_bin(path, probe), probe, 13)
    assert out["dirty_ids"] == ["q1"], "only the answer leak should drive the re-score"
    assert "q2" in out["question_ids"], "the question leak is still reported"


def test_the_clean_score_matches_the_real_result_schema():
    """The shape `eval run` actually writes: suites -> {score, items:[{id, correct}]}."""
    real = {"score": 0.5, "items": [{"id": "arc/Mercury_1", "gold": 0, "pred": 0,
                                     "correct": True},
                                    {"id": "arc/Mercury_2", "gold": 1, "pred": 0,
                                     "correct": False}]}
    out = con.clean_score(real, {"arc/Mercury_1"})
    assert out["kept"] == 1 and out["clean"] == 0.0 and out["dropped"] == 1


def test_a_saved_report_can_rescore_without_scanning_again(tmp_path):
    """`--report` reuses a scan instead of repeating it. A half-hour pass over ten billion
    tokens whose output cannot be re-used is a check that stops being run — so the report's
    own JSON has to be enough to re-score any result."""
    tok = Tok()
    items = [Item("q1", LONG, ["wrong one", "right answer here"], 1),
             Item("q2", LONG.replace("mitochondrion", "chloroplast"), ["no", "yes indeed"], 1)]
    probe = con.build_probe(con.item_texts("toy", items), tok, 13)
    path = write_bin(tmp_path, corpus(tok.encode(f"{LONG} right answer here"),
                                      tok.encode(items[1].context)))
    report = con.summarise(con.scan_bin(path, probe), probe, 13)

    # Round-trip through JSON exactly as the CLI writes and reads it.
    saved = json.loads(json.dumps(report))
    result = {"score": 1.0, "items": [{"id": "q1", "correct": True},
                                      {"id": "q2", "correct": False}]}
    clean = con.clean_score(result, set(saved["dirty_ids"]))
    assert clean["dropped"] == 1 and clean["kept"] == 1
    assert clean["clean"] == 0.0, "only q1 leaked its answer, and q1 was the correct one"
