"""Associative recall (longctx/recall.py): a task that is all retrieval.

The scorer is held to two models whose answers are known: one that really looks each key up
must score 100%, one that copies the previous token must score 0. Between them they pin the
off-by-one that matters most (the value at `pos` is read from the logits at `pos - 1`).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from aksharallm.longctx import recall
from aksharallm.longctx.recall import (KEY0, N_VALUES, QUERY, VALUE0, VOCAB, accuracy,
                                       document, max_len, prepare, score)


class _Lookup(torch.nn.Module):
    """After '?', at a key, puts all its mass on the value that followed that key earlier."""
    cfg = SimpleNamespace()

    def forward(self, x, full_logits=True):
        out = torch.zeros(*x.shape, len(VOCAB))
        for b, row in enumerate(x.tolist()):
            seen, after_q = {}, False
            for t, tok in enumerate(row):
                if tok == QUERY:
                    after_q = True
                if not after_q and KEY0 <= tok < VALUE0 and t + 1 < len(row):
                    seen[tok] = row[t + 1]
                if after_q and tok in seen:
                    out[b, t, seen[tok]] = 10.0
        return out, None


class _Copy(torch.nn.Module):
    cfg = SimpleNamespace()

    def forward(self, x, full_logits=True):
        return 10.0 * torch.nn.functional.one_hot(x, len(VOCAB)).float(), None


def test_a_document_hides_each_answer_behind_its_key():
    rng = np.random.default_rng(0)
    toks, answers = document(rng, pairs=10, gap=5, queries=4)
    assert len(answers) == 4 and toks.count(QUERY) == 1
    pairs = {toks[i]: toks[i + 1] for i in range(1, 21, 2)}
    assert len(pairs) == 10                                   # keys are distinct
    for pos in answers:
        assert toks[pos] == pairs[toks[pos - 1]]               # the value of the key before it
    assert max(toks) < len(VOCAB) == 148


def test_the_scorer_gives_a_real_lookup_everything_and_a_copier_nothing():
    for model, want in ((_Lookup(), 1.0), (_Copy(), 0.0)):
        r = accuracy(model, pairs=12, gap=40, n=8)
        assert r["accuracy"] == want and r["n"] == 8 * 8
        assert r["chance_any"] == 1 / N_VALUES and r["chance_in_doc"] == 1 / 12


def test_the_sweeps_reach_past_the_trained_window_and_fit_the_rope_table():
    r = score(_Lookup(), n=2, by_pairs=(8,), by_gap=(0, 768))
    assert [x["accuracy"] for x in r["by_pairs"] + r["by_gap"]] == [1.0, 1.0, 1.0]
    assert r["by_gap"][-1]["length"] > 512
    assert max_len() >= max(x["length"] for x in score(_Lookup(), n=1)["by_gap"])


def test_the_training_data_stays_in_its_vocabulary(tmp_path):
    meta = prepare(tmp_path, train_tokens=20_000, val_tokens=2_000, easy_tokens=5_000,
                   progress=None)
    easy = np.fromfile(tmp_path / "easy.bin", dtype=np.uint16)
    assert easy.size == 5_000 and easy.max() < len(VOCAB)
    assert not ((easy >= recall.FILLER0) & (easy < len(VOCAB))).any()   # no gap in easy docs
    arr = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert arr.size == 20_000 and arr.max() < len(VOCAB) and meta["documents"]["train"] > 50
    assert (arr == QUERY).sum() == meta["documents"]["train"] or (arr == QUERY).sum() + 1 == meta["documents"]["train"]


def test_two_results_pair_up_query_by_query():
    """The lookup against the copier on the same documents: every query is b-wrong-a-right,
    so the compare must count all of them in one column and none in the other. Mixing up the
    columns, or pairing queries from different documents, cannot pass this."""
    look = score(_Lookup(), n=2, by_pairs=(8,), by_gap=(0, 768))
    copy = score(_Copy(), n=2, by_pairs=(8,), by_gap=(0, 768))
    assert look["by_gap"][0]["correct"] == "1" * 16 and copy["by_gap"][0]["correct"] == "0" * 16
    rows = recall.compare(copy, look)
    assert [(r["b_only"], r["a_only"]) for r in rows] == [(16, 0)] * 3
    assert rows[0]["p"] < 1e-4
    other = score(_Lookup(), n=3, by_pairs=(8,), by_gap=(0, 768))
    try:
        recall.compare(copy, other)
    except ValueError as e:
        assert "different documents" in str(e)
    else:
        raise AssertionError("compared results scored on different documents")


def test_the_portal_pairs_two_saved_results_and_refuses_old_ones(tmp_path):
    import json

    import pytest

    from aksharallm.infer.checkpoints import InferError
    from aksharallm.portal.longctx import LongContext
    out = tmp_path / "logs" / "longctx"
    out.mkdir(parents=True)
    look = score(_Lookup(), n=2, by_pairs=(8,), by_gap=(0,))
    copy = score(_Copy(), n=2, by_pairs=(8,), by_gap=(0,))
    (out / "recall-a.json").write_text(json.dumps(copy))
    (out / "recall-b.json").write_text(json.dumps(look))
    for sweep in ("by_pairs", "by_gap"):
        for x in copy[sweep]:
            del x["correct"]
    (out / "recall-old.json").write_text(json.dumps(copy))
    lc = LongContext(None, root=tmp_path)
    rows = lc.recall_compare("recall-a.json", "recall-b.json")["rows"]
    assert [(r["b_only"], r["a_only"]) for r in rows] == [(16, 0), (16, 0)]
    with pytest.raises(InferError, match="per-query"):
        lc.recall_compare("recall-old.json", "recall-b.json")
