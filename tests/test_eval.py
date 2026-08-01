"""Tests for the eval harness.

The thing worth testing hardest is **scoring**, because a scoring bug does not fail — it
produces a plausible number that is quietly wrong, in the same direction, forever. The
log-likelihood tests below check it against a hand-computed answer rather than against
itself.

Nothing here downloads anything. The dataset layer is exercised against a cache written by
the test, which is also the honest way round: if the cache format changes, these fail.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.eval import judge, scoring, sources, suites
from aksharallm.model.transformer import Transformer


# ---- fixtures -------------------------------------------------------------------------

class FakeTokenizer:
    """One byte per token, so a test can reason about token counts by counting letters."""

    bos_id = 0
    eos_id = 0

    def encode(self, text, bos=False, eos=False):
        ids = [min(255, ord(c)) + 1 for c in text]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        return "".join(chr(i - 1) for i in ids if i > 0)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)
    return Transformer(cfg).eval()


@pytest.fixture
def tok():
    return FakeTokenizer()


# ---- the model change the harness needed ------------------------------------------------

def test_full_logits_matches_the_targets_path(tiny_model):
    """`full_logits=True` must return exactly what asking for a loss returns, minus the loss.

    It exists only to skip a cross-entropy the harness throws away. If it ever diverges from
    the training path, every benchmark number is computed on different logits from the ones
    the model is trained with — and nothing would say so.
    """
    idx = torch.randint(1, 300, (2, 9))
    with torch.no_grad():
        a, loss = tiny_model(idx, targets=idx)
        b, none = tiny_model(idx, full_logits=True)
    assert torch.allclose(a, b)
    assert none is None and loss is not None


def test_full_logits_does_not_change_plain_inference(tiny_model):
    """The default must still be last-position-only — that is the allocation that matters
    in generation, and a full logit tensor per decode step would be a silent slowdown."""
    idx = torch.randint(1, 300, (1, 9))
    with torch.no_grad():
        last, _ = tiny_model(idx)
        full, _ = tiny_model(idx, full_logits=True)
    assert last.shape[1] == 1 and full.shape[1] == 9
    assert torch.allclose(last[:, -1], full[:, -1])


# ---- log-likelihood scoring --------------------------------------------------------------

def _manual_logprob(model, tok, context, continuation):
    """The score, computed the slow obvious way, one sequence at a time and unbatched."""
    ctx = tok.encode(context, bos=True)
    cont = tok.encode(continuation)
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        logits, _ = model(ids[:, :-1], full_logits=True)
    lp = torch.log_softmax(logits.float(), dim=-1)[0]
    total = 0.0
    for i, token in enumerate(cont):
        total += float(lp[len(ctx) - 1 + i, token])
    return total


def test_loglikelihood_matches_a_hand_computed_score(tiny_model, tok):
    pairs = [("the cat sat on the", " mat"), ("hello there", " world")]
    got = scoring.loglikelihood(tiny_model, tok, pairs, device="cpu")
    for scored, (ctx, cont) in zip(got, pairs):
        assert scored.logprob == pytest.approx(_manual_logprob(tiny_model, tok, ctx, cont),
                                               abs=1e-3)


def test_batching_does_not_change_the_answer(tiny_model, tok):
    """Right-padding under a causal mask must not leak. A batch of mixed lengths has to
    score identically to the same pairs one at a time — this is the test that would catch
    a padding or an off-by-one in the target alignment."""
    pairs = [("a", " short"), ("a much longer context than the first one", " continuation"),
             ("mid length here", " x")]
    batched = scoring.loglikelihood(tiny_model, tok, pairs, device="cpu", batch_tokens=4096)
    one_by_one = [scoring.loglikelihood(tiny_model, tok, [p], device="cpu")[0] for p in pairs]
    for a, b in zip(batched, one_by_one):
        assert a.logprob == pytest.approx(b.logprob, abs=1e-4)


def test_results_come_back_in_the_order_they_were_asked(tiny_model, tok):
    """Scoring sorts by length internally for batching efficiency. If it forgot to undo
    that, every multiple-choice answer would be compared against a different question's
    score — and the accuracy would still look like a number."""
    pairs = [("z" * 30, " a"), ("z", " bbbbbbbb"), ("z" * 10, " cc")]
    got = scoring.loglikelihood(tiny_model, tok, pairs, device="cpu", batch_tokens=64)
    assert [s.n_chars for s in got] == [2, 9, 3]
    assert [s.n_tokens for s in got] == [2, 9, 3]


def test_the_continuation_is_encoded_separately_from_the_context(tiny_model, tok):
    """`encode(ctx + cont)` is not `encode(ctx) + encode(cont)` for a real BPE tokenizer,
    and counting continuation tokens backwards off the joined string is off by one wherever
    a merge crosses the boundary."""
    ids, n_cont = scoring._encode_pair(tok, "abc", " de", max_len=64)
    assert n_cont == 3                         # " de" is three bytes
    assert len(ids) == 1 + 3 + 3               # bos + "abc" + " de"


def test_an_empty_continuation_cannot_win_by_default(tiny_model, tok):
    """An empty string has a log-probability of zero, which beats every real answer. It has
    to become something scoreable or the model picks it every time."""
    ids, n_cont = scoring._encode_pair(tok, "abc", "", max_len=64)
    assert n_cont == 1


def test_overlong_pairs_keep_the_continuation_whole(tiny_model, tok):
    """Truncation trims the context, never the thing being scored — otherwise the score is
    of a different string from the one on the page."""
    ids, n_cont = scoring._encode_pair(tok, "x" * 200, " answer", max_len=32)
    assert len(ids) == 32
    assert n_cont == len(" answer")
    assert ids[-n_cont:] == tok.encode(" answer")


# ---- multiple choice ----------------------------------------------------------------------

def test_score_mc_reports_accuracy_and_a_baseline_aware_stderr(tiny_model, tok):
    items = [suites.MCItem(id=f"q{i}", context=f"Question {i}\nAnswer:",
                           choices=[" yes", " no"], gold=i % 2) for i in range(8)]
    result = scoring.score_mc(tiny_model, tok, items, device="cpu")
    assert result["n"] == 8
    assert 0.0 <= result["acc"] <= 1.0 and 0.0 <= result["acc_norm"] <= 1.0
    assert result["score"] == result["acc_norm"]
    # Binomial standard error, which is what says whether a two-point move means anything.
    p = result["acc_norm"]
    assert result["stderr"] == pytest.approx(math.sqrt(max(p * (1 - p), 1e-12) / 8))
    assert len(result["items"]) == 8


def test_length_normalisation_actually_changes_the_ranking(tiny_model, tok):
    """acc and acc_norm must be computed from different quantities. If normalisation were
    dropped they would be identical on every input, and HellaSwag — whose wrong endings are
    adversarially long — would be scored the way nobody scores it."""
    items = [suites.MCItem(id="q", context="ctx", gold=0,
                           choices=[" a", " a much much longer answer than the first"])]
    result = scoring.score_mc(tiny_model, tok, items, device="cpu")
    raw = scoring.loglikelihood(tiny_model, tok,
                                [("ctx", c) for c in items[0].choices], device="cpu")
    assert (raw[0].logprob > raw[1].logprob) == (result["acc"] == 1.0)
    assert ((raw[0].logprob / raw[0].n_chars > raw[1].logprob / raw[1].n_chars)
            == (result["acc_norm"] == 1.0))


def test_mmlu_groups_are_reported_per_subject(tiny_model, tok):
    items = [suites.MCItem(id=f"q{i}", context="c", choices=[" A", " B"], gold=0,
                           group="biology" if i < 3 else "history") for i in range(6)]
    result = scoring.score_mc(tiny_model, tok, items, device="cpu")
    assert set(result["groups"]) == {"biology", "history"}
    assert result["groups"]["biology"]["n"] == 3


# ---- suite construction --------------------------------------------------------------------

def test_mmlu_prompt_is_the_standard_letter_format():
    rows = [{"question": "What is 2+2?", "subject": "maths",
             "choices": ["3", "4", "5", "6"], "answer": 1}]
    shots = [{"question": "What is 1+1?", "subject": "maths",
              "choices": ["1", "2", "3", "4"], "answer": 1}]
    items = suites.build_mmlu(rows, shot_rows=shots, shots=5)
    assert len(items) == 1
    item = items[0]
    assert item.choices == [" A", " B", " C", " D"]
    assert item.gold == 1
    assert "about maths" in item.context
    assert "Answer: B\n\n" in item.context      # the shot, answered
    assert item.context.rstrip().endswith("Answer:")   # the question, not


def test_mmlu_shots_are_matched_by_subject():
    """The dev split exists to supply same-subject examples. Mixing subjects would teach
    the format and nothing else, which is a different (worse) benchmark."""
    rows = [{"question": "q", "subject": "biology", "choices": ["a", "b"], "answer": 0}]
    shots = [{"question": "chem shot", "subject": "chemistry", "choices": ["a", "b"],
              "answer": 0}]
    item = suites.build_mmlu(rows, shot_rows=shots, shots=5)[0]
    assert "chem shot" not in item.context


def test_arc_reads_the_gold_answer_out_of_the_label_list():
    """Not every ARC item has four options, and some label them 1-4 instead of A-D.
    Assuming a position is how you end up two points off the published number."""
    rows = [{"id": "x", "question": "q?",
             "choices": {"text": ["p", "q", "r"], "label": ["1", "2", "3"]},
             "answerKey": "3"}]
    item = suites.build_arc(rows)[0]
    assert item.gold == 2 and len(item.choices) == 3


def test_arc_skips_items_whose_answer_key_is_missing():
    rows = [{"id": "x", "question": "q", "choices": {"text": ["a"], "label": ["A"]},
             "answerKey": ""}]
    assert suites.build_arc(rows) == []


def test_hellaswag_strips_the_corpus_artifacts():
    rows = [{"ctx": "[header] Do the thing  [title] Then this", "activity_label": "Cooking",
             "endings": ["a [substeps] b", "c", "d", "e"], "label": "2"}]
    item = suites.build_hellaswag(rows)[0]
    assert "[header]" not in item.context and "[title]" not in item.context
    assert "[substeps]" not in item.choices[0]
    assert "  " not in item.context
    assert item.context.startswith("Cooking:")
    assert item.gold == 2


def test_hellaswag_skips_unlabelled_rows():
    """The test split ships without labels. Scoring those would silently count them all
    wrong and drag the number down."""
    assert suites.build_hellaswag([{"ctx": "c", "endings": ["a", "b"], "label": ""}]) == []


def test_gsm8k_shots_come_before_the_question_and_teach_the_answer_marker():
    rows = [{"question": "How many?", "answer": "Work.\n#### 7"}]
    shots = [{"question": "Shot?", "answer": "Reasoning here.\n#### 3"}]
    item = suites.build_gsm8k(rows, shot_rows=shots, shots=5)[0]
    assert item.gold == "7"
    assert item.prompt.index("Shot?") < item.prompt.index("How many?")
    assert "#### 3" in item.prompt              # the format is taught by example
    assert item.prompt.rstrip().endswith("Answer:")
    assert "\nQuestion:" in item.stop           # a base model writes the next question


@pytest.mark.parametrize("text,expected", [
    ("The answer is #### 42", "42"),
    ("blah #### 1,234", "1234"),
    ("#### -5", "-5"),
    ("#### 3.0", "3"),
    ("#### 3.5", "3.5"),
    ("no marker, just 17 at the end", "17"),
    ("$18 total", "18"),
    ("nothing numeric", None),
])
def test_gsm8k_answer_extraction(text, expected):
    assert suites.normalise_number(suites.extract_gsm_answer(text)) == expected


def test_gsm8k_grading_compares_numbers_not_strings():
    assert suites.gsm_correct("so #### 1,200", "1200") == (True, "1200")
    assert suites.gsm_correct("so #### 1200.0", "1200")[0]
    assert not suites.gsm_correct("so #### 1201", "1200")[0]


def test_humaneval_appends_the_check_call():
    """Without `check(entry_point)` the asserts are defined and never run, and every model
    scores 100%. This is the single most consequential line in the code suite."""
    rows = [{"task_id": "HumanEval/0", "prompt": "def f(x):\n", "test": "def check(f):\n    assert f(1) == 1",
             "entry_point": "f", "canonical_solution": ""}]
    item = suites.build_humaneval(rows)[0]
    assert item.tests.rstrip().endswith("check(f)")


# ---- the registry ---------------------------------------------------------------------------

def test_suite_groups_resolve_and_deduplicate():
    assert suites.resolve("fast") == list(suites.FAST_SUITES)
    assert suites.resolve("all") == list(suites.ALL_SUITES)
    assert suites.resolve("mmlu,mmlu,piqa") == ["mmlu", "piqa"]
    assert suites.resolve(None) == list(suites.DEFAULT_SUITES)
    assert all(suites.SUITES[n].kind == "mc" for n in suites.resolve("mc"))


def test_an_unknown_suite_names_the_known_ones():
    with pytest.raises(sources.EvalError, match="hellswag"):
        suites.resolve("hellswag")


def test_every_suite_says_what_to_expect():
    """The `expect` line is not decoration: 25% on MMLU is chance, and a reader who does
    not know that concludes the model is broken. A suite without one is a trap."""
    for name, suite in suites.SUITES.items():
        assert suite.expect and len(suite.expect) > 40, name
        assert suite.blurb, name


def test_datasets_for_includes_the_few_shot_source():
    needed = suites.datasets_for(["mmlu", "gsm8k"])
    assert set(needed) == {"mmlu", "mmlu-dev", "gsm8k", "gsm8k-train"}


# ---- the dataset cache -------------------------------------------------------------------------

def _write_cache(root: Path, name: str, rows: list[dict]):
    d = root / "data" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / f"{name}.meta.json").write_text(json.dumps({"name": name, "rows": len(rows),
                                                     "repo": "test/fixture"}))


def test_load_reads_the_cache_and_respects_a_limit(tmp_path):
    _write_cache(tmp_path, "piqa", [{"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": 0}
                                    for i in range(10)])
    rows = sources.load("piqa", tmp_path, limit=3)
    assert len(rows) == 3 and rows[0]["goal"] == "g0"


def test_limit_takes_the_first_rows_not_a_sample(tmp_path):
    """Deterministic beats representative: the point is that two checkpoints answer the
    same questions, and a seeded sample still changes the day anyone touches the seed."""
    _write_cache(tmp_path, "piqa", [{"goal": f"g{i}"} for i in range(10)])
    assert [r["goal"] for r in sources.load("piqa", tmp_path, limit=4)] == \
        ["g0", "g1", "g2", "g3"]


def test_missing_data_says_how_to_get_it(tmp_path):
    with pytest.raises(sources.EvalError, match="python -m aksharallm.eval fetch"):
        sources.load("mmlu", tmp_path, auto_fetch=False)


def test_status_reports_what_is_cached(tmp_path):
    _write_cache(tmp_path, "gsm8k", [{"question": "q", "answer": "#### 1"}])
    by_name = {row["name"]: row for row in sources.status(tmp_path)}
    assert by_name["gsm8k"]["cached"] and by_name["gsm8k"]["rows"] == 1
    assert not by_name["mmlu"]["cached"]


def test_piqa_lists_a_fallback_repository():
    """`ybisk/piqa` is a dataset script and stopped loading on datasets>=5. The fallback
    list is why PIQA still works; a single-repo spec would have silently lost the suite."""
    assert len(sources.spec("piqa").repos) > 1


# ---- the judge ---------------------------------------------------------------------------------

@pytest.mark.parametrize("reply,score", [
    ('{"score": 4, "reason": "correct and short"}', 4),
    ('Sure!\n{"score": 2, "reason": "wrong"}\n', 2),
    ('```json\n{"score": 5, "reason": "perfect"}\n```', 5),
    ("I would give this a 3 out of 5.", 3),
])
def test_judge_parses_whatever_shape_the_reply_arrives_in(reply, score):
    got, _ = judge.parse_grade(reply)
    assert got == score


def test_a_score_outside_one_to_five_is_rejected_rather_than_clamped():
    """A judge that answered 9 on a 1-5 scale did not understand the rubric, and clamping
    it to 5 would record a confident wrong grade as a perfect one. Missing is honest."""
    assert judge.parse_grade('{"score": 9}')[0] is None
    assert judge.parse_grade('{"score": 0}')[0] is None


def test_an_ungradeable_reply_is_missing_data_not_a_zero():
    """A judge that failed to answer must not be recorded as the model scoring badly."""
    score, reason = judge.parse_grade("the connection dropped")
    assert score is None and reason


def test_judge_score_is_rescaled_from_one_to_five():
    """1-5 maps to 0-1 as (mean-1)/4, so a model that scores 1 on everything sits at the
    floor rather than at a flattering 20%."""
    graded = [judge.Grade("a", "g", "p", "ans", 1, ""), judge.Grade("b", "g", "p", "ans", 5, "")]
    mean = sum(g.score for g in graded) / 2
    assert (mean - 1) / 4 == 0.5


def test_judge_config_reads_its_own_section(tmp_path):
    """The judge and the Code tab's explainer share one client and must not share one
    model: the explainer wants something small enough to run beside a training run, the
    judge wants the best model on the machine."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "portal.yaml").write_text(
        "explain:\n  model: small:1b\njudge:\n  model: big:70b\n  temperature: 0.0\n")
    cfg = judge.default_config(tmp_path)
    assert cfg.model == "big:70b" and cfg.temperature == 0.0

    from aksharallm.portal.explain import ExplainConfig
    assert ExplainConfig.load(tmp_path).model == "small:1b"


def test_every_judge_prompt_has_a_rubric():
    """The rubric is what makes the grade reproducible. Without one the judge grades
    against its own taste, which changes with its model version."""
    for item in suites.JUDGE_PROMPTS:
        assert item.rubric and len(item.rubric) > 20, item.id
        assert item.group


def test_a_real_bpe_merge_across_the_boundary_does_not_shift_the_score(tiny_model, tmp_path):
    """The counting rule, on a tokenizer that actually merges.

    Every other test in this file uses `FakeTokenizer`, which is one byte per token — so
    `encode(ctx + cont)` and `encode(ctx) + encode(cont)` are identical for it, and neither
    implementation of `_encode_pair` can be told from the other. That is fine for the rest of
    the scoring maths and useless for *this* property, which only exists because real BPE
    merges across the join.

    So this trains a small real tokenizer and picks a pair where the merge demonstrably
    happens. The first assertion is what makes the test able to fail at all: if the joined
    encoding ever stopped differing from the separate one, everything below would pass for
    the wrong reason.
    """
    from aksharallm.tokenizer.tokenizer import Tokenizer, train_bpe

    corpus = ["The quick brown fox jumps over the lazy dog again and again.",
              "She opened the door and saw a garden full of bright red flowers.",
              "He said hello to his friend and they walked to the park together."] * 60
    path = tmp_path / "tok.json"
    train_bpe(iter(corpus), vocab_size=512, out_path=path, min_frequency=1)
    tok = Tokenizer(path)

    context, continuation = "the la", "zy dog"
    joined = tok.encode(context + continuation)
    separate = tok.encode(context) + tok.encode(continuation)
    assert joined != separate, "this pair no longer merges — pick another or the test is moot"

    ids, n_cont = scoring._encode_pair(tok, context, continuation, max_len=64)

    # The continuation is scored as itself: its ids are exactly what it encodes to alone,
    # and they are the final n_cont ids of the pair.
    assert n_cont == len(tok.encode(continuation))
    assert ids[-n_cont:] == tok.encode(continuation)
    # And the naive rule -- encode the joined string, count backwards -- would have taken a
    # different number of tokens, which is the bug this guards.
    assert n_cont != len(joined) - len(tok.encode(context, bos=True)) + 1
