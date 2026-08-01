"""Tests for the synthetic-data pipeline.

The teacher is scripted throughout — nothing here talks to Ollama — which is the point of
`Teacher.ask` being the only method a recipe calls. What that leaves is everything that can
be wrong *about the data*, and those are the tests worth having: a filter that silently
passes everything, a duplicate check that only catches exact repeats, and above all the
mutation check, which is the one piece of this package whose failure mode is a dataset that
looks fine and teaches the model nothing.
"""

import json
import os

import pytest

from aksharallm.synth import filters, prompts, recipes, verify
from aksharallm.synth.dataset import Dataset, SynthError, list_datasets
from aksharallm.synth.run import GenerateOptions, generate
from aksharallm.synth.teacher import SynthConfig, Teacher, contention


# ---- fixtures ---------------------------------------------------------------------------

class ScriptedTeacher:
    """A teacher that answers from a list, and records what it was asked.

    Duck-typed against `Teacher`: the loop only ever calls `.ask()` and reads `.name` and
    `.cfg.host`, which is the whole reason the real client never has to be mocked.
    """

    def __init__(self, replies, name="fake:1b"):
        self.replies = list(replies)
        self.name = name
        self.asked = []
        self.cfg = type("cfg", (), {"host": "http://test"})()

    def ask(self, messages):
        from aksharallm.synth.teacher import Reply

        self.asked.append(messages)
        text = self.replies[min(len(self.asked) - 1, len(self.replies) - 1)]
        if isinstance(text, Exception):
            raise text
        return Reply(text=text, duration_s=0.01, model=self.name)


def py_reply(name="add_one", problem="Write a function that adds one to every number in a "
                                     "list and returns a new list.",
             solution=None, tests=None):
    solution = solution or (f"def {name}(values):\n"
                            "    return [v + 1 for v in values]")
    tests = tests or (f"assert {name}([1, 2]) == [2, 3]\n"
                      f"assert {name}([]) == []\n"
                      f"assert {name}([-1]) == [0]")
    return (f"### PROBLEM\n{problem}\n\n"
            f"### SOLUTION\n```python\n{solution}\n```\n\n"
            f"### TESTS\n```python\n{tests}\n```\n")


def chat_reply(prompt="Why does bread rise?", answer="Yeast eats the sugars in the dough "
                                                     "and gives off carbon dioxide, which "
                                                     "is trapped by the gluten."):
    return f"### PROMPT\n{prompt}\n\n### ANSWER\n{answer}\n"


def pref_reply(prompt="Name two uses for baking soda.",
               good="Deodorising a fridge and scrubbing a burnt pan.",
               bad="Baking soda has many uses around the home, and people have relied on "
                   "it for generations in all sorts of ways."):
    return f"### PROMPT\n{prompt}\n\n### GOOD\n{good}\n\n### BAD\n{bad}\n"


@pytest.fixture
def opts():
    return GenerateOptions(n=1, max_asks=4, sandbox_timeout_s=5)


# ---- the seed grid ----------------------------------------------------------------------

def test_seeds_do_not_repeat_a_cell_within_one_lap():
    seeds = prompts.seeds("python", 200, seed=3)
    cells = {(s.fields["topic"], s.fields["twist"], s.fields["difficulty"]) for s in seeds}
    assert len(cells) == 200, "200 seeds should be 200 distinct prompts, not 200 samples " \
                              "of a handful of popular ones"


def test_seeds_wrap_past_the_grid_rather_than_running_out():
    size = prompts.grid_size("python")
    seeds = prompts.seeds("python", size + 10, seed=0)
    assert len(seeds) == size + 10
    assert len({s.id for s in seeds}) == size + 10, "ids stay unique across a wrap"


def test_preference_seeds_carry_one_named_flaw():
    seeds = prompts.seeds("preference", 30)
    assert {s.fields["flaw"] for s in seeds} <= {f[0] for f in prompts.FLAWS}
    assert len({s.fields["flaw"] for s in seeds}) > 1, "flaws should vary across samples"


# ---- parsing ----------------------------------------------------------------------------

def test_sections_ignores_the_models_opening_sentence():
    text = "Sure! Here is an exercise.\n\n### PROBLEM\nDo a thing.\n\n### SOLUTION\nx = 1\n"
    got = recipes.sections(text)
    assert got["PROBLEM"] == "Do a thing."
    assert "Sure!" not in "".join(got.values())


def test_code_block_accepts_an_unfenced_answer():
    assert recipes.code_block("```python\ndef f():\n    pass\n```") == "def f():\n    pass"
    assert recipes.code_block("def f():\n    pass") == "def f():\n    pass"


def test_entry_point_prefers_the_ast_over_a_def_in_a_docstring():
    src = ('def helper(x):\n'
           '    """Not the entry point. def decoy(): pass"""\n'
           '    return x\n\n'
           'def solve(xs):\n'
           '    return [helper(x) for x in xs]\n')
    assert recipes.entry_point(src) == "solve"


def test_python_recipe_parses_all_three_sections():
    seed = prompts.seeds("python", 1)[0]
    sample = recipes.RECIPES["python"].parse(py_reply(), seed)
    assert sample["entry_point"] == "add_one"
    assert sample["tests"].count("assert") == 3
    assert sample["topic"] == seed.fields["topic"]


def test_a_missing_section_is_unparseable_not_a_half_sample():
    seed = prompts.seeds("python", 1)[0]
    with pytest.raises(SynthError):
        recipes.RECIPES["python"].parse("### PROBLEM\nDo a thing.\n", seed)


def test_preference_rejects_a_pair_that_is_the_same_answer_twice():
    seed = prompts.seeds("preference", 1)[0]
    with pytest.raises(SynthError) as exc:
        recipes.RECIPES["preference"].parse(pref_reply(good="Same.", bad="Same."), seed)
    assert "identical_pair" in str(exc.value)


# ---- filters ----------------------------------------------------------------------------

def test_boilerplate_is_dropped():
    assert filters.check_text("As an AI language model, I cannot bake bread for you at all.") \
        == "boilerplate"


def test_our_own_template_coming_back_is_a_reject():
    assert filters.check_text("### PROBLEM\nsomething that is quite long indeed here") \
        == "leaked_template"


def test_tests_must_mention_the_function_and_assert_more_than_once():
    sol = "def f(x):\n    return x"
    assert filters.check_code(sol, "assert f(1) == 1\nassert f(2) == 2", "f") is None
    assert filters.check_code(sol, "assert f(1) == 1", "f") == "bad_tests"
    assert filters.check_code(sol, "assert 1 == 1\nassert 2 == 2", "f") == "bad_tests"


def test_an_importing_test_block_is_rejected_because_there_is_no_module():
    sol = "def f(x):\n    return x"
    tests = "from solution import f\nassert f(1) == 1\nassert f(2) == 2"
    assert filters.check_code(sol, tests, "f") == "bad_tests"


def test_code_that_touches_the_machine_is_rejected():
    sol = "import os\ndef f(p):\n    return os.listdir(p)"
    assert filters.check_code(sol, "assert f('.')\nassert f('/')", "f") == "unsafe_code"


def test_deduper_catches_a_paraphrase_that_is_not_an_exact_repeat():
    dd = filters.Deduper(threshold=0.6)
    first = ("Write a function that counts how many times each word appears in a sentence "
             "and returns a dictionary of the counts.")
    dd.add(first)
    exact, _ = dd.check(first)
    assert exact == "duplicate"
    near, sim = dd.check(first.replace("dictionary of the counts", "dictionary with those "
                                       "counts"))
    assert near == "near_duplicate" and sim > 0.6
    other, _ = dd.check("Write a function that rotates a square matrix ninety degrees "
                        "clockwise, in place, without allocating a second matrix.")
    assert other is None


def test_the_dedup_threshold_is_the_knob_between_strict_and_permissive():
    """A heavier reword sits at ~0.5 and survives the default. That is the trade the
    threshold makes, and it is worth seeing rather than assuming."""
    first = ("Write a function that counts how many times each word appears in a sentence "
             "and returns a dictionary of the counts.")
    reworded = first.replace("sentence", "string of text")
    default = filters.Deduper(threshold=0.6)
    default.add(first)
    assert default.check(reworded)[0] is None
    strict = filters.Deduper(threshold=0.45)
    strict.add(first)
    assert strict.check(reworded)[0] == "near_duplicate"


# ---- verification: the part that matters ------------------------------------------------

def test_verify_passes_a_correct_solution_with_real_tests():
    verdict = verify.verify("def f(x):\n    return x * 2",
                            "assert f(2) == 4\nassert f(0) == 0", "f", timeout_s=5)
    assert verdict.ok and verdict.status == "pass"
    assert verdict.stub_status == "error", "the stub must raise, not pass"


def test_verify_fails_a_wrong_solution():
    verdict = verify.verify("def f(x):\n    return x * 3",
                            "assert f(2) == 4\nassert f(0) == 0", "f", timeout_s=5)
    assert not verdict.ok and verdict.status == "tests_failed"


def test_tests_that_never_call_the_function_are_caught_by_the_stub_run():
    """The whole reason `verify.py` runs the code twice.

    These tests pass, mention the function by name — which is all `check_code` can verify —
    and never depend on what it does. They pass against a function whose body has been
    deleted, so they would pass against any implementation at all.
    """
    verdict = verify.verify(
        "def dedupe(xs):\n    return list(dict.fromkeys(xs))",
        "assert callable(dedupe)\nassert dedupe.__name__ == 'dedupe'",
        "dedupe", timeout_s=5)
    assert not verdict.ok and verdict.status == "vacuous_tests"


def test_tests_that_swallow_the_exception_are_caught_too():
    """The other shape of the same bug, and the reason the stub *raises* rather than
    returning a default: a try/except around the call hides everything."""
    verdict = verify.verify(
        "def total(xs):\n    return sum(xs)",
        "try:\n    got = total([1, 2])\nexcept Exception:\n    got = 3\n"
        "assert got == 3\nassert isinstance(got, int)",
        "total", timeout_s=5)
    assert not verdict.ok and verdict.status == "vacuous_tests"


def test_a_weak_but_real_assertion_still_passes_and_that_is_honest():
    """`isinstance(f(x), list)` is a poor test, and the mutation check does NOT reject it —
    the stub raises, so the assert fails, which is the healthy signal. Knowing where the
    check stops matters more than believing it catches everything: what keeps this kind of
    sample honest is the two-assert floor and, in the end, the eval harness."""
    verdict = verify.verify(
        "def dedupe(xs):\n    return list(dict.fromkeys(xs))",
        "assert isinstance(dedupe([1, 1]), list)\nassert dedupe([1, 1]) is not None",
        "dedupe", timeout_s=5)
    assert verdict.ok


def test_stub_hollows_only_the_entry_point():
    src = ("def helper(x):\n    return x + 1\n\n"
           "def solve(xs):\n    return [helper(x) for x in xs]\n")
    hollow = verify.stub(src, "solve")
    assert "NotImplementedError" in hollow
    assert "return x + 1" in hollow, "the helper is left alone"


def test_stub_returns_none_when_the_entry_point_is_missing():
    assert verify.stub("def other(x):\n    return x", "solve") is None


def test_a_syntax_error_in_the_solution_fails_rather_than_crashing():
    verdict = verify.verify("def f(x)\n    return x", "assert f(1) == 1\nassert f(2) == 2",
                            "f", timeout_s=5)
    assert not verdict.ok and verdict.status == "tests_failed"


# ---- the loop ---------------------------------------------------------------------------

def test_a_good_python_sample_is_kept_with_its_provenance(tmp_path, opts):
    ds = Dataset("py-test", root=tmp_path)
    teacher = ScriptedTeacher([py_reply()])
    stats = generate(ds, "python", teacher, opts, root=tmp_path)

    assert stats.kept == 1 and stats.asked == 1
    row = ds.samples()[0]
    assert row["verified"] is True
    assert row["teacher"] == "fake:1b"
    assert row["seed"]["recipe"] == "python"
    meta = json.loads(ds.meta_path.read_text())
    assert meta["recipe"] == "python" and meta["teacher"] == "fake:1b"
    assert meta["options"]["verify"] is True


def test_the_loop_rejects_a_wrong_solution_and_records_why(tmp_path, opts):
    ds = Dataset("py-bad", root=tmp_path)
    bad = py_reply(solution="def add_one(values):\n    return values")
    stats = generate(ds, "python", ScriptedTeacher([bad]), opts, root=tmp_path)

    assert stats.kept == 0
    assert stats.rejected["tests_failed"] >= 1
    assert ds.rejects()[0]["reason"] == "tests_failed"
    assert ds.stats()["pass_rate"] == 0.0


def test_the_loop_drops_the_second_copy_of_the_same_problem(tmp_path):
    ds = Dataset("py-dupe", root=tmp_path)
    opts = GenerateOptions(n=2, max_asks=3, sandbox_timeout_s=5)
    stats = generate(ds, "python", ScriptedTeacher([py_reply()]), opts, root=tmp_path)

    assert stats.kept == 1, "the same reply every time is one sample, however often it is asked"
    assert stats.rejected.get("duplicate", 0) >= 1


def test_appending_to_a_dataset_still_sees_the_earlier_samples(tmp_path):
    """The dedup index is rebuilt from the file on open.

    Without this, stopping and restarting a generation run would let the second session
    happily re-generate everything the first one produced — and the duplicates would be
    invisible, because each session's own tally would look clean.
    """
    ds = Dataset("py-resume", root=tmp_path)
    opts = GenerateOptions(n=1, max_asks=2, sandbox_timeout_s=5)
    generate(ds, "python", ScriptedTeacher([py_reply()]), opts, root=tmp_path)

    again = Dataset("py-resume", root=tmp_path)
    stats = generate(again, "python", ScriptedTeacher([py_reply()]), opts, root=tmp_path)
    assert stats.kept == 0 and stats.rejected.get("duplicate", 0) >= 1
    assert again.n_samples() == 1


def test_a_dataset_is_one_recipe(tmp_path):
    ds = Dataset("mixed", root=tmp_path)
    generate(ds, "chat", ScriptedTeacher([chat_reply()]),
             GenerateOptions(n=1, max_asks=2), root=tmp_path)
    with pytest.raises(SynthError):
        Dataset("mixed", root=tmp_path).open("python", "fake:1b", "h", {}, 1)


def test_a_teacher_that_falls_over_is_counted_separately(tmp_path):
    ds = Dataset("py-broken", root=tmp_path)
    teacher = ScriptedTeacher([RuntimeError("ollama went away")])
    stats = generate(ds, "python", teacher, GenerateOptions(n=1, max_asks=2), root=tmp_path)
    assert stats.kept == 0
    assert stats.rejected["teacher_error"] >= 1
    assert "ollama went away" in stats.last_error


def test_stop_file_ends_the_run_at_the_number_of_kept_samples(tmp_path):
    """The same STOP contract the trainers use, with kept samples in place of steps."""
    stop = tmp_path / "STOP"
    stop.write_text("1")
    ds = Dataset("py-stop", root=tmp_path)
    replies = [py_reply(name=f"f{i}", problem=f"Exercise number {i} about lists and "
                                              f"slicing, returning a new list.",
                        solution=f"def f{i}(xs):\n    return list(xs)[:{i + 1}]",
                        tests=f"assert f{i}([1, 2, 3]) == [1, 2, 3][:{i + 1}]\n"
                              f"assert f{i}([]) == []")
               for i in range(5)]
    opts = GenerateOptions(n=5, max_asks=6, stop_file=stop, sandbox_timeout_s=5)
    stats = generate(ds, "python", ScriptedTeacher(replies), opts, root=tmp_path)
    assert stats.kept == 1
    assert "STOP file" in stats.stopped


def test_no_verify_marks_every_sample_rather_than_pretending(tmp_path):
    ds = Dataset("py-unverified", root=tmp_path)
    bad = py_reply(solution="def add_one(values):\n    return values")
    opts = GenerateOptions(n=1, max_asks=2, verify=False)
    stats = generate(ds, "python", ScriptedTeacher([bad]), opts, root=tmp_path)
    assert stats.kept == 1
    assert ds.samples()[0]["verified"] is False
    assert ds.stats()["options"]["verify"] is False


# ---- export -----------------------------------------------------------------------------

def test_python_export_is_the_shape_prepare_sft_reads(tmp_path):
    ds = Dataset("py-export", root=tmp_path)
    generate(ds, "python", ScriptedTeacher([py_reply()]),
             GenerateOptions(n=1, max_asks=2, sandbox_timeout_s=5), root=tmp_path)
    out = ds.export()
    rows = [json.loads(line) for line in open(out["path"])]
    assert out["rows"] == 1
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"].startswith("```python")
    assert "prepare_sft" in out["next"]


def test_preference_export_is_the_shape_prepare_dpo_reads(tmp_path):
    ds = Dataset("pref-export", root=tmp_path)
    generate(ds, "preference", ScriptedTeacher([pref_reply()]),
             GenerateOptions(n=1, max_asks=2), root=tmp_path)
    out = ds.export()
    row = json.loads(open(out["path"]).readline())
    assert set(row) == {"prompt", "chosen", "rejected"}
    assert row["chosen"] != row["rejected"]
    assert "prepare_dpo" in out["next"]


def test_export_refuses_an_empty_dataset(tmp_path):
    ds = Dataset("empty", root=tmp_path)
    ds.open("chat", "fake:1b", "h", {}, 1)
    with pytest.raises(SynthError):
        ds.export()


def test_list_datasets_reports_the_funnel(tmp_path):
    ds = Dataset("chat-listed", root=tmp_path)
    generate(ds, "chat", ScriptedTeacher([chat_reply()]),
             GenerateOptions(n=1, max_asks=2), root=tmp_path)
    rows = list_datasets(tmp_path)
    assert [r["name"] for r in rows] == ["chat-listed"]
    assert rows[0]["kept"] == 1 and rows[0]["recipe"] == "chat"


def test_sample_count_comes_from_the_file_not_the_metadata(tmp_path):
    """A process killed between the append and the metadata write must not leave a dataset
    that claims more samples than it holds."""
    ds = Dataset("py-truncated", root=tmp_path)
    generate(ds, "python", ScriptedTeacher([py_reply()]),
             GenerateOptions(n=1, max_asks=2, sandbox_timeout_s=5), root=tmp_path)
    ds.meta["counts"]["kept"] = 99
    ds.save()
    assert Dataset("py-truncated", root=tmp_path).stats()["kept"] == 1


# ---- configuration ----------------------------------------------------------------------

def test_synth_config_reads_its_own_section_and_per_recipe_models(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "portal.yaml").write_text(
        "explain:\n  model: small:1b\n"
        "synth:\n  model: mid:7b\n  temperature: 0.7\n"
        "  recipes:\n    python:\n      model: coder:3b\n      temperature: 0.4\n")
    cfg = SynthConfig.load(tmp_path)
    assert cfg.model == "mid:7b"
    assert cfg.model_for("python") == "coder:3b"
    assert cfg.temperature_for("python") == 0.4
    assert cfg.model_for("chat") == "mid:7b"
    assert cfg.temperature_for("chat") == 0.7


def test_per_recipe_default_is_used_when_nothing_is_configured(tmp_path, monkeypatch):
    for var in list(os.environ):
        if var.startswith("AKSHARALLM_SYNTH"):
            monkeypatch.delenv(var, raising=False)
    cfg = SynthConfig(path=None)
    cfg.reload()
    assert cfg.model_for("python") == "qwen2.5:14b"
    assert cfg.model_for("chat") == "gemma4:31b"


def test_contention_warns_about_a_big_teacher_and_clears_a_small_one(tmp_path, monkeypatch):
    """A 31B teacher beside a training run is not a slow tab, it is a dead run."""
    monkeypatch.setattr("aksharallm.portal.runs._alive", lambda pid: True)
    ckpts = tmp_path / "checkpoints" / "small-code"
    ckpts.mkdir(parents=True)
    (ckpts / "train.pid").write_text("4242")
    (ckpts / "ckpt_last.pt").write_bytes(b"not really a checkpoint")

    big = contention(tmp_path, "gemma4:31b")
    assert not big["safe"] and "small-code" in big["training"]
    small = contention(tmp_path, "starcoder2:3b")
    assert small["safe"] and "starcoder2:3b" in small["reason"]


def test_teacher_ask_is_the_only_thing_a_recipe_needs(tmp_path):
    """`Teacher` and the scripted stand-in are interchangeable — which is what lets every
    test above run without Ollama."""
    cfg = SynthConfig(path=None)
    real = Teacher(cfg, model="x:1b", temperature=0.5)
    assert real.name == "x:1b" and real.temperature == 0.5
    assert hasattr(real, "ask") and callable(real.ask)
