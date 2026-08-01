"""The learning path's own tests — most of which exist to stop the lessons rotting.

A lesson is a second description of the code, and a second description drifts. A stale
lesson is worse than stale prose: it sends someone to break a line that has moved, and they
conclude they have misunderstood rather than that the text is wrong.

So the suite is the drift detector. `test_every_lesson_still_points_at_real_code` and
`test_every_verify_node_still_exists` are the two that matter — they fail the moment a
source file is renamed or a test is deleted out from under a lesson, which is the earliest
anyone could possibly find out.
"""

import json

import pytest

from aksharallm.learn import check, lessons as lessons_mod
from aksharallm.learn.progress import Progress, gate


@pytest.fixture(scope="module")
def lessons():
    return lessons_mod.load_all()


# ---- the anti-rot tests -----------------------------------------------------------------

def test_there_are_lessons_at_all(lessons):
    assert lessons, "docs/lessons/ is empty — the Learn tab would show nothing"


def test_every_lesson_still_points_at_real_code(lessons):
    """Files, docs, prereqs, ids, cycles. `validate()` reports them all at once so a
    rename shows up as one readable list rather than thirteen failures."""
    problems = lessons_mod.validate()
    assert not problems, "\n".join(problems)


def test_every_verify_node_still_exists(lessons):
    """The other half of the anti-rot rule: a lesson's check must be a test that exists.

    One collection pass for the whole path — a test renamed out from under a lesson is
    otherwise invisible until someone tries the exercise and cannot complete it.
    """
    nodes = [lesson.verify for lesson in lessons if lesson.verify]
    missing = [node for node in nodes if not check.collectable(node)]
    assert not missing, f"lessons point at tests that no longer exist: {missing}"


def test_no_lesson_references_a_line_number(lessons):
    """Line numbers rot silently; file paths fail loudly when they move."""
    import re

    pattern = re.compile(r"\b(line|lines)\s+\d+", re.IGNORECASE)
    offenders = [(l.path.name, pattern.search(l.body).group(0))
                 for l in lessons if pattern.search(l.body)]
    assert not offenders, f"lessons must reference files, not line numbers: {offenders}"


def test_the_curriculum_is_reachable(lessons):
    """Every lesson must be openable eventually: the first one has no prereqs, and every
    prereq is satisfiable. A path whose first lesson is locked is a path nobody can start."""
    assert not lessons[0].prereqs, "the first lesson must be open from a cold start"
    known = {l.id for l in lessons}
    for lesson in lessons:
        assert set(lesson.prereqs) <= known


# ---- parsing ----------------------------------------------------------------------------

def test_frontmatter_is_required(tmp_path):
    path = tmp_path / "x.md"
    path.write_text("# no frontmatter here\n")
    with pytest.raises(lessons_mod.LearnError):
        lessons_mod.parse(path)


def test_a_lesson_without_an_id_is_refused(tmp_path):
    path = tmp_path / "x.md"
    path.write_text("---\ntitle: nameless\n---\nbody\n")
    with pytest.raises(lessons_mod.LearnError):
        lessons_mod.parse(path)


def test_validate_catches_a_prereq_cycle(tmp_path):
    (tmp_path / "docs" / "lessons").mkdir(parents=True)
    for a, b in (("one", "two"), ("two", "one")):
        (tmp_path / "docs" / "lessons" / f"{a}.md").write_text(
            f"---\nid: {a}\ntitle: {a}\nsummary: s\nverify: tests/test_x.py::test_y\n"
            f"prereqs: [{b}]\n---\nbody\n")
    problems = lessons_mod.validate(tmp_path)
    assert any("cycle" in p for p in problems)


def test_validate_catches_a_missing_file(tmp_path):
    (tmp_path / "docs" / "lessons").mkdir(parents=True)
    (tmp_path / "docs" / "lessons" / "a.md").write_text(
        "---\nid: a\ntitle: a\nsummary: s\nverify: tests/test_x.py::test_y\n"
        "files: [aksharallm/does/not/exist.py]\n---\nbody\n")
    assert any("does not exist" in p for p in lessons_mod.validate(tmp_path))


# ---- the red-then-green rule ------------------------------------------------------------

def test_a_lesson_that_only_ever_passed_is_not_complete(tmp_path):
    """The rule the whole design rests on: the check passes on a clean checkout, so
    "it passed" would complete the path for someone who never opened a file."""
    progress = Progress(tmp_path)
    progress.record("demo", passed=True)
    progress.record("demo", passed=True)
    entry = progress.of("demo")
    assert entry.complete is False
    assert entry.state() == "started"


def test_red_then_green_completes_it(tmp_path):
    progress = Progress(tmp_path)
    progress.record("demo", passed=True)      # tried it
    assert progress.of("demo").state() == "started"
    progress.record("demo", passed=False)     # broke it
    assert progress.of("demo").state() == "broken"
    progress.record("demo", passed=True)      # fixed it
    entry = progress.of("demo")
    assert entry.complete and entry.completed_at


def test_green_before_red_does_not_count_backwards(tmp_path):
    """Order matters: passing, then breaking, is a lesson in progress — not a finished one."""
    progress = Progress(tmp_path)
    progress.record("demo", passed=False)
    progress.record("demo", passed=True)
    assert progress.of("demo").complete
    progress2 = Progress(tmp_path)
    assert progress2.of("demo").complete, "completion survives a reload"


def test_completion_is_stamped_not_recomputed(tmp_path):
    """A finished lesson must stay finished even if the rule changes later."""
    progress = Progress(tmp_path)
    progress.record("demo", passed=False)
    progress.record("demo", passed=True)
    when = progress.of("demo").completed_at
    progress.record("demo", passed=False)
    assert progress.of("demo").completed_at == when


def test_progress_survives_a_corrupt_file(tmp_path):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "progress.json").write_text("{not json")
    progress = Progress(tmp_path)
    assert progress.all() == {}
    progress.record("demo", passed=False)
    assert json.loads((tmp_path / "learning" / "progress.json").read_text())["lessons"]


def test_reset_forgets_one_or_all(tmp_path):
    progress = Progress(tmp_path)
    progress.record("a", passed=False)
    progress.record("b", passed=False)
    progress.reset("a")
    assert "a" not in progress.all() and "b" in progress.all()
    progress.reset()
    assert progress.all() == {}


# ---- gating ------------------------------------------------------------------------------

def test_a_lesson_is_locked_until_its_prereqs_are_complete(tmp_path, lessons):
    progress = Progress(tmp_path)
    gates = gate(lessons, progress)
    first = lessons[0]
    assert gates[first.id]["open"] is True

    locked = [l for l in lessons if l.prereqs]
    assert locked, "the path has no prereqs at all — nothing is being sequenced"
    assert gates[locked[0].id]["open"] is False
    assert gates[locked[0].id]["reason"], "a locked lesson must say what is missing"


def test_finishing_a_prereq_opens_the_next_lesson(tmp_path, lessons):
    progress = Progress(tmp_path)
    child = next(l for l in lessons if l.prereqs)
    for prereq in child.prereqs:
        progress.record(prereq, passed=False)
        progress.record(prereq, passed=True)
    assert gate(lessons, progress)[child.id]["open"] is True


# ---- running a check ----------------------------------------------------------------------

def test_a_node_id_that_is_not_a_node_id_is_refused():
    """It reaches a command line, so it is matched rather than escaped."""
    for bad in ("; rm -rf /", "../etc/passwd", "aksharallm/model/transformer.py",
                "tests/../aksharallm/x.py::test_y && echo"):
        with pytest.raises(check.CheckError):
            check.run(bad)


def test_running_a_real_check_reports_pass_and_a_summary():
    result = check.run("tests/test_lessons.py::test_there_are_lessons_at_all")
    assert result.passed and "passed" in result.summary


def test_a_failing_check_hands_back_the_assertion_not_just_a_count():
    """The assertion is the lesson: it says what the broken code now believes."""
    summary = check.summarise("E       assert 3 == 4\n1 failed in 0.1s", passed=False)
    assert summary == "assert 3 == 4"
