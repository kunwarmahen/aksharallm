"""Sampled HumanEval (`humaneval-pass10`): the estimator, the stop cut, and the suite wiring."""
import math
from types import SimpleNamespace

import pytest

from aksharallm.eval import sampled, suites
from aksharallm.eval.runner import describe


def test_pass_at_k_edges_and_known_values():
    assert sampled.pass_at_k(20, 0, 10) == 0.0
    assert sampled.pass_at_k(20, 20, 10) == 1.0
    # pass@1 under the unbiased estimator is exactly the sampled pass rate.
    assert sampled.pass_at_k(20, 3, 1) == pytest.approx(3 / 20)
    # One correct among 20, draw 10 without replacement: half the draws contain it.
    assert sampled.pass_at_k(20, 1, 10) == pytest.approx(0.5)
    # Fewer wrong samples than k: every draw of k must include a correct one.
    assert sampled.pass_at_k(20, 15, 10) == 1.0


def test_the_estimator_is_not_the_biased_shortcut():
    """`1 - (1 - c/n)**k` overstates pass@k at small n; the comb form must not equal it."""
    n, c, k = 20, 2, 10
    biased = 1 - (1 - c / n) ** k
    exact = 1 - math.comb(n - c, k) / math.comb(n, k)
    assert sampled.pass_at_k(n, c, k) == pytest.approx(exact)
    assert biased < exact or biased > exact          # they differ...
    assert abs(biased - exact) > 0.05                # ...by more than rounding


def test_a_completion_is_cut_at_its_first_stop():
    text = "    return x + 1\n\ndef other():\n    pass\nprint(1)"
    assert sampled._cut(text) == "    return x + 1\n"
    assert sampled._cut("    return 1") == "    return 1"


def test_score_counts_per_problem_and_folds_into_pass_at_k(monkeypatch):
    from aksharallm.infer import sandbox
    items = [SimpleNamespace(id="p0", prompt="def f():\n", entry_point="f", tests="T0"),
             SimpleNamespace(id="p1", prompt="def g():\n", entry_point="g", tests="T1")]
    # p0: 2 of 10 pass, p1: none. The verdict is decided by what the completion says.
    comps = [["    return 1"] * 2 + ["    return 0"] * 8, ["    return 0"] * 10]
    monkeypatch.setattr(sandbox, "run_program",
                        lambda prog, timeout_s=0: SimpleNamespace(
                            ok="return 1" in prog, status="ok" if "return 1" in prog else "fail"))
    out = sampled.score(items, comps, ks=(1, 10))
    assert [r["correct"] for r in out["items"]] == [2, 0]
    assert out["pass@1"] == pytest.approx((0.2 + 0.0) / 2)
    assert out["pass@10"] == pytest.approx(0.5)          # p0 always solved at k=n, p1 never
    assert out["solved_any"] == 1 and out["samples"] == 10
    assert out["failures"] == {"fail": 18}
    assert "pass@10" in describe("humaneval-pass10", {**out, "kind": "code"})


def test_the_sampled_suite_is_separate_and_not_in_all():
    s = suites.get("humaneval-pass10")
    assert s.samples == 20 and s.temperature > 0
    assert suites.get("humaneval").samples == 0          # greedy suite unchanged (gotcha 4)
    assert "humaneval-pass10" not in suites.ALL_SUITES
    assert "humaneval" in suites.ALL_SUITES


# ---- MBPP -------------------------------------------------------------------------------

MBPP_ROW = {
    "task_id": 11, "text": "Write a python function to remove first and last occurrence "
    "of a given character from the string.",
    "code": "import re\r\ndef helper(x):\r\n\treturn x\r\ndef remove_Occ(s,ch): \r\n"
            "\tfor i in range(len(s)): \r\n\t\tif (s[i] == ch): \r\n\t\t\ts = s[0 : i] + s[i + 1:] \r\n"
            "\t\t\tbreak\r\n\tfor i in range(len(s) - 1,-1,-1):  \r\n\t\tif (s[i] == ch): \r\n"
            "\t\t\ts = s[0 : i] + s[i + 1:] \r\n\t\t\tbreak\r\n\treturn s ",
    "test_list": ['assert remove_Occ("hello","l") == "heo"',
                  'assert remove_Occ("abcda","a") == "bcd"',
                  'assert remove_Occ("PHP","P") == "H"'],
    "test_setup_code": "",
}


def test_an_mbpp_row_becomes_a_humaneval_shaped_task():
    it = suites.mbpp_item(MBPP_ROW)
    # The entry point is the function the first assert CALLS, not the first one defined.
    assert it.entry_point == "remove_Occ" and it.id == "mbpp/11"
    assert it.prompt.startswith("import re\n\ndef remove_Occ(s, ch):\n")
    assert '>>> remove_Occ("hello","l") == "heo"' in it.prompt       # the call shape
    assert "return s" not in it.prompt                                # no solution leaked
    assert it.tests.count("assert ") == 3                             # all three scored


def test_the_reference_solution_passes_and_a_wrong_one_fails():
    """The positive control the real run did on all 500: prompt + tests are assembled right."""
    from aksharallm.infer import sandbox
    ok, why = sandbox.available()
    if not ok:
        pytest.skip(why)
    it = suites.mbpp_item(MBPP_ROW)
    ref = MBPP_ROW["code"].replace("\r\n", "\n").replace("\t", "    ")
    assert sandbox.run_program(f"{ref}\n\n{it.tests}", timeout_s=10).ok
    wrong = it.prompt + "    return s\n"
    assert not sandbox.run_program(f"{wrong}\n\n{it.tests}", timeout_s=10).ok


def test_mbpp_suites_are_registered_and_sampled_one_is_not_in_all():
    assert suites.get("mbpp").builder == "build_mbpp" and suites.get("mbpp").samples == 0
    assert suites.get("mbpp-pass10").samples == 20
    assert "mbpp" in suites.ALL_SUITES and "mbpp-pass10" not in suites.ALL_SUITES
    assert suites.mbpp_item({**MBPP_ROW, "code": "def broken(:"}) is None
