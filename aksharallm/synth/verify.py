"""Running the generated tests — and then proving they were worth running.

This is the reason the Python recipe is the one worth building first. Everywhere else in
synthetic data you are trusting a model's opinion of its own output; here the tests are
executed in the sandbox that already grades the Playground's code tasks, and a sample is
kept only if they pass. Correctness is *checked*, not assumed.

Except that "the tests passed" is weaker than it sounds, and the weakness is systematic. Ask
a model for a function and some tests and it will occasionally write tests like

    assert callable(dedupe)
    assert dedupe.__name__ == "dedupe"

or wrap the call in a `try/except` and assert on the fallback. These pass. They also mention
the function by name, which is all a static check can confirm, and they would pass against a
function with no body at all. Nothing in the sandbox result tells them apart from a real test
— the exit code is 0 either way — and a dataset full of them is exactly the "trains smoothly,
model gets worse" failure this package is built to avoid.

So every sample is run **twice**:

    1. solution + tests            → must PASS   (the function is correct)
    2. stubbed solution + tests    → must FAIL   (the tests actually test it)

Step 2 is a one-line mutation: the entry point's body is replaced by `raise
NotImplementedError`, keeping its signature so the call still resolves. If the tests still
pass, they never depended on the answer, and the sample is rejected as `vacuous_tests`.
This is mutation testing at its smallest useful size, and it costs one extra subprocess —
about a hundred milliseconds against the several seconds the teacher took to write it.

Where the check stops, stated plainly, because a check believed to be stronger than it is
does more harm than no check:

* It catches tests that **do not depend on the implementation at all** — never calling the
  function, catching everything it raises, asserting on a constant.
* It does **not** catch a weak-but-real assertion. `assert isinstance(dedupe(xs), list)`
  fails against the stub (the stub raises), so the sample is kept, even though the assertion
  would hold for a function that always returns `[]`. The two-assert floor in
  `filters.check_code` is the only guard against that.
* It does **not** catch tests that are wrong in the same direction as the solution — a model
  that believes `is_prime(1)` is True and writes both the function and the assert to match.
  Nothing that treats the teacher as the oracle can. That is what the eval harness, rather
  than the pass rate, is for.

Read with: docs/13-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..infer import sandbox


@dataclass
class Verdict:
    """What happened when this exercise was executed."""

    ok: bool
    status: str            # pass | tests_failed | vacuous_tests | sandbox_error
    detail: str
    duration_s: float = 0.0
    #: What the sandbox said for the stubbed run — "fail" is the healthy value.
    stub_status: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status, "detail": self.detail,
                "duration_s": round(self.duration_s, 3), "stub_status": self.stub_status}


def program(solution: str, tests: str) -> str:
    """The single file that gets executed: the solution, then the asserts.

    Concatenated rather than imported, because the sandbox runs one isolated script with no
    package to import from — which is also why `filters.check_code` rejects tests that try
    `from solution import ...`.
    """
    return f"{solution.strip()}\n\n{tests.strip()}\n"


def stub(solution: str, entry_point: str) -> str | None:
    """`solution` with `entry_point`'s body replaced by `raise NotImplementedError`.

    Done through the AST rather than with a regex: a regex that finds the end of a function
    has to understand indentation, decorators, nested defs and strings that contain `def`,
    and getting it wrong produces a stub that is still the original function — which passes
    step 2 and silently disables the whole check.

    Returns None when the solution does not parse or the entry point is not found at module
    level; the caller treats that as "cannot verify" rather than as a pass.
    """
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None

    replaced = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == entry_point:
            node.body = [_raise()]
            replaced = True
        elif isinstance(node, ast.ClassDef) and node.name == entry_point:
            # A class entry point: hollow out every method, including __init__, so no
            # behaviour survives to satisfy an assert.
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    item.body = [_raise()]
            replaced = True
    if not replaced:
        return None
    try:
        return ast.unparse(tree)
    except Exception:                       # noqa: BLE001 — unparse is best-effort
        return None


def _raise() -> ast.Raise:
    return ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                                  args=[], keywords=[]), cause=None)


def verify(solution: str, tests: str, entry_point: str, timeout_s: float = 10.0,
           memory_mb: int = 512, enabled: bool = True, mutate: bool = True) -> Verdict:
    """Execute the tests, then execute them against a stub. See the module docstring."""
    ok, why = sandbox.available()
    if not enabled or not ok:
        return Verdict(False, "sandbox_error",
                       why or "running generated code is turned off, so this sample cannot "
                              "be verified.")

    real = sandbox.run_program(program(solution, tests), timeout_s=timeout_s,
                               memory_mb=memory_mb)
    if not real.ok:
        return Verdict(False, "tests_failed", f"{real.status}: {real.detail}",
                       duration_s=real.duration_s)

    if not mutate:
        return Verdict(True, "pass", "tests passed (mutation check skipped).",
                       duration_s=real.duration_s)

    hollow = stub(solution, entry_point)
    if hollow is None:
        return Verdict(False, "sandbox_error",
                       f"could not stub `{entry_point}` to check the tests are not vacuous.",
                       duration_s=real.duration_s)
    stubbed = sandbox.run_program(program(hollow, tests), timeout_s=timeout_s,
                                  memory_mb=memory_mb)
    if stubbed.ok:
        return Verdict(False, "vacuous_tests",
                       "the tests pass with the function's body removed — they assert "
                       "something that is true of any implementation.",
                       duration_s=real.duration_s + stubbed.duration_s,
                       stub_status=stubbed.status)
    return Verdict(True, "pass",
                   f"tests passed, and failed against the stub ({stubbed.status}) — they "
                   "depend on the implementation.",
                   duration_s=real.duration_s + stubbed.duration_s,
                   stub_status=stubbed.status)
