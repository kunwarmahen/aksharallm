"""A control that the server will refuse has to refuse itself first.

Several panels hold a one-job-at-a-time lock, and the server raises when a second job is
asked for -- `"a quantization job is already running"`, `"a measurement is already running"`,
`"a job is already running"`. That refusal is deliberate: a contamination scan streams ten
billion tokens, a per-domain split loads the model, and a quantization pass wants the card to
itself. None of them want company.

But a refusal that only exists server-side is a button that *looks* available, is pressed,
and fails into a toast. Three panels got this right (Quantize, Finetune, Synth) and two did
not: the Eval tab's four audit buttons and the Context tab's four measurement buttons stayed
live through a running job. This file is the check that a sixth panel cannot quietly join
them -- the same shape as `tests/test_portal_launchers.py`, which exists because three
identification bugs of one shape shipped in one evening.

Read with: docs/10-running-and-watching.md -- the chapter on the portal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "aksharallm" / "portal" / "static" / "js"
PORTAL = ROOT / "aksharallm" / "portal"

#: A click handler that asks the server to begin work. `stop`/`cancel` are excluded: those
#: are the controls you want live *precisely* when something is running.
STARTS = re.compile(r"post\('/api/[^']*?/(start|audit|run|generate|extend)")

HANDLER = re.compile(r"\$\('(#[\w-]+)'\)\.addEventListener\('click',")


def refusing_modules() -> set[str]:
    """Portal modules whose server side refuses a second concurrent job."""
    found = set()
    for path in PORTAL.glob("*.py"):
        if re.search(r'RunError\(|InferError\(', path.read_text()) and \
                "already running" in path.read_text():
            found.add(path.stem)
    return found


def function_body(js: str, name: str) -> str:
    """The body of a top-level `function name(...)` / `const name = ...`, or ""."""
    match = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\(", js) or \
        re.search(rf"const\s+{re.escape(name)}\s*=", js)
    return js[match.end(): match.end() + 1500] if match else ""


def starting_buttons(js: str) -> set[str]:
    """Buttons whose click asks the server to begin work.

    **One level of indirection matters.** Only the Eval tab posts inline; every other panel
    routes its click through a helper — `() => startQuant(false)`, `startSynth`,
    `() => start('curve')`. A version of this that only read the handler body found buttons
    in one file out of four and *skipped* the rest, which is a test that cannot fail
    pretending to be a test that passes. So a call in the handler is followed into the
    function it names.
    """
    out = set()
    for match in HANDLER.finditer(js):
        body = js[match.end(): match.end() + 900]
        if STARTS.search(body):
            out.add(match.group(1))
            continue
        for callee in re.findall(r"\b([A-Za-z_]\w*)\s*[(;)]", body[:200]):
            if callee in {"async", "await", "const", "return", "if"}:
                continue
            if STARTS.search(function_body(js, callee)):
                out.add(match.group(1))
                break
    return out


def gated_buttons(js: str) -> set[str]:
    """Ids this file disables, in either shape the portal uses.

    Two shapes, and a check that knew only the first is how the Context tab's four buttons
    were missed: a direct `$('#q-run').disabled = running || ...`, and a loop over a list of
    ids, which is what a panel with four sibling buttons naturally grows into.
    """
    ids = set()
    for line in js.splitlines():
        if ".disabled" in line:
            ids.update(re.findall(r"'(#[\w-]+)'", line))
    # `for (const id of [...]) { ... btn.disabled = ... }`
    for match in re.finditer(r"for \(const \w+ of \[(.*?)\]\)", js, re.S):
        tail = js[match.end(): match.end() + 500]
        if ".disabled" in tail:
            ids.update(re.findall(r"'(#[\w-]+)'", match.group(1)))
    return ids


#: (js file, portal module) for every tab whose server refuses a second job.
PANELS = [(p, p.stem) for p in sorted(JS.glob("*.js"))
          if p.stem in refusing_modules() or p.stem in {"evals", "longctx"}]


def test_there_are_panels_to_check():
    """Without this the parametrised tests below could pass on an empty list."""
    assert len(PANELS) >= 4, [p.name for p, _ in PANELS]


@pytest.mark.parametrize("path,module", PANELS, ids=lambda v: getattr(v, "stem", v))
def test_a_button_the_server_will_refuse_is_disabled_first(path: Path, module: str):
    js = path.read_text()
    starts = starting_buttons(js)
    if not starts:
        pytest.skip(f"{path.name} starts nothing")
    ungated = starts - gated_buttons(js)
    assert not ungated, (
        f"{path.name}: {sorted(ungated)} ask the server to start work, but nothing in this "
        f"file disables them. The server refuses a second job, so these stay clickable "
        f"during one and fail into an error toast instead of saying so up front.")


@pytest.mark.parametrize("path,module", PANELS, ids=lambda v: getattr(v, "stem", v))
def test_the_refusal_is_still_enforced_on_the_server(path: Path, module: str):
    """The gate above is a courtesy, not the guarantee. A second browser, a stale tab or a
    terminal can all still ask, so the server must keep saying no."""
    source = (PORTAL / f"{module}.py")
    if not source.exists():
        pytest.skip(f"no portal module named {module}")
    assert "already running" in source.read_text(), (
        f"{module}.py no longer refuses a concurrent job — the UI gate is not a lock")
