"""The one CSS bug in this portal that keeps coming back, made into a test.

Every working tab is built the same way: a two-column grid whose columns are fixed-height
flex panels, and inside each column exactly one thing scrolls.

    .ev-layout > .panel { height: calc(100vh - 150px); overflow: hidden; }
    .ev-form              { overflow-y: auto; }          <-- does nothing

The second rule looks like it makes the column scroll. It does not. `.ev-layout > .panel`
is two classes and `.ev-form` is one, so the panel's `overflow: hidden` wins the cascade and
the column is **clipped**: everything past the fold is not merely off-screen, it is
unreachable by any amount of scrolling.

It is invisible in every cheap check. The page renders, nothing errors, a screenshot of the
top looks perfect, and `document.body.scrollWidth` is clean. It only shows up when the
column's content grows past the panel — which happens long after the CSS was written. The
Learn tab broke the day the curriculum went from thirteen lessons to nineteen; the Eval tab
was hiding 1,482px of its own form, including the Evaluate button.

learn.css has carried a comment explaining this since it was fixed there. The comment did
not stop it happening in three more tabs, because a comment is only read by someone already
looking at that file. So: a test.

Read with: docs/10-running-and-watching.md -- the chapter on the portal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parents[1] / "aksharallm" / "portal" / "static" / "css"
PARTS = Path(__file__).resolve().parents[1] / "aksharallm" / "portal" / "static" / "parts"

#: `.foo-layout > .panel { ... overflow: hidden ... }` — a column that clips its children.
CLIPPING_PANEL = re.compile(
    r"\.([a-z-]+)-layout\s*>\s*\.panel\s*\{([^}]*)\}", re.S)

#: A rule body that turns scrolling on, in either the shorthand or the axis property.
SCROLLS = re.compile(r"overflow(-y)?\s*:\s*(auto|scroll)")
CLIPS = re.compile(r"overflow(-y)?\s*:\s*hidden")


def stylesheets() -> list[Path]:
    return sorted(CSS.glob("*.css"))


def strip_media(text: str) -> str:
    """Drop `@media` blocks. A narrow-screen override deliberately hands scrolling back to
    the page (`overflow: visible`), which is correct and must not be read as a clip."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("@media", i):
            depth_at = 0
            while i < len(text):
                if text[i] == "{":
                    depth_at += 1
                elif text[i] == "}":
                    depth_at -= 1
                    if depth_at == 0:
                        i += 1
                        break
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clipping_layouts() -> list[tuple[Path, str]]:
    """Every `(stylesheet, prefix)` whose panels clip — the places the trap can be sprung."""
    found = []
    for path in stylesheets():
        body = strip_media(path.read_text())
        for match in CLIPPING_PANEL.finditer(body):
            if CLIPS.search(match.group(2)):
                found.append((path, match.group(1)))
    return found


def test_the_trap_still_exists_so_this_file_still_has_a_job():
    """If every layout stopped clipping, the tests below would pass vacuously forever."""
    assert clipping_layouts(), "no clipping layouts found — has the portal been restyled?"


def panel_classes(prefix: str) -> set[str]:
    """The classes that sit ON a `.panel` element in the markup for this layout — the only
    ones that can be outranked by `.<prefix>-layout > .panel`. A class on a *child* of the
    panel is not in this competition and scrolls perfectly well with one class."""
    classes = set()
    for part in PARTS.glob("*.html"):
        for attr in re.findall(r'class="([^"]*)"', part.read_text()):
            names = attr.split()
            if "panel" in names:
                classes.update(n for n in names if n.startswith(f"{prefix}-"))
    return classes


@pytest.mark.parametrize("path,prefix", clipping_layouts(),
                         ids=lambda v: v if isinstance(v, str) else v.name)
def test_a_panels_own_scroll_rule_outranks_the_clip(path: Path, prefix: str):
    """A rule that makes a clipping panel scroll must name the layout too.

    `.ev-form { overflow-y: auto }` is a no-op; `.ev-layout > .ev-form { overflow-y: auto }`
    works. Both read as "this column scrolls" to someone skimming, which is precisely why
    this is asserted rather than left to review.
    """
    body = strip_media(path.read_text())
    on_panels = panel_classes(prefix)
    if not on_panels:
        pytest.skip(f"no .panel carries a .{prefix}-* class")

    for cls in sorted(on_panels):
        # Rules written as a bare `.foo-form { ... }`, i.e. one class and nothing else.
        for rule in re.finditer(rf"(?:^|\}})\s*\.{re.escape(cls)}\s*\{{([^}}]*)\}}", body, re.S):
            assert not SCROLLS.search(rule.group(1)), (
                f"{path.name}: `.{cls} {{ overflow-y: auto }}` is one class and loses to "
                f"`.{prefix}-layout > .panel {{ overflow: hidden }}`, so the column is "
                f"clipped, not scrolled. Write `.{prefix}-layout > .{cls}` instead.")


@pytest.mark.parametrize("path,prefix", clipping_layouts(),
                         ids=lambda v: v if isinstance(v, str) else v.name)
def test_every_clipping_column_can_actually_be_read(path: Path, prefix: str):
    """A fixed-height panel that neither scrolls nor is scrolled by a child is a column with
    unreachable content. `.quant-main` was exactly this: `overflow: hidden` from the panel
    rule and no scroll rule anywhere, so the bottom of the results column did not exist.

    **This check is deliberately weak, and here is its limit.** Statically, a stylesheet does
    not say which panel a descendant rule lands in — `.md { overflow-y: auto }` scrolls the
    Code tab's explain panel without naming it, and `.ln-lessons` scrolls `.ln-list` the same
    way. Binding scroller to panel needs a real cascade, which means a browser. So this only
    catches the total case: a layout whose panels clip and where *nothing at all* scrolls.
    The precise check is the test above; this one is the backstop.
    """
    body = strip_media(path.read_text())
    panels = sorted(panel_classes(prefix))
    if not panels:
        pytest.skip(f"no .panel carries a .{prefix}-* class")

    if any(SCROLLS.search(
            re.search(rf"\.{re.escape(prefix)}-layout\s*>\s*\.{re.escape(cls)}\s*\{{([^}}]*)\}}",
                      body, re.S).group(1))
           for cls in panels
           if re.search(rf"\.{re.escape(prefix)}-layout\s*>\s*\.{re.escape(cls)}\s*\{{([^}}]*)\}}",
                        body, re.S)):
        return

    # Nothing scrolls itself, so a descendant must — any selector, not just `.prefix-*`.
    others = [m.group(1) for m in re.finditer(r"\{([^}]*)\}", body)
              if not re.search(r">\s*\.panel\s*\{", m.group(0))]
    assert any(SCROLLS.search(b) for b in others), (
        f"{path.name}: the `.{prefix}-layout` panels clip their content and no rule in this "
        f"stylesheet scrolls anything — content past the fold cannot be reached at all.")


# ---- the second silent-width bug: a control that sets a floor under the page -------------
#
# Same family as the trap above (invisible until real data arrives), different mechanism. A
# flex/grid item's automatic minimum size is its min-content width, and for a `<select>` that
# is its LONGEST OPTION. The portal fills its pickers from disk — log filenames, checkpoint
# paths, corpus names — so the markup gives no hint how wide they get. One
# `train_20260807-082049.log (0.1 MB)` in the Log picker held the whole document at 317px on a
# 320px screen: the page scrolled sideways and every panel came along with it.
#
# base.css already does this for the panel grids. Controls need it for the same reason.

CONTROLS = CSS / "controls.css"


def _rule_body(text: str, selector: str) -> str:
    """The declarations of the first top-level rule whose selector list names `selector`.

    Comments are stripped first: prose commas would otherwise be read as selector
    separators, and a comment sitting above a rule becomes part of its selector text.
    """
    text = re.sub(r"/\*.*?\*/", " ", strip_media(text), flags=re.S)
    for match in re.finditer(r"([^{}]+)\{([^}]*)\}", text):
        names = [" ".join(s.split()) for s in match.group(1).split(",")]
        if selector in names:
            return match.group(2)
    return ""


@pytest.mark.parametrize("selector", [".field", "select"])
def test_a_control_cannot_set_a_floor_under_the_page(selector: str):
    """`min-width: 0` on the field and the control it holds.

    Without it a long option name is a hard minimum: it cannot shrink, so it pushes the
    document wider than the viewport and the whole page scrolls sideways on a phone.
    """
    body = _rule_body(CONTROLS.read_text(), selector)
    assert body, f"no `{selector}` rule in controls.css — has it been renamed?"
    assert re.search(r"min-width\s*:\s*0", body), (
        f"`{selector}` lost `min-width: 0`; a long <option> will widen the whole page again")


def test_a_control_stays_inside_the_column_it_was_given():
    body = _rule_body(CONTROLS.read_text(), "select")
    assert re.search(r"max-width\s*:\s*100%", body), (
        "`select` lost `max-width: 100%`; min-width:0 lets it shrink, this stops it growing")


CHROME = CSS / "chrome.css"


def test_the_top_bar_can_shrink_around_its_run_picker():
    """The same floor, one level up — and `max-width: 100%` on the select cannot catch it.

    `.field` and `select` already carry `min-width: 0`, so the picker itself will shrink.
    But it sits inside `.topbar-right`, which is a flex item of `.topbar` and so has its own
    automatic minimum of min-content — the widest option again. The parent is stretched to
    fit the option, and the select's `max-width: 100%` then resolves against that stretched
    parent and clamps nothing.

    Real instance: archiving a run ("Start fresh…") names it
    `small-code-sft.20260808-074957 — archived`. That option made `.topbar-right` 330px wide
    and a 320px phone scrolled sideways by 23px. Every ancestor between a picker and the
    page needs `min-width: 0`, not just the one holding it.
    """
    body = _rule_body(CHROME.read_text(), ".topbar-right")
    assert body, "no `.topbar-right` rule in chrome.css — has it been renamed?"
    assert re.search(r"min-width\s*:\s*0", body), (
        "`.topbar-right` lost `min-width: 0`; an archived run name in the picker will push "
        "the whole page sideways on a 320px screen again")
