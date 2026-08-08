"""The view menu: the two things about it that break silently.

The views used to be a strip of buttons in the top bar, where a missing one was obvious
because you could see all of them at once. They are a drawer now, closed by default, so a
view with no menu entry is reachable only by typing its `#hash` — and nobody types a hash.
That is the first test here.

The second is a bug this drawer actually shipped with for an hour. The open state is
`document.body.classList.add('nav-open')`, and the button that opens it was styled as
`.nav-open { display: inline-flex }`. That selector matches the BODY as well: opening the
menu turned the whole document into an inline-flex container — the top bar shrank to 140px
and dropped a thousand pixels down the page, the footer landed halfway up the right-hand
side, and a phone picked up 200px of horizontal scroll. Nothing errored, and the drawer
itself looked perfect, because the drawer is `position: fixed` and did not care.

A state class on <body> shares one namespace with every component class in the portal.
There is no warning for the collision in any tool, so: a test.

Read with: docs/10-running-and-watching.md -- the chapter on the portal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "aksharallm" / "portal" / "static"
INDEX = STATIC / "index.html"
CSS = STATIC / "css"
JS = STATIC / "js"


def views_in_router() -> list[str]:
    """The list the router switches on, which is the definition of "a view"."""
    src = (JS / "router.js").read_text()
    match = re.search(r"export const VIEWS\s*=\s*\[(.*?)\]", src, re.S)
    assert match, "router.js no longer declares VIEWS"
    return re.findall(r"'([^']+)'", match.group(1))


def views_in_menu() -> list[str]:
    return re.findall(r'<button class="tab" data-view="([^"]+)"', INDEX.read_text())


def test_every_view_has_a_menu_entry():
    """A view the router knows about but the drawer does not list is unreachable: the strip
    that used to show every view at once is gone, so nothing else advertises it."""
    missing = [v for v in views_in_router() if v not in views_in_menu()]
    assert not missing, (
        f"views with no entry in the menu in index.html: {missing}. "
        "They can only be reached by typing their #hash.")


def test_the_menu_lists_no_view_that_does_not_exist():
    """The other direction: a button whose data-view the router does not know hides every
    panel and shows none, because showView() iterates VIEWS and finds nothing to unhide."""
    unknown = [v for v in views_in_menu() if v not in views_in_router()]
    assert not unknown, f"menu entries for views the router does not know: {unknown}"


def test_no_view_is_listed_twice():
    menu = views_in_menu()
    dupes = sorted({v for v in menu if menu.count(v) > 1})
    assert not dupes, f"listed more than once in the menu: {dupes}"


def body_state_classes() -> set[str]:
    """Classes the JS puts on <body> — `document.body.classList.add/toggle('...')`."""
    found = set()
    for path in JS.glob("*.js"):
        for match in re.finditer(
                r"document\.body\.classList\.(?:add|toggle|remove)\(\s*'([^']+)'", path.read_text()):
            found.add(match.group(1))
    return found


def test_there_is_a_body_state_class_so_this_file_still_has_a_job():
    assert body_state_classes(), (
        "no class is set on <body> any more — has the drawer been rewritten?")


def selectors() -> list[tuple[Path, str]]:
    """Every individual selector in css/, comments and rule bodies removed."""
    out = []
    for path in sorted(CSS.glob("*.css")):
        text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        for head in re.findall(r"([^{}]+)\{", text):
            head = head.strip()
            if not head or head.startswith("@"):
                continue
            out.extend((path, sel.strip()) for sel in head.split(",") if sel.strip())
    return out


def subject(selector: str) -> str:
    """The last compound — the element a rule actually styles. In `body.nav-open .nav` that
    is `.nav`; in `.nav-open` it is `.nav-open`, i.e. anything at all with that class."""
    return re.split(r"[\s>+~]+", selector)[-1]


@pytest.mark.parametrize("cls", sorted(body_state_classes()) or ["none"])
def test_a_body_state_class_is_never_also_a_component_class(cls: str):
    """A rule whose SUBJECT is a bare body state class styles <body> itself.

    Only the subject matters. `body.nav-open .nav { }` is the intended shape and reads
    correctly; `.stale main { }` styles a `main` and cannot touch the body either. It is
    `.nav-open { display: inline-flex }` — the state class standing alone as the thing being
    styled — that quietly reaches the document root.

    The whole-token lookahead matters too: `.nav-open-label` is a different class.
    """
    token = re.compile(rf"\.{re.escape(cls)}(?![\w-])")
    for path, sel in selectors():
        last = subject(sel)
        if not token.search(last) or last.startswith("body"):
            continue
        pytest.fail(
            f"{path.name}: `{sel}` is styling `.{cls}` itself, but `{cls}` is a state class "
            f"on <body> — so this rule applies to the whole document whenever the state is "
            f"on. Write `body.{cls}`, or key the component off its id instead.")
