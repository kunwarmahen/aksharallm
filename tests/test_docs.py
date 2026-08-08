"""The docs are a second description of the code, and second descriptions drift.

`tests/test_lessons.py` keeps the *lessons* honest. This file keeps the two pointers that
tie the chapters and the source together honest, in both directions:

  docs/NN-*.md  --"The code, in reading order"-->  the files to open
  a module      --"Read with: docs/NN-*.md"----->  the chapter explaining it

Both are the kind of thing that rots silently: a file gets renamed, a chapter gets split,
and the reader is sent somewhere that no longer exists. Nothing else would notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: The heading every chapter ends with. It is the reading order for that chapter's files.
SECTION = "## The code, in reading order"

#: A shim with no content of its own has nothing to point at. Everything else must.
SHIM_LINES = 15

CHAPTERS = sorted(p for p in DOCS.glob("[0-9][0-9]-*.md"))
MODULES = sorted((ROOT / "aksharallm").rglob("*.py"))

LINK = re.compile(r"\]\(([^)]+)\)")
POINTER = re.compile(r"Read with: (docs/[0-9]{2}-[a-z-]+\.md)")
ALSO = re.compile(r"See also (docs/[0-9]{2}-[a-z-]+\.md)")


def section_of(text: str) -> str:
    """The reading-order section's body, or "" if the chapter has none."""
    if SECTION not in text:
        return ""
    after = text.split(SECTION, 1)[1]
    return after.split("\n## ", 1)[0]


def test_there_are_twentythree_chapters():
    assert len(CHAPTERS) == 23, [p.name for p in CHAPTERS]


@pytest.mark.parametrize("doc", CHAPTERS, ids=lambda p: p.stem)
def test_every_chapter_ends_with_a_reading_order(doc):
    text = doc.read_text()
    assert text.count(SECTION) == 1, f"{doc.name}: expected exactly one '{SECTION}'"
    assert section_of(text).strip(), f"{doc.name}: the section is empty"


@pytest.mark.parametrize("doc", CHAPTERS, ids=lambda p: p.stem)
def test_a_reading_order_points_at_real_files(doc):
    """Every link in the section resolves. A renamed module must break here, loudly, rather
    than send a reader to a path that no longer exists."""
    body = section_of(doc.read_text())
    targets = [t.split("#")[0] for t in LINK.findall(body)]
    code = [t for t in targets if t and not t.startswith(("http", "mailto"))]
    assert code, f"{doc.name}: the section links to nothing"
    for target in code:
        assert (doc.parent / target).resolve().exists(), f"{doc.name} -> {target}"


@pytest.mark.parametrize("doc", CHAPTERS, ids=lambda p: p.stem)
def test_chapter_links_resolve(doc):
    """The same check over the whole chapter, not just its last section."""
    for target in LINK.findall(doc.read_text()):
        target = target.split("#")[0]
        if not target or target.startswith(("http", "mailto", "<")):
            continue
        assert (doc.parent / target).resolve().exists(), f"{doc.name} -> {target}"


def test_lesson_links_are_repo_relative_and_real():
    """A lesson body is rendered in the Learn tab, which sends a link to the Code tab by its
    path — so the paths are relative to the repo root, not to `docs/lessons/`."""
    for lesson in sorted((DOCS / "lessons").glob("*.md")):
        for target in LINK.findall(lesson.read_text()):
            target = target.split("#")[0]
            if not target or target.startswith(("http", "mailto")):
                continue
            assert (ROOT / target).exists(), f"{lesson.name} -> {target}"


@pytest.mark.parametrize("mod", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_module_names_its_chapter(mod):
    """A reader who lands in a file from the Code tab, a traceback or a grep should be one
    line away from the chapter that explains why it looks like this."""
    text = mod.read_text()
    if len(text.splitlines()) < SHIM_LINES:
        return  # a `from .cli import main` shim; there is nothing to explain
    found = POINTER.search(text)
    assert found, f"{mod.relative_to(ROOT)}: no 'Read with: docs/...' in the module docstring"

    named = [found.group(1)]
    if (also := ALSO.search(text)) is not None:
        named.append(also.group(1))
    for doc in named:
        assert (ROOT / doc).is_file(), f"{mod.relative_to(ROOT)} -> {doc}"


def test_the_pointer_lives_in_the_module_docstring():
    """Not in a comment halfway down, where nobody opening the file would see it."""
    for mod in MODULES:
        text = mod.read_text()
        if not POINTER.search(text):
            continue
        head = text[: text.find('"""', text.find('"""') + 3) + 3]
        assert POINTER.search(head), f"{mod.relative_to(ROOT)}: pointer is outside the docstring"


def test_every_chapter_is_named_by_at_least_one_module():
    """A chapter nothing points to is either about the repo rather than the code (00, 07,
    08, 22) or has quietly lost its implementation."""
    named = set()
    for mod in MODULES:
        text = mod.read_text()
        named.update(POINTER.findall(text))
        named.update(ALSO.findall(text))
    #: 22 is the route map: it sequences the other chapters rather than explaining a module,
    #: so its "reading order" points at the launcher *scripts*, which are not Python modules
    #: and therefore carry no `Read with:` pointer back.
    prose_only = {"docs/00-overview.md", "docs/09-troubleshooting.md", "docs/01-journeys.md"}
    for doc in CHAPTERS:
        rel = f"docs/{doc.name}"
        if rel in prose_only:
            continue
        assert rel in named, f"no module says 'Read with: {rel}'"
