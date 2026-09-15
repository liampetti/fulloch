"""Markdown persistence and filename resolution for the shared notes root.

Raw Markdown, including YAML frontmatter, is preserved on read/append/save.
This module has no tool registrations or index/bridge imports: callers supply
the post-write hook so notifications happen only after successful persistence.
"""

import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from . import notes_root

logger = logging.getLogger(__name__)
FACTS_NOTE = "fulloch_facts"
_SAFE_TITLE_RE = re.compile(r"[^a-z0-9]+")
_FACT_LINE_RE = re.compile(r"^-\s*\[(\d{4}-\d{2}-\d{2})\]\s*(.*)$")
_FACTS_LOCK = threading.Lock()


def slugify(title: str) -> str:
    return _SAFE_TITLE_RE.sub("-", title.strip().lower()).strip("-") or "note"


def iter_notes() -> Iterable[Path]:
    return sorted(notes_root.get_notes_root().rglob("*.md"))


def find_note(query: str) -> Path | None:
    """Fuzzy-find a title: exact slug, slug substring, then raw substring."""
    if not query:
        return None
    slug = slugify(query)
    candidates = list(iter_notes())
    for path in candidates:
        if path.stem.lower() == slug:
            return path
    for path in candidates:
        if slug and slug in path.stem.lower():
            return path
    for path in candidates:
        if query.lower() in path.stem.lower():
            return path
    return None


def read_markdown(path: Path, *, errors: str = "strict") -> str:
    return path.read_text(encoding="utf-8", errors=errors)


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def append_markdown(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content)


def write_atomic(path: Path, content: str, *, prefix: str) -> None:
    """Replace a complete document without exposing partial content to readers."""
    body = content if content.endswith("\n") else content + "\n"
    fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def facts_path() -> Path:
    return notes_root.get_notes_root() / f"{FACTS_NOTE}.md"


def list_facts() -> list[dict]:
    path = facts_path()
    if not path.exists():
        return []
    try:
        content = read_markdown(path)
    except OSError as error:
        logger.error("Failed to read facts: %s", error)
        return []
    out: list[dict] = []
    for line in content.splitlines():
        match = _FACT_LINE_RE.match(line.strip())
        if match:
            out.append({"index": len(out), "date": match[1], "text": match[2].strip()})
    return out


def edit_fact(index: int, text: str | None, *, after_write: Callable[[Path], None]) -> bool:
    """Update one fact, or delete it when text is None, retaining other lines."""
    path = facts_path()
    with _FACTS_LOCK:
        if not path.exists():
            return False
        try:
            content = read_markdown(path)
        except OSError as error:
            logger.error("Failed to read facts: %s", error)
            return False
        lines = content.splitlines()
        fact_index = -1
        for position, line in enumerate(lines):
            match = _FACT_LINE_RE.match(line.strip())
            if match:
                fact_index += 1
                if fact_index == index:
                    if text is None:
                        del lines[position]
                    else:
                        lines[position] = f"- [{match[1]}] {text}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_atomic(path, "\n".join(lines), prefix=".facts-")
                    after_write(path)
                    return True
    return False


def resolve_note_file(name: str) -> Path | None:
    """Resolve a dashboard name, rejecting traversal and the dedicated facts file."""
    name = (name or "").strip()
    if not name:
        return None
    candidate = (notes_root.get_notes_root() / name).with_suffix(".md").resolve()
    try:
        candidate.relative_to(notes_root.get_notes_root())
    except ValueError:
        return None
    if candidate.name == f"{FACTS_NOTE}.md":
        return None
    return candidate


def list_note_files() -> list[dict]:
    return [
        {
            "name": path.relative_to(notes_root.get_notes_root()).with_suffix("").as_posix(),
            "title": path.stem.replace("-", " "),
        }
        for path in iter_notes()
        if path.name != f"{FACTS_NOTE}.md"
    ]


def read_note_file(name: str) -> str | None:
    path = resolve_note_file(name)
    if path is None or not path.exists():
        return None
    try:
        return read_markdown(path)
    except OSError as error:
        logger.error("Failed to read note %s: %s", name, error)
        return None


def save_note_file(name: str, content: str, *, after_write: Callable[[Path], None]) -> bool:
    path = resolve_note_file(name)
    if path is None or not path.exists():
        return False
    try:
        write_atomic(path, content, prefix=".note-")
    except OSError as error:
        logger.error("Failed to save note %s: %s", name, error)
        return False
    after_write(path)
    return True
