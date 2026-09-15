"""Registered voice note/fact tools, policy, and shared index notifications.

Markdown persistence lives in notes_storage; Obsidian bridge transport lives in
notes_obsidian. Dashboard CRUD wrappers pass the index notification callback
explicitly, keeping the existing tools.notes API stable.

Always loaded. The store is a flat (or nested) folder of `.md` files plus an
optional `daily/` subfolder for daily notes. Full-text search uses Python's
`re` over the folder — no external binary dependency. Semantic search lives
in `notes_index.py` and is loaded lazily on first use so the embedding model
isn't paid for until the first semantic query (or `warm_index()` call).

Optional config under `notes:` overrides defaults — `path` (default
`./data/notes`) and `daily_subdir` (default `"daily"`; set empty/null to keep
daily notes in the top-level folder).
"""

import logging
import queue
import re
import threading
from pathlib import Path
from typing import Optional

import utils.local_time as _local_time

from . import notes_obsidian, notes_root, notes_storage
from ._config import config
from .notes_storage import FACTS_NOTE
from .notes_storage import find_note as _find_note
from .notes_storage import iter_notes as _iter_notes
from .notes_storage import slugify as _slugify
from .thinking_playbooks import thinking_playbook
from .tool_registry import ArtifactText, tool

logger = logging.getLogger(__name__)

thinking_playbook(
    name="notes research",
    triggers=(r"\b(notes?|remember|recall|journal|saved fact|my vault)\b",),
    capabilities=("search_notes", "read_note"),
    solve_path=(
        "Search notes using the user's terms before making a claim about saved information.",
        "Read a specific matching note only when the search excerpt needs context.",
        "Distinguish stored facts from new analysis in the report.",
    ),
    completion_rule="The report identifies the note evidence used or states that none was found.",
)

_notes_config = config.get("notes", {}) or {}
# Stable source directory for vault migration.
NOTES_DIR_LEGACY = Path(_notes_config.get("path", "./data/notes")).expanduser().resolve()

# `append_to_today` writes to <NOTES_DIR>/<DAILY_SUBDIR>/YYYY-MM-DD.md so daily
# journals don't clutter the top-level notes folder. Defaults to "daily" when the
# key is absent; set it empty/null in config to keep daily notes at the top level.
DAILY_SUBDIR: Optional[str] = _notes_config.get("daily_subdir", "daily")
# Voice replies are read aloud — long bodies make for a tedious TTS, so cap
# the spoken content and tell the user we truncated.
MAX_READ_CHARS = 2000
MAX_SEARCH_MATCHES = 5
SEMANTIC_TOP_K = 5
# Semantic-search score threshold: BGE-small cosine similarities tend to
# sit around 0.4–0.7 for genuine matches and below ~0.25 for irrelevant ones.
# Kept deliberately loose: `search_notes` hands its hits back through the agent
# loop with a "may not actually contain X" caveat, so the SLM filters false
# positives — a missed real match (silent "found nothing") is the worse error.
SEMANTIC_MIN_SCORE = 0.25
# Total hits the hybrid search surfaces after fusing keyword + semantic lists.
MAX_HYBRID_MATCHES = 5
INDEX_BASENAME = "notes_index"
# The vault is commonly a read-only or externally mounted directory. Keep the
# derived cache with Fulloch's other writable runtime data instead of beside it.
INDEX_PATH = Path("./data") / INDEX_BASENAME

# Lightweight hand-off for the dashboard stats panel: the semantic-search paths
# record the number of matched chunks here, and the assistant pops it after a
# note-search dispatch. Avoids parsing the spoken result string.
last_retrieval: dict = {}

_notes_root_path = notes_root.get_notes_root()
_notes_root_path.mkdir(parents=True, exist_ok=True)
if DAILY_SUBDIR:
    (_notes_root_path / DAILY_SUBDIR).mkdir(parents=True, exist_ok=True)

_HEADER_RE = re.compile(r"^#+\s*", flags=re.MULTILINE)
_BULLET_RE = re.compile(r"^[-*+]\s+", flags=re.MULTILINE)
_EMPHASIS_RE = re.compile(r"[*_`]")

_TOKEN_RE = re.compile(r"\w+")
# Dropped from keyword queries so an AND-of-terms match isn't defeated by the
# filler words a spoken query carries ("what's your note about the X route").
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "my",
        "your",
        "our",
        "what",
        "whats",
        "which",
        "who",
        "about",
        "regarding",
        "note",
        "notes",
        "say",
        "says",
        "said",
        "tell",
        "me",
        "find",
        "anything",
        "something",
        "did",
        "do",
        "does",
        "i",
        "you",
        "it",
        "with",
        "from",
        "have",
        "has",
    }
)


def _match_plural(n: int) -> str:
    return "es" if n > 1 else ""


def _query_terms(query: str) -> list[str]:
    """Tokenise a search query into meaningful lowercase terms.

    Drops filler/stopwords so a keyword match is `AND` over the words that
    carry signal, not the whole spoken phrase. Falls back to the raw tokens
    if stripping stopwords would leave nothing (e.g. a one-word query that
    happens to be a stopword).
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
    meaningful = [t for t in tokens if len(t) > 1 and t not in _QUERY_STOPWORDS]
    return meaningful or tokens


def _term_in(term: str, text_lower: str) -> bool:
    """True if `term` occurs in already-lowercased `text_lower`.

    Substring match (so a singular query term hits a plural in the note), plus
    a singularised retry (trailing-'s' stripped) so a plural query term still
    matches the singular in the note ("routes" → "route").
    """
    if term in text_lower:
        return True
    if term.endswith("s") and len(term) > 3 and term[:-1] in text_lower:
        return True
    return False


def _strip_leading_title(md: str, title: str) -> str:
    """Drop a leading markdown header line when it just repeats the note title.

    Notes created by `write_note` start with `# <title>`, which `read_note`
    already announces via its `Note '<title>':` prefix — leaving it in the body
    makes TTS read the title twice. Only stripped when the header actually
    matches the title, so a meaningful first heading is preserved.
    """
    lines = md.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("#"):
        header_text = lines[i].lstrip("#").strip()
        if _slugify(header_text) == _slugify(title):
            return "\n".join(lines[i + 1 :]).strip()
    return md


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_daily_note(path: Path) -> bool:
    """True if `path` is a date-stamped daily/journal note.

    Daily notes live under `DAILY_SUBDIR` (or, when that's unset, as
    `YYYY-MM-DD.md` in the top-level folder). They're owned by
    `append_to_today`/`read_today`, never a named-append target — so a
    semantic *write* must never resolve onto one (see `append_to_note`).
    """
    if DAILY_SUBDIR:
        rel = path.relative_to(notes_root.get_notes_root())
        return bool(rel.parts) and rel.parts[0] == DAILY_SUBDIR
    return bool(_DATE_RE.match(path.stem))


def _find_note_semantic(
    query: str,
    min_score: float = SEMANTIC_MIN_SCORE,
    exclude_daily: bool = False,
) -> Optional[Path]:
    """Resolve a note by topic via the embedding index when literal title
    matching misses.

    `read_note`'s contract is "found by title or topic", but `_find_note`
    only covers the title. A topical query like "notes regarding climate
    change in Australia" won't match the slug `climate-living-advice-australia`,
    so fall through to the semantic index and take the top hit if it clears
    the relevance bar.

    Reads use the default (loose) bar and accept any note — a wrong guess just
    reads the wrong thing aloud. Writes (`append_to_note`) pass a stricter
    `min_score` and `exclude_daily=True`: a loose semantic match would silently
    *write* into the nearest-by-meaning note (a date-like title once landed an
    append in an unrelated daily journal), so the write path takes the first
    qualifying non-daily hit and otherwise misses.
    """
    if not query:
        return None
    try:
        # When filtering, look past the top hit so a leading daily note doesn't
        # mask a valid named note below it.
        results = _get_index().search(query, k=5 if exclude_daily else 1)
    except Exception as e:
        logger.error(f"Semantic note lookup failed for '{query}': {e}")
        return None
    for score, chunk in results:
        if score < min_score:
            break  # index results are score-sorted; nothing further qualifies
        path = notes_root.get_notes_root() / chunk.file
        if not path.exists():
            continue
        if exclude_daily and _is_daily_note(path):
            continue
        last_retrieval["chunks"] = 1
        return path
    return None


# A named append resolved only by topic is a *write*, so demand real confidence
# (well above the loose read-time `SEMANTIC_MIN_SCORE`) before touching a file.
WRITE_SEMANTIC_MIN_SCORE = 0.5


def _appendable_titles() -> list[str]:
    """Spoken titles of notes a named append may target (no facts, no daily)."""
    titles: list[str] = []
    for p in _iter_notes():
        if p.name == f"{FACTS_NOTE}.md" or _is_daily_note(p):
            continue
        titles.append(p.stem.replace("-", " "))
    return titles


def _daily_base() -> Path:
    base = (
        notes_root.get_notes_root() / DAILY_SUBDIR if DAILY_SUBDIR else notes_root.get_notes_root()
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _today_path() -> Path:
    return _daily_base() / f"{_local_time.now().strftime('%Y-%m-%d')}.md"


def _to_spoken(md: str) -> str:
    """Strip markdown markers TTS would otherwise read literally (`#`, `*`, bullets)."""
    text = _HEADER_RE.sub("", md)
    text = _BULLET_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Semantic index (lazy)
#
# The embedding model and its index sit behind `_get_index()` so loading them
# is paid for only on first semantic-search use or first write hook. The
# write hooks below call `_after_write(path)` to keep the index synchronous
# with disk — fine for personal-note counts (sub-second per file).
# ---------------------------------------------------------------------------

_index = None  # type: ignore[var-annotated]
_index_init_lock = threading.Lock()
_index_queue: "queue.Queue[Path]" = queue.Queue(maxsize=32)
_index_pending: set[Path] = set()
_index_pending_lock = threading.Lock()
_index_worker_started = False


def set_obsidian_cmd_q(q: "Optional[queue.Queue]") -> None:
    """Attach the dashboard's live Obsidian bridge to the transport."""
    notes_obsidian.set_command_queue(q)


def _send_obsidian_command(command: dict) -> bool:
    """Queue an explicit editor action when the Obsidian bridge is connected."""
    return notes_obsidian.send_command(command)


def _obsidian_edit_allowed() -> bool:
    return bool((config.get("obsidian") or {}).get("allow_edit_delete", False))


def _get_index():
    """Return a singleton `NotesIndex`, constructing it on first call."""
    global _index
    if _index is not None:
        return _index
    with _index_init_lock:
        if _index is not None:
            return _index
        from .notes_index import NotesIndex

        _index = NotesIndex(
            notes_root=notes_root.get_notes_root(),
            index_path=INDEX_PATH,
            spoken_filter=_to_spoken,
        )
        return _index


def _after_write(path: Path) -> None:
    """Queue one coalesced background re-index after a successful write."""
    global _index_worker_started
    with _index_pending_lock:
        if path not in _index_pending:
            try:
                _index_queue.put_nowait(path)
                _index_pending.add(path)
            except queue.Full:
                logger.warning("Skipping note re-index; queue is full: %s", path)
        if not _index_worker_started:
            _index_worker_started = True

            def _run():
                while True:
                    queued_path = _index_queue.get()
                    try:
                        _get_index().index_file(queued_path)
                    except Exception as e:
                        logger.error(f"Failed to re-index {queued_path}: {e}")
                    finally:
                        with _index_pending_lock:
                            _index_pending.discard(queued_path)

            threading.Thread(target=_run, daemon=True, name="notes-indexer").start()

    notes_obsidian.open_file(path)


@tool(
    name="insert_at_obsidian_cursor",
    description=(
        "Insert explicit text at the cursor in the currently active Obsidian note. "
        "Only use when the user explicitly asks to insert text at their cursor."
    ),
    aliases=["insert_at_cursor", "obsidian_insert"],
    available=_obsidian_edit_allowed,
)
def insert_at_obsidian_cursor(text: str) -> str:
    """Ask the connected Obsidian plugin to insert text at its active cursor."""
    text = text.strip()
    if not text:
        return "What would you like me to insert?"
    if not _obsidian_edit_allowed():
        return "Obsidian editing is disabled in the dashboard for safety."
    if not _send_obsidian_command({"type": "insert", "text": text}):
        return "Obsidian isn't connected, so I can't insert text at its cursor."
    return "Sent that text to the cursor in your active Obsidian note."


@tool(
    name="rename_active_obsidian_note",
    description=(
        "Rename the currently active note in Obsidian. Only use when the user "
        "explicitly asks to rename the note they currently have open."
    ),
    aliases=["rename_active_note", "obsidian_rename"],
    available=_obsidian_edit_allowed,
)
def rename_active_obsidian_note(title: str) -> str:
    """Ask the connected Obsidian plugin to rename its active note."""
    title = title.strip()
    if not title:
        return "What would you like to call the active note?"
    if not _obsidian_edit_allowed():
        return "Obsidian editing is disabled in the dashboard for safety."
    if not _send_obsidian_command({"type": "rename_active", "title": title}):
        return "Obsidian isn't connected, so I can't rename its active note."
    return f"Sent a request to rename the active Obsidian note to '{title}'."


@tool(
    name="delete_active_obsidian_note",
    description=(
        "Delete the currently active Obsidian note. Only use after an explicit "
        "user request, and only when Obsidian edit/delete access is enabled."
    ),
    aliases=["delete_active_note", "obsidian_delete"],
    available=_obsidian_edit_allowed,
)
def delete_active_obsidian_note() -> str:
    """Ask the connected Obsidian plugin to delete its active note."""
    if not _obsidian_edit_allowed():
        return "Obsidian editing is disabled in the dashboard for safety."
    if not _send_obsidian_command({"type": "delete_active"}):
        return "Obsidian isn't connected, so I can't delete its active note."
    return "Sent a request to delete the active Obsidian note."


@tool(
    name="replace_selected_obsidian_text",
    description=(
        "Replace the text currently selected in Obsidian's active editor. Only "
        "use for an explicit request to replace selected text when edit/delete access is enabled."
    ),
    aliases=["replace_selection", "obsidian_replace_selected"],
    available=_obsidian_edit_allowed,
)
def replace_selected_obsidian_text(text: str) -> str:
    """Ask the connected plugin to replace its active editor selection."""
    text = text.strip()
    if not text:
        return "What should replace the selected text?"
    if not _obsidian_edit_allowed():
        return "Obsidian editing is disabled in the dashboard for safety."
    if not _send_obsidian_command({"type": "replace_selection", "text": text}):
        return "Obsidian isn't connected, so I can't replace selected text."
    return "Sent a request to replace the selected text in Obsidian."


def warm_index() -> bool:
    """Pre-load the BGE embedding model and persisted index at startup.

    Calls `scan()`, which loads the model, restores any persisted
    `.npy`/`.json` index, and walks the notes folder to embed anything
    new or mtime-stale. Returns True on success.

    Called from `core.assistant.Assistant._warm_and_announce` so the first
    user-facing semantic-search query isn't slowed by a cold model.
    """
    try:
        _get_index().scan()
        return True
    except Exception:
        logger.exception("Failed to warm notes index")
        return False


@tool(
    name="list_notes",
    description="List the titles of every saved markdown note.",
    aliases=["my_notes", "show_notes"],
)
def list_notes() -> str:
    notes = list(_iter_notes())
    if not notes:
        return "You don't have any notes saved yet."
    titles = [p.stem.replace("-", " ") for p in notes]
    return f"You have {len(titles)} notes: {', '.join(titles)}."


@tool(
    name="read_note",
    description=(
        "Read a saved markdown note aloud, found by title or topic. Use when "
        "the user asks to read, open, or recall a specific note."
    ),
    aliases=["open_note", "recall_note"],
)
def read_note(title: str) -> str:
    note = _find_note(title) or _find_note_semantic(title)
    if note is None:
        return f"I couldn't find a note about {title}."
    try:
        raw = notes_storage.read_markdown(note)
    except OSError as e:
        logger.error(f"Failed to read {note}: {e}")
        return f"I couldn't read the {title} note."
    title_spoken = note.stem.replace("-", " ")
    # Don't speak the title twice — the prefix below already announces it, so
    # strip a leading `# <title>` header from the body if present.
    text = _to_spoken(_strip_leading_title(raw, title_spoken))
    truncated = ""
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS]
        truncated = " (note continues)"
    return ArtifactText(
        f"Note '{title_spoken}': {text}{truncated}",
        {
            "type": "note",
            "title": title_spoken,
            "excerpt": text[:500],
            "truncated": bool(truncated),
        },
    )


@tool(
    name="write_note",
    description=(
        "Create or overwrite a markdown note. Only use when the user explicitly "
        "asks to save, write, or create a note — never proactively."
    ),
    aliases=["create_note", "save_note", "new_note"],
)
def write_note(title: str, content: str) -> str:
    slug = _slugify(title)
    path = notes_root.get_notes_root() / f"{slug}.md"
    body = content.strip()
    # Note already exists — append rather than erroring, so the agent doesn't
    # have to re-dispatch to append_to_note.
    if path.exists():
        if not body:
            return f"The {title} note already exists; there was nothing to add."
        try:
            notes_storage.append_markdown(path, f"\n{body}\n")
        except OSError as e:
            logger.error(f"Failed to append to {path}: {e}")
            return f"I couldn't update the {title} note."
        _after_write(path)
        return f"Added to your existing '{title}' note."
    header = f"# {title.strip()}\n\n"
    try:
        notes_storage.write_markdown(path, header + body + "\n")
    except OSError as e:
        logger.error(f"Failed to write {path}: {e}")
        return f"I couldn't save the {title} note."
    _after_write(path)
    return f"Saved a new note called '{title}'."


@tool(
    name="append_to_note",
    description=(
        "Append content to an existing markdown note. Only use when the user "
        "explicitly asks to add, append, or extend a note — never proactively."
    ),
    aliases=["add_to_note", "extend_note"],
)
def append_to_note(title: str, content: str) -> str:
    line = content.strip()
    if not line:
        return "There was nothing to append."
    # A literal title can exact-match a dated journal (e.g. "2026 06 05" →
    # daily/2026-06-05.md). Refuse it: a named append must never edit a daily
    # log — that's another day's journal. Daily entries go through
    # append_to_today (today only).
    literal = _find_note(title)
    if literal is not None and _is_daily_note(literal):
        return (
            f"Reactive question: {title!r} is a dated daily journal — I won't "
            "write to a past day's log. Save this to a general note instead: "
            "write_note to create one, or append_to_note to an existing named note."
        )
    # Semantic fallback is write-guarded (stricter score, no daily notes) so a
    # loose nearest-by-meaning match can't silently corrupt an unrelated file.
    note = literal or _find_note_semantic(
        title, min_score=WRITE_SEMANTIC_MIN_SCORE, exclude_daily=True
    )
    if note is None:
        # No confident match — bounce back so the agent picks a real title or
        # creates the note, rather than guessing or speaking a dead end.
        existing = ", ".join(_appendable_titles()) or "none"
        return (
            f"Reactive question: I couldn't find an existing note titled "
            f"{title!r} to append to. Existing notes: {existing}. Append to one "
            f"of those by its exact title, or create a new note with write_note."
        )
    try:
        notes_storage.append_markdown(note, f"\n- {line}\n")
    except OSError as e:
        logger.error(f"Failed to append to {note}: {e}")
        return f"I couldn't append to the {title} note."
    _after_write(note)
    return f"Added to your {note.stem.replace('-', ' ')} note."


@tool(
    name="append_to_today",
    description=(
        "Append a line to today's daily markdown note. Use for journal-style "
        "entries, 'add to today', or 'log this' style requests."
    ),
    aliases=["daily_note", "log_today", "add_to_today"],
)
def append_to_today(content: str) -> str:
    line = content.strip()
    if not line:
        return "There was nothing to log."
    path = _today_path()
    now = _local_time.now()
    timestamp = now.strftime("%H:%M")
    try:
        if not path.exists():
            header = f"# {now.strftime('%A %d %B %Y')}\n\n"
            notes_storage.write_markdown(path, header)
        notes_storage.append_markdown(path, f"- {timestamp} {line}\n")
    except OSError as e:
        logger.error(f"Failed to append to {path}: {e}")
        return "I couldn't update today's note."
    _after_write(path)
    return f"Logged at {timestamp} in today's note."


@tool(
    name="read_today",
    description=(
        "Read back the daily note — the dated journal of entries saved with "
        "append_to_today. Pass an optional YYYY-MM-DD date for a past day; "
        "omit for today. This is the correct tool for the daily log — never "
        "use semantic search to find today's note."
    ),
    aliases=["read_daily_note", "todays_note", "read_today_note"],
)
def read_today(date: Optional[str] = None) -> str:
    # Guard the date arg: it builds a filename, and the SLM may pass a literal
    # word ("today") or a relative phrase. Anything that isn't a YYYY-MM-DD
    # stamp falls back to today (also blocks path-traversal via the date).
    date_str = (date or "").strip()
    if not _DATE_RE.match(date_str):
        date_str = _local_time.now().strftime("%Y-%m-%d")
    today_str = _local_time.now().strftime("%Y-%m-%d")
    when = "today" if date_str == today_str else f"on {date_str}"

    path = _daily_base() / f"{date_str}.md"
    if not path.exists():
        return f"You don't have any notes logged {when}."
    try:
        raw = notes_storage.read_markdown(path)
    except OSError as e:
        logger.error(f"Failed to read daily note {path}: {e}")
        return f"I couldn't read your note {when}."
    text = _to_spoken(raw)
    truncated = ""
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS]
        truncated = " (note continues)"
    return f"Your note {when}: {text}{truncated}"


def _truncate_snippet(snippet: str, limit: int = 240) -> str:
    snippet = snippet.strip()
    if len(snippet) > limit:
        return snippet[:limit].rstrip() + "..."
    return snippet


def _keyword_search(query: str) -> list[tuple[str, str]]:
    """`AND`-of-terms full-text search → `(title, snippet)` hits.

    A note matches when *every* meaningful query term appears somewhere in it
    (order-independent, plural-tolerant — see `_term_in`). The surfaced snippet
    is the single line carrying the most query terms, so the agent sees the
    most relevant part rather than the first incidental mention. Replaces the
    old exact-phrase `re.escape` match, which missed any rewording.
    """
    terms = _query_terms(query)
    if not terms:
        return []
    hits: list[tuple[str, str]] = []
    for path in _iter_notes():
        try:
            content = notes_storage.read_markdown(path, errors="ignore")
        except OSError:
            continue
        if not all(_term_in(t, content.lower()) for t in terms):
            continue
        best_line, best_score = "", 0
        for line in content.splitlines():
            line_lower = line.lower()
            score = sum(1 for t in terms if _term_in(t, line_lower))
            if score > best_score:
                best_line, best_score = line, score
        snippet = _to_spoken(best_line).strip()
        if snippet:
            hits.append((path.stem.replace("-", " "), snippet))
        if len(hits) >= MAX_SEARCH_MATCHES:
            break
    return hits


def _semantic_search(query: str) -> list[tuple[str, str]]:
    """Embedding search → `(title, snippet)` hits clearing `SEMANTIC_MIN_SCORE`."""
    try:
        results = _get_index().search(query, k=SEMANTIC_TOP_K)
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
    out: list[tuple[str, str]] = []
    for score, chunk in results:
        if score < SEMANTIC_MIN_SCORE:
            continue
        out.append((Path(chunk.file).stem.replace("-", " "), chunk.text.strip()))
    return out


@tool(
    name="search_notes",
    description=(
        "Search the user's saved markdown notes by keyword, name, meaning or "
        "topic (hybrid keyword + semantic, so exact words or just the gist "
        "both work). Confirm a returned note actually covers the topic from "
        "its text before answering."
    ),
    aliases=[
        "find_notes",
        "lookup_notes",
        "search_notes_semantic",
        "semantic_notes",
        "find_notes_about",
    ],
)
def search_notes(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Please give me something to search for."

    # Fuse both backends. Keyword hits lead (an exact term match is the
    # strongest signal); semantic hits fill in reworded / topical matches the
    # keyword pass can't reach. Dedupe so a paragraph that also satisfied the
    # keyword match isn't surfaced twice.
    fused: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for title, snippet in _keyword_search(query) + _semantic_search(query):
        if not snippet:
            continue
        key = (title.lower(), snippet.lower())
        # Skip a hit whose snippet is already contained in (or contains) one we
        # kept for the same note — keyword lines are often a subset of the
        # semantic paragraph from the same file.
        if any(
            t == title.lower() and (snippet.lower() in s or s in snippet.lower()) for t, s in seen
        ):
            continue
        seen.add(key)
        fused.append((title, _truncate_snippet(snippet)))
        if len(fused) >= MAX_HYBRID_MATCHES:
            break

    last_retrieval["chunks"] = len(fused)
    if not fused:
        return f"I didn't find anything about {query} in your notes."
    parts = "; ".join(f"in '{title}': {snippet}" for title, snippet in fused)
    # `Reactive question:` routes the hits back through the agent loop so the
    # SLM filters / summarises them rather than speaking raw matches. The
    # caveat matters: semantic hits are nearest-by-meaning and may not contain
    # the query term, so the agent must not claim a note mentions it blindly.
    n = len(fused)
    return ArtifactText(
        f"Reactive question: Found {n} possible match{_match_plural(n)} for "
        f"'{query}' in the user's notes (matched by keyword or by topic, so "
        f"some may not contain '{query}' literally). {parts}.",
        {
            "type": "notes_search",
            "query": query[:120],
            "matches": [{"title": title, "excerpt": snippet[:300]} for title, snippet in fused],
        },
    )


@tool(
    name="remember_fact",
    description=(
        "Save a long-term fact across sessions. Only use when the user explicitly "
        "asks you to remember or save a fact — never proactively."
    ),
    aliases=["save_fact", "remember_this", "remember"],
)
def remember_fact(content: str) -> str:
    fact = (content or "").strip()
    if not fact:
        return "There was nothing to remember."
    path = notes_root.get_notes_root() / f"{FACTS_NOTE}.md"
    timestamp = _local_time.now().strftime("%Y-%m-%d")
    try:
        if not path.exists():
            notes_storage.write_markdown(path, "# Long-term facts\n\n")
        notes_storage.append_markdown(path, f"- [{timestamp}] {fact}\n")
    except OSError as e:
        logger.error(f"Failed to append fact: {e}")
        return "I couldn't save that fact."
    _after_write(path)
    return "Got it, I'll remember that."


def recall_facts() -> str:
    """Return saved facts as a prompt-ready 'Known facts' block (or '').

    Read fresh on every call: `utils.prompts.get_agent_system_prompt`
    is rebuilt per turn, so edits via the dashboard or a new `remember_fact`
    are picked up on the next agent call without a restart.
    """
    path = notes_root.get_notes_root() / f"{FACTS_NOTE}.md"
    if not path.exists():
        return ""
    try:
        content = notes_storage.read_markdown(path)
    except OSError as e:
        logger.error(f"Failed to read facts: {e}")
        return ""
    # Drop the markdown header(s) so the system prompt doesn't double up
    # on the "Long-term facts" framing.
    body_lines = [
        line for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not body_lines:
        return ""
    body = "\n".join(body_lines)
    return f"## Known facts about the user\n{body}"


# ---------------------------------------------------------------------------
# Facts CRUD — dashboard-facing helpers (no @tool decorator)
#
# Voice users add facts via `remember_fact` (append-only). The dashboard
# needs view / edit / delete, so we parse the structured `- [DATE] text`
# lines and rewrite the file under a lock. Atomic via tmp+rename so a
# concurrent `recall_facts()` read never sees a partial file.
# ---------------------------------------------------------------------------


def list_facts() -> list[dict]:
    """Return parsed facts in file order. Empty list if fulloch_facts.md is missing."""
    return notes_storage.list_facts()


def update_fact(index: int, text: str) -> bool:
    """Replace the text portion of the indexed fact. Date stamp preserved."""
    text = (text or "").strip()
    if not text:
        return False
    return notes_storage.edit_fact(index, text, after_write=_after_write)


def delete_fact(index: int) -> bool:
    """Remove the indexed fact line."""
    return notes_storage.edit_fact(index, None, after_write=_after_write)


# ---------------------------------------------------------------------------
# Notes CRUD — dashboard-facing helpers (no @tool decorator)
#
# The dashboard lists note files and lets the user read / edit the raw
# markdown. Notes are addressed by `name` — the path relative to NOTES_DIR
# without the `.md` suffix (so a daily note is `daily/2026-05-28`).
# `notes_storage.resolve_note_file` guards against traversal so a crafted name can't
# escape NOTES_DIR. `fulloch_facts.md` is excluded — it has its own dashboard tab and
# its `- [DATE] text` structure would break under free-form editing.
# ---------------------------------------------------------------------------


def list_note_files() -> list[dict]:
    """Return saved notes as `{name, title}` dicts, excluding the facts note."""
    return notes_storage.list_note_files()


def read_note_file(name: str) -> Optional[str]:
    """Return the raw markdown of a note, or None if it can't be read."""
    return notes_storage.read_note_file(name)


def save_note_file(name: str, content: str) -> bool:
    """Overwrite an existing note's content. Atomic via tmp+rename so a
    concurrent semantic-index read never sees a partial file."""
    return notes_storage.save_note_file(name, content, after_write=_after_write)
