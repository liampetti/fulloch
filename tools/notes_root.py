"""Sticky override for the notes root directory.

The default notes root is read from `data/config.yml` (`notes.path` or
`./data/notes`). When the Obsidian plugin connects with a `vault_path`, that
path becomes a *sticky* override that persists across restarts and plugin
disconnects. Voice commands work whether or not Obsidian is running, because
the override survives both.

The override is stored at `data/notes_root_override.json`:
  {"path": "/abs/path/to/vault", "set_at": "...", "migrated": false}

`tools.notes_index.NotesIndex` registers as a listener so the embedding index
is invalidated and rebuilt on root change.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

logger = logging.getLogger(__name__)

# Module-level cache of the obsidian path-translation map (read from
# data/config.yml `obsidian.path_translation`). Cleared on override changes so
# the dashboard's "switch vault" + wizard "save vault" endpoints pick up
# edits without a restart.
_translation_map: dict[str, str] = {}
_translation_lock = threading.Lock()


def _read_translation_map() -> dict[str, str]:
    """Read `obsidian.path_translation` from config.yml (empty dict if absent)."""
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    obsidian_cfg = cfg.get("obsidian") or {}
    raw = obsidian_cfg.get("path_translation") or {}
    if not isinstance(raw, dict):
        logger.warning("obsidian.path_translation is not a dict — ignoring")
        return {}
    # Normalise keys/values to stripped strings; drop empty pairs.
    out: dict[str, str] = {}
    for k, v in raw.items():
        kk = str(k).strip()
        vv = str(v).strip() if v is not None else ""
        if kk and vv:
            out[kk] = vv
    return out


def reload_translation_map() -> None:
    """Refresh the cached translation map from disk. Cheap; called at the
    WebSocket handler and at the switch-vault endpoints before each use so
    config edits show up without a restart."""
    global _translation_map
    with _translation_lock:
        _translation_map = _read_translation_map()


def translate_vault_path(raw: str) -> Path:
    """Apply the host→container prefix translation to a vault path.

    The Obsidian plugin runs on the host, so it reports paths in the host's
    filesystem namespace. When Fulloch runs in Docker, the same path is
    invisible unless the user has bind-mounted it. The translation map
    (configured under `obsidian.path_translation`) lets the user point
    `/Users/jane/Documents/MyVault` (host) at `/vault/Documents/MyVault`
    (container) so file operations land in the right place.

    Longest matching host prefix wins. Unmatched paths are returned
    unchanged — so a native install with no translation map keeps working
    as before.
    """
    if not raw:
        return Path(raw)
    s = str(raw)
    with _translation_lock:
        # Sort by length so the longest matching prefix wins.
        candidates = sorted(_translation_map.items(), key=lambda kv: -len(kv[0]))
        for host_prefix, container_prefix in candidates:
            if s == host_prefix or s.startswith(host_prefix + os.sep):
                translated = container_prefix + s[len(host_prefix):]
                return Path(translated)
        return Path(s)


_DATA_DIR = Path("./data")
# Overridable so the test suite can point at a scratch file instead of the
# real sticky override (which may name a path — e.g. a Docker-only mount —
# that doesn't exist on the native dev machine running pytest).
_OVERRIDE_PATH = Path(
    os.environ.get("FULLOCH_NOTES_ROOT_OVERRIDE_PATH", str(_DATA_DIR / "notes_root_override.json"))
)
_CONFIG_PATH = Path(os.environ.get("FULLOCH_CONFIG_PATH", str(_DATA_DIR / "config.yml")))

_lock = threading.Lock()
_override: Optional[Path] = None
_migrated: bool = False
_index_listener: Optional[Callable[[Optional[Path], Optional[Path]], None]] = None


def _load_default_path() -> Path:
    """Read `notes.path` from config.yml; fall back to ./data/notes if absent."""
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    notes_cfg = cfg.get("notes") or {}
    raw = notes_cfg.get("path", "./data/notes")
    return Path(raw).expanduser().resolve()


def _read_override_file() -> tuple[Optional[Path], bool]:
    """Load (override_path, migrated_flag) from disk; default to (None, False)."""
    if not _OVERRIDE_PATH.is_file():
        return None, False
    try:
        data = json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8"))
        p = data.get("path")
        migrated = bool(data.get("migrated", False))
        return (Path(p).expanduser().resolve() if p else None), migrated
    except Exception as e:
        logger.warning("Failed to read %s: %s", _OVERRIDE_PATH, e)
        return None, False


def _write_override_file(path: Optional[Path], migrated: bool) -> None:
    """Atomically write the override JSON; tmp+rename so a concurrent read
    never sees a partial file."""
    payload: dict = {
        "migrated": migrated,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    if path is not None:
        payload["path"] = str(path)
    _OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OVERRIDE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, _OVERRIDE_PATH)
    finally:
        if tmp.exists():
            tmp.unlink()


def register_index_listener(
    listener: Callable[[Optional[Path], Optional[Path]], None],
) -> None:
    """Subscribe `NotesIndex` to root-change events. Replaces any prior listener."""
    global _index_listener
    with _lock:
        _index_listener = listener


def get_notes_root() -> Path:
    """Return the current effective notes directory: override or config default."""
    with _lock:
        if _override is not None:
            return _override
    return _load_default_path()


def set_notes_root(path: Optional[Path], persist: bool = True) -> bool:
    """Set/clear the sticky override. Returns True if it actually changed.

    On change, fires the registered index listener so the embedding index
    can be invalidated and rebuilt.
    """
    global _override
    new_path = path.expanduser().resolve() if path is not None else None
    old_path: Optional[Path]
    changed = False
    listener: Optional[Callable[[Optional[Path], Optional[Path]], None]] = None
    with _lock:
        old_path = _override
        if old_path != new_path:
            _override = new_path
            changed = True
            if persist:
                try:
                    _write_override_file(new_path, _migrated)
                except Exception as e:
                    logger.error("Failed to persist notes-root override: %s", e)
            listener = _index_listener
    if changed and listener is not None:
        try:
            listener(old_path, new_path)
        except Exception:
            logger.exception("Index listener raised on notes-root change")
    return changed


def get_migrated() -> bool:
    with _lock:
        return _migrated


def set_migrated(value: bool) -> None:
    global _migrated
    with _lock:
        _migrated = value
        try:
            _write_override_file(_override, value)
        except Exception as e:
            logger.error("Failed to persist migrated flag: %s", e)


def init() -> None:
    """Read the override file at startup. Called once on module import."""
    global _override, _migrated
    p, m = _read_override_file()
    with _lock:
        _override = p
        _migrated = m
    if p is not None:
        logger.info("Loaded notes-root override: %s", p)


init()

# Seed the path-translation cache so the WebSocket handler and HTTP
# endpoints can translate without an explicit init() call. Callers that
# edit config.yml at runtime should invoke reload_translation_map() to
# refresh.
reload_translation_map()
