"""Configured notes-root directory and Obsidian path translation.

`notes.path` in config.yml is the sole source of truth for note storage.
The Obsidian plugin reports live editor context but never changes where
Fulloch writes notes while the plugin is disconnected.
"""
import logging
import os
import threading
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
_CONFIG_PATH = Path(os.environ.get("FULLOCH_CONFIG_PATH", str(_DATA_DIR / "config.yml")))

_lock = threading.Lock()
_current_root: Optional[Path] = None
# Test-only in-memory override retained for isolated note-tool tests. No
# production path calls this helper and it is never written to disk.
_override: Optional[Path] = None
_migrated = False  # Compatibility sentinel for isolated legacy tests.
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


def register_index_listener(
    listener: Callable[[Optional[Path], Optional[Path]], None],
) -> None:
    """Subscribe `NotesIndex` to root-change events. Replaces any prior listener."""
    global _index_listener
    with _lock:
        _index_listener = listener


def get_notes_root() -> Path:
    """Return the configured notes directory."""
    with _lock:
        return _override or _current_root or _load_default_path()


def set_notes_root(path: Optional[Path], persist: bool = False) -> bool:
    """Test helper for temporarily selecting an isolated notes directory.

    Production storage is configured only through notes.path; `persist` is
    accepted for old callers but intentionally ignored.
    """
    del persist
    global _override
    new_path = path.expanduser().resolve() if path is not None else None
    with _lock:
        old_path = _override or _current_root
        if old_path == new_path:
            return False
        _override = new_path
        listener = _index_listener
    if listener is not None:
        listener(old_path, new_path or _current_root)
    return True


def refresh_notes_root() -> bool:
    """Reload notes.path and notify the search index if it changed."""
    global _current_root
    new_path = _load_default_path()
    with _lock:
        old_path = _current_root
        if old_path == new_path:
            return False
        _current_root = new_path
        listener = _index_listener
    if listener is not None:
        try:
            listener(old_path, new_path)
        except Exception:
            logger.exception("Index listener raised on notes-root change")
    return True


refresh_notes_root()

# Seed the path-translation cache so the WebSocket handler and HTTP
# endpoints can translate without an explicit init() call. Callers that
# edit config.yml at runtime should invoke reload_translation_map() to
# refresh.
reload_translation_map()
