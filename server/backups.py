"""Timestamped setup snapshots of user state, excluding models and certificates."""

import json
import logging
import re
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_BACKUP_FILES = ("config.yml", "credentials.json", ".env", "voice_denylist.json")
_BACKUP_DIRS = ("voices",)
_SAFE_BACKUP_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}(?:-\d+)?$")
_MAX_SETUP_BACKUPS = 10


def _prune_backups(backups_root: Path) -> None:
    """Keep only the ten newest valid setup snapshots."""
    try:
        backups = sorted(
            (path for path in backups_root.iterdir() if path.is_dir() and _SAFE_BACKUP_NAME_RE.match(path.name)),
            key=lambda path: path.name,
            reverse=True,
        )
        for backup in backups[_MAX_SETUP_BACKUPS:]:
            shutil.rmtree(backup)
    except OSError as exc:
        logger.warning("Could not prune setup backups in %s: %s", backups_root, exc)


def _create_backup(data_dir: Path) -> Path:
    """Snapshot state into backups/<UTC timestamp>, adding a suffix on collision."""
    backups_root = data_dir / "backups"
    backups_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
    target = backups_root / ts
    suffix = 0
    while target.exists():
        suffix += 1
        target = backups_root / f"{ts}-{suffix}"
    target.mkdir(parents=True, exist_ok=True)
    backed_up: list[str] = []
    for rel in _BACKUP_FILES:
        src = data_dir / rel
        if not src.is_file():
            continue
        shutil.copy2(src, target / rel)
        backed_up.append(rel)
    for rel in _BACKUP_DIRS:
        src = data_dir / rel
        if not src.is_dir():
            continue
        shutil.copytree(src, target / rel)
        backed_up.append(f"{rel}/")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": backed_up,
                "reason": "setup reset",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _prune_backups(backups_root)
    return target


def _list_backups(data_dir: Path) -> list[dict]:
    """Return backup summaries, newest first."""
    backups_root = data_dir / "backups"
    if not backups_root.is_dir():
        return []
    out: list[dict] = []
    for d in sorted(backups_root.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir() or not _SAFE_BACKUP_NAME_RE.match(d.name):
            continue
        meta = d / "meta.json"
        files: list[str] = []
        created_at = None
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                files = m.get("files", [])
                created_at = m.get("created_at")
            except Exception:
                pass
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        out.append({"name": d.name, "created_at": created_at, "files": files, "size_bytes": size})
    return out


def _restore_backup(backup_dir: Path, data_dir: Path) -> list[str]:
    """Copy each file/dir in the backup back to data_dir, overwriting."""
    restored: list[str] = []
    for rel in _BACKUP_FILES:
        src = backup_dir / rel
        if not src.exists():
            continue
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
    for rel in _BACKUP_DIRS:
        src = backup_dir / rel
        if not src.is_dir():
            continue
        dst = data_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        restored.append(f"{rel}/")
    return restored
