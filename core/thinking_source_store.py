"""Content-addressed storage for immutable cleaned research documents."""

import hashlib
import json
import os
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("THINKING_SOURCE_STORE_DIR", "./data/thinking-sources"))


def store_document(url: str, text: str) -> dict:
    """Persist one exact body and return a stable version reference."""
    body = str(text or "")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    key = hashlib.sha256(str(url or "").encode("utf-8")).hexdigest()[:24]
    path = _root() / key / f"{digest}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"url": str(url or ""), "content_hash": digest, "text": body},
                       ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        try:
            temporary.replace(path)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
    return {"key": key, "content_hash": digest}


def load_document(reference: dict) -> dict:
    """Rehydrate and integrity-check a stored body from its version reference."""
    key = reference.get("key") if isinstance(reference, dict) else None
    digest = reference.get("content_hash") if isinstance(reference, dict) else None
    if not isinstance(key, str) or not isinstance(digest, str):
        return {}
    path = _root() / key / f"{digest}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    text = value.get("text")
    if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
        return {}
    return value


def archive_source(source: dict) -> dict:
    """Replace a source's long body with an immutable external version reference."""
    if not isinstance(source, dict):
        return source
    body_key = "source_text" if isinstance(source.get("source_text"), str) else "text"
    body = source.get(body_key)
    if not isinstance(body, str) or not body:
        return source
    reference = store_document(str(source.get("canonical_url") or source.get("url") or ""), body)
    archived = {key: value for key, value in source.items() if key != body_key}
    archived["source_ref"] = reference
    return archived


def resolve_source(source: dict) -> dict:
    """Return an ephemeral hydrated source view without mutating stored state."""
    if not isinstance(source, dict) or not isinstance(source.get("source_ref"), dict):
        return dict(source) if isinstance(source, dict) else {}
    document = load_document(source["source_ref"])
    return {**source, **({"source_text": document["text"]} if document else {})}
