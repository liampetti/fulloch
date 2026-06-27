"""Read / validate / write `config.yml`, preserving comments (v2.2 Step 4).

`config.yml` is the single source of truth. The settings console reads it
through `settings_view()` (schema + current values), and writes go through
`update_config()`, which validates every key against `config_schema.SCHEMA`,
coerces it to the declared type, and writes atomically. Writes produce a clean,
**comment-free** config (see `_strip_comments`) so the active values are easy to
read; the shipped `config.example.yml` is the inline documentation reference. A
brand-new config is still seeded from the example to inherit its default values.
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedBase, CommentedMap

from .config_schema import (
    GROUPS,
    field_for,
    schema_as_dicts,  # noqa: F401  (re-exported for the endpoint)
    tier_presets_as_dicts,
    wakeword_presets_as_dicts,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "./data/config.yml"


class ConfigValidationError(ValueError):
    """One or more updates failed schema validation. `.errors` is {path: msg}."""

    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096  # don't wrap long help-comment lines
    return y


def _load_doc(path: str, example_path: Optional[str] = None) -> CommentedMap:
    """Load the round-trip doc, seeding from the example when absent/empty.

    The seed example defaults to a `config.example.yml` sibling of `path`, so a
    brand-new config inherits the template's default values. Comments are dropped
    on write (`_strip_comments`), so the seed's docs don't persist into config.yml.
    """
    y = _yaml()
    p = Path(path)
    if p.is_file():
        with p.open() as f:
            data = y.load(f)
        if data is not None:
            return data
    ex = Path(example_path) if example_path else p.with_name("config.example.yml")
    if ex.is_file():
        with ex.open() as f:
            data = y.load(f)
        if data is not None:
            logger.info("Seeding new config from %s", example_path)
            return data
    return CommentedMap()


def read_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Current config as a plain-ish dict (ruamel CommentedMap is dict-like)."""
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open() as f:
        return _yaml().load(f) or {}


def _coerce(field, raw: Any) -> Optional[Any]:
    """Coerce a submitted value to the field's declared type.

    An empty string / None means "unset" → returns None (the key is removed
    on write). Raises ValueError on a type mismatch.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None
    t = field.type
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if t == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"expected an integer, got {raw!r}") from None
    if t == "float":
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"expected a number, got {raw!r}") from None
    if t == "enum":
        s = str(raw)
        if s not in field.choices:
            raise ValueError(f"must be one of {', '.join(field.choices)}")
        return s
    if t == "list":
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        else:
            items = [s.strip() for s in re.split(r"[\n,]", str(raw)) if s.strip()]
        return items or None
    return str(raw)


def _set_value(doc: CommentedMap, section: str, name: str, value: Any) -> None:
    sec = doc.get(section)
    if value is None:
        if isinstance(sec, dict) and name in sec:
            del sec[name]
        return
    if not isinstance(sec, dict):
        sec = CommentedMap()
        doc[section] = sec
    sec[name] = value


def _strip_comments(node) -> None:
    """Recursively drop every comment from a ruamel doc (in place).

    The seed `config.example.yml` carries the full inline documentation, which
    made a written `config.yml` hard to read — the active values were buried in
    template comments, and console-set keys appeared appended among them. We keep
    the example as the documentation reference and write a clean, comment-free
    config so it's obvious what's actually set.
    """
    if isinstance(node, CommentedBase):
        try:
            node.ca.comment = None
            node.ca.items.clear()
        except Exception:  # noqa: BLE001 — comment metadata is best-effort
            pass
    if isinstance(node, dict):
        for v in node.values():
            _strip_comments(v)
    elif isinstance(node, list):
        for v in node:
            _strip_comments(v)


def _atomic_dump(doc: CommentedMap, path: str) -> None:
    _strip_comments(doc)  # write a clean config (docs live in config.example.yml)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            _yaml().dump(doc, f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update_config(updates: dict, path: str = DEFAULT_CONFIG_PATH) -> list:
    """Validate + apply a {dotted_path: value} map, writing atomically.

    Every key must be a known schema path. Raises ConfigValidationError
    (with all offending keys) before writing anything. Returns the applied
    changes as `[{"path", "apply", "value"}]` — `apply` lets the caller flag
    restart-required edits and `value` (already coerced) lets it push the
    change to the running assistant for live hot-apply.
    """
    errors: dict = {}
    coerced: list = []
    for dotted, raw in updates.items():
        f = field_for(dotted)
        if f is None:
            errors[dotted] = "unknown config key"
            continue
        try:
            coerced.append((f, _coerce(f, raw)))
        except ValueError as e:
            errors[dotted] = str(e)
    if errors:
        raise ConfigValidationError(errors)

    doc = _load_doc(path)
    for f, value in coerced:
        _set_value(doc, f.section, f.name, value)
    _atomic_dump(doc, path)
    return [{"path": f.path, "apply": f.apply, "value": value} for f, value in coerced]


def write_models(models: dict, path: str = DEFAULT_CONFIG_PATH) -> None:
    """Write the structured `models:` block (tier/backends)."""
    doc = _load_doc(path)
    block = CommentedMap()
    for domain in ("asr", "tts", "llm"):
        if domain in models:
            inner = CommentedMap()
            for k, v in models[domain].items():
                inner[k] = v
            block[domain] = inner
    doc["models"] = block
    _atomic_dump(doc, path)


def set_llm_model_name(name: str, path: str = DEFAULT_CONFIG_PATH) -> None:
    """Persist only `models.llm.model`, preserving the rest of the block.

    Used by the live model switch (POST /llm/model) so the chosen model survives
    a restart without rewriting backend / base_url / api_key. Raises KeyError if
    there's no structured `models.llm` block to patch.
    """
    doc = _load_doc(path)
    models = doc.get("models")
    if models is None or "llm" not in models:
        raise KeyError("models.llm not configured")
    models["llm"]["model"] = name
    _atomic_dump(doc, path)


def _backends_view() -> dict:
    """Per-domain backend options for the wizard dropdowns.

    `offerable` reflects the running image variant (the CPU image can't offer
    GPU-only backends), so the wizard renders only what this image can run.
    """
    from core.backends import DOMAINS, is_offerable, list_backends
    out: dict = {}
    for domain in DOMAINS:
        out[domain] = [
            {
                "backend": s.backend, "display_name": s.display_name,
                "implemented": s.implemented, "offerable": is_offerable(s),
                "cpu_ok": s.cpu_ok, "gpu_only": s.gpu_only,
                "vram_gb": s.vram_gb, "download_size_gb": s.download_size_gb,
                "notes": s.notes, "experimental": s.experimental,
            }
            for s in list_backends(domain)
        ]
    return out


def _tiers_view() -> list:
    """Tier presets tagged with `offerable` for the running image variant."""
    from core.backends import get_spec, is_offerable
    tiers = tier_presets_as_dicts()
    for tier in tiers:
        models = tier["models"]
        tier["offerable"] = all(
            is_offerable(get_spec(domain, models[domain]["backend"]))
            for domain in models
        )
    return tiers


def settings_view(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Schema + current values + presets — everything the console UI needs."""
    cfg = read_config(path)
    fields = []
    for spec in schema_as_dicts():
        section = cfg.get(spec["section"])
        present = isinstance(section, dict) and spec["name"] in section
        value = section.get(spec["name"]) if present else None
        fields.append({**spec, "value": value, "set": present})
    from core.backends import variant
    return {
        "groups": list(GROUPS),
        "fields": fields,
        "models": cfg.get("models"),
        "variant": variant(),
        "backends": _backends_view(),
        "wakeword_presets": wakeword_presets_as_dicts(),
        "tier_presets": _tiers_view(),
        # Whether a remote-LLM API key is already in the environment, so the UI
        # can say "found in env" and skip prompting for one. Mirrors the
        # precedence in core/llm_openai.py:_resolve_api_key.
        "llm_api_key_in_env": bool(
            os.environ.get("FULLOCH_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        ),
    }
