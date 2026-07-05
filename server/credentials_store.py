"""Atomic JSON credentials store for data/credentials.json.

All runtime secrets live in one file inside the ./data Docker volume.
Copying the data folder transfers everything to a new machine — no hunting
for scattered .env files.

Credentials map to os.environ names injected at startup via inject_env():
  ha_token              → HA_TOKEN
  llm_api_key           → LLM_API_KEY
  dashboard_password    → DASHBOARD_PASSWORD
  obsidian_token        → OBSIDIAN_TOKEN
  spotify_client_id     → SPOTIFY_CLIENT_ID
  spotify_client_secret → SPOTIFY_CLIENT_SECRET
  spotify_refresh_token → SPOTIFY_REFRESH_TOKEN

System-level env vars (Docker compose, shell exports) always take precedence —
inject_env() uses os.environ.setdefault() and never overwrites existing values.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = "./data/credentials.json"

# Maps JSON keys → os.environ names.
_ENV_MAP: dict[str, str] = {
    "ha_token": "HA_TOKEN",
    "llm_api_key": "LLM_API_KEY",
    "dashboard_password": "DASHBOARD_PASSWORD",
    "obsidian_token": "OBSIDIAN_TOKEN",
    "spotify_client_id": "SPOTIFY_CLIENT_ID",
    "spotify_client_secret": "SPOTIFY_CLIENT_SECRET",
    "spotify_refresh_token": "SPOTIFY_REFRESH_TOKEN",
}


def load(path: str = DEFAULT_PATH) -> dict:
    """Return the parsed credentials dict (empty dict if file absent or unreadable)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text()) or {}
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}


def inject_env(path: str = DEFAULT_PATH) -> None:
    """Load credentials.json and inject into os.environ via setdefault (no overwrite)."""
    creds = load(path)
    for key, env_name in _ENV_MAP.items():
        value = creds.get(key, "")
        if value:
            os.environ.setdefault(env_name, str(value))


def set_credential(key: str, value: str, path: str = DEFAULT_PATH) -> None:
    """Set/replace a single credential atomically and reflect into os.environ."""
    p = Path(path)
    creds = load(path)
    creds[key] = value
    _write(creds, p)
    env_name = _ENV_MAP.get(key)
    if env_name:
        os.environ[env_name] = value
    logger.debug("Credential '%s' updated", key)


def get_credential(key: str, path: str = DEFAULT_PATH) -> str:
    """Read a single credential value (empty string if absent)."""
    return load(path).get(key, "")


def _write(creds: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(creds, f, indent=2)
            f.write("\n")
        os.replace(tmp, str(p))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
