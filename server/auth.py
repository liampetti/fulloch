"""Password hashing and session management for the dashboard auth gate."""

import hashlib
import hmac
import json
import os
import secrets
import time

SESSION_COOKIE = "fulloch_session"
# Cookie max-age sent to the browser — long enough that the user never notices
# it expiring. Sessions are persisted to SESSIONS_PATH (see load_sessions/
# save_sessions) so a restart — including the rebuild-and-restart that follows
# any code/prompt edit — doesn't force a re-login within this window.
COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 year
_PBKDF2_ITERS = 600_000

SESSIONS_PATH = os.environ.get("FULLOCH_SESSIONS_PATH", "data/dashboard_sessions.json")


def hash_password(password: str) -> str:
    """Return 'salt$digest_hex' suitable for storage in credentials.json."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERS)
    return hmac.compare_digest(expected.hex(), digest_hex)


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def load_sessions(path: str = SESSIONS_PATH) -> dict:
    """Read persisted sessions ({session_id: created_timestamp}), dropping any
    past COOKIE_MAX_AGE. Empty if the file is absent/invalid.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    now = time.time()
    return {
        sid: ts for sid, ts in data.items()
        if isinstance(ts, (int, float)) and now - ts < COOKIE_MAX_AGE
    }


def save_sessions(sessions: dict, path: str = SESSIONS_PATH) -> None:
    """Persist the session store atomically (temp file + os.replace)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
    os.replace(tmp, path)
