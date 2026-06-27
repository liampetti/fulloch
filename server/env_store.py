"""Minimal `.env` updater for the post-setup token step (v2.2 Step 5).

The wizard generates a dashboard access token and persists it to `.env` as
`FULLOCH_DASHBOARD_TOKEN` so it survives a restart. We only need to set/replace
a single key while leaving the rest of the file (other vars, comments) intact.
"""

import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENV_PATH = ".env"


def set_env_var(key: str, value: str, path: str = DEFAULT_ENV_PATH) -> None:
    """Set/replace `KEY=value` in `path` (creating it from .env.example/empty).

    Replaces the first uncommented `KEY=` line; appends if absent. Writes
    atomically. Other lines (vars, comments) are preserved verbatim.
    """
    p = Path(path)
    if p.is_file():
        lines = p.read_text().splitlines()
    else:
        example = p.with_name(".env.example")
        lines = example.read_text().splitlines() if example.is_file() else []

    new_line = f"{key}={value}"
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replaced = False
    out = []
    for line in lines:
        if not replaced and pattern.match(line):
            out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(new_line)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    # Reflect immediately for this process too.
    os.environ[key] = value
