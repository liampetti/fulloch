"""Shared config loader for tool modules.

Loaded once at import time so individual tools don't each re-parse the YAML.
Keys nested under `internal_tools:` are flattened onto the top-level mapping
(top-level wins) so tool modules can read `config['<integration>']` regardless
of where the user put the block in the YAML.

Path is overridable via `FULLOCH_CONFIG_PATH` — the test suite points this at
`data/config.example.yml` (see conftest.py) so tests are deterministic across
machines and CI instead of picking up whatever a developer's real, gitignored
`data/config.yml` happens to have configured (a real Spotify/HA block would
silently change which code paths tests exercise).
"""

import os

import yaml

CONFIG_PATH = os.environ.get("FULLOCH_CONFIG_PATH", "./data/config.yml")

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}

for _k, _v in (config.get("internal_tools") or {}).items():
    config.setdefault(_k, _v)
