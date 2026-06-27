"""Shared config loader for tool modules.

Loaded once at import time so individual tools don't each re-parse the YAML.
Keys nested under `internal_tools:` are flattened onto the top-level mapping
(top-level wins) so tool modules can read `config['<integration>']` regardless
of where the user put the block in the YAML.
"""

import yaml

try:
    with open("./data/config.yml", "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}

for _k, _v in (config.get("internal_tools") or {}).items():
    config.setdefault(_k, _v)
