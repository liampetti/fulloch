"""Tools package.

Loads each tool module conditionally based on the keys present in
data/config.yml — commenting out an integration's config block disables it.
"""

import importlib
import logging

from ._config import config
from .tool_registry import tool, tool_registry  # noqa: F401  (public surface)

logger = logging.getLogger(__name__)

# Tools that load whenever their top-level config key is present.
_OPTIONAL_TOOLS = {
    "home_assistant": "home_assistant",
    "search": "search_web",
    "spotify": "spotify",
}

# Tools that are always loaded — no config dependency.
_ALWAYS_LOAD = ["time_tools", "thinking", "notes", "calculator"]


def _load(module_name: str) -> None:
    try:
        importlib.import_module(f".{module_name}", package=__name__)
        logger.info(f"Loaded tool: {module_name}")
    except Exception as e:
        logger.error(f"Failed to load tool {module_name}: {e}")


for _name in _ALWAYS_LOAD:
    _load(_name)

for _key, _name in _OPTIONAL_TOOLS.items():
    if _key in config:
        _load(_name)
    else:
        logger.info(f"Skipping tool {_name} ('{_key}' not in config)")


__all__ = ["tool_registry", "tool"]
