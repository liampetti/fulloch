"""
Tools package for the voice assistant.

This package contains all the tools that can be used by the voice assistant.
Tools are conditionally loaded based on which integrations are configured
in data/config.yml. Commenting out or removing a config section disables
that tool entirely.
"""

import yaml
import logging
import importlib

logger = logging.getLogger(__name__)

# Load configuration to determine which tools to activate
try:
    with open("./data/config.yml", "r") as f:
        _config = yaml.safe_load(f) or {}
except FileNotFoundError:
    _config = {}

# Mapping of tool module names to their required config key.
# A tool is only loaded if its config key is present in config.yml.
_TOOL_CONFIG_MAP = {
    'spotify': 'spotify',
    'lighting': 'philips',
    'google_calendar': 'google',
    'airtouch': 'airtouch',
    'thinq': 'thinq',
    'webos': 'webos',
    'search_web': 'search',
    'pioneer_avr': 'pioneer',
    'home_assistant': 'home_assistant',
}

# Always load these tools (no config dependency)
_ALWAYS_LOAD = ['weather_time']

for _module_name in _ALWAYS_LOAD:
    try:
        importlib.import_module(f'.{_module_name}', package=__name__)
    except Exception as e:
        logger.error(f"Failed to load tool {_module_name}: {e}")

for _module_name, _config_key in _TOOL_CONFIG_MAP.items():
    if _config_key in _config:
        try:
            importlib.import_module(f'.{_module_name}', package=__name__)
            logger.info(f"Loaded tool: {_module_name}")
        except Exception as e:
            logger.error(f"Failed to load tool {_module_name}: {e}")
    else:
        logger.info(f"Skipping tool {_module_name} ('{_config_key}' not in config)")

# Import the tool registry
from .tool_registry import tool_registry, tool

__all__ = ['tool_registry', 'tool']
