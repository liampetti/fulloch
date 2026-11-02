"""Compatibility imports for the deep-think job context in core."""

from core.thinking_context import (
    get_artifact,
    get_artifacts,
    get_handoff,
    reset_artifacts,
    reset_handoff,
    set_artifacts,
    set_handoff,
)

__all__ = [
    "get_artifact", "get_artifacts", "get_handoff", "reset_artifacts",
    "reset_handoff", "set_artifacts", "set_handoff",
]
