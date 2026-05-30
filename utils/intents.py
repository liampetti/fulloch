"""Action dispatch + replan predicate for the unified agent loop."""

import logging
from typing import Any, Dict, Optional

from tools.tool_registry import tool_registry, UnknownToolError

logger = logging.getLogger(__name__)

# Sentinel prefixes that signal "the agent should be re-called with this
# observation in history". Tools that need SLM follow-up emit one of these
# at the start of their return string. Centralised here so additions don't
# require touching multiple call sites.
REPLAN_SENTINEL_PREFIXES = (
    "User question:",
    "Thinking question:",
    "Summary question:",
    "Reactive question:",
)

# Hard cap on agent re-calls per turn. 1 initial + up to 5 replans = 6.
# With grammar-cap-3 actions per emission, worst-case is 18 tool calls;
# typical multi-step turn is 1–3.
MAX_AGENT_CALLS_PER_TURN = 6

# Canonical name of the web-search tool. The orchestrator uses
# `is_web_search` to play its "searching the web" stall *before* dispatching
# this tool, since the SearXNG round-trip is the slow part of the turn.
WEB_SEARCH_TOOL = "external_information"


def describe_tools() -> str:
    """Human-readable tool listing for the agent system prompt."""
    return tool_registry.describe_tools()


def handle_action(action: Dict[str, Any]) -> Optional[str]:
    """Execute one action from an agent `actions` list.

    `action` is `{"intent": "<name>", "args": [...]}`. Returns the tool's
    output string. On unknown tool, returns a `Reactive question:` sentinel
    so `should_replan` triggers and the next agent call sees the failure as
    an observation it can correct from. On any other dispatch exception
    returns `None` (also triggers replan via `should_replan`).
    """
    if not isinstance(action, dict):
        logger.error(f"Action is not a dict: {action!r}")
        return None
    name = action.get("intent")
    args = action.get("args", [])
    if not name:
        logger.error(f"Action missing intent: {action!r}")
        return None
    try:
        return tool_registry.execute_tool(name, args=args)
    except UnknownToolError as e:
        logger.warning(f"Agent picked unknown tool: {e}")
        return (
            f"Reactive question: The tool {name!r} is not available. "
            f"Use one of the tools listed in the system prompt."
        )
    except Exception as e:
        logger.exception(f"Error executing action {name}: {e}")
        return None


def is_web_search(intent_name: Optional[str]) -> bool:
    """True if `intent_name` resolves (directly or via alias) to the web-search
    tool. Used by the orchestrator to play the search stall before dispatch."""
    if not intent_name:
        return False
    return tool_registry.canonical_name(intent_name) == WEB_SEARCH_TOOL


def should_replan(step_result: Optional[str]) -> bool:
    """True if a step's result should trigger another agent call.

    Triggers:
      - `None` (unknown tool, exception)
      - Output starts with one of the routing sentinels
    """
    if step_result is None:
        return True
    if not isinstance(step_result, str):
        return False
    stripped = step_result.lstrip()
    return any(stripped.startswith(p) for p in REPLAN_SENTINEL_PREFIXES)
