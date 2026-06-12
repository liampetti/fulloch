"""Action dispatch + typed step results for the unified agent loop."""

import enum
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.thinking import SUMMARY_PREFIX, THINKING_PREFIX
from tools.tool_registry import UnknownToolError, tool_registry

logger = logging.getLogger(__name__)

# Leading sentinels a tool can emit to request an SLM follow-up. Tools that
# need a re-call embed one at the start of their return string. `classify_step`
# is the SINGLE place these prefixes are matched — the agent loop then routes on
# the typed `StepKind`, never on the raw string. Centralising the match removes
# the hijack risk where any downstream code re-checking prefixes could mis-route
# a tool output that merely happened to begin with a sentinel.
WEB_QUESTION_PREFIX = "User question:"
REACTIVE_PREFIX = "Reactive question:"


class StepKind(enum.Enum):
    """Routing class of a dispatched action's result."""

    NORMAL = "normal"          # plain output — joined into the spoken reply
    WEB_SEARCH = "web_search"  # raw SearXNG payload — summarise inline, replan
    THINKING = "thinking"      # deep_think flagged — run the /think branch
    SUMMARY = "summary"        # summarize_thinking — surface captured partial
    REACTIVE = "reactive"      # tool error / HA 4xx — replan with failure shown
    ERROR = "error"            # dispatch returned None / raised — replan


# Prefix → kind, in match priority order.
_SENTINEL_KINDS = (
    (WEB_QUESTION_PREFIX, StepKind.WEB_SEARCH),
    (THINKING_PREFIX, StepKind.THINKING),
    (SUMMARY_PREFIX, StepKind.SUMMARY),
    (REACTIVE_PREFIX, StepKind.REACTIVE),
)

# Kinds that hand control back to the agent for another call.
_REPLAN_KINDS = frozenset({
    StepKind.WEB_SEARCH, StepKind.THINKING, StepKind.SUMMARY,
    StepKind.REACTIVE, StepKind.ERROR,
})


@dataclass(frozen=True)
class StepResult:
    """Typed outcome of one dispatched action.

    `text` is the history / observation representation (the raw tool string,
    or "<error>" for a failed dispatch). `in_output` is whether the text should
    be joined into the final spoken reply — true only for plain string results;
    sentinel/error kinds force a replan so they never reach the join anyway.
    """

    kind: StepKind
    text: str
    in_output: bool

    @property
    def should_replan(self) -> bool:
        return self.kind in _REPLAN_KINDS


def classify_step(raw: Optional[object]) -> StepResult:
    """Convert a raw tool result into a typed `StepResult`.

    The single boundary where sentinel prefixes are matched. `None` (a failed
    dispatch) becomes `ERROR`; a non-string result is treated as plain output
    that isn't spoken; a string is matched against the sentinel prefixes (after
    `lstrip`, so leading whitespace doesn't hide one) and otherwise `NORMAL`.
    """
    if raw is None:
        return StepResult(StepKind.ERROR, "<error>", in_output=False)
    if not isinstance(raw, str):
        return StepResult(StepKind.NORMAL, str(raw), in_output=False)
    stripped = raw.lstrip()
    for prefix, kind in _SENTINEL_KINDS:
        if stripped.startswith(prefix):
            return StepResult(kind, raw, in_output=True)
    return StepResult(StepKind.NORMAL, raw, in_output=True)


# Hard cap on agent re-calls per turn. 1 initial + up to 5 replans = 6.
# With grammar-cap-3 actions per emission, worst-case is 18 tool calls;
# typical multi-step turn is 1–3.
MAX_AGENT_CALLS_PER_TURN = 6

# Canonical name of the web-search tool. The orchestrator uses
# `is_web_search` to play its "searching the web" stall *before* dispatching
# this tool, since the SearXNG round-trip is the slow part of the turn.
WEB_SEARCH_TOOL = "external_information"
NOTE_WRITE_TOOLS = frozenset({"write_note", "append_to_note", "remember_fact"})


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


def is_note_write(intent_name: Optional[str]) -> bool:
    """True if `intent_name` resolves to a note-write tool."""
    if not intent_name:
        return False
    return tool_registry.canonical_name(intent_name) in NOTE_WRITE_TOOLS


def is_web_search(intent_name: Optional[str]) -> bool:
    """True if `intent_name` resolves (directly or via alias) to the web-search
    tool. Used by the orchestrator to play the search stall before dispatch."""
    if not intent_name:
        return False
    return tool_registry.canonical_name(intent_name) == WEB_SEARCH_TOOL


def should_replan(step_result: Optional[object]) -> bool:
    """True if a step's result should trigger another agent call.

    Thin wrapper over `classify_step` for callers/tests that work with the raw
    string form (`None` or a sentinel-prefixed string). New code in the agent
    loop routes on `StepResult.kind` / `StepResult.should_replan` instead.
    """
    return classify_step(step_result).should_replan
