"""Action dispatch + typed step results for the unified agent loop."""

import enum
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.thinking import SUMMARY_PREFIX, THINKING_PREFIX
from tools.tool_registry import UnknownToolError, tool_registry

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAG = re.compile(r"</?think>")


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring, respecting JSON strings.

    Brace-matched and string-aware (braces inside string values don't count),
    so trailing content after the first complete object is ignored.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_agent_emission(text: str) -> Dict[str, Any]:
    """Parse the agent's JSON emission, tolerating reasoning-model noise.

    A remote LLM without the GBNF grammar can wrap or trail the JSON with
    reasoning artefacts — a ``<think>...</think>`` block, a stray ``</think>``,
    or even the same object repeated. Strip think tags, then parse only the
    FIRST balanced top-level object, so trailing junk or repeats don't break the
    turn. Raises ValueError when no balanced object is present (caller falls
    back to a clarification prompt).
    """
    cleaned = _THINK_TAG.sub("", _THINK_BLOCK.sub("", text))
    obj = _first_json_object(cleaned)
    if obj is None:
        raise ValueError("no balanced JSON object in emission")
    return json.loads(obj)


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

    NORMAL = "normal"  # plain output — joined into the spoken reply
    WEB_SEARCH = "web_search"  # raw SearXNG payload — summarise inline, replan
    THINKING = "thinking"  # deep_think flagged — run the /think branch
    SUMMARY = "summary"  # summarize_thinking — surface captured partial
    REACTIVE = "reactive"  # tool error / HA 4xx — replan with failure shown
    ERROR = "error"  # dispatch returned None / raised — replan


# Prefix → kind, in match priority order.
_SENTINEL_KINDS = (
    (WEB_QUESTION_PREFIX, StepKind.WEB_SEARCH),
    (THINKING_PREFIX, StepKind.THINKING),
    (SUMMARY_PREFIX, StepKind.SUMMARY),
    (REACTIVE_PREFIX, StepKind.REACTIVE),
)

# Kinds that hand control back to the agent for another call.
_REPLAN_KINDS = frozenset(
    {
        StepKind.WEB_SEARCH,
        StepKind.THINKING,
        StepKind.SUMMARY,
        StepKind.REACTIVE,
        StepKind.ERROR,
    }
)


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


# Sentences in a reactive message that instruct the (absent) agent rather than
# inform the user — dropped when speaking the observation directly on the no-LLM
# tier, which has no SLM to act on them.
_AGENT_DIRECTIVE = re.compile(r"^(tell|ask|let|suggest|confirm|remind|prompt)\b", re.I)


def reactive_to_speech(text: str) -> str:
    """Turn a `Reactive question:` sentinel into a directly-speakable observation.

    The no-LLM tier dispatches a tool but has no SLM to replan with, so when a
    tool returns a reactive sentinel (e.g. HA "couldn't find that entity") we
    surface the underlying message to the user instead of the generic "no AI"
    phrase — the tool ran and produced a real result. Strips the prefix and any
    agent-directed instruction sentences ("Tell the user…", "Ask the user…").
    """
    msg = text.lstrip()
    if msg.startswith(REACTIVE_PREFIX):
        msg = msg[len(REACTIVE_PREFIX) :].strip()
    kept = [
        s.strip()
        for s in re.split(r"(?<=[.?!])\s+", msg)
        if s.strip() and not _AGENT_DIRECTIVE.match(s.strip())
    ]
    return " ".join(kept).strip() or "Sorry, I couldn't do that."


# Hard cap on agent re-calls per turn. 1 initial + up to 5 replans = 6.
# With grammar-cap-3 actions per emission, worst-case is 18 tool calls;
# typical multi-step turn is 1–3.
MAX_AGENT_CALLS_PER_TURN = 6

# Canonical name of the web-search tool. The orchestrator uses
# `is_web_search` to play its "searching the web" stall *before* dispatching
# this tool, since the SearXNG round-trip is the slow part of the turn.
WEB_SEARCH_TOOL = "external_information"
NOTE_WRITE_TOOLS = frozenset({"write_note", "append_to_note", "remember_fact"})

# Data-retrieval tools whose result is raw records — a state-change dump, a
# conversation transcript, fused note chunks — rather than a spoken-ready
# answer. Unlike an action tool ("turn on the lights" -> "Done") or a tool that
# pre-formats its own reply (the calendar/weather summaries), these return data
# the agent still has to *distill* into one answer. The loop reads an ordinary
# NORMAL result aloud verbatim, which for these means regurgitating the whole
# dump (e.g. "when did the lights last turn on" reading back 15 state changes);
# so it instead hands a lookup result back for one composing replan, letting the
# agent answer the actual question from the records. See `is_lookup` + the agent
# loop's dispatch step.
LOOKUP_TOOLS = frozenset(
    {
        "get_entity_history",
        "get_conversation_history",
        "search_notes",
    }
)


def describe_tools() -> str:
    """Human-readable tool listing for the agent system prompt."""
    return tool_registry.describe_tools()


def coerce_args(raw: Any) -> tuple:
    """Normalise an action's `args` into `(positional_list, kwargs_dict)`.

    The GBNF grammar guarantees a list of primitives on the local path, but a
    grammar-less remote model may emit args as a JSON object (kwargs-style, e.g.
    `{"query": "x"}`) or a bare scalar. Coerce so dispatch never assumes a list:
    a dict became `KeyError(0)` the moment anything indexed `args[0]`.
      - dict   -> ([], dict)      kwargs by name (matches the tool's params)
      - list   -> (list, {})      positional, as the grammar intends
      - None   -> ([], {})        no-arg call
      - scalar -> ([scalar], {})  a single bare positional value
    """
    if raw is None:
        return [], {}
    if isinstance(raw, dict):
        return [], dict(raw)
    if isinstance(raw, (list, tuple)):
        return list(raw), {}
    return [raw], {}


# Non-tool intents a grammar-less model uses to mean "speak this" when it wrongly
# bundles its answer inside `actions` instead of the {"reply": ...} envelope.
REPLY_PSEUDO_INTENTS = frozenset({"reply", "respond", "say", "answer", "speak"})


def coerce_reply_text(raw: Any) -> Optional[str]:
    """Extract the spoken text from a bundled `reply` pseudo-action's args.

    Shape-robust like `coerce_args` — the text may arrive as `["..."]`,
    `{"text": "..."}`, or a bare string. Returns the first non-empty string.
    """
    args, kwargs = coerce_args(raw)
    if args and isinstance(args[0], str) and args[0].strip():
        return args[0].strip()
    for v in kwargs.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def handle_action(action: Dict[str, Any]) -> Optional[str]:
    """Execute one action from an agent `actions` list.

    `action` is `{"intent": "<name>", "args": [...]}` (or `args` as a kwargs
    object from a grammar-less remote model — see `coerce_args`). Returns the
    tool's output string. On unknown tool, returns a `Reactive question:`
    sentinel so `should_replan` triggers and the next agent call sees the
    failure as an observation it can correct from. On any other dispatch
    exception returns `None` (also triggers replan via `should_replan`).
    """
    if not isinstance(action, dict):
        logger.error(f"Action is not a dict: {action!r}")
        return None
    name = action.get("intent")
    args, kwargs = coerce_args(action.get("args"))
    if not name:
        logger.error(f"Action missing intent: {action!r}")
        return None
    try:
        return tool_registry.execute_tool(name, args=args, kwargs=kwargs)
    except UnknownToolError as e:
        logger.warning(f"Agent picked unknown tool: {e}")
        return (
            f"Reactive question: The tool {name!r} is not available. "
            f"Use one of the tools listed in the system prompt."
        )
    except Exception as e:
        logger.exception(f"Error executing action {name}: {e}")
        return None


def is_registered_tool(name: Optional[str]) -> bool:
    """True if `name` (or one of its aliases) is a currently-loaded tool.

    A direct registry match, not a heuristic read of the observation: used by
    the agent loop's hallucinated-tool guard to block fabrication when the SLM
    invents a tool that was never in its prompt.
    """
    if not name:
        return False
    return tool_registry.canonical_name(name) is not None


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


def is_lookup(intent_name: Optional[str]) -> bool:
    """True if `intent_name` resolves to a data-retrieval tool whose raw result
    the agent should compose into a spoken answer (one replan) instead of having
    it read aloud verbatim. See `LOOKUP_TOOLS`."""
    if not intent_name:
        return False
    return tool_registry.canonical_name(intent_name) in LOOKUP_TOOLS


# A sentence claiming the assistant saved/logged something to notes. The SLM —
# especially a remote one without the GBNF grammar — sometimes tacks "I've saved
# this to your notes" onto a plain answer when no note-write tool actually ran.
_SAVE_VERB = (
    r"saved|added|noted|logged|recorded|wrote|written|jotted|stored|"
    r"made a note|making a note|taken a note|put (?:this|it|that) in"
)
_NOTE_NOUN = r"notes?|journal|diary|daily log|log|memo|to-?do list"
_SAVE_CLAIM_RE = re.compile(
    rf"(?:\b(?:{_SAVE_VERB})\b.*?\b(?:{_NOTE_NOUN})\b"
    rf"|\b(?:{_NOTE_NOUN})\b.*?\b(?:{_SAVE_VERB})\b)",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


def strip_unfounded_save_claim(text: str, note_written: bool) -> str:
    """Drop a confabulated 'I saved this to your notes' from a spoken reply.

    When `note_written` is False (no note-write tool ran this turn), remove any
    sentence asserting such a save so we never tell the user we did something we
    didn't. Returns the original text unchanged when a write did happen, when
    there's no claim, or when stripping would leave nothing.
    """
    if note_written or not text:
        return text
    sentences = _SENTENCE_RE.findall(text)
    kept = [s for s in sentences if not _SAVE_CLAIM_RE.search(s)]
    if len(kept) == len(sentences):
        return text  # no claim found
    result = "".join(kept).strip()
    if result != text:
        logger.warning("Stripped unfounded note-save claim from reply")
    return result or text  # never blank out the whole reply


def should_replan(step_result: Optional[object]) -> bool:
    """True if a step's result should trigger another agent call.

    Thin wrapper over `classify_step` for callers/tests that work with the raw
    string form (`None` or a sentinel-prefixed string). New code in the agent
    loop routes on `StepResult.kind` / `StepResult.should_replan` instead.
    """
    return classify_step(step_result).should_replan
