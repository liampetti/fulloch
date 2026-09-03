"""Registry of callable tools exposed to the SLM via intent prompts."""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tools describe what happened in their own bounded operation. The controller,
# not a tool, decides whether the overall investigation is exhausted.
THINKING_OUTCOME_STATUSES = frozenset(
    {"evidence", "rejected", "needs_input", "failed", "unavailable"}
)


class UnknownToolError(KeyError):
    """Raised when execute_tool is called with a name the registry doesn't know.

    Distinct from generic errors so callers can fall back to chat mode when
    the SLM hallucinates a tool name.
    """


class ArtifactText(str):
    """A normal tool observation with an optional, UI-only structured artifact."""

    def __new__(cls, text: str, artifact: Optional[dict] = None):
        value = super().__new__(cls, text)
        value.artifact = artifact
        return value


class ThinkingResult(ArtifactText):
    """Standard evidence envelope for a deep-think-capable tool.

    It remains a string so existing dispatch stays compatible, while the
    background controller reads typed metadata instead of reverse-engineering
    prose. Every result has a bounded-operation ``status`` and non-empty
    ``scope``; only the controller may determine investigation exhaustion.
    """

    def __new__(
        cls,
        text: str,
        *,
        status: str = "evidence",
        evidence: Optional[dict] = None,
        scope: str = "",
        next_actions: tuple[str, ...] = (),
        artifact: Optional[dict] = None,
    ):
        value = super().__new__(cls, text, artifact=artifact)
        value.thinking_status = status
        value.evidence = {} if evidence is None else evidence
        value.scope = scope
        value.next_actions = next_actions
        return value


def thinking_result_error(result: ThinkingResult) -> str | None:
    """Return the contract violation for a typed outcome, if any.

    Validation lives beside the public envelope so every capability source can
    share it. The controller turns an invalid result into a safe failure rather
    than allowing untrusted tool metadata into its evidence ledger.
    """
    if result.thinking_status not in THINKING_OUTCOME_STATUSES:
        return f"unknown status {result.thinking_status!r}"
    if not isinstance(result.scope, str) or not result.scope.strip():
        return "scope must be a non-empty string"
    if not isinstance(result.evidence, dict):
        return "evidence must be a mapping"
    if not isinstance(result.next_actions, tuple) or not all(
        isinstance(action, str) and action for action in result.next_actions
    ):
        return "next_actions must be a tuple of non-empty names"
    if result.artifact is not None and not isinstance(result.artifact, dict):
        return "artifact must be a mapping or None"
    return None


@dataclass
class Param:
    name: str
    required: bool
    default: Any = None


@dataclass
class ToolSchema:
    name: str
    description: str
    params: List[Param]
    deep_think_only: bool = False
    thinking_outcome: bool = False


def _extract_params(func: Callable) -> List[Param]:
    params = []
    for name, p in inspect.signature(func).parameters.items():
        if name == "self":
            continue
        required = p.default is inspect.Parameter.empty
        params.append(
            Param(
                name=name,
                required=required,
                default=None if required else p.default,
            )
        )
    return params


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._aliases: Dict[str, str] = {}
        self._availability: Dict[str, Callable[[], bool]] = {}

    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        available: Optional[Callable[[], bool]] = None,
        deep_think_only: bool = False,
        thinking_outcome: bool = False,
    ) -> Callable:
        tool_name = name or func.__name__

        if tool_name in self._tools or tool_name in self._aliases:
            # Two integrations registering the same name (e.g. lighting and
            # home_assistant both wanting `turn_on`) — first one wins so the
            # assistant still starts. User can disambiguate via config.
            logger.warning(
                "Tool name collision: '%s' already registered; skipping new registration",
                tool_name,
            )
            return func

        self._tools[tool_name] = func
        self._schemas[tool_name] = ToolSchema(
            name=tool_name,
            description=description or func.__doc__ or f"Function {tool_name}",
            params=_extract_params(func),
            deep_think_only=deep_think_only,
            thinking_outcome=thinking_outcome,
        )
        if available is not None:
            self._availability[tool_name] = available

        for alias in aliases or []:
            if alias in self._aliases or alias in self._tools:
                # Existing alias/name collision is a real bug — log loudly but
                # don't raise: tool modules import at startup and one bad
                # decorator shouldn't break the whole assistant.
                logger.warning(
                    "Alias collision: '%s' already maps to '%s'; ignoring on '%s'",
                    alias,
                    self._aliases.get(alias, alias),
                    tool_name,
                )
                continue
            self._aliases[alias] = tool_name

        logger.info(f"Registered tool: {tool_name}")
        return func

    def get_tool(self, name: str) -> Optional[Callable]:
        if name in self._tools:
            return self._tools[name]
        if name in self._aliases:
            return self._tools[self._aliases[name]]
        return None

    def canonical_name(self, name: str) -> Optional[str]:
        """Resolve a name or alias to its canonical registered tool name.

        Returns None when the name isn't a known tool or alias. Lets callers
        recognise a tool the agent invoked under any of its aliases without
        re-implementing the alias lookup.
        """
        if name in self._tools:
            return name
        return self._aliases.get(name)

    def is_available(self, name: str) -> bool:
        """Whether a registered tool is enabled for agent exposure."""
        canonical = self.canonical_name(name)
        if canonical is None:
            return False
        available = self._availability.get(canonical)
        return available is None or available()

    def is_deep_think_only(self, name: str) -> bool:
        """Whether foreground calls to this tool must use the planning worker."""
        canonical = self.canonical_name(name)
        return bool(canonical and self._schemas[canonical].deep_think_only)

    def execute_tool(
        self,
        name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        func = self.get_tool(name)
        if not func:
            raise UnknownToolError(name)
        # Let tool exceptions propagate to the caller. `handle_action` maps them
        # to None -> StepKind.ERROR -> replan, so a tool that raises (e.g. the
        # agent omitted a required arg) hands control back to the agent instead
        # of being swallowed into "" — which would surface as an empty
        # observation and a misleading "Done." spoken reply.
        result = func(*(args or []), **(kwargs or {}))
        return result if result is not None else ""

    def describe_tools(self) -> str:
        """Render the registered tools as a human-readable block for the intent prompt.

        Each line is `- name: description` followed by the argument signature
        derived from the function (`Args: a, b (optional, default: ...)`), so
        the agent sees exact arg names/order without each tool author having to
        restate them in the description.
        """
        lines = []
        for schema in self._schemas.values():
            available = self._availability.get(schema.name)
            if available is not None and not available():
                continue
            parts = []
            for p in schema.params:
                if p.required:
                    parts.append(p.name)
                else:
                    parts.append(f"{p.name} (optional, default: {p.default})")
            args = f" Args: {', '.join(parts)}." if parts else ""
            lines.append(f"- {schema.name}: {schema.description}{args}")
        return "\n".join(lines)


tool_registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    available: Optional[Callable[[], bool]] = None,
    deep_think_only: bool = False,
    thinking_outcome: bool = False,
):
    """Decorator: register a function as a tool callable by the SLM."""

    def decorator(func):
        return tool_registry.register_tool(
            func, name, description, aliases, available, deep_think_only, thinking_outcome
        )

    return decorator
