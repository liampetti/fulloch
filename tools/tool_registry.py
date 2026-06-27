"""Registry of callable tools exposed to the SLM via intent prompts."""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class UnknownToolError(KeyError):
    """Raised when execute_tool is called with a name the registry doesn't know.

    Distinct from generic errors so callers can fall back to chat mode when
    the SLM hallucinates a tool name.
    """


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

    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None,
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
        )

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
):
    """Decorator: register a function as a tool callable by the SLM."""

    def decorator(func):
        return tool_registry.register_tool(func, name, description, aliases)

    return decorator
