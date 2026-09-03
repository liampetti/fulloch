"""Capability metadata and adapters for background thinking workers.

Native tools remain registered and dispatched through ``tool_registry``. This
module only adds the policy metadata needed to expose a deliberately restricted
subset to a worker or future MCP integration.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

from .tool_registry import tool_registry

AccessClass = Literal["read", "propose", "execute"]
CapabilitySource = Literal["native", "mcp"]


def _format_result(result: str) -> str:
    return result


@dataclass(frozen=True)
class ToolCapability:
    """One policy-controlled tool exposed to a thinking worker."""

    name: str
    invoke: Callable[[list[Any], dict[str, Any]], str]
    source: CapabilitySource
    timeout_seconds: float
    format_result: Callable[[str], str]
    access_class: AccessClass


# Unlisted tools intentionally fall back to ``execute``. That conservative
# default prevents a new native mutation tool from silently reaching workers.
_NATIVE_ACCESS: dict[str, AccessClass] = {
    "calculate": "read",
    "convert_units": "read",
    "days_between": "read",
    "external_information": "read",
    "find_calendar_event": "read",
    "get_conversation_history": "read",
    "get_entity_history": "read",
    "get_entity_state": "read",
    "get_energy_overview": "read",
    "get_home_overview": "read",
    "get_security_overview": "read",
    "get_todo_items": "read",
    "get_weather_forecast": "read",
    "list_entities_in_area": "read",
    "read_note": "read",
    "search_notes": "read",
    "search_papers": "read",
    "get_paper_detail": "read",
    "search_flights": "read",
    "search_hotels": "read",
    "whats_on": "read",
    "deep_think": "propose",
    "assess_itinerary": "read",
    "plan_travel": "read",
}

def native_access_class(name: str) -> AccessClass:
    """Return the policy access class for a registered native tool."""
    canonical = tool_registry.canonical_name(name)
    return _NATIVE_ACCESS.get(canonical or name, "execute")


def native_requires_deep_think(name: str) -> bool:
    """Whether a foreground call must be delegated to the planning worker."""
    return tool_registry.is_deep_think_only(name)


def _native_invoke(name: str) -> Callable[[list[Any], dict[str, Any]], str]:
    def invoke(args: list[Any], kwargs: dict[str, Any]) -> str:
        return tool_registry.execute_tool(name, args=args, kwargs=kwargs)

    return invoke


def native_capabilities(names: list[str] | None = None) -> dict[str, ToolCapability]:
    """Return currently available native tools as policy-aware adapters."""
    selected = names or list(tool_registry._schemas)
    capabilities = {}
    for name in selected:
        canonical = tool_registry.canonical_name(name)
        if canonical is None or not tool_registry.is_available(canonical):
            continue
        capabilities[canonical] = ToolCapability(
            name=canonical,
            invoke=_native_invoke(canonical),
            source="native",
            timeout_seconds=30.0,
            format_result=_format_result,
            access_class=native_access_class(canonical),
        )
    return capabilities
