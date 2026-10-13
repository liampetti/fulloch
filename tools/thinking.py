"""Deliberate-work tool for bounded, background research and planning."""

from core.satellite_context import current_satellite_id, get_current_assistant

from .thinking_context import get_handoff
from .tool_registry import ThinkingResult, tool


@tool(
    name="deep_think",
    description=(
        "Queue a bounded background investigation for substantial research, "
        "complex planning, comparing options, or reviewing current information. "
        "It can use approved read-only tools and will report back when ready. "
        "Do not use for a quick lookup, small talk, or a simple factual answer."
    ),
    aliases=["research", "investigate", "plan_complex_task"],
)
def deep_think(query: str) -> str:
    """Queue work and return the concise acknowledgement for this turn."""
    assistant = get_current_assistant()
    if assistant is None:
        return ThinkingResult("Deliberate thinking is not ready yet.", status="unavailable",
                              scope="The background research service is unavailable.")
    try:
        handoff = get_handoff()
        job = assistant.run_thinking_task(
            query,
            origin_satellite_id=current_satellite_id.get(),
            origin_source="conversation",
            **({"handoff": handoff} if handoff is not None else {}),
        )
    except RuntimeError as exc:
        return ThinkingResult(str(exc), status="unavailable",
                              scope="The background research job could not be queued.")
    active = assistant.active_thinking_task()
    if job.get("status") == "QUEUED" and active and active.get("id") != job.get("id"):
        return "I'm still working on an earlier task. I've queued this one next."
    return "I'll look into that and let you know when I'm done."
