"""Deliberate-work tool for bounded, background research and planning."""

from core.satellite_context import current_satellite_id, get_current_assistant

from .tool_registry import tool


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
        return "Deliberate thinking is not ready yet."
    try:
        job = assistant.run_thinking_task(
            query,
            origin_satellite_id=current_satellite_id.get(),
            origin_source="conversation",
        )
    except RuntimeError as exc:
        return str(exc)
    if job.get("status") == "QUEUED" and assistant.active_thinking_task() != job:
        return "I'm still working on an earlier task. I've queued this one next."
    return "I'll look into that and let you know when I'm done."
