"""Thinking-mode marker tools.

`deep_think` flags a query for Qwen3 thinking mode; `summarize_thinking`
asks the assistant to summarise whatever it was thinking through when the
user last interrupted. Both follow the `search_web` pattern of returning a
sentinel-prefixed string that `Assistant._handle_wakeword` recognises and
routes accordingly — letting the intent SLM (or regex fast-path) pick them
without the chat path needing to introspect every utterance.
"""

from .tool_registry import tool

# Distinct from search_web's "User question:" marker so `_handle_wakeword`
# can route the two differently — search_web wants the spoken stall, deep
# thinking wants both the stall and the thinking-enabled chat prompt.
THINKING_PREFIX = "Thinking question:"
# Marker for the interrupt-and-summarise path. The assistant looks up the
# partial `<think>` content captured from the last cancelled thinking turn
# and runs a brief summary call against it.
SUMMARY_PREFIX = "Summary question:"


@tool(
    name="deep_think",
    description=(
        "Enter thinking mode for a complex query that benefits from "
        "deliberate reasoning, weighing tradeoffs, or careful analysis. "
        "Use when the user explicitly asks to think, consider, reflect, "
        "or weigh options. Do NOT use for quick lookups, simple chitchat, "
        "or any query a one-line answer covers."
    ),
    aliases=["think_about", "consider", "reflect", "ponder"],
)
def deep_think(query: str) -> str:
    """Tag a query so the chat path runs with Qwen3 thinking enabled."""
    return f"{THINKING_PREFIX}\n{query}"


@tool(
    name="summarize_thinking",
    description=(
        "Summarise where the assistant was leaning when the user last "
        "interrupted a thinking-mode response. Use when the user asks "
        "'what do you have so far', 'summarise your thoughts', or "
        "'tell me what you were thinking'. Has no effect outside that "
        "specific interrupt-mid-thinking flow."
    ),
    aliases=["summarise_thinking", "thoughts_so_far"],
)
def summarize_thinking() -> str:
    """Tag a request so the assistant summarises its last partial thoughts."""
    return SUMMARY_PREFIX
