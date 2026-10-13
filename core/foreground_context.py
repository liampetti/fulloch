"""Conservative prompt specialization; ambiguous/action turns keep full routing.

This is an input-size policy, not an intent dispatcher. Tools still run through
the agent plan, availability checks and execution-evidence guards.
"""

import re

_SOCIAL_TURN = re.compile(
    r"\s*(?:hello|hi|hey|good (?:morning|afternoon|evening)|"
    r"how are you(?: doing)?|thanks(?: a lot)?|thank you(?: very much)?|"
    r"(?:please )?tell me a joke)\s*[.!?]*\s*", re.I,
)
_RESEARCH_TURN = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:research|investigate)\s+\S", re.I,
)
_CURRENT_DISCOVERY_TURN = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:find|look up|search for|check)\b"
    r"(?=.*\b(?:latest|recent|current|trending)\b)"
    r"(?=.*\b(?:info(?:rmation)?|models?|trends?|news|sources?)\b).+\Z",
    re.I | re.S,
)
_CONTEXT_REFERENCE = re.compile(
    r"\b(?:it|that|this|them|those|these|above|earlier|previous|same|again|"
    r"continue|instead|more|also)\b", re.I,
)
_MIXED_ACTION = re.compile(
    r"\b(?:save|write|append|remember|remind|schedule|book|buy|order|delete|"
    r"send|announce|play|pause|stop|resume|turn|switch|dim|set|lock|unlock|"
    r"open|close|cancel|tell|calendar|timer|todo)\b", re.I,
)


def is_standalone_research_request(user_prompt: str) -> bool:
    """Recognize an explicit research instruction without references or actions."""
    return bool(
        (_RESEARCH_TURN.search(user_prompt) or _CURRENT_DISCOVERY_TURN.search(user_prompt))
        and not _CONTEXT_REFERENCE.search(user_prompt)
        and not _MIXED_ACTION.search(user_prompt)
    )


def select_prompt_profile(user_prompt: str, history: list[dict], *, plan_active: bool = False) -> str:
    """Narrow only standalone social exchanges or explicit research handoffs.

    A confirmation, anaphoric request, ongoing plan, or latest unresolved
    question needs the full schema and its retained conversation context.
    """
    if plan_active:
        return "full"
    # The current user message has already been appended by AgentLoop.
    prior = history[:-1] if history and history[-1].get("role") == "user" else history
    if prior:
        latest = str(prior[-1].get("content") or "")
        if (prior[-1].get("role") == "tool" or "?" in latest
                or '"actions"' in latest
                or any(marker in latest.casefold() for marker in ("open task:", "pending task:"))):
            return "full"
    if _SOCIAL_TURN.fullmatch(user_prompt):
        return "conversation"
    if is_standalone_research_request(user_prompt):
        return "research"
    return "full"
