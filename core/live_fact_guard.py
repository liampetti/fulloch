"""High-confidence live questions that must not be answered from model guesses.

This is a reply guard, not a fast-path dispatcher: the agent still chooses and
calls its tools. Unrecognized questions retain the normal agent policy.
"""

import re
from dataclasses import dataclass

from utils.intent_catch import (
    _OUTPUT_LANGUAGE_RE,
    _english_clock_emission,
    requested_output_language,
)


@dataclass(frozen=True)
class LiveFactRequirement:
    tools: tuple[str, ...]
    guidance: str
    clock_intent: str | None = None


_NEWS = re.compile(r"\b(?:news|headlines|nachrichten|schlagzeilen|actualités|nouvelles)\b", re.I)
_CURRENT = re.compile(
    r"\b(?:today|tonight|now|current|currently|latest|recent|this morning|this evening|"
    r"heute|jetzt|aktuell(?:e[nmrs]?)?|neueste[nmrs]?|aujourd['’]hui|maintenant|récent(?:e|es|s)?)\b", re.I,
)
_QUESTION = re.compile(
    r"\b(?:what|which|how|tell|show|give|find|check|search|summari[sz]e|look up|"
    r"was|welche[nmrs]?|wie|zeig|zeige|quel(?:le)?s?|quoi|donne|montre)\b", re.I,
)
_WEATHER = re.compile(r"\b(?:weather|forecast|wetter|wettervorhersage|météo)\b", re.I)
_MATH_REQUEST = re.compile(r"\b(?:what(?:'s| is)|how much is|calculate|compute|evaluate)\b", re.I)
_ARITHMETIC = re.compile(
    r"(?<![\w./:-])\d+(?:\.\d+)?\s*(?:\+|\*|/|\s-\s|\bplus\b|\bminus\b|\btimes\b|\bdivided by\b)"
    r"\s*\d+(?:\.\d+)?(?![\w./:-])", re.I,
)


def live_fact_requirement(prompt: str, matcher=None) -> LiveFactRequirement | None:
    """Identify narrow, explicit live requests without guessing a tool action."""
    clock_prompt = prompt
    if requested_output_language(prompt):
        match = _OUTPUT_LANGUAGE_RE.search(prompt)
        if match and not prompt[match.end():].strip(" \t\r\n.,?!;:"):
            clock_prompt = prompt[:match.start()].rstrip(" \t\r\n.,?!;:")
    clock = _english_clock_emission(clock_prompt)
    if clock is None and matcher is not None:
        from utils.hassil_fastpath import MatchMetadata

        try:
            candidate = matcher.match(clock_prompt)
            metadata = getattr(candidate, "metadata", None)
            if (isinstance(metadata, MatchMetadata)
                    and metadata.intent in {"HassGetCurrentTime", "HassGetCurrentDate"}
                    and candidate.get("actions") == [{"intent": "get_current_time", "args": []}]):
                clock = candidate
        except Exception:  # noqa: BLE001 — a matcher failure must not break the agent turn
            pass
    if clock is not None:
        return LiveFactRequirement(
            ("get_current_time",),
            "Use get_current_time for the current local clock or date before answering.",
            clock.metadata.intent,
        )
    if _QUESTION.search(prompt) and _NEWS.search(prompt) and _CURRENT.search(prompt):
        return LiveFactRequirement(
            ("external_information",),
            "Use external_information to retrieve current news. Preserve the requested place, time and output language.",
        )
    if _QUESTION.search(prompt) and _WEATHER.search(prompt) and _CURRENT.search(prompt):
        return LiveFactRequirement(
            ("get_weather_forecast", "external_information"),
            "Use an appropriate available weather or live-search tool. Preserve the requested place, time and output language.",
        )
    if _MATH_REQUEST.search(prompt) and _ARITHMETIC.search(prompt):
        return LiveFactRequirement(
            ("calculate",), "Use calculate to obtain the result before answering; preserve the requested output language.",
        )
    return None
