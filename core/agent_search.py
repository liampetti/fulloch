"""Invocation-local search summaries; no dispatch or conversation-history ownership."""

import re

from utils.intents import StepKind, StepResult

PROMPT_STRIP_CHARS = " ,.!?;:"
_SUMMARY_REFUSAL = re.compile(
    r"^(?:i(?:'m|’m| am) (?:not able|unable) to answer|"
    r"i (?:can't|cannot|couldn't|could not) (?:answer|summari[sz]e))\b",
    re.IGNORECASE,
)


def normalise_search_query(args) -> str | None:
    """Deduplicate positional, keyword, and bare queries, including default news."""
    if isinstance(args, dict):
        query = next(iter(args.values()), None)
    elif isinstance(args, (list, tuple)):
        query = args[0] if args else None
    elif isinstance(args, str):
        query = args
    else:
        query = None
    if query is None:
        return "__default__"
    if not isinstance(query, str):
        return None
    query = re.sub(r"\s+", " ", query).strip().lower().strip(PROMPT_STRIP_CHARS)
    return query or "__default__"


def last_user_question(history: list) -> str | None:
    """Most recent non-empty user message for a topic-less search follow-up."""
    for message in reversed(history):
        if message.get("role") == "user" and (content := message.get("content", "").strip()):
            return content
    return None


class TurnSearch:
    """Own cache equivalence and grounding for exactly one run invocation.

    Summary generation is explicit and separate from accepting its result, so the
    loop can check cancellation before committing it to the cache and history.
    A cache hit still needs the same composing replan as a fresh summary.
    """

    def __init__(self):
        self._cache: dict[str, StepResult] = {}
        self.latest: str | None = None
        self.summaries: list[str] = []

    def cached(self, query: str | None) -> StepResult | None:
        step = self._cache.get(query) if query is not None else None
        if step is None:
            return None
        self.latest = step.text
        return step

    def summarise(
        self, payload: str, *, summariser, watchdog, clips, play_chunks,
        session, sink, tts_active_event, cancel_check, stats,
    ) -> str:
        """Use bounded search-specific progress around the injected model call."""
        with watchdog(
            clips, play_chunks, session, sink=sink,
            tts_active_event=tts_active_event, max_stalls=2,
        ):
            summary = summariser(payload, cancel_check, stats=stats)
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("The search summariser returned no usable text")
            return summary

    def accept(self, query: str | None, summary: str, original: StepResult) -> StepResult:
        self.latest = summary
        outcome = original.outcome
        if _SUMMARY_REFUSAL.match(summary.strip()) and (outcome is None or outcome.get("status") == "evidence"):
            outcome = {"evidence": {}, "next_actions": [], "artifact": original.artifact,
                       "recovery": {}, **(outcome or {}), "status": "rejected",
                       "scope": "The search summary could not answer from the retrieved sources."}
        successful = outcome is None or outcome.get("status") == "evidence"
        if successful and summary not in self.summaries:
            self.summaries.append(summary)
        step = StepResult(StepKind.NORMAL, summary, in_output=True,
                          artifact=original.artifact, outcome=outcome,
                          raw_text=original.raw_text if original.raw_text is not None else str(original.text))
        # A failed retrieval must remain retryable, including with the same query.
        if query is not None and successful:
            self._cache[query] = step
        return step

    def grounded_reply(self) -> str | None:
        return self.latest.strip() if self.latest and self.latest.strip() else None

    def output_parts(self, results: list[str]) -> list[str]:
        """Surface findings before follow-up tool output without repeating them."""
        parts = [s.strip() for s in results if s and s.strip()]
        if self.latest:
            summary = self.latest.strip().rstrip(".")
            if summary and not any(p.rstrip(".") == summary for p in parts):
                parts.insert(0, summary)
        return parts
