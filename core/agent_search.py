"""Invocation-local search summaries; no dispatch or conversation-history ownership."""

import re

from utils.intents import StepKind, StepResult

PROMPT_STRIP_CHARS = " ,.!?;:"


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
        self._cache: dict[str, str] = {}
        self.latest: str | None = None

    def cached(self, query: str | None) -> StepResult | None:
        summary = self._cache.get(query) if query is not None else None
        if summary is None:
            return None
        self.latest = summary
        return StepResult(StepKind.NORMAL, summary, in_output=True)

    def summarise(
        self, payload: str, *, summariser, watchdog, clips, play_chunks,
        session, sink, tts_active_event, cancel_check, stats,
    ) -> str:
        """Use bounded search-specific progress around the injected model call."""
        with watchdog(
            clips, play_chunks, session, sink=sink,
            tts_active_event=tts_active_event, max_stalls=2,
        ):
            return summariser(payload, cancel_check, stats=stats)

    def accept(self, query: str | None, summary: str, original: StepResult) -> StepResult:
        self.latest = summary
        if query is not None:
            self._cache[query] = summary
        return StepResult(StepKind.NORMAL, summary, in_output=True, artifact=original.artifact)

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
