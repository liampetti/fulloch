"""Foreground routing of completed-report and startup-greeting follow-ups."""

import re
from dataclasses import dataclass

_REPORT_FOLLOW_UP_RE = re.compile(
    r"\b(?:report|summary|findings|conclusion|recommendation|what did (?:it|the report) say|"
    r"does (?:it|the report) (?:say|show)|is (?:it|that) (?:feasible|possible|worth it|safe)|"
    r"(?:check|read|explain|clarify) (?:that|it|again)|more (?:detail|details))\b", re.I,
)
_REPORT_BYPASS_RE = re.compile(
    r"\b(?:new question|something else|search again|look up|research again|use other sources)\b",
    re.I,
)
_STARTUP_GREETING_FOLLOW_UP_RE = re.compile(
    r"\s*(?:(?:can|could|would)\s+you\s+)?(?:"
    r"(?:tell|give|share)\s+me\s+(?:more|more\s+(?:information|details))\s+(?:about|on)\s+|"
    r"(?:explain|expand|elaborate)\s+(?:(?:more\s+)?(?:about|on)\s+)?)"
    r"(?:(?:that|it)(?:\s+(?:topic|fact))?|the\s+(?:topic|fact))"
    r"\s*(?:please)?[.!?]*\s*\Z", re.I,
)


def startup_greeting_follow_up(response: str | None, user_prompt: str) -> str | None:
    """Select referential context; the caller consumes the one-shot greeting."""
    return response if response and _STARTUP_GREETING_FOLLOW_UP_RE.fullmatch(user_prompt) else None


@dataclass(frozen=True)
class ReportRoute:
    caught: dict | None = None
    reply: str | None = None


def route_report_follow_up(
    user_prompt: str, *, satellite_id, cancel_check, stats,
    consume_report, answer_report, active_task, catch_intent,
) -> ReportRoute:
    """Consume first, then prefer deterministic commands over report questions.

    Services are explicit callbacks; persistence and grounded model reading stay
    with the report owner. No history is added for these early-return routes.
    """
    affirmation = re.fullmatch(
        r"\s*(?:yes|yeah|yep|go ahead|please do)\s*[.!]?\s*", user_prompt, re.I,
    )
    report_request = re.fullmatch(
        r"\s*(?:(?:yes|yeah|yep|go ahead|please do)[,!]?\s*)?"
        r"(?:(?:give|read|tell)\s+me\s+)?(?:(?:a|the)\s+)?"
        r"(?:(?:short|full)\s+)?(?:summary|report)(?:\s+please)?[.!]?\s*",
        user_prompt, re.I,
    )
    if (affirmation or report_request) and consume_report is not None:
        if report := consume_report(satellite_id):
            return ReportRoute(reply=report)
    caught = catch_intent(user_prompt)
    caught = caught if isinstance(caught, dict) else None
    if (
        caught is None and _REPORT_FOLLOW_UP_RE.search(user_prompt)
        and not _REPORT_BYPASS_RE.search(user_prompt) and answer_report is not None
    ):
        if answer := answer_report(satellite_id, user_prompt, cancel_check, stats):
            return ReportRoute(reply=answer)
    job = active_task() if active_task is not None else None
    if (
        job is not None and job.get("status") in {"QUEUED", "RUNNING", "PAUSED"}
        and (affirmation or report_request)
    ):
        return ReportRoute(
            caught, "I'm still working on that. I'll let you know as soon as the report is ready.",
        )
    return ReportRoute(caught)
