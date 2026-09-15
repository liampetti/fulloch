"""Shared conversation policy.

Callers own the assistant's turn lock for the entire foreground turn, including
generation and retries. These functions mutate the supplied list in place so a
model request holding that list sees recovery edits. They never acquire the
non-reentrant turn lock themselves.
"""

import json
import logging
import time

from .slm import ContextExhaustedError

logger = logging.getLogger(__name__)

# Full turn traces are kept during a turn; completed tool payloads retain only
# evidence of execution. Recovery preserves approximately the in-flight turn.
HISTORY_MAX_MESSAGES = 50
CONTEXT_TRIM_KEEP_MIN = 4
CHAT_SESSION_TIMEOUT_S = 1800.0
COMPACTED_TOOL_TRACE_MAX_CHARS = 160
CONTEXT_EXHAUSTED_REPLY = "I've lost some conversation context. Could you ask that again?"


def exhausted_reply(history: list) -> str:
    logger.warning(
        "SLM context exhausted; clearing %d history entries and apologising", len(history)
    )
    history.clear()
    return CONTEXT_EXHAUSTED_REPLY


def reset_session(history, satellites, timeout: float) -> None:
    most_recent = max((s.last_turn_end for s in satellites.values()), default=0.0)
    if most_recent and history and time.monotonic() - most_recent > timeout:
        logger.info("Session timeout (%.0fs) — clearing %d history entries", timeout, len(history))
        history.clear()
        for satellite in satellites.values():
            satellite.skip_followup_self_echo = False


def trim(history: list, maximum: int) -> None:
    if len(history) > maximum:
        del history[:-maximum]


def compact(history: list, trace_limit: int) -> None:
    """Retain short tool evidence, replies and prose; discard planning scaffolding."""
    kept = []
    for message in history:
        role = message.get("role")
        if role == "tool":
            content = message.get("content") or ""
            if len(content) > trace_limit:
                message = {**message, "content": content[:trace_limit].rstrip() + "…"}
        elif role == "assistant":
            try:
                emission = json.loads(message.get("content") or "")
            except (TypeError, ValueError):
                emission = None
            if isinstance(emission, dict) and "reply" not in emission and "actions" in emission:
                continue
        kept.append(message)
    history[:] = kept


def shed_oldest(history: list, keep_minimum: int) -> bool:
    """Halve over-floor history, advancing to a user boundary where possible."""
    count = len(history)
    if count <= keep_minimum:
        return False
    drop = max(2, (count - keep_minimum + 1) // 2)
    del history[:drop]
    while len(history) > keep_minimum and history[0].get("role") != "user":
        del history[0]
    logger.info(
        "Context overflow: shed %d oldest history entries, %d remain",
        count - len(history),
        len(history),
    )
    return True


def generate_with_recovery(
    model, *, conversation: list, generate, keep_minimum: int, background_active: bool, **kwargs
) -> str:
    """Retry under the caller-owned turn lock, retaining the current question.

    A foreground timeout must not restart the shared server under a durable
    background request. An oversized current request is retried once at the
    floor, then its ContextExhaustedError is allowed to reach the caller.
    """
    kwargs.setdefault("recover_on_failure", not background_active)
    history = conversation
    cleared_history = False
    while True:
        try:
            return generate(model, **kwargs)
        except ContextExhaustedError:
            if shed_oldest(history, keep_minimum):
                continue
            if cleared_history:
                raise
            logger.warning(
                "Context overflow persists at history floor; clearing history and retrying turn"
            )
            current_user = next(
                (message.copy() for message in reversed(history) if message.get("role") == "user"),
                None,
            )
            history.clear()
            if current_user is not None:
                history.append(current_user)
            cleared_history = True
