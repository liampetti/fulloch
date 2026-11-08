"""Tools for sending spoken announcements to connected voice satellites."""

import contextvars
import re
import threading

from core.satellite_context import get_current_assistant
from utils.response_language import response

from .tool_registry import tool

_announcement_slots = threading.BoundedSemaphore(2)


def _normalise_name(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower()))


def _matching_satellites(assistant, target: str) -> list[tuple[str, str]]:
    """Return connected, speakable satellites whose room/device name matches."""
    target_name = _normalise_name(target)
    target_words = set(target_name.split())
    matches = []
    for satellite_id, session in assistant.satellites.items():
        if satellite_id == "dashboard-text" or session.tts_sink is None:
            continue
        names = (session.label, session.ha_area_name, session.ha_area)
        for name in names:
            if not name:
                continue
            candidate = _normalise_name(name)
            candidate_words = set(candidate.split())
            if target_name == candidate or (target_words and target_words <= candidate_words):
                matches.append((satellite_id, name))
                break
    return matches


@tool(
    name="send_satellite_message",
    description=(
        "Speak a short announcement through a connected voice satellite. Use for "
        "requests such as 'tell downstairs that dinner is ready'. target is the "
        "connected satellite's room or device name; message is concise spoken delivery. "
        "Preserve exact wording only when the user asks to quote or send it verbatim."
    ),
    aliases=["announce_to_satellite", "tell_satellite", "message_satellite"],
)
def send_satellite_message(target: str, message: str) -> str:
    """Queue a spoken message for one named, connected voice satellite."""
    assistant = get_current_assistant()
    if assistant is None:
        return "Reactive question: " + response("satellite_unavailable")
    matches = _matching_satellites(assistant, target)
    if not matches:
        return "Reactive question: " + response("satellite_target_not_found", target=repr(target))
    if len(matches) > 1:
        names = ", ".join(name for _satellite_id, name in matches)
        return "Reactive question: " + response("satellite_target_ambiguous", target=repr(target), names=names) + " Ask which one to use."

    satellite_id, name = matches[0]
    text = message.strip()
    if not text:
        return "Reactive question: " + response("satellite_message_empty")
    if not _announcement_slots.acquire(blocking=False):
        return "Reactive question: " + response("satellite_announcement_busy")

    def _deliver() -> None:
        try:
            assistant.speak_proactive(text=text, satellite_id=satellite_id)
        finally:
            _announcement_slots.release()

    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(_deliver,), daemon=True, name="satellite-message").start()
    return response("satellite_announcement_queued", target=name)
