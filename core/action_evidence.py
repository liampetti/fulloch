"""Cheap, deliberately conservative guard for model-written completion claims.

This is not a semantic verifier. Never authorize prose merely because *some*
tool ran: operation results are rendered verbatim instead, preserving target
and outcome, including plain-string failures that classify as NORMAL.
"""

import re
from dataclasses import dataclass, field

from utils.intents import StepKind, reactive_to_speech

OPERATION_TOOLS = frozenset({
    "turn_on", "turn_off", "toggle", "ha_set_brightness", "ha_set_color",
    "ha_service", "ha_set_climate", "ha_lock", "ha_unlock", "ha_open_cover",
    "ha_close_cover", "ha_stop_cover", "ha_set_cover_position", "ha_set_fan_speed",
    "ha_vacuum", "ha_run_script", "ha_activate_scene", "ha_volume_set",
    "ha_volume_up", "ha_volume_down", "ha_select_source", "ha_mute", "pause",
    "resume", "skip", "previous", "play_song", "start_countdown", "cancel_timer",
    "extend_timer", "create_calendar_event", "add_todo_item", "complete_todo_item",
    "write_note", "append_to_note", "append_to_today", "remember_fact",
    "send_satellite_message", "insert_at_obsidian_cursor", "rename_active_obsidian_note",
    "delete_active_obsidian_note", "replace_selected_obsidian_text",
})

_PAST = (
    r"turned|switched|opened|closed|locked|unlocked|set|adjusted|dimmed|brightened|"
    r"activated|deactivated|started|stopped|paused|resumed|skipped|played|saved|"
    r"added|deleted|removed|updated|created|sent|scheduled|cancelled|canceled|"
    r"extended|muted|unmuted|toggled|renamed"
)
_CLAIM = re.compile(
    rf"\b(?:I|we)\s+(?:(?:have|'ve|’ve)\s+)?(?:just\s+|already\s+|successfully\s+)*"
    rf"(?:{_PAST})\b|\b(?:I|we)['’]ve\s+(?:just\s+|already\s+)?(?:{_PAST})\b|"
    rf"\b(?:has|have)\s+been\s+(?:successfully\s+)?(?:{_PAST})\b|"
    rf"^(?:done|all done|all set|taken care of)(?:[!.,:]|\s*$)|"
    rf"^(?:done[!,.:]\s*)?(?:{_PAST})\b|"
    r"\b(?:lamp|light|lights|blind|blinds|cover|door|lock|timer|music)\b"
    r"[^.!?\n]{0,60}\b(?:is|are)\s+(?:now|already)\s+"
    r"(?:on|off|open|closed|locked|unlocked|set|playing|paused)\b",
    re.I,
)


def claims_execution(text: str) -> bool:
    """Detect affirmative completion wording, not requests or future promises."""
    return bool(_CLAIM.search(text.strip()))


FALLBACK = "I couldn't confirm that the requested action was completed."
REPAIR = (
    "The proposed reply was withheld: it claims an action without a matching "
    "recorded execution outcome. Previous conversation text and unrelated or failed "
    "tool calls are not proof. If the user requested an action that has not run, "
    "call the appropriate tool. Do not repeat an already completed operation. "
    "Otherwise explain that it was not completed, or answer without claiming execution."
)


@dataclass
class ActionEvidence:
    # Actual dispatch records only, scoped to one run, never reconstructed from history.
    records: list = field(default_factory=list)
    repairs: int = 0

    def record(self, action, step) -> bool:
        if action.get("intent") in OPERATION_TOOLS:
            self.records.append((dict(action), step))
            return True
        return False

    def response(self) -> str:
        parts = []
        for action, step in self.records:
            if step.kind is StepKind.REACTIVE:
                text = reactive_to_speech(step.text)
            elif step.kind is StepKind.ERROR or not step.in_output:
                text = f"I couldn't confirm the result of {action['intent'].replace('_', ' ')}."
            else:
                text = step.text.strip()
            if text and text not in parts:
                parts.append(text)
        return " ".join(parts) or FALLBACK
