"""Tests for announcements sent to named connected voice satellites."""

from unittest.mock import MagicMock, patch

from core.satellite import SatelliteSession
from core.satellite_context import set_current_assistant


def _assistant_with_satellites():
    assistant = MagicMock()
    assistant.satellites = {
        "kitchen": SatelliteSession(id="kitchen", label="Kitchen", tts_sink=MagicMock()),
        "downstairs": SatelliteSession(
            id="downstairs", ha_area="downstairs", ha_area_name="Downstairs", tts_sink=MagicMock()
        ),
        "dashboard-text": SatelliteSession(id="dashboard-text"),
    }
    return assistant


class TestSendSatelliteMessage:
    def teardown_method(self):
        set_current_assistant(None)

    def test_queues_announcement_for_matching_browser_area(self):
        from tools.satellite_messages import send_satellite_message

        assistant = _assistant_with_satellites()
        set_current_assistant(assistant)

        with patch("tools.satellite_messages.threading.Thread") as thread:
            result = send_satellite_message("downstairs", "Dinner is ready")

        assert result == "Announcement queued for Downstairs."
        thread.assert_called_once_with(
            target=assistant.speak_proactive,
            kwargs={"text": "Dinner is ready", "satellite_id": "downstairs"},
            daemon=True,
            name="satellite-message",
        )
        thread.return_value.start.assert_called_once_with()

    def test_returns_reactive_question_when_target_is_not_connected(self):
        from tools.satellite_messages import send_satellite_message

        set_current_assistant(_assistant_with_satellites())

        assert send_satellite_message("bedroom", "Dinner is ready") == (
            "Reactive question: No connected voice satellite is named 'bedroom'."
        )

    def test_requires_an_unambiguous_target(self):
        from tools.satellite_messages import send_satellite_message

        assistant = _assistant_with_satellites()
        assistant.satellites["downstairs-speaker"] = SatelliteSession(
            id="downstairs-speaker", label="Downstairs Speaker", tts_sink=MagicMock()
        )
        set_current_assistant(assistant)

        assert send_satellite_message("downstairs", "Dinner is ready").startswith(
            "Reactive question: More than one connected satellite matches"
        )
