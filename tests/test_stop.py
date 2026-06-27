"""Dashboard-driven stop: `Assistant.request_stop` aborts the in-flight turn
(voice or text) silently and stands down without a follow-up window, and the
`/stop` endpoint wires the button to it. `get_state` reports work even outside
barge-in mode so the button knows when to appear.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.tts_session import TtsSession
from server.dashboard import create_app


def _make_assistant(**kwargs):
    """Bare Assistant with AudioCapture mocked out."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus", **kwargs)
    # MagicMock tts_active would read truthy; force a real idle reading.
    a.audio_capture.tts_active.is_set.return_value = False
    return a


class TestRequestStop:
    def test_stops_active_session_and_stands_down(self):
        a = _make_assistant()
        events = []
        a.register_turn_listener(events.append)
        sess = TtsSession()
        a._active_session = sess
        a._last_turn_end = 123.0

        a.request_stop()

        assert sess.cancelled is True  # SLM/TTS told to abort
        assert a._last_turn_end == 0.0  # no follow-up window
        a.audio_capture.clear_follow_up.assert_called()
        assert any(e.get("role") == "stopped" for e in events)

    def test_idle_is_safe_noop(self):
        a = _make_assistant()
        a._active_session = None
        a.request_stop()  # must not raise

    def test_barge_in_flushes_and_drops(self):
        a = _make_assistant()
        a._turn_active = True
        a._active_session = TtsSession()
        a.request_stop()
        a.audio_capture.flush.assert_called()
        assert a._drop_results_until > 0


class TestGetStateBusy:
    def test_active_session_reads_as_thinking(self):
        a = _make_assistant()
        a._turn_active = False
        a._active_session = None
        assert a.get_state() == "idle"
        # A half-duplex / text turn sets only _active_session (no _turn_active).
        a._active_session = TtsSession()
        assert a.get_state() == "thinking"


class TestStopEndpoint:
    def test_post_stop_calls_request_stop(self, monkeypatch):
        monkeypatch.delenv("FULLOCH_DASHBOARD_TOKEN", raising=False)
        a = MagicMock()
        a.register_turn_listener = MagicMock()
        client = TestClient(create_app(a))
        r = client.post("/stop")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        a.request_stop.assert_called_once()
