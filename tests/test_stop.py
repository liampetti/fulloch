"""Dashboard-driven stop: `Assistant.request_stop` aborts the in-flight turn
(voice or text) silently and stands down without a follow-up window, and the
`/stop` endpoint wires the button to it. `get_state` reports work even outside
barge-in mode so the button knows when to appear.

`request_stop`/`get_state` scan `Assistant.satellites` (per-satellite turn
state) rather than reading a single global flag, so these tests register a
bare `SatelliteSession` to stand in for a real `connect_satellite` call.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.satellite import SatelliteSession
from core.tts_session import TtsSession
from server.dashboard import create_app


def _make_assistant(**kwargs):
    """Bare Assistant with AudioCapture mocked out."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus", **kwargs)
    # A fresh SatelliteSession's tts_active is a real, unset threading.Event
    # (reads False), unlike the MagicMock audio_capture it sits alongside.
    a.satellites["sat-a"] = SatelliteSession(id="sat-a", chunk_q=None)
    return a


class TestRequestStop:
    def test_stops_active_session_and_stands_down(self):
        a = _make_assistant()
        events = []
        a.register_turn_listener(events.append)
        a._turn_arbiter.try_acquire("sat-a")  # a real turn always holds this
        sess = TtsSession()
        a.satellites["sat-a"].active_session = sess
        a.satellites["sat-a"].last_turn_end = 123.0

        a.request_stop()

        assert sess.cancelled is True  # SLM/TTS told to abort
        assert a.satellites["sat-a"].last_turn_end == 0.0  # no follow-up window
        a.audio_capture.clear_follow_up.assert_called()
        assert any(e.get("role") == "stopped" for e in events)
        assert a._turn_arbiter.owner is None  # released

    def test_idle_is_safe_noop(self):
        a = _make_assistant()
        a.satellites["sat-a"].active_session = None
        a.request_stop()  # must not raise

    def test_barge_in_flushes_and_drops(self):
        a = _make_assistant()
        a._turn_arbiter.try_acquire("sat-a")
        a.satellites["sat-a"].turn_active = True
        a.satellites["sat-a"].active_session = TtsSession()
        a.request_stop()
        a.audio_capture.flush.assert_called()
        assert a.satellites["sat-a"].drop_results_until > 0

    def test_stops_the_explicitly_named_satellite(self):
        a = _make_assistant()
        a.satellites["sat-b"] = SatelliteSession(id="sat-b", chunk_q=None)
        a._turn_arbiter.try_acquire("sat-b")
        sess_b = TtsSession()
        a.satellites["sat-b"].active_session = sess_b

        a.request_stop("sat-b")

        assert sess_b.cancelled is True
        assert a._turn_arbiter.owner is None


class TestGetStateBusy:
    def test_active_session_reads_as_thinking(self):
        a = _make_assistant()
        a.satellites["sat-a"].turn_active = False
        a.satellites["sat-a"].active_session = None
        assert a.get_state() == "idle"
        # A half-duplex / text turn sets only active_session (no turn_active).
        a.satellites["sat-a"].active_session = TtsSession()
        assert a.get_state() == "thinking"


class TestStopEndpoint:
    def test_post_stop_calls_request_stop(self, monkeypatch):
        a = MagicMock()
        a.register_turn_listener = MagicMock()
        client = TestClient(create_app(a))
        r = client.post("/stop")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        a.request_stop.assert_called_once()
