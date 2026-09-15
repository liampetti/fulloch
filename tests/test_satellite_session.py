"""Two `/ws/satellite` connects must not clobber each other's queues.

Before this refactor, `Assistant` tracked the single connected satellite as
bare instance attributes (`_satellite_sink`, `_satellite_chunk_q`); a second
connect overwrote them and the first satellite's recorder thread was never
told to stop. `Assistant.satellites` (keyed by caller-supplied satellite_id)
fixes both: each connect gets its own `SatelliteSession`, and disconnecting
one only tears down that session.
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from core.satellite import SatelliteSession


def _make_assistant(**kwargs):
    """Bare Assistant with AudioCapture mocked out (see tests/test_stop.py)."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus", **kwargs)
    return a


def _blocking_recorder(session):
    """Stand-in for AudioCapture.satellite_recorder_thread: blocks until the
    None sentinel arrives, like the real recorder does on disconnect."""
    while True:
        item = session.chunk_q.get()
        if item is None:
            return


class TestConnectSatellite:
    def test_two_connects_produce_distinct_sessions(self):
        a = _make_assistant(max_voice_satellites=2)
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        q_a = a.connect_satellite("sat-a")
        q_b = a.connect_satellite("sat-b")

        # "dashboard-text" is always present (the reserved pseudo-session for
        # typed turns, seeded in __init__) alongside the two real connects.
        assert set(a.satellites) == {"dashboard-text", "sat-a", "sat-b"}
        assert a.satellites["sat-a"].chunk_q is q_a
        assert a.satellites["sat-b"].chunk_q is q_b
        assert q_a is not q_b

        a.disconnect_satellite("sat-a")
        a.disconnect_satellite("sat-b")

    def test_third_voice_connect_is_rejected_to_bound_model_instances(self):
        from core.assistant import ConversationModeUnavailable

        a = _make_assistant(max_voice_satellites=2)
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.connect_satellite("sat-a")
        a.connect_satellite("sat-b")

        with pytest.raises(ConversationModeUnavailable, match="maximum number"):
            a.connect_satellite("sat-c")

        a.disconnect_satellite("sat-a")
        a.disconnect_satellite("sat-b")

    def test_second_connect_does_not_overwrite_first(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("sat-a")
        sink_a: "queue.Queue" = queue.Queue()
        a.set_satellite_sink("sat-a", sink_a)

        a.connect_satellite("sat-b")

        assert a.satellites["sat-a"].tts_sink is sink_a
        assert a.satellites["sat-b"].tts_sink is None

        a.disconnect_satellite("sat-a")
        a.disconnect_satellite("sat-b")

    def test_session_id_matches_caller_supplied_id(self):
        # Task 0: connect_satellite takes the id as a required argument
        # rather than minting its own — the caller (the WS handler) needs to
        # know the id synchronously to key set_satellite_sink/disconnect_satellite.
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("my-id")
        assert a.satellites["my-id"].id == "my-id"
        a.disconnect_satellite("my-id")

    def test_native_session_starts_wakeword_gated(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("native", device_id="kitchen-01")

        session = a.satellites["native"]
        assert session.last_turn_end == 0.0
        a.audio_capture.arm_follow_up.assert_not_called()
        a.disconnect_satellite("native")

    def test_browser_initial_grace_allows_immediate_speech(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("browser", initial_grace=True)

        session = a.satellites["browser"]
        assert session.last_turn_end > 0.0
        a.audio_capture.arm_follow_up.assert_called_once_with(session, 60)
        a.disconnect_satellite("browser")

    def test_same_device_id_replaces_the_existing_session(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = lambda session: None
        old_chunks = a.connect_satellite("old-session", device_id="kitchen-01")
        old_tts: "queue.Queue" = queue.Queue()
        a.set_satellite_sink("old-session", old_tts)

        a.connect_satellite("new-session", device_id="kitchen-01")

        assert set(a.satellites) == {"dashboard-text", "new-session"}
        assert old_chunks.get_nowait() is None
        assert old_tts.get_nowait() == ("stop",)
        assert a.satellites["new-session"].device_id == "kitchen-01"
        a.disconnect_satellite("new-session")

    def test_conversation_mode_is_exclusive(self):
        from core.assistant import ConversationModeUnavailable

        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.connect_satellite("sat-a", conversation_mode=True)

        assert a.conversation_owner_id == "sat-a"
        assert a.satellites["sat-a"].conversation_mode is True
        with pytest.raises(ConversationModeUnavailable):
            a.connect_satellite("sat-b")

        a.disconnect_satellite("sat-a")
        assert a.conversation_owner_id is None

    def test_conversation_mode_disconnects_other_voice_satellites(self):
        from core.assistant import ConversationModeUnavailable

        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.connect_satellite("sat-a")
        a.connect_satellite("sat-b")
        sink_b: "queue.Queue" = queue.Queue()
        a.set_satellite_sink("sat-b", sink_b)

        enabled, _ = a.set_satellite_conversation_mode("sat-a", True)

        assert enabled is True
        assert set(a.satellites) == {"dashboard-text", "sat-a"}
        assert a.conversation_owner_id == "sat-a"
        assert sink_b.get_nowait() == ("stop",)

        with pytest.raises(ConversationModeUnavailable):
            a.connect_satellite("sat-c")

        a.set_satellite_conversation_mode("sat-a", False)
        a.connect_satellite("sat-c")
        a.disconnect_satellite("sat-a")
        a.disconnect_satellite("sat-c")

    def test_native_conversation_mode_opens_a_listening_turn(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        events = []
        a.register_turn_listener(events.append)
        a.connect_satellite("native", device_id="kitchen-01")

        enabled, _ = a.set_satellite_conversation_mode("native", True)

        assert enabled is True
        assert a.satellites["native"].protocol_turn_id is not None
        assert events[-1] == {
            "type": "assistant.state",
            "state": "listening",
            "satellite_id": "native",
            "turn_id": a.satellites["native"].protocol_turn_id,
        }
        a.disconnect_satellite("native")

    def test_conversation_mode_announces_listening_once_sink_is_attached(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.conversation_listening_cache = [("audio", 24000)]
        a.play_chunks = MagicMock()

        a.connect_satellite("sat-a", conversation_mode=True)

        a.play_chunks.assert_not_called()
        a.set_satellite_sink("sat-a", queue.Queue())

        a.play_chunks.assert_called_once_with("audio", 24000, session=a.tts_session)
        a.set_satellite_conversation_mode("sat-a", True)
        a.play_chunks.assert_called_once()
        a.disconnect_satellite("sat-a")

    def test_enabling_conversation_mode_announces_listening_immediately(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.conversation_listening_cache = [("audio", 24000)]
        a.play_chunks = MagicMock()
        a._mark_turn_end = MagicMock()
        a.connect_satellite("sat-a")
        a.set_satellite_sink("sat-a", queue.Queue())

        a.set_satellite_conversation_mode("sat-a", True)

        a.play_chunks.assert_called_once_with("audio", 24000, session=a.tts_session)
        a._mark_turn_end.assert_called_once()
        assert a._mark_turn_end.call_args.args[0] == "sat-a"
        a.disconnect_satellite("sat-a")


class TestDisconnectSatellite:
    def test_disconnect_leaves_other_session_intact(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("sat-a")
        a.connect_satellite("sat-b")
        sink_b: "queue.Queue" = queue.Queue()
        a.set_satellite_sink("sat-b", sink_b)

        a.disconnect_satellite("sat-a")

        assert "sat-a" not in a.satellites
        assert "sat-b" in a.satellites
        assert a.satellites["sat-b"].tts_sink is sink_b

        a.disconnect_satellite("sat-b")

    def test_disconnect_joins_mid_drain_recorder_thread(self):
        import time

        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        before = threading.active_count()
        a.connect_satellite("sat-a")
        assert threading.active_count() == before + 1

        # disconnect_satellite doesn't itself join; it sentinels the queue
        # and the recorder thread exits on its own once it sees None — give
        # it a moment, mirroring the existing test_audio_*.py thread-count
        # checks (no leaked thread after a disconnect).
        a.disconnect_satellite("sat-a")
        deadline = time.monotonic() + 2.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.01)
        assert threading.active_count() == before

    def test_disconnect_unknown_satellite_is_a_noop(self):
        a = _make_assistant()
        a.disconnect_satellite("never-connected")  # must not raise

    def test_disconnect_flushes_queued_asr_work_for_that_session(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.connect_satellite("sat-a")

        a.disconnect_satellite("sat-a")

        a.audio_capture.flush.assert_called_once_with("sat-a")


class TestSetSatelliteSink:
    def test_keyed_not_blind_overwrite(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder

        a.connect_satellite("sat-a")
        a.connect_satellite("sat-b")
        sink_a: "queue.Queue" = queue.Queue()
        sink_b: "queue.Queue" = queue.Queue()

        a.set_satellite_sink("sat-a", sink_a)
        a.set_satellite_sink("sat-b", sink_b)

        assert a.satellites["sat-a"].tts_sink is sink_a
        assert a.satellites["sat-b"].tts_sink is sink_b

        a.disconnect_satellite("sat-a")
        a.disconnect_satellite("sat-b")

    def test_sink_for_unknown_satellite_is_noop(self):
        a = _make_assistant()
        a.set_satellite_sink("never-connected", queue.Queue())  # must not raise


class TestTargetedProactiveSpeech:
    def test_explicit_target_uses_its_own_sink(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        a._tts_module.speak_stream.return_value = 0.0
        sink_a: "queue.Queue" = queue.Queue()
        sink_b: "queue.Queue" = queue.Queue()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", tts_sink=sink_a)
        a.satellites["sat-b"] = SatelliteSession(id="sat-b", tts_sink=sink_b)

        a.speak_proactive("Hello downstairs", emit_event=False, satellite_id="sat-b")

        assert a._tts_module.speak_stream.call_args.kwargs["sink"] is sink_b

    def test_missing_explicit_target_does_not_fall_back_to_latest_satellite(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", tts_sink=queue.Queue())

        a.speak_proactive("Hello downstairs", satellite_id="missing")

        a._tts_module.speak_stream.assert_not_called()

    def test_proactive_speech_stands_down_without_follow_up(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        a._tts_module.speak_stream.return_value = 0.0
        session = SatelliteSession(id="sat-a", tts_sink=queue.Queue())
        a.satellites["sat-a"] = session
        emitted = []
        a._dispatch_event = emitted.append

        a.speak_proactive("Calendar reminder", emit_event=False)

        assert a.audio_capture.clear_follow_up.called
        assert emitted[0]["state"] == "thinking"
        assert emitted[1]["type"] == "assistant.stand_down_after_tts"
        assert emitted[1]["turn_id"] == emitted[0]["turn_id"]
        assert session.protocol_turn_id is None

    def test_proactive_question_opens_a_follow_up_window(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        a._tts_module.speak_stream.return_value = 42.0
        session = SatelliteSession(id="sat-a", tts_sink=queue.Queue())
        a.satellites["sat-a"] = session
        a._mark_turn_end = MagicMock()

        a.speak_proactive(
            "Would you like a summary?", emit_event=False, satellite_id="sat-a", follow_up=True
        )

        a._mark_turn_end.assert_called_once_with("sat-a", 42.0)

    def test_untargeted_proactive_speech_broadcasts_to_every_satellite(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        sink_a: "queue.Queue" = queue.Queue()
        sink_b: "queue.Queue" = queue.Queue()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", tts_sink=sink_a)
        a.satellites["sat-b"] = SatelliteSession(id="sat-b", tts_sink=sink_b)

        a.speak_proactive("Calendar reminder", emit_event=False)

        tts_sink = a._tts_module.speak_stream.call_args.kwargs["sink"]
        tts_sink.put(("start", 16000))
        assert sink_a.get_nowait() == ("start", 16000)
        assert sink_b.get_nowait() == ("start", 16000)


def test_satellite_session_defaults():
    s = SatelliteSession(id="x", chunk_q=queue.Queue())
    assert s.tts_sink is None
    assert s.recorder_thread is None
    assert s.conversation_mode is False
    assert s.label is None
    assert s.ha_area is None
    assert s.server_vad is True
    assert s.auth_token is None
    assert s.device_id is None
