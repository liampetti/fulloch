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
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.audio import AudioCapture
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
        a = _make_assistant()
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
        a._last_connected_satellite_id = "sat-a"

        a.speak_proactive("Hello downstairs", emit_event=False, satellite_id="sat-b")

        assert a._tts_module.speak_stream.call_args.kwargs["sink"] is sink_b

    def test_missing_explicit_target_does_not_fall_back_to_latest_satellite(self):
        a = _make_assistant()
        a.models_ready.set()
        a._tts_module = MagicMock()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", tts_sink=queue.Queue())
        a._last_connected_satellite_id = "sat-a"

        a.speak_proactive("Hello downstairs", satellite_id="missing")

        a._tts_module.speak_stream.assert_not_called()


class TestSinkFor:
    def test_resolves_connected_satellite(self):
        a = _make_assistant()
        a.audio_capture.satellite_recorder_thread = _blocking_recorder
        a.connect_satellite("sat-a")
        sink = queue.Queue()
        a.set_satellite_sink("sat-a", sink)

        assert a._sink_for("sat-a") is sink
        assert a._sink_for("sat-b") is None
        assert a._sink_for(None) is None

        a.disconnect_satellite("sat-a")


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


class TestServerVadHook:
    """Forward-compat hook for the Phase 5 satellite-v2 protocol: a client
    that does its own VAD (`server_vad=False`) sends already-endpointed
    audio, so the recorder shouldn't run RMS/VAD endpointing over it. It may
    still stream several chunks per utterance, though, so unlike a naive
    "every chunk is an utterance" reading, the recorder accumulates chunks
    and only pushes them as one utterance when the client's `audio.flush`
    message arrives (relayed as the `FLUSH` sentinel on `chunk_q`) — matching
    the satellite-v2 message table's separate `audio.frame`/`audio.flush`
    types.
    """

    def test_server_vad_false_accumulates_until_flush(self):
        from core.audio import FLUSH, AudioCapture

        ac = AudioCapture(use_vad=False)
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q, server_vad=False)

        import numpy as np

        chunk1 = np.ones(160, dtype=np.float32) * 0.1
        chunk2 = np.ones(160, dtype=np.float32) * 0.2
        chunk_q.put(chunk1)
        chunk_q.put(chunk2)
        chunk_q.put(FLUSH)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        pushed = ac.audio_queue.get_nowait()
        assert ac.audio_queue.empty()  # exactly one utterance, not two
        buf, _onset, _dbfs, _provisional, satellite_id, _endpoint_t = pushed
        assert len(buf) == len(chunk1) + len(chunk2)  # both chunks concatenated
        assert satellite_id == "sat-a"

    def test_server_vad_false_flush_with_no_audio_pushes_nothing(self):
        from core.audio import FLUSH, AudioCapture

        ac = AudioCapture(use_vad=False)
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q, server_vad=False)
        chunk_q.put(FLUSH)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        assert ac.audio_queue.empty()

    def test_server_vad_false_second_utterance_after_flush_is_independent(self):
        from core.audio import FLUSH, AudioCapture

        ac = AudioCapture(use_vad=False)
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q, server_vad=False)

        import numpy as np

        first = np.ones(160, dtype=np.float32) * 0.1
        second = np.ones(320, dtype=np.float32) * 0.2
        chunk_q.put(first)
        chunk_q.put(FLUSH)
        chunk_q.put(second)
        chunk_q.put(FLUSH)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        pushed = [ac.audio_queue.get_nowait(), ac.audio_queue.get_nowait()]
        assert ac.audio_queue.empty()
        assert len(pushed[0][0]) == len(first)
        assert len(pushed[1][0]) == len(second)

    def test_server_vad_false_still_honours_mic_gates(self):
        from core.audio import FLUSH, AudioCapture

        ac = AudioCapture(use_vad=False)
        ac.mic_globally_enabled = False
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q, server_vad=False)

        import numpy as np

        chunk_q.put(np.ones(320, dtype=np.float32) * 0.1)
        chunk_q.put(FLUSH)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        assert ac.audio_queue.empty()

    def test_server_vad_true_is_the_default_and_unchanged(self):
        s = SatelliteSession(id="x", chunk_q=queue.Queue())
        assert s.server_vad is True


class _ScriptedEndpointer:
    """Duck-typed stand-in for `core.vad.VadEndpointer` (A0).

    Each `process()` call pops the next scripted state dict and applies it
    over the current attributes (unset keys carry over), like `FakeIterator`
    in `tests/test_audio_vad.py` but at the endpointer level rather than the
    raw `VADIterator` level — the recorder only ever touches the endpointer's
    public attributes, never the iterator underneath.
    """

    def __init__(self, script):
        self._script = list(script)
        self.speech_started = False
        self.soft_endpointed = False
        self.endpointed = False
        self.last_speech_samples = 0
        self.speech_onset = None
        self.voiced_rms = None
        self.reset_calls = 0

    def process(self, samples) -> None:
        if not self._script:
            return
        state = self._script.pop(0)
        self.speech_started = state.get("speech_started", self.speech_started)
        self.soft_endpointed = state.get("soft_endpointed", self.soft_endpointed)
        self.endpointed = state.get("endpointed", self.endpointed)
        self.last_speech_samples = state.get("last_speech_samples", self.last_speech_samples)
        if self.speech_started and self.speech_onset is None:
            self.speech_onset = time.monotonic()

    def reset(self) -> None:
        self.reset_calls += 1
        self.speech_started = False
        self.soft_endpointed = False
        self.endpointed = False
        self.last_speech_samples = 0
        self.speech_onset = None
        self.voiced_rms = None


class TestVadEndpointing:
    """A0: the recorder's VAD path (`AudioCapture.satellite_recorder_thread`),
    driven by a scripted `_ScriptedEndpointer` in place of real Silero so the
    windowing/state transitions matter, not the model. `_build_endpointer` is
    monkeypatched per test rather than loading real Silero.
    """

    def test_soft_endpoint_emits_one_provisional_per_pause_then_final(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100  # low floor so the scripted span passes
        script = [
            {"speech_started": True},
            {"soft_endpointed": True},  # pause -> one provisional
            {},  # still paused -> debounced, no second provisional
            {"endpointed": True, "last_speech_samples": 5000},  # hard endpoint
        ]
        ac._build_endpointer = lambda: _ScriptedEndpointer(script)

        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        chunk = np.zeros(2000, dtype=np.float32)
        for _ in range(4):
            chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        results = []
        while not ac.audio_queue.empty():
            results.append(ac.audio_queue.get_nowait())

        assert [r[3] for r in results] == [True, False]  # one provisional, one final
        assert results[0][4] == results[1][4] == "sat-a"

    def test_hard_endpoint_keeps_its_vad_onset_when_tts_starts_after_soft_probe(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100
        # The old RMS fallback would immediately finalise its accumulated buffer
        # after TTS began, but assign a new onset to that duplicate utterance.
        ac.tts_max_utterance_samples = 1000
        script = [
            {"speech_started": True},
            {"soft_endpointed": True},
            {"endpointed": True, "last_speech_samples": 5000},
        ]
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)

        class TtsStartingEndpointer(_ScriptedEndpointer):
            def process(self, samples) -> None:
                super().process(samples)
                if self.soft_endpointed:
                    session.tts_active.set()

        ac._build_endpointer = lambda: TtsStartingEndpointer(script)
        chunk = np.zeros(2000, dtype=np.float32)
        for _ in range(3):
            chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        results = []
        while not ac.audio_queue.empty():
            results.append(ac.audio_queue.get_nowait())

        assert [r[3] for r in results] == [True, False]
        assert results[0][1] == results[1][1]

    def test_soft_endpoint_probes_short_wake_phrase_below_normal_minimum(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=1500, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        script = [
            {"speech_started": True},
            {"soft_endpointed": True},
        ]
        ac._build_endpointer = lambda: _ScriptedEndpointer(script)

        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        # 500ms: enough for a short wake probe, below the normal 1.5s floor.
        chunk = np.zeros(4000, dtype=np.float32)
        chunk_q.put(chunk)
        chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        result = ac.audio_queue.get_nowait()
        assert result[0].size == 8000
        assert result[3] is True
        assert result[4] == "sat-a"
        assert ac.audio_queue.empty()

    def test_speech_onset_emits_one_wake_only_probe_without_a_pause(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=1500, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac._build_endpointer = lambda: _ScriptedEndpointer([{"speech_started": True}, {}])

        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        # Start the probe clock in the past to test the recorder decision without
        # sleeping for the production 600 ms delay.
        session.early_wake_probe_started_at = time.monotonic() - ac.early_wake_probe_seconds
        chunk_q.put(np.zeros(2000, dtype=np.float32))
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        result = ac.audio_queue.get_nowait()
        assert result[3] is True
        assert result[6] is True
        assert session.early_wake_probe_emitted is True
        assert ac.audio_queue.empty()

    def test_endpointer_reset_after_hard_endpoint_prevents_stale_duplicate(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100
        script = [
            {"speech_started": True},
            {"endpointed": True, "last_speech_samples": 5000},  # commits + resets
            {},  # post-reset: speech_started is False again, nothing to commit
        ]
        ac._build_endpointer = lambda: _ScriptedEndpointer(script)

        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        chunk = np.zeros(2000, dtype=np.float32)
        for _ in range(3):
            chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        results = []
        while not ac.audio_queue.empty():
            results.append(ac.audio_queue.get_nowait())

        assert len(results) == 1  # not a second, stale commit from leftover state
        assert session.vad_endpointer.reset_calls == 1
        assert session.early_wake_probe_emitted is False
        assert session.early_wake_probe_started_at == 0.0

    def test_two_satellites_get_independent_endpointer_instances(self):
        ac = AudioCapture(use_vad=False)
        ac._use_vad_enabled = True
        built = []

        def fake_build():
            ep = _ScriptedEndpointer([])
            built.append(ep)
            return ep

        ac._build_endpointer = fake_build

        session_a = SatelliteSession(id="sat-a", chunk_q=queue.Queue())
        session_b = SatelliteSession(id="sat-b", chunk_q=queue.Queue())
        session_a.chunk_q.put(None)
        session_b.chunk_q.put(None)

        ac.satellite_recorder_thread(session_a)
        assert "sat-a" not in ac._live_endpointers  # cleaned up on thread exit
        ac.satellite_recorder_thread(session_b)
        assert "sat-b" not in ac._live_endpointers

        assert len(built) == 2
        assert session_a.vad_endpointer is not session_b.vad_endpointer

    def test_concurrent_satellites_endpoint_independently(self):
        # Two satellites, each with a distinct scripted pause pattern, run on
        # real threads at once — a shared endpointer (the pre-A0 singleton
        # design) would corrupt one satellite's state machine with the
        # other's window sequence. `_build_endpointer` is keyed off the
        # calling thread's name (set to the satellite id below) so each
        # thread gets its own script deterministically despite the race.
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100

        scripts_by_thread = {
            "sat-a": [
                {"speech_started": True},
                {"endpointed": True, "last_speech_samples": 5000},
            ],
            "sat-b": [
                {"speech_started": True},
                {"soft_endpointed": True},
            ],
        }
        ac._build_endpointer = lambda: _ScriptedEndpointer(
            scripts_by_thread[threading.current_thread().name]
        )

        session_a = SatelliteSession(id="sat-a", chunk_q=queue.Queue())
        session_b = SatelliteSession(id="sat-b", chunk_q=queue.Queue())
        chunk = np.zeros(2000, dtype=np.float32)
        for session in (session_a, session_b):
            session.chunk_q.put(chunk)
            session.chunk_q.put(chunk)
            session.chunk_q.put(None)

        t_a = threading.Thread(target=ac.satellite_recorder_thread, args=(session_a,), name="sat-a")
        t_b = threading.Thread(target=ac.satellite_recorder_thread, args=(session_b,), name="sat-b")
        t_a.start()
        t_b.start()
        t_a.join(timeout=2)
        t_b.join(timeout=2)

        by_sat: dict = {}
        while not ac.audio_queue.empty():
            buf, onset, db, provisional, sid, endpoint_t = ac.audio_queue.get_nowait()
            by_sat.setdefault(sid, []).append(provisional)

        assert by_sat["sat-a"] == [False]  # hard endpoint only
        assert by_sat["sat-b"] == [True]  # still paused mid-utterance, provisional only
        assert session_a.vad_endpointer is not session_b.vad_endpointer
