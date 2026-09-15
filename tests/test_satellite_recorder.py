"""Client flush and per-satellite VAD recorder behavior."""

import queue
import threading
import time

import numpy as np
import pytest

from core.audio import AudioCapture
from core.satellite import SatelliteSession
from core.wakeword import WakewordResult


@pytest.mark.parametrize("accepted", [True, False])
def test_hard_endpoint_retains_final_until_delayed_wake_verdict(accepted):
    """ASR may finish verification after VAD has closed the command buffer."""
    capture = AudioCapture(use_vad=False, min_utterance_ms=100)
    capture._use_vad_enabled = True
    capture.vad_min_speech_samples = 100
    capture._build_endpointer = lambda: _ScriptedEndpointer(
        [
            {"speech_started": True},
            {"endpointed": True, "last_speech_samples": 4000},
        ]
    )

    class Backend:
        def reset(self, satellite_id):
            assert satellite_id == "sat-a"

    capture.set_wakeword_backend(Backend())

    class VerdictAfterEndpoint(queue.Queue):
        reads = 0

        def get(self, *args, **kwargs):
            self.reads += 1
            if self.reads == 3:
                assert session.kws_candidate
                assert session.kws_pending_final is not None
                assert capture.audio_queue.empty()
                capture.resolve_wakeword_candidate(session, 7, accepted)
                self.put(None)
            return super().get(*args, **kwargs)

    session = SatelliteSession(id="sat-a", chunk_q=VerdictAfterEndpoint())
    session.kws_candidate = True
    session.kws_capture_id = 7
    chunk = np.full(2000, 0.25, dtype=np.float32)
    session.chunk_q.put(chunk)
    session.chunk_q.put(chunk)

    capture.satellite_recorder_thread(session)

    assert not session.kws_candidate
    assert session.kws_pending_final is None
    assert "sat-a" not in capture._live_endpointers
    if accepted:
        final = capture.audio_queue.get_nowait()
        np.testing.assert_array_equal(final[0], np.concatenate([chunk, chunk]))
        assert final[3] is False
        assert final[7] is True
        assert final[10:] == (False, 7)
    assert capture.audio_queue.empty()


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
        self.processed = []
        self.speech_started = False
        self.soft_endpointed = False
        self.endpointed = False
        self.last_speech_samples = 0
        self.speech_onset = None
        self.voiced_rms = None
        self.reset_calls = 0

    def process(self, samples) -> None:
        self.processed.append(samples.copy())
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

    def test_live_asr_only_receives_vad_confirmed_speech(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100
        ac._build_endpointer = lambda: _ScriptedEndpointer(
            [{}, {"speech_started": True}, {"endpointed": True, "last_speech_samples": 5000}]
        )

        class LiveWorker:
            def __init__(self):
                self.starts = []
                self.frames = []
                self.finals = []

            def start(self, satellite_id, pcm):
                self.starts.append((satellite_id, pcm.copy()))

            def feed_frame(self, satellite_id, pcm):
                self.frames.append((satellite_id, pcm.copy()))

            def finish(self, item):
                self.finals.append(item)
                return True

            def discard(self, _satellite_id):
                pass

        worker = LiveWorker()
        ac.set_live_asr_backend(worker)
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        chunk = np.ones(2000, dtype=np.float32) * 0.1
        for _ in range(3):
            chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        assert len(worker.starts) == 1
        assert len(worker.frames) == 1
        assert len(worker.finals) == 1

    def test_stereo_utterance_selects_one_best_channel_after_endpointing(self):
        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100
        endpointer = _ScriptedEndpointer(
            [
                {"speech_started": True},
                {"endpointed": True, "last_speech_samples": 5000},
            ]
        )
        ac._build_endpointer = lambda: endpointer
        chunk_q: "queue.Queue" = queue.Queue()
        session = SatelliteSession(id="sat-a", chunk_q=chunk_q)
        chunk = np.column_stack(
            (np.full(2000, 0.1, dtype=np.float32), np.full(2000, 0.5, dtype=np.float32))
        )
        chunk_q.put(chunk)
        chunk_q.put(chunk)
        chunk_q.put(None)

        ac.satellite_recorder_thread(session)

        buf, *_ = ac.audio_queue.get_nowait()
        assert buf.shape == (4000,)
        assert np.allclose(buf, 0.5)
        assert np.allclose(endpointer.processed[0], 0.5)

    def test_wakeword_match_discards_pre_wake_room_audio(self):
        class Endpointer:
            def __init__(self):
                self.speech_started = False
                self.soft_endpointed = False
                self.endpointed = False
                self.last_speech_samples = 0
                self.speech_onset = None
                self.voiced_rms = None

            def process(self, samples) -> None:
                self.speech_started = True
                self.last_speech_samples = len(samples)
                self.endpointed = samples[0] == 9.0

            def reset(self) -> None:
                self.speech_started = False
                self.soft_endpointed = False
                self.endpointed = False
                self.last_speech_samples = 0

        class WakewordBackend:
            def feed_pcm(self, _satellite_id, samples):
                return WakewordResult(samples[0] == 7.0, 0.9)

            def reset(self, _satellite_id) -> None:
                pass

        ac = AudioCapture(use_vad=False, min_utterance_ms=100, max_utterance_ms=10000)
        ac._use_vad_enabled = True
        ac.vad_min_speech_samples = 100
        ac._build_endpointer = Endpointer
        ac.set_wakeword_backend(WakewordBackend())
        session = SatelliteSession(id="sat-a", chunk_q=queue.Queue())
        detected = threading.Event()
        ac.set_wakeword_detected_callback(lambda _id: detected.set())
        recorder = threading.Thread(target=ac.satellite_recorder_thread, args=(session,))
        recorder.start()
        for value in range(1, 8):
            session.chunk_q.put(np.full(3200, value, dtype=np.float32))
        assert detected.wait(timeout=1)
        ac.resolve_wakeword_candidate(session, session.kws_capture_id, True)
        for value in range(8, 10):
            session.chunk_q.put(np.full(3200, value, dtype=np.float32))
        session.chunk_q.put(None)
        recorder.join(timeout=1)
        assert not recorder.is_alive()

        verification, *_ = ac.audio_queue.get_nowait()
        buf, *_ = ac.audio_queue.get_nowait()
        assert verification.size == 20000
        # The one-second pre-roll keeps chunks 3-7; chunks 1-2 precede the
        # wakeword and must never reach ASR with the command.
        assert buf[0] == 3.0
        assert buf[-1] == 9.0

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
