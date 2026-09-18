"""Tests for openWakeWord runtime selection."""

import queue
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.asr import stream_generator  # noqa: E402
from core.audio import AudioCapture  # noqa: E402
from core.satellite import SatelliteSession  # noqa: E402
from core.wakeword import WakewordResult  # noqa: E402
from core.wakeword_openwakeword import OpenWakeWordBackend  # noqa: E402


def test_onnx_model_selects_onnx_runtime(tmp_path, monkeypatch):
    model_path = tmp_path / "wakeword.onnx"
    model_path.touch()
    calls = []

    class Model:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def predict(self, _frame):
            return {}

    monkeypatch.setitem(sys.modules, "openwakeword.model", SimpleNamespace(Model=Model))

    backend = OpenWakeWordBackend(str(model_path), 0.5, 3, 1500)
    backend.feed_pcm("satellite", np.zeros(1280, dtype=np.float32))

    assert calls == [{"wakeword_models": [str(model_path)], "inference_framework": "onnx"}]


def test_wakeword_activation_records_score():
    class Backend:
        def feed_pcm(self, _satellite_id, _pcm):
            return WakewordResult(True, 0.873)

        def reset(self, _satellite_id):
            pass

    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(Backend())
    matched_ids = []
    capture.set_wakeword_detected_callback(matched_ids.append)
    session = SatelliteSession("satellite")

    assert (
        capture.wake_candidates._feed_wakeword_gate(session, np.zeros(1280, dtype=np.float32))
        is True
    )

    assert session.kws_candidate is True
    assert capture.wakeword_metrics["candidates"] == 1
    assert capture.wakeword_metrics["last_score"] == 0.873
    assert matched_ids == ["satellite"]


def test_wakeword_activation_optionally_saves_timestamped_wav(tmp_path):
    class Backend:
        def feed_pcm(self, _satellite_id, _pcm):
            return WakewordResult(True, 0.873)

        def reset(self, _satellite_id):
            pass

    capture = AudioCapture(use_vad=False, save_wakeword_wavs=True)
    capture.wakeword_wav_dir = tmp_path
    capture.set_wakeword_backend(Backend())
    session = SatelliteSession("kitchen/phone")
    capture.wake_candidates._feed_wakeword_gate(session, np.ones(1280, dtype=np.float32))
    early = capture.audio_queue.get_nowait()
    capture.mark_wakeword_wav(early[8], accepted=False)
    assert session.kws_verified is False
    capture.wake_candidates._enqueue(
        session, np.ones(16000, dtype=np.float32), 0.0, -10.0, False, 0.0
    )

    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 2
    final = capture.audio_queue.get_nowait()
    early_path = Path(early[8].replace("_pending.wav", "_rejected.wav"))
    final_path = Path(final[8])
    assert final_path.name == early_path.name.replace("_early_rejected.wav", "_final_pending.wav")
    assert final_path.stat().st_size == 32044
    assert early_path.stat().st_size == 2604
    assert early[10] is True
    assert final[10] is False
    capture.mark_wakeword_wav(final[8], accepted=True)
    assert len(list(tmp_path.glob("*_final_accepted.wav"))) == 1


def test_wakeword_candidate_does_not_prepend_overlapping_preroll():
    class Backend:
        def reset(self, _satellite_id):
            pass

    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(Backend())
    session = SatelliteSession("satellite")
    session.kws_candidate = True
    session.kws_verified = True
    session.kws_pre_roll.append(np.full(1600, -1.0, dtype=np.float32))
    utterance = np.full(3200, 0.5, dtype=np.float32)

    capture.wake_candidates._enqueue(session, utterance, 0.0, -10.0, False, 0.0)

    queued = capture.audio_queue.get_nowait()[0]
    assert np.array_equal(queued, utterance)


def test_wakeword_candidate_waits_for_final_endpoint_before_asr():
    capture = AudioCapture(use_vad=False)
    session = SatelliteSession("satellite")
    session.kws_candidate = True

    capture.wake_candidates._enqueue(
        session, np.ones(3200, dtype=np.float32), 0.0, -10.0, True, 0.0
    )

    assert capture.audio_queue.empty()
    assert session.kws_candidate is True


def test_wakeword_activation_queues_a_1250ms_verification_snapshot():
    class Backend:
        def __init__(self):
            self.calls = 0

        def feed_pcm(self, _satellite_id, _pcm):
            self.calls += 1
            return WakewordResult(self.calls == 7, 0.873)

        def reset(self, _satellite_id):
            pass

    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(Backend())
    session = SatelliteSession("satellite")
    chunks = [np.full(3200, value, dtype=np.float32) for value in range(7)]

    for chunk in chunks:
        capture.wake_candidates._feed_wakeword_gate(session, chunk)

    queued = capture.audio_queue.get_nowait()
    assert queued[0].size == 20000
    assert np.array_equal(queued[0], np.concatenate(chunks)[-20000:])
    assert queued[7] is True
    assert queued[10] is True


def test_rejected_verification_rearms_the_wakeword_gate():
    class Backend:
        def __init__(self):
            self.calls = 0

        def feed_pcm(self, _satellite_id, _pcm):
            self.calls += 1
            return WakewordResult(self.calls in (1, 2), 0.9)

        def reset(self, _satellite_id):
            pass

    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(Backend())
    session = SatelliteSession("satellite", chunk_q=queue.Queue(), server_vad=False)
    detected = []
    first_detected = threading.Event()
    second_detected = threading.Event()

    def on_detected(_satellite_id):
        detected.append(session.kws_capture_id)
        (first_detected if len(detected) == 1 else second_detected).set()

    capture.set_wakeword_detected_callback(on_detected)
    recorder = threading.Thread(target=capture.satellite_recorder_thread, args=(session,))
    recorder.start()
    session.chunk_q.put(np.ones(1280, dtype=np.float32))
    assert first_detected.wait(timeout=1)
    capture.resolve_wakeword_candidate(session, detected[0], False)
    session.chunk_q.put(np.ones(1280, dtype=np.float32))
    assert second_detected.wait(timeout=1)
    session.chunk_q.put(None)
    recorder.join(timeout=1)

    assert detected == [1, 2]


def test_stereo_capture_stays_mono_after_wakeword_preroll():
    class Backend:
        def feed_pcm(self, _satellite_id, _pcm):
            return WakewordResult(True, 0.9)

        def reset(self, _satellite_id):
            pass

    class Endpointer:
        def __init__(self):
            self.process_calls = 0
            self.speech_started = False
            self.speech_onset = 0.0
            self.voiced_rms = 0.1
            self.soft_endpointed = False
            self.endpointed = False
            self.last_speech_samples = 16000

        def process(self, _pcm):
            self.process_calls += 1
            self.speech_started = True
            self.endpointed = self.process_calls >= 2

        def reset(self):
            self.process_calls = 0
            self.endpointed = False

    capture = AudioCapture(use_vad=True)
    capture.set_wakeword_backend(Backend())
    capture._build_endpointer = Endpointer
    session = SatelliteSession("satellite", chunk_q=queue.Queue())
    recorder = threading.Thread(target=capture.satellite_recorder_thread, args=(session,))
    stereo_chunk = np.ones((1280, 2), dtype=np.float32)

    recorder.start()
    session.chunk_q.put(stereo_chunk)
    session.chunk_q.put(stereo_chunk)
    session.chunk_q.put(None)
    recorder.join(timeout=1)

    assert not recorder.is_alive()


def test_wakeword_wav_is_labelled_after_asr_verification(tmp_path):
    capture = AudioCapture(use_vad=False, save_wakeword_wavs=True)
    capture.wakeword_wav_dir = tmp_path
    path = capture.wake_candidates._save_wakeword_wav(
        "kitchen", np.ones(1280, dtype=np.float32), 0.873
    )

    capture.mark_wakeword_wav(path, accepted=True)

    assert len(list(tmp_path.glob("*_accepted.wav"))) == 1


def test_wakeword_wav_preserves_stereo_samples(tmp_path):
    capture = AudioCapture(use_vad=False, save_wakeword_wavs=True)
    capture.wakeword_wav_dir = tmp_path
    pcm = np.array([[0.25, -0.5], [-0.75, 1.0]], dtype=np.float32)

    path = capture.wake_candidates._save_wakeword_wav("kitchen", pcm, 0.873)

    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() == 2
        assert np.array_equal(
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2"),
            (pcm * 32767).astype("<i2").ravel(),
        )


def test_wakeword_wav_path_stays_with_queued_candidate():
    items = queue.Queue()
    items.put(
        (
            np.ones(1280, dtype=np.float32),
            0.0,
            -10.0,
            False,
            "kitchen",
            0.0,
            False,
            True,
            "/tmp/candidate.wav",
        )
    )
    items.put(None)
    path_sink = {}

    next(stream_generator(items, kws_wav_path_sink=path_sink))

    assert path_sink == {"path": "/tmp/candidate.wav"}


def test_wakeword_gate_skips_idle_audio_until_vad_detects_speech():
    class Backend:
        def __init__(self):
            self.frames = []

        def feed_pcm(self, _satellite_id, pcm):
            self.frames.append(pcm.copy())
            return WakewordResult(False, 0.0)

    backend = Backend()
    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(backend)
    session = SatelliteSession("satellite")
    session.vad_endpointer = SimpleNamespace(speech_started=False)
    chunk = np.zeros(1280, dtype=np.float32)

    capture.wake_candidates._feed_wakeword_gate(session, chunk)

    assert backend.frames == []
    session.vad_endpointer.speech_started = True
    capture.wake_candidates._feed_wakeword_gate(session, chunk)

    assert len(backend.frames) == 1
    assert backend.frames[0].size == 2 * chunk.size


def test_wakeword_boundary_reset_clears_idle_gate_without_candidate():
    class Backend:
        def __init__(self):
            self.reset_ids = []

        def reset(self, satellite_id):
            self.reset_ids.append(satellite_id)

    backend = Backend()
    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(backend)
    session = SatelliteSession("satellite")
    session.kws_speech_active = True
    session.kws_pre_roll.append(np.zeros(1280, dtype=np.float32))

    capture.wake_candidates._discard_wakeword_candidate(session)

    assert session.kws_speech_active is False
    assert session.kws_pre_roll == []
    assert backend.reset_ids == ["satellite"]


def test_wakeword_boundary_without_classification_keeps_backend_model():
    class Backend:
        def __init__(self):
            self.reset_ids = []

        def reset(self, satellite_id):
            self.reset_ids.append(satellite_id)

    backend = Backend()
    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(backend)

    capture.wake_candidates._discard_wakeword_candidate(SatelliteSession("satellite"))

    assert backend.reset_ids == []


def test_wakeword_gate_stays_closed_until_browser_playback_ends():
    import time

    class Backend:
        def feed_pcm(self, _satellite_id, _pcm):
            return WakewordResult(True, 0.873)

    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(Backend())
    session = SatelliteSession("satellite")
    session.last_turn_end = time.monotonic() + 10.0

    capture.wake_candidates._feed_wakeword_gate(session, np.zeros(1280, dtype=np.float32))

    assert session.kws_candidate is False
    assert capture.wakeword_metrics["candidates"] == 0


def test_discard_wakeword_candidate_resets_backend_and_state():
    class Backend:
        def __init__(self):
            self.reset_ids = []

        def reset(self, satellite_id):
            self.reset_ids.append(satellite_id)

    backend = Backend()
    capture = AudioCapture(use_vad=False)
    capture.set_wakeword_backend(backend)
    session = SatelliteSession("satellite")
    session.kws_candidate = True
    session.kws_score = 0.873
    session.kws_detected_at = 123.0
    session.kws_pre_roll.append(np.zeros(1280, dtype=np.float32))

    capture.wake_candidates._discard_wakeword_candidate(session)

    assert session.kws_candidate is False
    assert session.kws_score == 0.0
    assert session.kws_detected_at == 0.0
    assert session.kws_pre_roll == []
    assert backend.reset_ids == ["satellite"]
