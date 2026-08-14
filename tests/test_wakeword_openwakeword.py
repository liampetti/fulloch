"""Tests for openWakeWord runtime selection."""

import queue
import sys
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

    capture._feed_wakeword_gate(session, np.zeros(1280, dtype=np.float32))

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
    capture._feed_wakeword_gate(session, np.ones(1280, dtype=np.float32))
    capture._enqueue(session, np.ones(16000, dtype=np.float32), 0.0, -10.0, False, 0.0)

    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 1
    assert files[0].name.endswith("_kitchen_phone_0.873_pending.wav")
    assert files[0].read_bytes()[:4] == b"RIFF"
    assert files[0].stat().st_size == 32044
    assert capture.audio_queue.get_nowait()[8] == str(files[0])


def test_wakeword_wav_is_labelled_after_asr_verification(tmp_path):
    capture = AudioCapture(use_vad=False, save_wakeword_wavs=True)
    capture.wakeword_wav_dir = tmp_path
    path = capture._save_wakeword_wav("kitchen", np.ones(1280, dtype=np.float32), 0.873)

    capture.mark_wakeword_wav(path, accepted=True)

    assert len(list(tmp_path.glob("*_accepted.wav"))) == 1


def test_wakeword_wav_path_stays_with_queued_candidate():
    items = queue.Queue()
    items.put((np.ones(1280, dtype=np.float32), 0.0, -10.0, False, "kitchen", 0.0, False, True, "/tmp/candidate.wav"))
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

    capture._feed_wakeword_gate(session, chunk)

    assert backend.frames == []
    session.vad_endpointer.speech_started = True
    capture._feed_wakeword_gate(session, chunk)

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

    capture._discard_wakeword_candidate(session)

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

    capture._discard_wakeword_candidate(SatelliteSession("satellite"))

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

    capture._feed_wakeword_gate(session, np.zeros(1280, dtype=np.float32))

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

    capture._discard_wakeword_candidate(session)

    assert session.kws_candidate is False
    assert session.kws_score == 0.0
    assert session.kws_detected_at == 0.0
    assert session.kws_pre_roll == []
    assert backend.reset_ids == ["satellite"]
