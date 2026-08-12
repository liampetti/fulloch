"""Tests for openWakeWord runtime selection."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

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
