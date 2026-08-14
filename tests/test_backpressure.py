"""Regression coverage for bounded audio and browser-playback handoff queues."""

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from core.assistant import _safe_sink
from core.audio import AUDIO_QUEUE_MAX_ITEMS, AudioCapture
from core.satellite import SatelliteSession
from server.dashboard import create_app
from server.integration_api import create_integration_app
from server.lifecycle import READY, AppContext, Lifecycle


def _assistant():
    with patch("core.assistant.AudioCapture") as capture:
        capture.return_value = MagicMock()
        from core.assistant import Assistant

        return Assistant(barge_in="wakeword", wakeword="hey atticus")


def test_completed_audio_queue_is_bounded_and_drops_when_asr_is_busy():
    capture = AudioCapture(use_vad=False)
    assert capture.audio_queue.maxsize == AUDIO_QUEUE_MAX_ITEMS
    capture.audio_queue = queue.Queue(maxsize=1)
    session = SatelliteSession(id="sat-a")
    audio = np.ones(160, dtype=np.float32)

    capture._enqueue(session, audio, 1.0, -30.0, False, 2.0)
    capture._enqueue(session, audio * 2, 2.0, -20.0, False, 3.0)

    item = capture.audio_queue.get_nowait()
    assert item[0].tolist() == audio.tolist()
    assert capture.audio_queue.empty()


def test_tts_sink_replaces_stale_audio_with_cancel():
    output = queue.Queue(maxsize=1)
    sink = _safe_sink(output)
    sink.put((np.ones(160, dtype=np.float32), None))

    sink.put(("cancel",))

    assert output.get_nowait() == ("cancel",)


def test_tts_sink_backpressures_instead_of_dropping_audio():
    output = queue.Queue(maxsize=1)
    sink = _safe_sink(output)
    first = np.ones(160, dtype=np.float32)
    second = np.full(160, 2.0, dtype=np.float32)
    sink.put((first, None))

    producer = threading.Thread(target=lambda: sink.put((second, None)))
    producer.start()
    producer.join(timeout=0.05)
    assert producer.is_alive()

    np.testing.assert_array_equal(output.get_nowait()[0], first)
    producer.join(timeout=1)
    assert not producer.is_alive()
    np.testing.assert_array_equal(output.get_nowait()[0], second)


def test_disconnect_does_not_block_on_a_full_microphone_queue_and_cancels_turn():
    assistant = _assistant()
    chunks = queue.Queue(maxsize=1)
    chunks.put(np.ones(160, dtype=np.float32))
    active = MagicMock()
    assistant.satellites["sat-a"] = SatelliteSession(
        id="sat-a", chunk_q=chunks, tts_sink=queue.Queue(maxsize=1), active_session=active
    )

    assistant.disconnect_satellite("sat-a")

    active.stop.assert_called_once()
    assert chunks.get_nowait() is None
    assert "sat-a" not in assistant.satellites


def _blocking_context():
    entered = threading.Event()
    release = threading.Event()
    assistant = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.mic_globally_enabled = True

    def speak(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2)

    assistant.speak_proactive.side_effect = speak
    return AppContext(lifecycle=Lifecycle(phase=READY), assistant=assistant), entered, release


def test_dashboard_limits_pending_proactive_request_threads():
    from fastapi.testclient import TestClient

    context, entered, release = _blocking_context()
    client = TestClient(create_app(context=context))
    try:
        assert client.post("/speak", json={"text": "one"}).status_code == 200
        assert entered.wait(timeout=1)
        assert client.post("/speak", json={"text": "two"}).status_code == 200
        assert client.post("/speak", json={"text": "three"}).status_code == 429
    finally:
        release.set()


def test_integration_limits_pending_proactive_request_threads(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("server.integration_api.get_credential", lambda _key: ["token"])
    context, entered, release = _blocking_context()
    client = TestClient(create_integration_app(context))
    headers = {"Authorization": "Bearer token"}
    try:
        assert client.post("/speak", headers=headers, json={"text": "one"}).status_code == 200
        assert entered.wait(timeout=1)
        assert client.post("/speak", headers=headers, json={"text": "two"}).status_code == 200
        assert client.post("/speak", headers=headers, json={"text": "three"}).status_code == 429
    finally:
        release.set()
