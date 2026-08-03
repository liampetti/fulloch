"""Wire-contract tests for the ESP32 satellite-v2 endpoint."""

import json
import queue
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.dashboard import create_app


@pytest.fixture(autouse=True)
def _empty_satellite_token_config(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(config_store, "read_config", lambda *a, **k: {})


def _stub_assistant():
    assistant = MagicMock()
    assistant.connect_satellite.return_value = queue.Queue()
    assistant.satellites = {}
    return assistant


def _hello(**overrides):
    message = {
        "type": "satellite.hello",
        "token": "device-token",
        "protocol": {"name": "satellite-v2", "major": 2, "minor": 2},
        "device": {"id": "kitchen-01", "name": "Kitchen"},
        "capabilities": {"audio_input": True, "audio_output": True},
    }
    message.update(overrides)
    return message


def _client(assistant):
    return TestClient(create_app(assistant))


def test_hello_returns_exact_welcome_audio_contract():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        welcome = ws.receive_json()

    assert welcome == {
        "type": "satellite.welcome",
        "session_id": welcome["session_id"],
        "protocol": {"major": 2, "minor": 2},
        "audio": {
            "uplink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1, "frame_duration_ms": 20},
            "downlink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
        },
        "heartbeat_interval_ms": 15000,
    }
    assert welcome["session_id"]
    assert assistant.connect_satellite.call_args.kwargs["device_id"] == "kitchen-01"


@pytest.mark.parametrize("message", [
    {"type": "audio.frame"},
    _hello(protocol={"name": "satellite-v2", "major": 1, "minor": 1}),
    _hello(protocol={"name": "satellite-v2", "major": 2, "minor": 3}),
])
def test_invalid_initial_protocol_is_rejected(message):
    with _client(_stub_assistant()).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(message)
        assert ws.receive_json()["type"] == "error"


def test_invalid_token_is_rejected(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(config_store, "read_config", lambda *a, **k: {"satellite_tokens": ["right-token"]})
    with _client(_stub_assistant()).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(token="wrong-token"))
        assert ws.receive_json()["code"] == "auth"


def test_only_640_byte_s16le_uplink_frames_are_accepted():
    assistant = _stub_assistant()
    chunk_q = assistant.connect_satellite.return_value
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        samples = np.array([1000, -1000] * 160, dtype="<i2")
        ws.send_bytes(samples.tobytes())

    pushed = chunk_q.get_nowait()
    assert np.allclose(pushed[:2], [1000 / 32768, -1000 / 32768])


def test_bad_binary_frame_and_oversized_control_are_rejected():
    with _client(_stub_assistant()).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        ws.send_bytes(b"bad")
        assert ws.receive_json()["code"] == "protocol"

    with _client(_stub_assistant()).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        ws.send_text(json.dumps({"type": "satellite.health", "padding": "x" * 2048}))
        assert ws.receive_json()["code"] == "protocol"


def test_health_report_receives_a_heartbeat_acknowledgement():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        ws.send_json({
            "type": "satellite.health",
            "dropped_uplink_frames": 0,
            "dropped_downlink_frames": 0,
            "capture_overruns": 0,
            "playback_underruns": 0,
        })
        assert ws.receive_json() == {"type": "satellite.heartbeat"}


def test_tts_is_s16le_bounded_and_completes_with_turn_id():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 22050))
        speaking = ws.receive_json()
        assert speaking["type"] == "assistant.state"
        assert speaking["state"] == "speaking"
        tts_q.put((np.ones(3000, dtype=np.float32), None))
        first_audio = ws.receive_json()
        assert first_audio == {"type": "tts.audio", "turn_id": speaking["turn_id"], "seq": 0, "bytes": 4096}
        assert len(ws.receive_bytes()) <= 4096
        assert ws.receive_json()["seq"] == 1
        assert len(ws.receive_bytes()) <= 4096
        tts_q.put(("end",))
        assert ws.receive_json() == {"type": "tts.end", "turn_id": speaking["turn_id"]}


def test_tts_cancel_uses_the_active_turn_id():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 22050))
        speaking = ws.receive_json()
        tts_q.put(("cancel",))
        assert ws.receive_json() == {"type": "tts.cancel", "turn_id": speaking["turn_id"]}


def test_tts_paces_existing_downlink_frames_after_initial_buffer(monkeypatch):
    import server.satellite_v2 as satellite_v2

    # Keep the test short while proving the sender leaves the initial burst and
    # begins rate-limiting its unchanged 4 KiB PCM frames.
    monkeypatch.setattr(satellite_v2, "SATELLITE_V2_DOWNLINK_INITIAL_BUFFER_SECONDS", 0.1)
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 22050))
        ws.receive_json()
        # Three existing 4 KiB wire frames. The first contains 128 ms of audio
        # and is sent immediately; later frames are paced at their playout rate.
        tts_q.put((np.ones(6144, dtype=np.float32), None))
        assert ws.receive_json()["type"] == "tts.audio"
        assert len(ws.receive_bytes()) == 4096
        assert ws.receive_json()["type"] == "tts.audio"
        assert len(ws.receive_bytes()) == 4096
        started_waiting = time.monotonic()
        assert ws.receive_json()["type"] == "tts.audio"
        assert len(ws.receive_bytes()) == 724

    assert time.monotonic() - started_waiting >= 0.05


def test_authoritative_lifecycle_events_are_forwarded_without_tts_queue_delay():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        listener = assistant.register_turn_listener.call_args.args[0]

        listener({
            "type": "assistant.state", "satellite_id": assistant.connect_satellite.call_args.args[0],
            "state": "wake_detected", "turn_id": "turn-123",
        })
        listener({
            "type": "assistant.state", "satellite_id": assistant.connect_satellite.call_args.args[0],
            "state": "listening", "turn_id": "turn-123",
        })

        assert ws.receive_json() == {
            "type": "assistant.state", "state": "wake_detected", "turn_id": "turn-123"
        }
        assert ws.receive_json() == {
            "type": "assistant.state", "state": "listening", "turn_id": "turn-123"
        }


def test_satellite_stop_requests_an_immediate_server_stop():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        ws.send_json({"type": "satellite.stop"})

    assistant.request_stop.assert_called_once()


def test_minor_one_keeps_raw_downlink_frames():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(protocol={"name": "satellite-v2", "major": 2, "minor": 1}))
        assert ws.receive_json()["protocol"] == {"major": 2, "minor": 1}
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 16000))
        ws.receive_json()
        tts_q.put((np.ones(20, dtype=np.float32), None))
        assert len(ws.receive_bytes()) == 40


def test_minor_two_reconnect_replays_requested_tts_frames():
    assistant = _stub_assistant()
    client = _client(assistant)
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        welcome = ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 16000))
        speaking = ws.receive_json()
        tts_q.put((np.ones(4096, dtype=np.float32), None))
        first_meta = ws.receive_json()
        first_frame = ws.receive_bytes()

    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(resume={
            "session_id": welcome["session_id"], "turn_id": speaking["turn_id"], "next_seq": first_meta["seq"],
        }))
        resumed = ws.receive_json()
        assert resumed["session_id"] == welcome["session_id"]
        assert ws.receive_json() == {"type": "assistant.state", "state": "speaking", "turn_id": speaking["turn_id"]}
        assert ws.receive_json() == first_meta
        assert ws.receive_bytes() == first_frame
        assert assistant.connect_satellite.call_count == 1


def test_minor_two_reconnect_delivers_terminal_event_queued_while_disconnected():
    assistant = _stub_assistant()
    client = _client(assistant)
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        welcome = ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 16000))
        speaking = ws.receive_json()

    tts_q.put(("end",))
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(resume={
            "session_id": welcome["session_id"], "turn_id": speaking["turn_id"], "next_seq": 0,
        }))
        ws.receive_json()  # welcome
        assert ws.receive_json() == {"type": "assistant.state", "state": "speaking", "turn_id": speaking["turn_id"]}
        assert ws.receive_json() == {"type": "tts.end", "turn_id": speaking["turn_id"]}
