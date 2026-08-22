"""Wire-contract tests for the ESP32 satellite-v2 endpoint."""

import json
import queue
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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
        "protocol": {"name": "satellite-v2", "major": 2, "minor": 4},
        "device": {"id": "kitchen-01", "name": "Kitchen"},
        "capabilities": {"audio_input": True, "audio_output": True, "conversation_mode_control": True},
    }
    message.update(overrides)
    return message


def _stereo_hello(**overrides):
    message = _hello(protocol={"name": "satellite-v2", "major": 2, "minor": 5})
    message["capabilities"]["aec_uplink_channels"] = [1, 2]
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
        "protocol": {"major": 2, "minor": 4},
        "audio": {
            "uplink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1, "frame_duration_ms": 20},
            "downlink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
        },
    }
    assert welcome["session_id"]
    assert assistant.connect_satellite.call_args.kwargs["device_id"] == "kitchen-01"


@pytest.mark.parametrize("message", [
    {"type": "audio.frame"},
    _hello(protocol={"name": "satellite-v2", "major": 1, "minor": 1}),
    _hello(protocol={"name": "satellite-v2", "major": 2, "minor": 2}),
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


def test_v25_negotiates_stereo_only_when_both_sides_opt_in(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(config_store, "read_config", lambda *a, **k: {"satellite": {"uplink_channels": 2}})
    assistant = _stub_assistant()
    chunk_q = assistant.connect_satellite.return_value
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_stereo_hello())
        welcome = ws.receive_json()
        assert welcome["protocol"] == {"major": 2, "minor": 5}
        assert welcome["audio"]["uplink"]["channels"] == 2
        ws.send_bytes(np.array([1000, 500, -1000, 400] * 160, dtype="<i2").tobytes())

    pushed = chunk_q.get_nowait()
    assert np.allclose(pushed[:2], [[1000 / 32768, 500 / 32768], [-1000 / 32768, 400 / 32768]])


def test_v25_stereo_preference_falls_back_to_mono_without_client_capability(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(config_store, "read_config", lambda *a, **k: {"satellite": {"uplink_channels": 2}})
    with _client(_stub_assistant()).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        welcome = ws.receive_json()
    assert welcome["protocol"] == {"major": 2, "minor": 4}
    assert welcome["audio"]["uplink"]["channels"] == 1


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


def test_server_health_challenge_keeps_an_idle_satellite_connected(monkeypatch):
    import server.satellite_v2 as satellite_v2

    monkeypatch.setattr(satellite_v2, "SATELLITE_V2_HEALTH_CHALLENGE_SECONDS", 0.01)
    monkeypatch.setattr(satellite_v2, "SATELLITE_V2_HEALTH_RESPONSE_SECONDS", 0.1)
    assistant = _stub_assistant()
    chunk_q = assistant.connect_satellite.return_value
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        request = ws.receive_json()
        assert request["type"] == "satellite.health_request"
        ws.send_json({
            "type": "satellite.health_response", "id": request["id"],
            "dropped_uplink_frames": 0, "dropped_downlink_frames": 0,
            "capture_overruns": 0, "playback_underruns": 0,
        })
        ws.send_bytes(np.zeros(320, dtype="<i2").tobytes())

    assert chunk_q.get_nowait().shape == (320,)


def test_server_disconnects_after_three_unanswered_health_challenges(monkeypatch):
    import server.satellite_v2 as satellite_v2

    monkeypatch.setattr(satellite_v2, "SATELLITE_V2_HEALTH_CHALLENGE_SECONDS", 0.01)
    monkeypatch.setattr(satellite_v2, "SATELLITE_V2_HEALTH_RESPONSE_SECONDS", 0.01)
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        for _ in range(3):
            assert ws.receive_json()["type"] == "satellite.health_request"
        # The third miss is recorded only after its response deadline expires.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert assistant.disconnect_satellite.called


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


def test_busy_response_stands_down_only_after_tts_ends():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        listener = assistant.register_turn_listener.call_args.args[0]
        satellite_id = assistant.connect_satellite.call_args.args[0]
        tts_q = assistant.set_satellite_sink.call_args.args[1]

        listener({
            "type": "assistant.state", "satellite_id": satellite_id,
            "state": "listening", "turn_id": "turn-123",
        })
        assert ws.receive_json() == {
            "type": "assistant.state", "state": "listening", "turn_id": "turn-123"
        }
        tts_q.put(("start", 16000))
        assert ws.receive_json() == {
            "type": "assistant.state", "state": "speaking", "turn_id": "turn-123"
        }
        listener({
            "type": "assistant.stand_down_after_tts", "satellite_id": satellite_id,
            "turn_id": "turn-123",
        })
        tts_q.put(("end",))

        assert ws.receive_json() == {"type": "tts.end", "turn_id": "turn-123"}
        assert ws.receive_json() == {
            "type": "assistant.state", "state": "idle", "turn_id": "turn-123"
        }


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


def test_lifecycle_change_cancels_active_tts_before_forwarding_state():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        listener = assistant.register_turn_listener.call_args.args[0]
        satellite_id = assistant.connect_satellite.call_args.args[0]

        tts_q.put(("start", 16000))
        speaking = ws.receive_json()
        tts_q.put((np.ones(1280, dtype=np.float32), None))
        listener({
            "type": "assistant.state", "satellite_id": satellite_id,
            "state": "listening", "turn_id": "next-turn",
        })

        assert ws.receive_json() == {"type": "tts.cancel", "turn_id": speaking["turn_id"]}
        assert ws.receive_json() == {
            "type": "assistant.state", "state": "listening", "turn_id": "next-turn"
        }


def test_satellite_stop_requests_an_immediate_server_stop():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        ws.send_json({"type": "satellite.stop"})

    assistant.request_stop.assert_called_once()


def test_conversation_mode_controls_require_declared_capability():
    assistant = _stub_assistant()
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(capabilities={"audio_input": True, "audio_output": True}))
        ws.receive_json()
        ws.send_json({"type": "conversation_mode.enable"})
        assert ws.receive_json() == {
            "type": "error", "code": "protocol",
            "message": "satellite did not declare conversation mode control",
        }


def test_conversation_mode_controls_return_authoritative_state():
    assistant = _stub_assistant()
    session = MagicMock()
    session.conversation_mode = True
    assistant.satellites = {}
    assistant.set_satellite_conversation_mode.return_value = (True, "")
    with _client(assistant).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        ws.receive_json()
        satellite_id = assistant.connect_satellite.call_args.args[0]
        assistant.satellites[satellite_id] = session
        assistant.conversation_owner_id = satellite_id
        ws.send_json({"type": "conversation_mode.enable"})
        assert ws.receive_json() == {
            "type": "conversation_mode.changed", "enabled": True, "owner": True, "message": ""
        }
        assistant.set_satellite_conversation_mode.assert_called_once_with(satellite_id, True)


def test_reconnect_replays_requested_tts_frames():
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


def test_reconnect_rejects_a_cursor_outside_the_replay_range():
    assistant = _stub_assistant()
    client = _client(assistant)
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello())
        welcome = ws.receive_json()
        tts_q = assistant.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 16000))
        speaking = ws.receive_json()
        tts_q.put((np.ones(4096, dtype=np.float32), None))
        ws.receive_json()
        ws.receive_bytes()

    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(_hello(resume={
            "session_id": welcome["session_id"], "turn_id": speaking["turn_id"], "next_seq": 999,
        }))
        assert ws.receive_json() == {
            "type": "error", "code": "protocol", "message": "resume next_seq is outside the replay range"
        }


def test_reconnect_delivers_terminal_event_queued_while_disconnected():
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
