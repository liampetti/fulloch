"""`/ws/satellite-v2` — a documented WebSocket protocol for a non-browser client,
translating the same internal TTS-sink-tuple contract `/ws/satellite` uses into
the v2 JSON/binary frame shapes. Pins the exact frame shapes so any drift
fails CI.
"""

import queue
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from server.dashboard import create_app


def _stub_assistant():
    a = MagicMock()
    a.connect_satellite.return_value = queue.Queue()
    a.satellites = {}
    return a


def _client(a):
    return TestClient(create_app(a))


def test_session_start_returns_session_started_with_satellite_id():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        msg = ws.receive_json()
    assert msg["type"] == "session.started"
    assert isinstance(msg["satellite_id"], str) and msg["satellite_id"]


def test_connect_satellite_receives_the_session_start_fields():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "label": "kitchen",
                "ha_area": "kitchen",
                "server_vad": False,
                "always_listen": True,
            }
        )
        msg = ws.receive_json()

    args, kwargs = a.connect_satellite.call_args
    assert args[0] == msg["satellite_id"]
    assert kwargs["wakeword_bypass"] is True
    assert kwargs["label"] == "kitchen"
    assert kwargs["ha_area"] == "kitchen"
    assert kwargs["server_vad"] is False


def test_non_session_start_first_message_is_rejected():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "audio.frame"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "protocol"


def test_no_tokens_configured_accepts_unauthenticated(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(config_store, "read_config", lambda *a, **k: {})
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        msg = ws.receive_json()
    assert msg["type"] == "session.started"


def test_auth_rejects_wrong_token(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(
        config_store, "read_config", lambda *a, **k: {"satellite_tokens": ["right-token"]}
    )
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start", "auth_token": "wrong-token"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "auth"


def test_auth_rejects_missing_token_when_tokens_configured(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(
        config_store, "read_config", lambda *a, **k: {"satellite_tokens": ["right-token"]}
    )
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "auth"


def test_auth_accepts_right_token(monkeypatch):
    import server.config_store as config_store

    monkeypatch.setattr(
        config_store, "read_config", lambda *a, **k: {"satellite_tokens": ["right-token"]}
    )
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start", "auth_token": "right-token"})
        msg = ws.receive_json()
    assert msg["type"] == "session.started"


def test_tts_sink_tuples_translate_to_v2_frame_sequence():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        ws.receive_json()  # session.started

        tts_q = a.set_satellite_sink.call_args[0][1]
        tts_q.put(("start", 24000))
        assert ws.receive_json() == {"type": "turn.tts_start", "sample_rate": 24000}

        chunk = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        tts_q.put((chunk, None))
        raw = ws.receive_bytes()
        assert np.allclose(np.frombuffer(raw, dtype=np.float32), chunk)

        tts_q.put(("end",))
        assert ws.receive_json() == {"type": "turn.tts_end"}


def test_cancel_sink_tuple_translates_to_turn_tts_cancel():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        ws.receive_json()

        tts_q = a.set_satellite_sink.call_args[0][1]
        tts_q.put(("start", 24000))
        ws.receive_json()
        tts_q.put(("cancel",))
        assert ws.receive_json() == {"type": "turn.tts_cancel"}


def test_binary_audio_frame_is_pushed_to_chunk_q():
    a = _stub_assistant()
    chunk_q = a.connect_satellite.return_value
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        ws.receive_json()
        pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        ws.send_bytes(pcm.tobytes())

    pushed = chunk_q.get_nowait()
    assert np.allclose(pushed, pcm)


def test_debug_json_audio_frame_is_pushed_to_chunk_q():
    import base64

    a = _stub_assistant()
    chunk_q = a.connect_satellite.return_value
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        ws.receive_json()
        pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        ws.send_json({"type": "audio.frame", "data": base64.b64encode(pcm.tobytes()).decode()})

    pushed = chunk_q.get_nowait()
    assert np.allclose(pushed, pcm)


def test_audio_flush_pushes_the_flush_sentinel():
    from core.audio import FLUSH

    a = _stub_assistant()
    chunk_q = a.connect_satellite.return_value
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start", "server_vad": False})
        ws.receive_json()
        ws.send_json({"type": "audio.flush"})

    assert chunk_q.get_nowait() is FLUSH


def test_wake_word_toggle_calls_set_satellite_wakeword():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        ws.receive_json()
        ws.send_json({"type": "wake_word.disable"})
        ws.send_json({"type": "wake_word.enable"})

    a.set_satellite_wakeword.assert_any_call(True)
    a.set_satellite_wakeword.assert_any_call(False)


def test_session_stop_ends_the_connection_cleanly():
    a = _stub_assistant()
    with _client(a).websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json({"type": "session.start"})
        satellite_id = ws.receive_json()["satellite_id"]
        ws.send_json({"type": "session.stop"})

    a.disconnect_satellite.assert_called_once_with(satellite_id)
