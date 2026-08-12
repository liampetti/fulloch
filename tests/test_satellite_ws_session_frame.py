"""Phase 4 (#13) forward-compat hook: `/ws/satellite` tells the connecting
browser its own `satellite_id` immediately on connect, via a `{"type":
"session", "satellite_id": ...}` frame — shape-aligned with the
satellite-v2 protocol's `session.started` frame (Phase 5). The dashboard UI
uses this to tell whether a busy turn (from `/status`'s `active_owner_id`)
is its own or another satellite's.
"""

import queue
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from server.dashboard import create_app


def _stub_assistant(barge_in="off"):
    a = MagicMock()
    a.connect_satellite.return_value = queue.Queue()
    a.satellites = {}
    a.barge_in = barge_in
    return a


def test_session_frame_is_the_first_message_on_connect():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "session"
    assert isinstance(msg["satellite_id"], str) and msg["satellite_id"]


def test_connect_satellite_called_with_the_same_id_sent_to_the_browser():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        msg = ws.receive_json()

    sent_id = a.connect_satellite.call_args[0][0]
    assert sent_id == msg["satellite_id"]


def test_session_marks_default_mode_as_half_duplex():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        msg = ws.receive_json()

    assert msg["half_duplex"] is True


def test_session_keeps_microphone_live_for_wakeword_barge_in():
    a = _stub_assistant(barge_in="wakeword")
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        msg = ws.receive_json()

    assert msg["half_duplex"] is False


def test_disconnect_tears_down_the_same_satellite_id():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        msg = ws.receive_json()

    a.disconnect_satellite.assert_called_once_with(msg["satellite_id"])


def test_area_query_param_is_forwarded_to_connect_satellite():
    """6b: the browser's one-time area-picker choice arrives as `?area=`."""
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite?area=kitchen"):
        pass

    assert a.connect_satellite.call_args.kwargs["ha_area"] == "kitchen"


def test_missing_area_query_param_forwards_none():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite"):
        pass

    assert a.connect_satellite.call_args.kwargs["ha_area"] is None


def test_tts_audio_sends_into_the_initial_browser_playback_buffer():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        ws.receive_json()
        tts_q = a.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 24000))
        assert ws.receive_json() == {"type": "tts_start", "sr": 24000}
        tts_q.put((np.ones(2400, dtype=np.float32), None))
        assert len(ws.receive_bytes()) == 2400 * 4


def test_tts_does_not_send_end_before_all_pcm():
    a = _stub_assistant()
    client = TestClient(create_app(a))

    with client.websocket_connect("/ws/satellite") as ws:
        ws.receive_json()
        tts_q = a.set_satellite_sink.call_args.args[1]
        tts_q.put(("start", 24000))
        assert ws.receive_json() == {"type": "tts_start", "sr": 24000}
        tts_q.put((np.ones(12000, dtype=np.float32), None))
        tts_q.put(("end",))

        for _ in range(5):
            assert len(ws.receive_bytes()) == 2400 * 4
        assert ws.receive_json() == {"type": "tts_end"}
