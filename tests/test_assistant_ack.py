"""Acknowledgements reach the calling satellite and finish before the reply."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.assistant import Assistant
from core.satellite import SatelliteSession
from core.tts_session import TtsSession
from core.turn_arbiter import TurnArbiter


@pytest.mark.parametrize("fails", [False, True])
def test_ack_uses_explicit_sink_and_restores_thread_local_state(fails):
    old_sink, sink = queue.Queue(), queue.Queue()
    old_active, active = threading.Event(), threading.Event()
    session = TtsSession()
    local = threading.local()
    local.sink, local.tts_active_event = old_sink, old_active
    observed = []

    def play(chunks, rate, **kwargs):
        observed.append((chunks, rate, kwargs["session"], local.sink, local.tts_active_event))
        if fails:
            raise RuntimeError("playback failed")

    host = SimpleNamespace(
        ack_cache=[(["ack"], 24000)], _turn_local=local, play_chunks=play,
    )
    Assistant._play_random_ack(host, session, sink=sink, tts_active_event=active)

    assert observed == [(["ack"], 24000, session, sink, active)]
    assert local.sink is old_sink
    assert local.tts_active_event is old_active


@pytest.mark.parametrize("cache", [None, [], [(["alternate"], 16000)]])
def test_ack_cache_selection(cache):
    session = TtsSession()
    host = SimpleNamespace(
        ack_cache=[(["default"], 24000)], _turn_local=threading.local(), play_chunks=Mock(),
    )
    Assistant._play_random_ack(host, session, cache=cache)
    if cache == []:
        host.play_chunks.assert_not_called()
    else:
        chunks, rate = (cache if cache is not None else host.ack_cache)[0]
        host.play_chunks.assert_called_once_with(chunks, rate, session=session)


@pytest.mark.parametrize("mode", ["half_duplex", "barge_in"])
@pytest.mark.parametrize("needs_model", [False, True])
def test_voice_turn_ack_thread_captures_satellite_sink(mode, needs_model):
    assistant = Assistant.__new__(Assistant)
    satellite = SatelliteSession(id="kitchen", chunk_q=queue.Queue(), tts_sink=queue.Queue())
    assistant.satellites = {satellite.id: satellite}
    assistant._turn_local = threading.local()
    assistant._turn_arbiter = TurnArbiter()
    assistant._tts_backend = "qwen"
    assistant.voice_clone_prompt = None
    assistant.audio_capture = Mock()
    assistant._maybe_reset_session = Mock()
    assistant._satellite_thinking = Mock()
    assistant._emit_turn_event = Mock()
    assistant._prepare_delivery_request = Mock(side_effect=lambda text, sat: text)
    assistant._mark_turn_end = Mock()
    assistant._note_runtime_error = Mock()
    parent_thread = threading.get_ident()
    events = []

    def ack(session, *, sink, tts_active_event):
        events.append(("ack", threading.get_ident(), session, tts_active_event))
        sink.put(("ack",))

    def handle(prompt, **kwargs):
        if needs_model:
            kwargs["on_slm_start"]()
        return "The answer."

    assistant._play_random_ack = Mock(side_effect=ack)
    assistant._handle_wakeword = Mock(side_effect=handle)
    assistant.speak_stream = Mock(side_effect=lambda *args, **kwargs: events.append(("reply",)))

    if mode == "half_duplex":
        assistant._run_half_duplex("A question", satellite_id=satellite.id)
    else:
        assistant._turn_arbiter.try_acquire(satellite.id)
        assistant._run_turn("A question", TtsSession(), satellite_id=satellite.id)

    assistant._note_runtime_error.assert_not_called()
    assistant.speak_stream.assert_called_once()
    if needs_model:
        assistant._play_random_ack.assert_called_once()
        assert [event[0] for event in events] == ["ack", "reply"]
        assert events[0][1] != parent_thread
        assert events[0][2] is assistant.speak_stream.call_args.kwargs["session"]
        assert events[0][3] is satellite.tts_active
        assert satellite.tts_sink.get_nowait() == ("ack",)
    else:
        assistant._play_random_ack.assert_not_called()
        assert events == [("reply",)]
        assert satellite.tts_sink.empty()
