"""Lifecycle timing for wakewords found in soft-endpoint transcripts."""

from unittest.mock import MagicMock, patch

import pytest

from core.satellite import SatelliteSession


@pytest.fixture
def assistant():
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        instance = Assistant(wakeword="atticus", barge_in="wakeword")
        instance.satellites["sat-a"] = SatelliteSession(id="sat-a")
        instance._start_turn = MagicMock()
        return instance


def _run_transcripts(assistant, transcripts, before_final=None, wake_probe_indexes=()):
    items = iter(transcripts)

    def stream_generator(
        _queue,
        onset_sink,
        loudness_sink,
        provisional_sink,
        audio_sink,
        satellite_id_sink,
        endpoint_wait_sink,
        wake_probe_sink,
    ):
        for index, _text in enumerate(transcripts):
            onset_sink["t"] = 10.0
            loudness_sink["db"] = -20.0
            provisional_sink["flag"] = index < len(transcripts) - 1
            audio_sink["buf"] = object()
            satellite_id_sink["id"] = "sat-a"
            endpoint_wait_sink["s"] = 0.0
            wake_probe_sink["flag"] = index in wake_probe_indexes
            yield object()

    def asr_pipe(stream, **_kwargs):
        for index in range(len(transcripts)):
            next(stream)
            if index == len(transcripts) - 1 and before_final is not None:
                before_final()
            yield {"text": next(items)}

    assistant.asr_stream_generator = stream_generator
    assistant.asr_pipe = asr_pipe
    assistant._run_transcriber_loop()


def test_soft_wake_is_emitted_before_final_transcript(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    def assert_listening_before_final():
        assert [event["state"] for event in events] == ["wake_detected", "listening"]
        assert assistant.satellites["sat-a"].protocol_wake_pending is True

    _run_transcripts(
        assistant,
        ["atticus", "atticus tell me a story"],
        before_final=assert_listening_before_final,
    )

    assert [event["state"] for event in events] == ["wake_detected", "listening", "thinking"]
    assert len({event["turn_id"] for event in events}) == 1
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "tell me a story"


def test_rejected_soft_wake_returns_satellite_to_idle(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(assistant, ["atticus", "television noise"])

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert assistant.satellites["sat-a"].protocol_turn_id is None
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assistant._start_turn.assert_not_called()


def test_bare_hard_wake_starts_lifecycle_before_follow_up(assistant):
    events = []
    assistant.register_turn_listener(events.append)
    assistant._verify_bare_wakeword = MagicMock(return_value=True)
    assistant._mark_turn_end = MagicMock()

    _run_transcripts(assistant, ["atticus"])

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assistant._mark_turn_end.assert_called_once_with("sat-a")


def test_speech_onset_wake_probe_only_emits_feedback(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(assistant, ["atticus"], wake_probe_indexes={0})

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assert assistant.satellites["sat-a"].protocol_wake_pending is True
    assistant._start_turn.assert_not_called()


def test_speech_onset_wake_probe_rejects_context_bias_echo(assistant):
    assistant.asr_context_terms = ["kiama"]
    assistant._asr_context_tokens = frozenset({"kiama"})
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(assistant, ["atticus kiama"], wake_probe_indexes={0})

    assert events == []
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assistant._start_turn.assert_not_called()


def test_follow_up_replaces_the_prior_protocol_turn_before_thinking(assistant):
    sat = assistant.satellites["sat-a"]
    sat.protocol_turn_id = "prior-turn"
    expiry = MagicMock()
    sat.protocol_follow_up_timer = expiry
    sat.last_turn_end = 9.0
    assistant.follow_up_seconds = 10.0
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(assistant, ["what time is it"])

    assert sat.protocol_turn_id != "prior-turn"
    expiry.cancel.assert_called_once()
    assert [event["state"] for event in events] == ["listening", "thinking"]
    assert len({event["turn_id"] for event in events}) == 1
