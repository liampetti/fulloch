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


def _run_transcripts(
    assistant,
    transcripts,
    before_final=None,
    onsets=None,
    wake_probe_indexes=(),
    kws_candidate_indexes=(),
    early_verification_indexes=(),
):
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
        kws_candidate_sink=None,
        kws_early_verification_sink=None,
    ):
        for index, _text in enumerate(transcripts):
            onset_sink["t"] = onsets[index] if onsets is not None else 10.0
            loudness_sink["db"] = -20.0
            provisional_sink["flag"] = index < len(transcripts) - 1
            audio_sink["buf"] = object()
            satellite_id_sink["id"] = "sat-a"
            endpoint_wait_sink["s"] = 0.0
            wake_probe_sink["flag"] = index in wake_probe_indexes
            if kws_candidate_sink is not None:
                kws_candidate_sink["flag"] = index in kws_candidate_indexes
            if kws_early_verification_sink is not None:
                kws_early_verification_sink["flag"] = index in early_verification_indexes
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

    # Thinking is emitted by the turn worker after it owns the arbiter; this
    # transcriber test replaces that worker with a mock.
    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assert len({event["turn_id"] for event in events}) == 1
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "tell me a story"


def test_delayed_hard_endpoint_drops_matching_committed_soft_turn(assistant):
    _run_transcripts(
        assistant,
        ["atticus turn off downstairs", "atticus turn off downstairs"],
        # A VAD reset/flush can shift the final segment's onset. The text match
        # must still consume its authoritative endpoint even after a slow tail.
        onsets=[10.0, 11.0],
    )

    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "turn off downstairs"


def test_rolling_tail_partial_keeps_pending_wake_until_final_verifies(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(
        assistant,
        ["atticus", "turn on the kitchen lights", "atticus turn on the kitchen lights"],
    )

    # The second partial no longer contains the wakeword, but it must not
    # stand down or dispatch before the full endpoint confirms the capture.
    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "turn on the kitchen lights"


def test_repeated_wakewords_route_the_latest_command(assistant):
    _run_transcripts(assistant, ["atticus, atticus, dim upstairs lights"])

    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "dim upstairs lights"


def test_rejected_soft_wake_returns_satellite_to_idle(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    _run_transcripts(assistant, ["atticus", "television noise"])

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert assistant.satellites["sat-a"].protocol_turn_id is None
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assistant._start_turn.assert_not_called()


def test_model_wake_is_emitted_before_asr_and_can_be_rejected(assistant):
    events = []
    assistant.register_turn_listener(events.append)

    assistant._on_wakeword_model_match("sat-a")

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assert assistant.satellites["sat-a"].protocol_wake_pending is True

    assistant._reject_pending_satellite_wake(assistant.satellites["sat-a"])

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert assistant.satellites["sat-a"].protocol_turn_id is None


def test_unresolved_model_wake_times_out_to_idle(assistant, monkeypatch):
    class Timer:
        instances = []

        def __init__(self, _seconds, callback):
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            self.instances.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("core.assistant.threading.Timer", Timer)
    events = []
    assistant.register_turn_listener(events.append)

    assistant._on_wakeword_model_match("sat-a")
    Timer.instances[-1].callback()

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert assistant.satellites["sat-a"].protocol_wake_pending is False
    assert assistant.satellites["sat-a"].protocol_turn_id is None


def test_early_verification_rejection_stands_down_without_dispatch(assistant):
    events = []
    assistant.register_turn_listener(events.append)
    assistant._on_wakeword_model_match("sat-a")

    _run_transcripts(
        assistant,
        ["television noise"],
        kws_candidate_indexes={0},
        early_verification_indexes={0},
    )

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assistant._start_turn.assert_not_called()


def test_accepted_early_verification_dispatches_the_final_query(assistant):
    events = []
    assistant.register_turn_listener(events.append)
    assistant._on_wakeword_model_match("sat-a")

    _run_transcripts(
        assistant,
        ["atticus", "atticus set a timer for six minutes"],
        kws_candidate_indexes={0, 1},
        early_verification_indexes={0},
    )

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "set a timer for six minutes"


def test_openwakeword_can_dispatch_without_asr_wakeword_verification(assistant):
    assistant.verify_asr_wakeword = False

    _run_transcripts(assistant, ["turn off the office lights"], kws_candidate_indexes={0})

    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "turn off the office lights"


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
    assert [event["state"] for event in events] == ["listening"]
    assert len({event["turn_id"] for event in events}) == 1


def test_dropping_ordinary_work_keeps_a_pending_wake(assistant):
    sat = assistant.satellites["sat-a"]
    events = []
    assistant.register_turn_listener(events.append)

    assistant._on_wakeword_model_match("sat-a")
    assistant._on_asr_work_dropped("sat-a", "final", False, "evicted")

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assert sat.protocol_wake_pending is True
    assert sat.protocol_turn_id is not None


def test_dropping_wake_candidate_stands_down_pending_wake(assistant):
    sat = assistant.satellites["sat-a"]
    events = []
    assistant.register_turn_listener(events.append)

    assistant._on_wakeword_model_match("sat-a")
    assistant._on_asr_work_dropped("sat-a", "wake_candidate", True, "full")

    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert sat.protocol_wake_pending is False
    assert sat.protocol_turn_id is None
