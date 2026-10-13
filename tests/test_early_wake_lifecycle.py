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
    capture_ids=None,
    provisional_indexes=None,
    satellite_ids=None,
    before_each=None,
    audio_values=None,
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
        capture_id_sink=None,
    ):
        for index, _text in enumerate(transcripts):
            if before_each is not None:
                before_each(index)
            onset_sink["t"] = onsets[index] if onsets is not None else 10.0
            loudness_sink["db"] = -20.0
            provisional_sink["flag"] = (
                index < len(transcripts) - 1
                if provisional_indexes is None
                else index in provisional_indexes
            )
            audio_sink["buf"] = audio_values[index] if audio_values is not None else object()
            satellite_id_sink["id"] = satellite_ids[index] if satellite_ids is not None else "sat-a"
            endpoint_wait_sink["s"] = 0.0
            wake_probe_sink["flag"] = index in wake_probe_indexes
            if kws_candidate_sink is not None:
                kws_candidate_sink["flag"] = index in kws_candidate_indexes
            if kws_early_verification_sink is not None:
                kws_early_verification_sink["flag"] = index in early_verification_indexes
            if capture_id_sink is not None:
                capture_id_sink["value"] = capture_ids[index] if capture_ids is not None else None
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


def test_committed_capture_drops_repeated_partials_despite_rolling_onset_drift(assistant):
    command = "atticus turn on kitchen lights"
    _run_transcripts(
        assistant, [command, command, command], capture_ids=[1, 1, 1], onsets=[10, 20, 30],
    )
    assistant._start_turn.assert_called_once()
    assert assistant.satellites["sat-a"].committed_capture_id == 1
    assistant.audio_capture.flush.assert_not_called()


def test_prioritised_hard_final_retires_older_queued_live_partial(assistant):
    command = "atticus turn on kitchen lights"
    _run_transcripts(
        assistant, [command, command], capture_ids=[1, 1], provisional_indexes={1},
        onsets=[10.0, 12.0],
    )
    assistant._start_turn.assert_called_once()
    assert assistant.satellites["sat-a"].committed_capture_id == 1


def test_new_identical_capture_commits_immediately_without_cooldown(assistant):
    command = "atticus turn on kitchen lights"

    def check_first_commit(index):
        if index == 1:
            assistant._start_turn.assert_called_once()
            assert assistant.satellites["sat-a"].drop_results_until == 0

    _run_transcripts(
        assistant, [command, command, command], capture_ids=[1, 2, 2],
        onsets=[10, 10.1, 20], before_each=check_first_commit,
    )
    assert assistant._start_turn.call_count == 2
    assert assistant.satellites["sat-a"].committed_capture_id == 2


@pytest.mark.parametrize("final", ["", "yeah", "television noise", "atticus turn on kitchen lights"])
def test_empty_or_filtered_committed_final_cannot_suppress_next_command(assistant, final):
    command = "atticus turn on kitchen lights"
    _run_transcripts(assistant, [command, final], capture_ids=[1, 1])
    _run_transcripts(assistant, [command], capture_ids=[2], onsets=[10.1])
    assert assistant._start_turn.call_count == 2
    assert assistant.satellites["sat-a"].provisional_committed_onset == 0


def test_missing_final_does_not_block_next_safe_partial(assistant):
    command = "atticus turn on kitchen lights"
    for capture_id in (1, 2):
        _run_transcripts(assistant, [command], capture_ids=[capture_id], provisional_indexes={0})
    assert assistant._start_turn.call_count == 2


def test_retired_capture_stays_retired_after_a_new_capture_commits(assistant):
    command = "atticus turn on kitchen lights"
    _run_transcripts(assistant, [command, command, command], capture_ids=[1, 2, 1])
    assert assistant._start_turn.call_count == 2


@pytest.mark.parametrize("conversation_mode", [False, True])
def test_new_capture_stop_can_barge_in_after_early_commit(assistant, conversation_mode):
    sat = assistant.satellites["sat-a"]
    sat.conversation_mode = conversation_mode
    assistant._cancel_turn = MagicMock()
    # Conversation turns use a settle timer; replace only their dispatch seam.
    assistant._queue_conversation_turn = MagicMock()

    def speaking_after_commit(index):
        if index == 1:
            sat.turn_active = True
            assistant._is_speaking = MagicMock(return_value=True)

    _run_transcripts(
        assistant, ["atticus turn on kitchen lights", "atticus stop"],
        capture_ids=[1, 2], before_each=speaking_after_commit,
    )
    assistant._cancel_turn.assert_called_once_with("sat-a")
    assert sat.last_turn_end == 0


def test_capture_identity_is_per_satellite(assistant):
    assistant.satellites["sat-b"] = SatelliteSession(id="sat-b")
    command = "atticus turn on kitchen lights"
    _run_transcripts(
        assistant, [command, command], capture_ids=[1, 1], satellite_ids=["sat-a", "sat-b"],
    )
    assert assistant._start_turn.call_count == 2


def test_fresh_wakeword_free_follow_up_is_not_flushed_by_early_commit(assistant):
    def open_follow_up(index):
        if index == 1:
            assistant.satellites["sat-a"].last_turn_end = 10.0
            assistant.follow_up_seconds = 5.0

    _run_transcripts(
        assistant, ["atticus turn on kitchen lights", "turn off kitchen lights"],
        capture_ids=[1, 2], onsets=[10.0, 10.1], before_each=open_follow_up,
    )
    assert assistant._start_turn.call_count == 2
    assert assistant._start_turn.call_args.args[0] == "turn off kitchen lights"


def test_legacy_empty_final_clears_fallback_before_text_filtering(assistant):
    command = "atticus turn on kitchen lights"
    _run_transcripts(assistant, [command, ""])
    assert assistant.satellites["sat-a"].provisional_committed_onset == 0


@pytest.mark.parametrize("scoped", [False, True])
def test_transcriber_rechecks_live_result_cancelled_after_queueing(assistant, scoped):
    import numpy as np

    from core.asr_parakeet import LiveAsrResult, ParakeetLiveWorker

    worker = ParakeetLiveWorker(object())  # no inference should run
    assistant.audio_capture.live_asr_backend = worker
    command = "atticus turn on kitchen lights"
    result = LiveAsrResult(command, np.zeros(4), generation=0, capture_id=1)
    try:
        if scoped:
            worker.discard_capture("sat-a", 1)
        else:
            worker.discard("sat-a")
        _run_transcripts(assistant, [command], capture_ids=[1], audio_values=[result])
        assistant._start_turn.assert_not_called()
    finally:
        worker.close()


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


def test_early_verification_miss_remains_pending_without_dispatch(assistant):
    events = []
    assistant.register_turn_listener(events.append)
    assistant._on_wakeword_model_match("sat-a")

    _run_transcripts(
        assistant,
        ["television noise"],
        kws_candidate_indexes={0},
        early_verification_indexes={0},
    )

    assert [event["state"] for event in events] == ["wake_detected", "listening"]
    assert assistant.wakeword_metrics["rejected"] == 0
    assistant._start_turn.assert_not_called()


def test_early_miss_can_be_recovered_by_complete_wakeword(assistant):
    assistant._on_wakeword_model_match("sat-a")
    _run_transcripts(
        assistant,
        ["hey at", "atticus set a timer for six minutes"],
        kws_candidate_indexes={0, 1},
        early_verification_indexes={0},
    )
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.args[0] == "set a timer for six minutes"
    assert assistant.wakeword_metrics["rejected"] == 0


def test_early_miss_is_rejected_only_when_final_audio_also_misses(assistant):
    events = []
    assistant.register_turn_listener(events.append)
    assistant._on_wakeword_model_match("sat-a")
    _run_transcripts(
        assistant,
        ["hey at", "television noise"],
        kws_candidate_indexes={0, 1},
        early_verification_indexes={0},
    )
    assert [event["state"] for event in events] == ["wake_detected", "listening", "idle"]
    assert assistant.wakeword_metrics["rejected"] == 1
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
