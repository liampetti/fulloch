"""Cross-satellite independence for turn/echo/follow-up/noise-baseline state.

These live on each satellite's own `SatelliteSession`, so a turn on satellite A
must never be visible to (or suppress/trigger) satellite B's own barge-in,
follow-up, or self-echo checks.
"""

from unittest.mock import MagicMock, patch

from core.satellite import SatelliteSession
from utils.phrases import BUSY_PHRASES


def _make_assistant(**kwargs):
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus", **kwargs)
    return a


def _connect(a, satellite_id, **overrides):
    overrides.setdefault("chunk_q", None)
    a.satellites[satellite_id] = SatelliteSession(id=satellite_id, **overrides)
    return a.satellites[satellite_id]


class TestIsSpeakingIndependence:
    def test_a_speaking_does_not_make_b_speak(self):
        a = _make_assistant()
        _connect(a, "sat-a", turn_active=True)
        _connect(a, "sat-b", turn_active=False)

        assert a._is_speaking("sat-a") is True
        assert a._is_speaking("sat-b") is False

    def test_unknown_satellite_is_not_speaking(self):
        a = _make_assistant()
        assert a._is_speaking("never-connected") is False


class TestSelfEchoIndependence:
    def test_a_spoken_text_does_not_suppress_b_self_echo_check(self):
        a = _make_assistant()
        _connect(a, "sat-a", turn_active=True, last_spoken_text="atticus is a local voice assistant")
        _connect(a, "sat-b", turn_active=True, last_spoken_text="")

        # B never said anything, so an utterance echoing A's speech must not
        # register as B's own self-echo — B has nothing to echo.
        assert a._is_self_echo("sat-b", "atticus is a local") is False
        # The same text against A (who actually said it) is a real echo.
        assert a._is_self_echo("sat-a", "atticus is a local") is True

    def test_check_barge_in_judged_per_satellite(self):
        a = _make_assistant()
        _connect(a, "sat-a", turn_active=True, last_spoken_text="it is sunny today")
        _connect(a, "sat-b", turn_active=False, last_spoken_text="it is sunny today")

        # A is mid-turn, so a real barge-in phrase fires for A...
        assert a._check_barge_in("sat-a", "atticus stop") is True
        # ...but B isn't speaking at all, so the same utterance is not a
        # barge-in for B regardless of what B's last_spoken_text says.
        assert a._check_barge_in("sat-b", "atticus stop") is False


class TestMarkTurnEndIndependence:
    def test_mark_turn_end_only_updates_the_named_satellite(self):
        a = _make_assistant(follow_up_time="5s")
        sat_a = _connect(a, "sat-a")
        sat_b = _connect(a, "sat-b")
        a.audio_capture.arm_follow_up = MagicMock()

        a._mark_turn_end("sat-a", None)

        assert sat_a.last_turn_end > 0.0
        assert sat_b.last_turn_end == 0.0
        # arm_follow_up was called for A's session, not B's.
        called_session = a.audio_capture.arm_follow_up.call_args[0][0]
        assert called_session is sat_a


class TestRequestStopAndGetStateScanAllSatellites:
    def test_get_state_reports_speaking_if_any_satellite_is(self):
        a = _make_assistant()
        _connect(a, "sat-a")
        sat_b = _connect(a, "sat-b")
        sat_b.tts_active.set()

        assert a.get_state() == "speaking"

    def test_request_stop_only_stops_the_active_satellite(self):
        from core.tts_session import TtsSession

        a = _make_assistant()
        sat_a = _connect(a, "sat-a")
        sat_b = _connect(a, "sat-b")
        a._turn_arbiter.try_acquire("sat-a")  # a real turn always holds this
        session_a = TtsSession()
        sat_a.active_session = session_a
        sat_b.active_session = None

        a.request_stop()

        assert session_a.cancelled is True
        assert sat_a.last_turn_end == 0.0
        # B was never active, so nothing about it should have been touched.
        assert sat_b.active_session is None
        assert sat_b.last_turn_end == 0.0


class TestTurnArbiterBusyBounce:
    """Phase 3: when two turn sources race, the loser gets an audible/text
    "busy" bounce on its own channel and the winner's turn is untouched.
    `_start_turn`'s arbiter check runs before anything else, so calling it
    against an already-held arbiter never reaches the SLM/TTS pipeline —
    safe to exercise directly without a loaded model.
    """

    def test_satellite_b_bounced_while_a_holds_the_arbiter(self):
        import queue

        a = _make_assistant()
        a._tts_module = MagicMock()
        a.busy_cache = [(["busy_chunk"], 24000)]
        sat_a = _connect(a, "sat-a", turn_active=True)
        sink_b: "queue.Queue" = queue.Queue()
        _connect(a, "sat-b", tts_sink=sink_b)
        a._turn_arbiter.try_acquire("sat-a")  # A's real turn already owns it

        a._start_turn("what time is it", satellite_id="sat-b")

        # The arbiter is untouched — A's turn is still the owner.
        assert a._turn_arbiter.owner == "sat-a"
        assert sat_a.turn_active is True
        # B never got a turn_thread — it was bounced before spawning one.
        assert a.satellites["sat-b"].turn_thread is None
        # The busy phrase played on B's own sink specifically.
        a._tts_module.play_chunks.assert_called_once()
        assert a._tts_module.play_chunks.call_args.kwargs["sink"] is sink_b

    def test_voice_turn_bounced_while_text_turn_holds_the_arbiter(self):
        a = _make_assistant()
        a._tts_module = MagicMock()
        a.busy_cache = [(["busy_chunk"], 24000)]
        _connect(a, "sat-a")
        a._turn_arbiter.try_acquire("dashboard-text")  # a text turn is mid-SLM-call

        a._start_turn("turn on the lights", satellite_id="sat-a")

        # Not interleaved into the text turn: sat-a's turn never started,
        # and the arbiter is still held by the text turn.
        assert a._turn_arbiter.owner == "dashboard-text"
        assert a.satellites["sat-a"].turn_active is False
        assert a.satellites["sat-a"].turn_thread is None

    def test_text_turn_bounced_while_a_satellite_holds_the_arbiter(self):
        a = _make_assistant()
        a.models_ready.set()  # skip handle_text_turn's startup wait
        _connect(a, "sat-a")
        a._turn_arbiter.try_acquire("sat-a")  # a voice turn is mid-SLM-call
        history_before = list(a._history)

        reply = a.handle_text_turn("what's the weather")

        # Bounced with a busy phrase, not interleaved into the voice turn's
        # history write — handle_text_turn returned before touching _history.
        assert reply in BUSY_PHRASES
        assert a._history == history_before
        assert a._turn_arbiter.owner == "sat-a"


class TestDisconnectMidTurnReleasesArbiter:
    def test_disconnect_releases_the_arbiter_and_frees_it_for_another_satellite(self):
        import queue

        a = _make_assistant()
        _connect(a, "sat-a", chunk_q=queue.Queue())
        a._turn_arbiter.try_acquire("sat-a")

        a.disconnect_satellite("sat-a")

        assert a._turn_arbiter.owner is None
        assert a._turn_arbiter.try_acquire("sat-b") is True
