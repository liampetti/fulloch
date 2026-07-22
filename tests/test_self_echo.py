"""Tests for self-echo suppression in barge-in detection.

Targets the symptom where the assistant interrupts its own TTS because AEC
residue leaked the wakeword (or a near-homophone of it) back through ASR.
The unit under test is `Assistant._is_self_echo` plus the integration with
`_check_barge_in`. Both can be exercised against a bare instance — no
models loaded — because the relevant code paths only touch in-process
per-satellite state (`SatelliteSession.last_spoken_text`/`turn_active`) plus
`barge_in`. Every fixture here uses a single satellite keyed `"sat-a"`.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.satellite import SatelliteSession  # noqa: E402


def _add_satellite(a, satellite_id="sat-a", turn_active=True):
    """Register a bare SatelliteSession on `a`, standing in for a real
    `connect_satellite` call (which would open a real recorder thread)."""
    a.satellites[satellite_id] = SatelliteSession(
        id=satellite_id, chunk_q=None, turn_active=turn_active
    )


@pytest.fixture
def assistant():
    """Bare `Assistant` instance with AudioCapture mocked out.

    AudioCapture probes the host audio system at __init__ time (via
    sounddevice). We don't need that here, so patch it before construction.
    """
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(
            wakeword="atticus",
            barge_in="wakeword",
        )
        _add_satellite(a)  # pretend a turn is running so checks engage
        return a


class TestIsSelfEcho:
    def test_no_recent_spoken_text_means_not_echo(self, assistant):
        assistant.satellites["sat-a"].last_spoken_text = ""
        assert assistant._is_self_echo("sat-a", "atticus what time is it") is False

    def test_multi_word_overlap_above_threshold(self, assistant):
        # Assistant said this; mic picks up most of it as a barge-in.
        assistant.satellites["sat-a"].last_spoken_text = "atticus is a local voice assistant"
        assert assistant._is_self_echo("sat-a", "atticus is a local") is True

    def test_multi_word_overlap_below_threshold(self, assistant):
        # User's real request shares only stop-words with the prior answer.
        assistant.satellites["sat-a"].last_spoken_text = "the time is half past four"
        assert assistant._is_self_echo("sat-a", "play some jazz music for me") is False

    def test_single_word_wakeword_in_recent_speech_is_echo(self, assistant):
        # The assistant's answer mentioned "atticus" — a stray single-word
        # "atticus" transcript is almost certainly AEC residue.
        assistant.satellites["sat-a"].last_spoken_text = "atticus is a great narrator"
        assert assistant._is_self_echo("sat-a", "atticus") is True

    def test_single_word_wakeword_not_in_recent_speech_is_not_echo(self, assistant):
        # The prior answer didn't mention the wakeword, so an "atticus" transcript
        # is a legitimate barge-in attempt.
        assistant.satellites["sat-a"].last_spoken_text = "it is sunny and twenty two degrees"
        assert assistant._is_self_echo("sat-a", "atticus") is False


class TestCheckBargeIn:
    def test_does_not_barge_when_self_echo(self, assistant):
        assistant.satellites["sat-a"].last_spoken_text = "atticus is a local voice assistant"
        assert assistant._check_barge_in("sat-a", "atticus is a local") is False

    def test_barges_on_real_user_utterance(self, assistant):
        assistant.satellites["sat-a"].last_spoken_text = "it is sunny and twenty two degrees"
        assert assistant._check_barge_in("sat-a", "atticus stop") is True

    def test_does_not_barge_when_turn_inactive(self, assistant):
        assistant.satellites["sat-a"].turn_active = False
        assistant.satellites["sat-a"].last_spoken_text = "it is sunny"
        assert assistant._check_barge_in("sat-a", "atticus stop") is False

    def test_does_not_barge_when_barge_in_off(self, assistant):
        assistant.barge_in = "off"
        assistant.satellites["sat-a"].last_spoken_text = "it is sunny"
        assert assistant._check_barge_in("sat-a", "atticus stop") is False

    def test_conversation_mode_interrupts_natural_speech(self, assistant):
        assistant.barge_in = "off"
        assistant.satellites["sat-a"].conversation_mode = True
        assistant.satellites["sat-a"].last_spoken_text = "it is sunny"
        assert assistant._check_barge_in("sat-a", "actually check tomorrow") is True


class TestBargeInTranscriptDispatch:
    """A non-stop interruption must become the next request, not be discarded."""

    @staticmethod
    def _run_transcript(a, text):
        def stream_generator(
            _queue,
            onset_sink,
            loudness_sink,
            provisional_sink,
            audio_sink,
            satellite_id_sink,
            endpoint_wait_sink,
        ):
            onset_sink["t"] = 1.0
            loudness_sink["db"] = -30.0
            provisional_sink["flag"] = False
            audio_sink["buf"] = object()
            satellite_id_sink["id"] = "sat-a"
            endpoint_wait_sink["s"] = 0.0
            yield object()

        def asr_pipe(stream, **_kwargs):
            next(stream)
            return iter([{"text": text}])

        a.asr_stream_generator = stream_generator
        a.asr_pipe = asr_pipe
        a._run_transcriber_loop()

    def test_wakeword_barge_in_dispatches_its_command(self, assistant):
        assistant._cancel_turn = MagicMock()
        assistant._start_turn = MagicMock()
        assistant._verify_strict_barge_wakeword = MagicMock(return_value=True)

        self._run_transcript(assistant, "atticus, tell me more about that thing")

        assistant._cancel_turn.assert_called_once_with("sat-a")
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.args[0] == "tell me more about that thing"

    def test_conversation_barge_in_dispatches_its_command(self, assistant, monkeypatch):
        import time

        monkeypatch.setattr("core.assistant.CONVERSATION_TURN_SETTLE_S", 0.01)
        assistant.barge_in = "off"
        assistant.satellites["sat-a"].conversation_mode = True
        assistant._cancel_turn = MagicMock()
        assistant._start_turn = MagicMock()
        assistant._run_half_duplex = MagicMock()

        self._run_transcript(assistant, "tell me more about that thing")
        time.sleep(0.05)

        assistant._cancel_turn.assert_called_once_with("sat-a")
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.args[0] == "tell me more about that thing"
        assistant._run_half_duplex.assert_not_called()


class TestConversationTurnSettle:
    def test_only_latest_rapid_request_starts_a_turn(self, assistant, monkeypatch):
        import time

        monkeypatch.setattr("core.assistant.CONVERSATION_TURN_SETTLE_S", 0.01)
        assistant.satellites["sat-a"].conversation_mode = True
        assistant._start_turn = MagicMock()

        assistant._queue_conversation_turn("first", "sat-a")
        assistant._queue_conversation_turn("second", "sat-a")
        assistant._queue_conversation_turn("last", "sat-a")
        time.sleep(0.05)

        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.args[0] == "last"

    def test_stop_cancels_an_unstarted_request(self, assistant, monkeypatch):
        import time

        monkeypatch.setattr("core.assistant.CONVERSATION_TURN_SETTLE_S", 0.05)
        sat = assistant.satellites["sat-a"]
        sat.conversation_mode = True
        assistant._start_turn = MagicMock()

        assistant._queue_conversation_turn("do not run", "sat-a")
        assistant._clear_pending_conversation_turn(sat)
        time.sleep(0.1)

        assistant._start_turn.assert_not_called()


class TestStrictBargeVerification:
    def test_strict_wakeword_uses_unbiased_verification(self, assistant, monkeypatch):
        verified = []
        monkeypatch.setattr(
            assistant,
            "_verify_bare_wakeword",
            lambda loudness, baseline: verified.append((loudness, baseline)) or False,
        )

        assert assistant._verify_strict_barge_wakeword("hey atticus kiama", -38, -60) is False
        assert verified == [(-38, -60)]

    def test_stop_skips_strict_verification(self, assistant, monkeypatch):
        monkeypatch.setattr(
            assistant,
            "_verify_bare_wakeword",
            lambda *_args: (_ for _ in ()).throw(AssertionError("should not verify")),
        )

        assert assistant._verify_strict_barge_wakeword("atticus stop", -38, -60) is True

    def test_bare_name_redirect_skips_strict_verification(self, multiword_assistant, monkeypatch):
        monkeypatch.setattr(
            multiword_assistant,
            "_verify_bare_wakeword",
            lambda *_args: (_ for _ in ()).throw(AssertionError("should not verify")),
        )

        assert multiword_assistant._verify_strict_barge_wakeword(
            "atticus what time is it", -38, -60
        ) is True


@pytest.fixture
def multiword_assistant():
    """Bare Assistant with a two-word wakeword for tolerant-matcher tests."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(
            wakeword="hey atticus",
            barge_in="wakeword",
        )
        _add_satellite(a)
        return a


class TestTolerantWakewordMatcher:
    """`_wakeword_re` tolerates ASR-inserted punctuation between tokens
    and respects word boundaries, so multi-word wakewords stay usable.
    """

    def test_word_boundary_rejects_substring_match(self, assistant):
        # "atticus" should not fire inside a longer word like "atticuses".
        assert assistant._wakeword_re.search("atticuses chemistry") is None
        assert assistant._wakeword_re.search("the atticuser arrived") is None

    def test_word_boundary_accepts_clean_match(self, assistant):
        assert assistant._wakeword_re.search("atticus stop") is not None
        assert assistant._wakeword_re.search("ok atticus, what time") is not None

    def test_two_word_wakeword_plain(self, multiword_assistant):
        assert multiword_assistant._wakeword_re.search("hey atticus stop") is not None

    def test_two_word_wakeword_with_comma_between(self, multiword_assistant):
        # ASR routinely inserts a comma after the leading word.
        assert multiword_assistant._wakeword_re.search("hey, atticus what time is it") is not None

    def test_two_word_wakeword_with_extra_whitespace(self, multiword_assistant):
        assert multiword_assistant._wakeword_re.search("hey   atticus play some music") is not None

    def test_two_word_wakeword_partial_does_not_match(self, multiword_assistant):
        # "atticus" alone is not the wakeword.
        assert multiword_assistant._wakeword_re.search("atticus what time") is None
        # "hey" alone is not either.
        assert multiword_assistant._wakeword_re.search("hey what time") is None

    def test_two_word_wakeword_self_echo(self, multiword_assistant):
        # The assistant's reply mentioned the wakeword, so a stray
        # "hey atticus" transcript is AEC residue, not a real barge-in.
        multiword_assistant.satellites["sat-a"].last_spoken_text = "hey atticus is a friendly australian voice"
        assert multiword_assistant._is_self_echo("sat-a", "hey atticus") is True

    def test_s_in_wakeword_matches_z_transcript(self, multiword_assistant):
        # ASR routinely transcribes the trailing /s/ of names like
        # "Atticus" as "z" (and vice versa). Either spelling should hit.
        assert multiword_assistant._wakeword_re.search("hey atticuz stop") is not None
        assert multiword_assistant._wakeword_re.search("hey, atticuz what time is it") is not None

    def test_z_in_wakeword_matches_s_transcript(self):
        """A configured wakeword spelt with `z` should accept the `s`
        variant too — same tolerance, applied symmetrically.
        """
        with patch("core.assistant.AudioCapture") as mock_ac:
            mock_ac.return_value = MagicMock()
            from core.assistant import Assistant

            a = Assistant(wakeword="atticuz", barge_in="wakeword")
            _add_satellite(a)
        assert a._wakeword_re.search("atticuz stop") is not None
        assert a._wakeword_re.search("atticus stop") is not None

    def test_two_word_wakeword_barge_in_with_comma(self, multiword_assistant):
        multiword_assistant.satellites["sat-a"].last_spoken_text = "it is sunny and warm today"
        # Even with a comma inserted by ASR, the barge-in should fire.
        assert multiword_assistant._check_barge_in("sat-a", "hey, atticus stop") is True


def _make_assistant(**kwargs):
    """Build a bare Assistant with AudioCapture mocked out."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", **kwargs)
    _add_satellite(a)
    return a


class TestWakewordPatternOverride:
    """An explicit `wakeword_pattern` overrides the auto-built matcher and
    can express phonetic swaps the builder can't (e.g. f↔t for
    "hey fulloch" / "hey tulloch").
    """

    PATTERN = r"\b(?:hey|hay|hi)\W+[ft][ua]ll?o(?:ch|ck|k)\b"

    @pytest.fixture
    def fulloch(self):
        return _make_assistant(wakeword="hey fulloch", wakeword_pattern=self.PATTERN)

    def test_matches_configured_phonetics(self, fulloch):
        assert fulloch._wakeword_re.search("hey fulloch stop") is not None
        # f↔t swap — "tulloch" should hit too.
        assert fulloch._wakeword_re.search("hey tulloch what time is it") is not None
        # u↔a swap and ASR-inserted comma.
        assert fulloch._wakeword_re.search("hey, falloch play music") is not None
        # single-l + 'k' ending.
        assert fulloch._wakeword_re.search("hi fulok stop") is not None

    def test_rejects_non_matches(self, fulloch):
        # Requires the hey/hay/hi prefix.
        assert fulloch._wakeword_re.search("fulloch what time") is None
        # Word boundary guards against substring fires.
        assert fulloch._wakeword_re.search("the waterfall ocean is loud") is None
        assert fulloch._wakeword_re.search("follow the lock") is None

    def test_overrides_auto_builder(self):
        # Without the override, the auto-builder can't reach "tulloch".
        auto = _make_assistant(wakeword="hey fulloch")
        assert auto._wakeword_re.search("hey tulloch stop") is None
        # With it, it can.
        override = _make_assistant(wakeword="hey fulloch", wakeword_pattern=self.PATTERN)
        assert override._wakeword_re.search("hey tulloch stop") is not None

    def test_invalid_pattern_falls_back_to_auto_builder(self):
        # A broken regex must not crash startup — it degrades to the
        # auto-built matcher for the plain wakeword.
        a = _make_assistant(wakeword="hey fulloch", wakeword_pattern="(unclosed[")
        assert a._wakeword_re.search("hey fulloch stop") is not None
        # And the bad custom phonetics are simply not honoured.
        assert a._wakeword_re.search("hey tulloch stop") is None


class TestPrefixOptionalBargeIn:
    """While the assistant is speaking, users interrupt with the bare name
    ("Atticus, stop") far more often than the full "Hey Atticus". `_barge_re`
    makes the greeting prefix optional for the barge-in check, while idle
    detection (`_wakeword_re`) still requires it to keep false-positives low.
    """

    @pytest.fixture
    def atticus(self):
        # Mirror the shipped default: explicit pattern tolerating the
        # "attic us" split, behind a "hey/hay/hi" prefix.
        return _make_assistant(
            wakeword="hey atticus",
            wakeword_pattern=r"\b(?:hey|hay|hi)\W+attic\W*u[sz]\b",
        )

    def test_idle_detection_still_requires_prefix(self, atticus):
        # Bare name must NOT fire idle detection (guards against random
        # mentions waking the assistant when no turn is running).
        assert atticus._wakeword_re.search("atticus stop") is None
        assert atticus._wakeword_re.search("hey atticus stop") is not None

    def test_barge_fires_without_prefix(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "it is sunny and warm today"
        assert atticus._check_barge_in("sat-a", "atticus stop") is True
        assert atticus._check_barge_in("sat-a", "atticus, be quiet") is True

    def test_barge_still_fires_with_prefix(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "it is sunny and warm today"
        # Full wakeword (incl. the "attic us" split) still routes through
        # the base pattern.
        assert atticus._check_barge_in("sat-a", "hey atticus stop") is True
        assert atticus._check_barge_in("sat-a", "hey attic us stop") is True

    def test_barge_respects_word_boundaries(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "it is sunny and warm today"
        # The bare-name branch must not fire inside a longer word.
        assert atticus._check_barge_in("sat-a", "fanaticus rules everything") is False

    def test_bare_name_barge_for_auto_built_wakeword(self):
        # No explicit pattern — the greeting prefix is stripped off the
        # auto-built wakeword too.
        a = _make_assistant(wakeword="hey atticus")
        a.satellites["sat-a"].last_spoken_text = "it is sunny and warm today"
        assert a._wakeword_re.search("atticus stop") is None  # idle: needs prefix
        assert a._check_barge_in("sat-a", "atticus stop") is True  # barge: prefix optional


class TestBareStopBargeIn:
    """Bare stop commands (no wakeword) must trigger barge-in.

    _BARGE_STOP_RE fires independently of the name-based _barge_re so that
    even when ASR mangles the wakeword into something unrecognisable (e.g.
    "Aricus" / "Arica" instead of "Atticus"), a clearly-spoken "stop" in
    the same utterance still interrupts the active turn.
    """

    @pytest.fixture
    def atticus(self):
        return _make_assistant(
            wakeword="hey atticus",
            wakeword_pattern=r"\b(?:hey|hay|hi)\W+[ao][dtl]{1,2}i?c\W*u[sz]\b",
        )

    # --- Should fire ---

    def test_bare_stop_triggers_barge_in(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "stop") is True

    def test_mangled_name_plus_stop_triggers_barge_in(self, atticus):
        # Real-world failure: ASR transcribes "Atticus" as "Aricus" or
        # "Arica" at normal mic distance.  Name fails the barge_re, but
        # "stop" must still fire via _BARGE_STOP_RE.
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "aricus, stop") is True
        assert atticus._check_barge_in("sat-a", "arica, stop") is True

    def test_other_stop_words_trigger_barge_in(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        for word in ("halt", "cancel", "pause", "quiet", "enough"):
            assert atticus._check_barge_in("sat-a", word) is True, f"{word!r} should fire"

    def test_stop_in_longer_utterance_triggers_barge_in(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "okay stop that") is True
        assert atticus._check_barge_in("sat-a", "please cancel that") is True

    # --- Should NOT fire ---

    def test_stop_suppressed_when_tts_said_stop(self, atticus):
        # If the assistant's TTS just said "stop", a bare "stop" transcript
        # is likely AEC residue — _is_self_echo must catch it before
        # _BARGE_STOP_RE has a chance to fire.
        atticus.satellites["sat-a"].last_spoken_text = "i will stop the music right away"
        assert atticus._check_barge_in("sat-a", "stop") is False

    def test_stop_does_not_fire_when_turn_inactive(self, atticus):
        atticus.satellites["sat-a"].turn_active = False
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "stop") is False

    def test_stop_does_not_fire_when_barge_in_off(self, atticus):
        atticus.barge_in = "off"
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "stop") is False

    def test_word_boundary_not_triggered_by_stopped(self, atticus):
        # "stopped" contains "stop" but _BARGE_STOP_RE requires \b on both
        # sides, so inflected forms must not fire.
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "i stopped the timer") is False
        assert atticus._check_barge_in("sat-a", "that is unstoppable") is False

    def test_word_boundary_not_triggered_by_paused(self, atticus):
        atticus.satellites["sat-a"].last_spoken_text = "the model supports a range of modalities"
        assert atticus._check_barge_in("sat-a", "music is paused") is False
        assert atticus._check_barge_in("sat-a", "it was cancelled yesterday") is False


class TestIsPureStop:
    """A bare stop barge-in ("Atticus, stop") must terminate the turn and
    stand down — no follow-up window. A redirect ("stop — play jazz instead")
    carries a real command after the stop word, so it keeps the follow-up.
    `_is_pure_stop` draws that line; the transcriber uses it to decide whether
    to call `_mark_turn_end()` after a barge-in cancel.
    """

    @pytest.fixture
    def atticus(self):
        return _make_assistant(
            wakeword="hey atticus",
            wakeword_pattern=r"\b(?:hey|hay|hi)\W+attic\W*u[sz]\b",
        )

    # --- Pure stop: stand down, no follow-up ---

    def test_bare_stop_word(self, atticus):
        assert atticus._is_pure_stop("stop") is True

    def test_name_plus_stop(self, atticus):
        assert atticus._is_pure_stop("atticus stop") is True
        assert atticus._is_pure_stop("hey atticus, stop") is True

    def test_other_stop_words(self, atticus):
        for word in ("halt", "cancel", "pause", "quiet", "enough"):
            assert atticus._is_pure_stop(f"atticus {word}") is True, word

    def test_stop_with_innocuous_filler(self, atticus):
        # Filler around the stop word doesn't make it a new command.
        assert atticus._is_pure_stop("atticus, stop talking now please") is True
        assert atticus._is_pure_stop("ok stop that") is True
        assert atticus._is_pure_stop("be quiet please") is True

    # --- Redirect: keep the follow-up window ---

    def test_stop_then_new_command(self, atticus):
        assert atticus._is_pure_stop("atticus stop, play jazz instead") is False
        assert atticus._is_pure_stop("stop and turn on the lights") is False

    def test_no_stop_word_is_not_pure_stop(self, atticus):
        # A wakeword-alone barge-in (user pauses before asking) carries no
        # stop word — it must keep the follow-up window, so not a pure stop.
        assert atticus._is_pure_stop("atticus") is False
        assert atticus._is_pure_stop("hey atticus") is False

    def test_question_is_not_pure_stop(self, atticus):
        assert atticus._is_pure_stop("atticus what time is it") is False


class TestContextEchoGuard:
    """A wakeword command built solely from `asr_context_terms` is the ASR
    decoder echoing its bias prompt off non-speech (a sigh/cough Silero
    scored as speech). `_is_context_echo` must drop those while leaving real
    commands (which carry non-bias words) untouched. The token pool spans
    every term, single- or multi-word, so the guard works regardless of how
    many context terms are configured.
    """

    def test_single_short_term_echo_is_dropped(self):
        # One short context term, command == that term (the failure mode:
        # a cough transcribed as wakeword + the lone bias word).
        a = _make_assistant(wakeword="hey atticus", asr_context_terms=["zephyr"])
        assert a._is_context_echo("zephyr") is True

    def test_real_command_with_bias_term_kept(self):
        a = _make_assistant(wakeword="hey atticus", asr_context_terms=["zephyr"])
        # A genuine query carries non-bias words, so it must survive.
        assert a._is_context_echo("what is the weather in zephyr") is False

    def test_multiple_terms_single_word_echo_dropped(self):
        a = _make_assistant(
            wakeword="hey atticus",
            asr_context_terms=["zephyr", "north study"],
        )
        assert a._is_context_echo("zephyr") is True

    def test_multiple_terms_multiword_echo_dropped(self):
        # A command equal to a whole multi-word term is still pure-bias.
        a = _make_assistant(
            wakeword="hey atticus",
            asr_context_terms=["zephyr", "north study"],
        )
        assert a._is_context_echo("north study") is True

    def test_mixed_bias_and_real_words_kept(self):
        a = _make_assistant(
            wakeword="hey atticus",
            asr_context_terms=["zephyr", "north study"],
        )
        assert a._is_context_echo("turn on north study") is False

    def test_punctuation_and_case_ignored(self):
        a = _make_assistant(wakeword="hey atticus", asr_context_terms=["zephyr"])
        assert a._is_context_echo("Zephyr.") is True

    def test_no_context_terms_means_guard_off(self):
        # With no terms configured the guard must never fire — otherwise it
        # could swallow real one-word commands.
        a = _make_assistant(wakeword="hey atticus", asr_context_terms=[])
        assert a._is_context_echo("zephyr") is False

    def test_empty_command_is_not_echo(self):
        # Wakeword-alone (empty command) is handled by the follow-up path,
        # not the echo guard — it must not register as an echo.
        a = _make_assistant(wakeword="hey atticus", asr_context_terms=["zephyr"])
        assert a._is_context_echo("") is False


class TestPromptStripCharset:
    """The post-wakeword strip must peel off leading/trailing punctuation
    (notably "!"/"?") so a barge-in like "Hey Atticus! Stop." reduces to
    "stop" before being dispatched to the agent.
    """

    @pytest.fixture
    def strip_chars(self):
        from core.assistant import _PROMPT_STRIP_CHARS

        return _PROMPT_STRIP_CHARS

    @pytest.mark.parametrize(
        "residue,expected",
        [
            ("! stop.", "stop"),
            ("! stop", "stop"),
            ("? stop talking?", "stop talking"),
            (", be quiet!", "be quiet"),
            (". stop ", "stop"),
        ],
    )
    def test_strip_yields_bare_command(self, strip_chars, residue, expected):
        assert residue.strip(strip_chars) == expected
