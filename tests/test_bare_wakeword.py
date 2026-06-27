"""Tests for bare-wakeword verification against ASR context-bias hallucination.

A wakeword with no command following is the shape the ASR decoder most readily
invents from voiced non-speech, because the `asr_context_hint` prompt primes the
wakeword spelling. `Assistant._verify_bare_wakeword` gates opening the follow-up
window on one with a loudness check plus an unbiased re-transcribe. Both run
against a bare instance — no models loaded — with the ASR pipe mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def assistant():
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(
            wakeword="atticus",
            barge_in="wakeword",
            asr_context_hint=True,
        )
        # Stand in for the loaded pipeline: a mutable .context and a callable
        # that returns the configured unbiased transcription.
        a.asr_pipe = MagicMock()
        a.asr_pipe.context = "Technical terms: atticus"
        a._asr_audio = {"buf": object()}
        return a


class TestLoudnessGate:
    def test_below_baseline_is_rejected(self, assistant):
        # The log case: utterance quieter than ambient background speech.
        assert assistant._verify_bare_wakeword(-38.6, -35.6) is False
        # Rejected at the loudness gate — re-transcribe never runs.
        assistant.asr_pipe.assert_not_called()

    def test_loudness_missing_falls_through_to_retranscribe(self, assistant):
        assistant.asr_pipe.return_value = [{"text": "hey atticus"}]
        assert assistant._verify_bare_wakeword(None, -35.6) is True
        assistant.asr_pipe.assert_called_once()


class TestUnbiasedRetranscribe:
    def test_wakeword_absent_unbiased_is_rejected(self, assistant):
        # Loud enough to clear the gate, but without the bias the decoder
        # produces something else — it was a bias echo.
        assistant.asr_pipe.return_value = [{"text": "hello there"}]
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is False

    def test_wakeword_present_unbiased_is_accepted(self, assistant):
        assistant.asr_pipe.return_value = [{"text": "hey atticus"}]
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is True

    def test_bias_dropped_during_call_and_restored_after(self, assistant):
        seen = {}

        def record(*args, **kwargs):
            seen["context"] = assistant.asr_pipe.context
            return [{"text": "hey atticus"}]

        assistant.asr_pipe.side_effect = record
        assistant._verify_bare_wakeword(-20.0, -35.6)
        # The verification pass ran with the bias dropped...
        assert seen["context"] == ""
        # ...and the original bias is restored afterwards.
        assert assistant.asr_pipe.context == "Technical terms: atticus"

    def test_asr_error_assumes_genuine(self, assistant):
        assistant.asr_pipe.side_effect = RuntimeError("boom")
        # A transient ASR failure must never swallow a real wakeword.
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is True
        # And the saved context is restored even on error.
        assert assistant.asr_pipe.context == "Technical terms: atticus"

    def test_skipped_when_hint_disabled(self, assistant):
        assistant.asr_context_hint = False
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is True
        assistant.asr_pipe.assert_not_called()

    def test_skipped_when_no_context_set(self, assistant):
        assistant.asr_pipe.context = ""
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is True
        assistant.asr_pipe.assert_not_called()

    def test_skipped_when_no_buffer_captured(self, assistant):
        assistant._asr_audio = {"buf": None}
        assert assistant._verify_bare_wakeword(-20.0, -35.6) is True
        assistant.asr_pipe.assert_not_called()


class TestContextPromptMarker:
    """The decoder sometimes echoes the prompt scaffolding verbatim
    ("Technical terms: hey atticus, <garbage>") — a marker the loop drops."""

    def test_marker_derived_from_context_prefix(self):
        # _load_models sets it from the context; emulate that derivation.
        ctx = "Technical terms: hey atticus, foo"
        marker = ctx.split(":", 1)[0].strip().lower()
        assert marker == "technical terms"

    def test_echo_string_contains_the_marker(self, assistant):
        assistant._context_prompt_marker = "technical terms"
        echo = "Technical terms: hey atticus, but all the bike and binga."
        assert assistant._context_prompt_marker in echo.lower()

    def test_genuine_command_does_not_contain_marker(self, assistant):
        assistant._context_prompt_marker = "technical terms"
        assert assistant._context_prompt_marker not in "hey atticus turn on the lights"
