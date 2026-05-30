"""Tests for self-echo suppression in barge-in detection.

Targets the symptom where the assistant interrupts its own TTS because AEC
residue leaked the wakeword (or a near-homophone of it) back through ASR.
The unit under test is `Assistant._is_self_echo` plus the integration with
`_check_barge_in`. Both can be exercised against a bare instance — no
models loaded — because the relevant code paths only touch in-process
state (`_last_spoken_text`, `_turn_active`, `barge_in`).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


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
            wakeword="morgan",
            barge_in="wakeword",
        )
        a._turn_active = True  # pretend a turn is running so checks engage
        return a


class TestIsSelfEcho:
    def test_no_recent_spoken_text_means_not_echo(self, assistant):
        assistant._last_spoken_text = ""
        assert assistant._is_self_echo("morgan what time is it") is False

    def test_multi_word_overlap_above_threshold(self, assistant):
        # Assistant said this; mic picks up most of it as a barge-in.
        assistant._last_spoken_text = "morgan freeman was born in memphis"
        assert assistant._is_self_echo("morgan freeman was born") is True

    def test_multi_word_overlap_below_threshold(self, assistant):
        # User's real request shares only stop-words with the prior answer.
        assistant._last_spoken_text = "the time is half past four"
        assert assistant._is_self_echo("play some jazz music for me") is False

    def test_single_word_wakeword_in_recent_speech_is_echo(self, assistant):
        # The assistant's answer mentioned "morgan" — a stray single-word
        # "morgan" transcript is almost certainly AEC residue.
        assistant._last_spoken_text = "morgan freeman was a great narrator"
        assert assistant._is_self_echo("morgan") is True

    def test_single_word_wakeword_not_in_recent_speech_is_not_echo(self, assistant):
        # The prior answer didn't mention the wakeword, so a "morgan" transcript
        # is a legitimate barge-in attempt.
        assistant._last_spoken_text = "it is sunny and twenty two degrees"
        assert assistant._is_self_echo("morgan") is False


class TestCheckBargeIn:
    def test_does_not_barge_when_self_echo(self, assistant):
        assistant._last_spoken_text = "morgan freeman was born in memphis"
        assert assistant._check_barge_in("morgan freeman was born") is False

    def test_barges_on_real_user_utterance(self, assistant):
        assistant._last_spoken_text = "it is sunny and twenty two degrees"
        assert assistant._check_barge_in("morgan stop") is True

    def test_does_not_barge_when_turn_inactive(self, assistant):
        assistant._turn_active = False
        assistant._last_spoken_text = "it is sunny"
        assert assistant._check_barge_in("morgan stop") is False

    def test_does_not_barge_when_barge_in_off(self, assistant):
        assistant.barge_in = "off"
        assistant._last_spoken_text = "it is sunny"
        assert assistant._check_barge_in("morgan stop") is False


@pytest.fixture
def multiword_assistant():
    """Bare Assistant with a two-word wakeword for tolerant-matcher tests."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant
        a = Assistant(
            wakeword="hey fraser",
            barge_in="wakeword",
        )
        a._turn_active = True
        return a


class TestTolerantWakewordMatcher:
    """`_wakeword_re` tolerates ASR-inserted punctuation between tokens
    and respects word boundaries, so multi-word wakewords stay usable.
    """

    def test_word_boundary_rejects_substring_match(self, assistant):
        # "morgan" should not fire inside "morganic" / "morgans".
        assert assistant._wakeword_re.search("morganic chemistry") is None
        assert assistant._wakeword_re.search("the morgans arrived") is None

    def test_word_boundary_accepts_clean_match(self, assistant):
        assert assistant._wakeword_re.search("morgan stop") is not None
        assert assistant._wakeword_re.search("ok morgan, what time") is not None

    def test_two_word_wakeword_plain(self, multiword_assistant):
        assert multiword_assistant._wakeword_re.search("hey fraser stop") is not None

    def test_two_word_wakeword_with_comma_between(self, multiword_assistant):
        # ASR routinely inserts a comma after the leading word.
        assert multiword_assistant._wakeword_re.search(
            "hey, fraser what time is it"
        ) is not None

    def test_two_word_wakeword_with_extra_whitespace(self, multiword_assistant):
        assert multiword_assistant._wakeword_re.search(
            "hey   fraser play some music"
        ) is not None

    def test_two_word_wakeword_partial_does_not_match(self, multiword_assistant):
        # "fraser" alone is not the wakeword.
        assert multiword_assistant._wakeword_re.search("fraser what time") is None
        # "hey" alone is not either.
        assert multiword_assistant._wakeword_re.search("hey what time") is None

    def test_two_word_wakeword_self_echo(self, multiword_assistant):
        # The assistant's reply mentioned the wakeword, so a stray
        # "hey fraser" transcript is AEC residue, not a real barge-in.
        multiword_assistant._last_spoken_text = (
            "hey fraser is a friendly australian voice"
        )
        assert multiword_assistant._is_self_echo("hey fraser") is True

    def test_s_in_wakeword_matches_z_transcript(self, multiword_assistant):
        # ASR routinely transcribes the trailing /s/ of names like
        # "Fraser" as "z" (and vice versa). Either spelling should hit.
        assert multiword_assistant._wakeword_re.search("hey frazer stop") is not None
        assert multiword_assistant._wakeword_re.search(
            "hey, frazer what time is it"
        ) is not None

    def test_z_in_wakeword_matches_s_transcript(self):
        """A configured wakeword spelt with `z` should accept the `s`
        variant too — same tolerance, applied symmetrically.
        """
        with patch("core.assistant.AudioCapture") as mock_ac:
            mock_ac.return_value = MagicMock()
            from core.assistant import Assistant
            a = Assistant(wakeword="frazer", barge_in="wakeword")
            a._turn_active = True
        assert a._wakeword_re.search("frazer stop") is not None
        assert a._wakeword_re.search("fraser stop") is not None

    def test_two_word_wakeword_barge_in_with_comma(self, multiword_assistant):
        multiword_assistant._last_spoken_text = "it is sunny and warm today"
        # Even with a comma inserted by ASR, the barge-in should fire.
        assert multiword_assistant._check_barge_in(
            "hey, fraser stop"
        ) is True


def _make_assistant(**kwargs):
    """Build a bare Assistant with AudioCapture mocked out."""
    with patch("core.assistant.AudioCapture") as mock_ac:
        mock_ac.return_value = MagicMock()
        from core.assistant import Assistant
        a = Assistant(barge_in="wakeword", **kwargs)
    a._turn_active = True
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
        override = _make_assistant(
            wakeword="hey fulloch", wakeword_pattern=self.PATTERN
        )
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
        atticus._last_spoken_text = "it is sunny and warm today"
        assert atticus._check_barge_in("atticus stop") is True
        assert atticus._check_barge_in("atticus, be quiet") is True

    def test_barge_still_fires_with_prefix(self, atticus):
        atticus._last_spoken_text = "it is sunny and warm today"
        # Full wakeword (incl. the "attic us" split) still routes through
        # the base pattern.
        assert atticus._check_barge_in("hey atticus stop") is True
        assert atticus._check_barge_in("hey attic us stop") is True

    def test_barge_respects_word_boundaries(self, atticus):
        atticus._last_spoken_text = "it is sunny and warm today"
        # The bare-name branch must not fire inside a longer word.
        assert atticus._check_barge_in("fanaticus rules everything") is False

    def test_bare_name_barge_for_auto_built_wakeword(self):
        # No explicit pattern — the greeting prefix is stripped off the
        # auto-built wakeword too.
        a = _make_assistant(wakeword="hey morgan")
        a._last_spoken_text = "it is sunny and warm today"
        assert a._wakeword_re.search("morgan stop") is None       # idle: needs prefix
        assert a._check_barge_in("morgan stop") is True            # barge: prefix optional


class TestBargeQuietRegex:
    """`_BARGE_QUIET_RE` only matches "silence the assistant" phrases —
    not music-control phrases like 'stop the music'.
    """

    @pytest.fixture
    def regex(self):
        from core.assistant import _BARGE_QUIET_RE
        return _BARGE_QUIET_RE

    @pytest.mark.parametrize("phrase", [
        "stop",
        "stop talking",
        "stop speaking",
        "stop it",
        "stop please",
        "stop now",
        "please stop",
        "ok stop",
        "okay stop talking",
        "just stop",
        "be quiet",
        "be quiet please",
        "quiet",
        "quiet please",
        "shut up",
        "hush",
        "silence",
        "enough",
        "that's enough",
        "that is enough",
        "stop.",
        "stop talking.",
    ])
    def test_quiet_phrases_match(self, regex, phrase):
        assert regex.match(phrase) is not None, f"{phrase!r} should match"

    @pytest.mark.parametrize("phrase", [
        "stop the music",
        "stop the song",
        "stop playing",
        "stop the timer",
        "stop the alarm",
        "pause",
        "pause the music",
        "halt",
        "skip",
        "resume",
        "what time is it",
        "play some jazz",
        "turn off the lights",
        "stop reading the news",
    ])
    def test_non_quiet_phrases_fall_through(self, regex, phrase):
        assert regex.match(phrase) is None, f"{phrase!r} should NOT match"


class TestPromptStripCharset:
    """The post-wakeword strip must peel off leading/trailing punctuation
    (notably "!"/"?") so a barge-in "Hey Frasier! Stop." reduces to "stop"
    and matches `_BARGE_QUIET_RE` — otherwise the leading "!" defeats the
    barge-quiet check and the utterance gets dispatched to the agent, which
    then talks again instead of going silent.
    """

    @pytest.fixture
    def strip_chars(self):
        from core.assistant import _PROMPT_STRIP_CHARS
        return _PROMPT_STRIP_CHARS

    @pytest.mark.parametrize("residue,expected", [
        ("! stop.", "stop"),
        ("! stop", "stop"),
        ("? stop talking?", "stop talking"),
        (", be quiet!", "be quiet"),
        (". stop ", "stop"),
    ])
    def test_strip_yields_bare_command(self, strip_chars, residue, expected):
        assert residue.strip(strip_chars) == expected

    def test_stripped_barge_quiet_matches(self, strip_chars):
        from core.assistant import _BARGE_QUIET_RE
        # The exact failure from the debug log: wakeword stripped to "! stop".
        assert _BARGE_QUIET_RE.match("! stop".strip(strip_chars)) is not None
