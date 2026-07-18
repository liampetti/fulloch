"""Regression coverage for the follow-up-window timing bug.

`_mark_turn_end` used to stamp `_last_turn_end` at `time.monotonic()` the
instant TTS finished *generating and queuing* audio — not when the browser
actually finished *playing* it. For a long reply, generation can finish
before the audio actually stops playing — silently expiring the 5s
follow-up window while the user was still listening, then reporting a huge
"gap" once they replied the instant the voice actually stopped.

`_mark_turn_end` now optionally takes `speak_stream`'s return value (an
estimated playback-end monotonic timestamp) and uses whichever is later.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_assistant_module():
    """Import core.assistant without triggering the heavy submodules (mirrors
    tests/test_assistant_ack.py's stubbing approach)."""
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice",
            "warmup_model",
            "synthesize",
            "play_chunks",
            "speak_stream",
            "set_output_device",
            "set_tts_active_event",
            "model",
        ],
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        if name == "core.slm":
            mod.ContextExhaustedError = type("ContextExhaustedError", (RuntimeError,), {})
            mod.RemoteUnreachable = type("RemoteUnreachable", (RuntimeError,), {})
        sys.modules[name] = mod

    import core.assistant as assistant  # noqa: E402

    return assistant


class _StubAudioCapture:
    def __init__(self):
        self.armed_window = None

    def arm_follow_up(self, session, seconds):
        self.armed_window = seconds


def _make_stub_assistant(follow_up_seconds=5.0, conversation_mode=False):
    from core.satellite import SatelliteSession

    assistant = _import_assistant_module()
    self = assistant.Assistant.__new__(assistant.Assistant)
    self.audio_capture = _StubAudioCapture()
    self.follow_up_seconds = follow_up_seconds
    sat = SatelliteSession(id="sat-a", chunk_q=None)
    sat.conversation_mode = conversation_mode
    self.satellites = {"sat-a": sat}
    return self, assistant, sat


def test_mark_turn_end_uses_playback_end_when_later_than_now():
    import time

    self, assistant, sat = _make_stub_assistant()
    future = time.monotonic() + 30.0

    assistant.Assistant._mark_turn_end(self, "sat-a", future)

    assert sat.last_turn_end == future


def test_mark_turn_end_falls_back_to_now_without_estimate():
    import time

    self, assistant, sat = _make_stub_assistant()
    before = time.monotonic()

    assistant.Assistant._mark_turn_end(self, "sat-a", None)

    after = time.monotonic()
    assert before <= sat.last_turn_end <= after


def test_mark_turn_end_never_regresses_before_now():
    """A stale/short estimate must not pull last_turn_end into the past —
    that would open the follow-up window artificially early relative to now."""
    import time

    self, assistant, sat = _make_stub_assistant()
    stale_past = time.monotonic() - 100.0
    before = time.monotonic()

    assistant.Assistant._mark_turn_end(self, "sat-a", stale_past)

    after = time.monotonic()
    assert before <= sat.last_turn_end <= after
