"""The opening greeting is synthesised during startup to warm the TTS model
and the LLM cache. Once a satellite connects, `replay_greeting()` is called
but must NOT play audio — the warming effect is already done, and playback
would inject the greeting into the active microphone, causing a self-echo.
"""

import types

import core.assistant as a


def _stub_self(cache, delivered=False):
    calls = []
    tts_module = types.SimpleNamespace(
        play_chunks=lambda chunks, sr, session=None, sink=None, tts_active_event=None: calls.append(
            (chunks, sr)
        )
    )
    self = types.SimpleNamespace(
        greeting_cache=cache,
        _greeting_delivered=delivered,
        _tts_module=tts_module,
    )
    return self, calls


def test_replay_greeting_marks_delivered_with_cache():
    cache = [(["c1"], 24000), (["c2"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == []  # no audio played
    assert self._greeting_delivered is True


def test_replay_greeting_is_idempotent_on_reconnect():
    cache = [(["c1"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")
    a.Assistant.replay_greeting(self, "sat-a")  # simulated reconnect

    assert calls == []
    assert self._greeting_delivered is True


def test_replay_greeting_noop_without_cache():
    self, calls = _stub_self([])

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == []
    assert self._greeting_delivered is False


def test_replay_greeting_skips_empty_chunk_lists():
    cache = [([], 24000), (["c1"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == []
    assert self._greeting_delivered is True
