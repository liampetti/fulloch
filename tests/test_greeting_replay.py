"""The opening greeting is synthesised during startup, before any browser
satellite could possibly be connected — `_warm_and_announce`'s own playback
attempt reaches nobody. `replay_greeting()` (called from the satellite
WebSocket handler once a connection lands) is what actually delivers it, and
must only do so once even if the browser reconnects.
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
    satellite = types.SimpleNamespace(tts_sink=None, tts_active=None)
    self = types.SimpleNamespace(
        greeting_cache=cache,
        _greeting_delivered=delivered,
        tts_session=None,
        _tts_module=tts_module,
        satellites={"sat-a": satellite},
    )
    return self, calls


def test_replay_greeting_plays_cached_chunks_once():
    cache = [(["c1"], 24000), (["c2"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == [(["c1"], 24000), (["c2"], 24000)]
    assert self._greeting_delivered is True


def test_replay_greeting_is_idempotent_on_reconnect():
    cache = [(["c1"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")
    a.Assistant.replay_greeting(self, "sat-a")  # simulated reconnect

    assert calls == [(["c1"], 24000)]


def test_replay_greeting_noop_without_cache():
    self, calls = _stub_self([])

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == []
    assert self._greeting_delivered is False


def test_replay_greeting_skips_empty_chunk_lists():
    cache = [([], 24000), (["c1"], 24000)]
    self, calls = _stub_self(cache)

    a.Assistant.replay_greeting(self, "sat-a")

    assert calls == [(["c1"], 24000)]
