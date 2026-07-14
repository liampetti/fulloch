"""CrispASR GGUF TTS resilience helpers (no native runtime required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.tts_crispasr as tts  # noqa: E402


def test_synthesis_fragments_bound_long_clauses_without_losing_text(monkeypatch):
    monkeypatch.setattr(tts, "_MAX_SYNTHESIS_CHARS", 16)

    fragments = list(tts._synthesis_fragments("one two three four five six seven"))

    assert fragments == ["one two three", "four five six", "seven"]
    assert all(len(fragment) <= 16 for fragment in fragments)


def test_restart_dead_worker_restores_the_voice_prompt(monkeypatch):
    created = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.alive = bool(created)
            self.calls = []
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

        def call(self, command, **payload):
            self.calls.append((command, payload))

    dead = FakeWorker(model_path="old")
    dead.alive = False
    monkeypatch.setattr(tts, "CrispASRWorker", FakeWorker)
    monkeypatch.setattr(tts, "_session", dead)
    monkeypatch.setattr(tts, "_worker_config", {"model_path": "model", "lib_dir": "runtime"})
    monkeypatch.setattr(tts, "_voice_prompt", ("voice.wav", "voice text"))

    tts._restart_dead_worker()

    assert dead.closed is True
    assert tts._session is created[-1]
    assert tts._session.calls == [("set_voice", {"audio": "voice.wav", "text": "voice text"})]

    prior = tts._session
    monkeypatch.setattr(tts, "_worker_stream_count", 2)
    monkeypatch.setattr(tts, "_MAX_STREAMS_PER_WORKER", 2)
    tts._recycle_worker_if_needed()

    assert prior.closed is True
    assert tts._session is created[-1]
    assert tts._worker_stream_count == 0
    assert tts._session.calls == [("set_voice", {"audio": "voice.wav", "text": "voice text"})]
