"""Contract tests for the official Pocket TTS PyTorch adapter."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.tts_pocket as tts  # noqa: E402


def test_load_tts_uses_numeric_default_temperature(monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def load_model(cls, **kwargs):
            calls.update(kwargs)
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            pass

    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=Model))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )

    tts.load_tts()

    assert calls == {"language": "english_2026-04", "temp": 0.7, "lsd_decode_steps": 1}


def test_pocket_rollover_keeps_first_sentence_early(monkeypatch):
    monkeypatch.setattr(tts, "_MAX_SYNTHESIS_CHARS", 80)

    fragments = list(
        tts._synthesis_fragments(
            "Hello, there friend. The weather is warm, dry, and clear. It stays pleasant tonight."
        )
    )

    assert fragments == [
        "Hello, there friend.",
        "The weather is warm, dry, and clear. It stays pleasant tonight.",
    ]


def test_pocket_native_stream_forwards_pcm_without_waiting_for_completion(monkeypatch):
    calls = []

    class Audio:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def to(self, **kwargs):
            calls.append(kwargs)
            return self

        def numpy(self):
            return np.asarray(self.values, dtype=np.float32)

    class Model:
        def generate_audio_stream(self, **kwargs):
            assert kwargs == {"model_state": "voice", "text_to_generate": "Hello"}
            yield Audio([0.1, 0.2])
            yield Audio([0.3])

    monkeypatch.setattr(tts, "_model", Model())
    monkeypatch.setattr(tts, "_torch", SimpleNamespace(float32="float32"))

    chunks = list(tts._stream("Hello", "voice"))

    np.testing.assert_allclose(chunks[0], [0.1, 0.2])
    np.testing.assert_allclose(chunks[1], [0.3])
    assert calls == [
        {"device": "cpu", "dtype": "float32"},
        {"device": "cpu", "dtype": "float32"},
    ]
