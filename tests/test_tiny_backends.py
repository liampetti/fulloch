"""Moonshine (experimental edge) ASR backend — verifies the v2.1.9 pipeline
contract against a fake pipe (no model load). Kokoro TTS moved to ONNX
(test_tts_onnx)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.asr_tiny as asr_tiny  # noqa: E402


def _gen(items):
    for it in items:
        yield it


def test_moonshine_wrapper_streams_text_dicts():
    calls = []

    def fake_pipe(inputs, generate_kwargs=None):
        calls.append((inputs, generate_kwargs))
        return {"text": "hello"}

    w = asr_tiny.MoonshineASRPipelineWrapper(fake_pipe)
    buf = np.zeros(1600, dtype=np.float32)
    out = list(w(_gen([buf, buf]), batch_size=1, generate_kwargs={"max_new_tokens": 256}))

    assert out == [{"text": "hello"}, {"text": "hello"}]
    assert calls[0][0]["sampling_rate"] == asr_tiny.SAMPLE_RATE
    assert isinstance(w.last_transcribe_seconds, float)


def test_moonshine_wrapper_has_context_and_is_noop():
    w = asr_tiny.MoonshineASRPipelineWrapper(lambda *a, **k: {"text": "x"})
    assert w.context == ""
    w.context = "Technical terms: atticus"
    assert w.context == "Technical terms: atticus"


def test_moonshine_non_streaming_path_returns_list():
    w = asr_tiny.MoonshineASRPipelineWrapper(lambda *a, **k: {"text": "single"})
    res = w(np.zeros(1600, dtype=np.float32))
    assert res == [{"text": "single"}]


def test_moonshine_load_signature_matches_registry():
    import inspect
    params = inspect.signature(asr_tiny.load_asr_model).parameters
    assert "model_name" in params and "language" in params
