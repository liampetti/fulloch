"""Unit seams for the CPU-only Parakeet ONNX backend."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import asr_parakeet_onnx as backend


class _Model:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        return SimpleNamespace(text="  hello world  ")


def test_wrapper_transcribes_float32_pcm_and_uses_16khz():
    model = _Model()
    pipe = backend.ParakeetOnnxASRPipelineWrapper(model)

    assert pipe(np.array([0, 1], dtype=np.int16)) == [{"text": "hello world"}]
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32
    assert kwargs == {"sample_rate": 16000}
    assert pipe.last_transcribe_seconds is not None
    assert pipe.supports_context_unbias is False


def test_context_phrase_seam_is_a_safe_noop():
    pipe = backend.ParakeetOnnxASRPipelineWrapper(_Model())
    pipe.set_context_phrases(["hey atticus"])
    assert pipe.context == ""


def test_loader_requests_int8_cpu_only(monkeypatch):
    calls = {}

    class _Options:
        graph_optimization_level = None
        intra_op_num_threads = None
        log_severity_level = None

    fake_ort = SimpleNamespace(
        SessionOptions=_Options,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )

    def load_model(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _Model()

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model))

    pipe = backend.load_asr_model("/models/parakeet", num_threads=3)

    assert isinstance(pipe, backend.ParakeetOnnxASRPipelineWrapper)
    assert calls["args"] == (backend.MODEL_TYPE, "/models/parakeet")
    assert calls["kwargs"]["quantization"] == "int8"
    assert calls["kwargs"]["providers"] == ["CPUExecutionProvider"]
    assert calls["kwargs"]["sess_options"].intra_op_num_threads == 3


def test_loader_rejects_unknown_quantization():
    with pytest.raises(ValueError, match="quantize"):
        backend.load_asr_model(quantize="fp16")
