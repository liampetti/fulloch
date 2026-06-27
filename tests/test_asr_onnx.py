"""Qwen3-ASR-0.6B ONNX CPU backend (registry + a guarded functional test)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import backends as b  # noqa: E402

_MODEL_DIR = Path("data/models/qwen3-asr-0.6b-onnx")
_HAVE_MODEL = (_MODEL_DIR / "onnx_models").is_dir()


# --- registry (always) ------------------------------------------------------


def test_qwen_onnx_registered_and_cpu_offerable():
    spec = b.get_spec("asr", "qwen-onnx-small")
    assert spec.implemented and spec.cpu_ok and not spec.gpu_only
    # Offered on both CPU and GPU images.
    assert b.is_offerable(spec, "cpu") and b.is_offerable(spec, "gpu")
    # Loader resolves to the module function.
    assert b.get_loader("asr", "qwen-onnx-small").__name__ == "load_asr_model"


def test_qwen_onnx_default_is_local_dir():
    r = b.resolve_models({"asr": {"backend": "qwen-onnx-small"}})
    assert r["asr"]["model"].endswith("qwen3-asr-0.6b-onnx")


# --- functional (skips without the model + onnxruntime/librosa) --------------


@pytest.mark.skipif(not _HAVE_MODEL, reason="ONNX model dir not present")
def test_warmup_primes_via_short_silent_buffer():
    # No model needed: a fake pipeline records the call. Warmup must walk the
    # full transcribe path with a short buffer so ORT sessions go warm at load.
    from core import asr_onnx

    calls = {}

    class _FakePipe:
        def transcribe(self, wav, context="", language=None, max_new_tokens=256):
            calls.update(length=len(wav), context=context, max_new_tokens=max_new_tokens)
            return ""

    w = asr_onnx.QwenOnnxASRPipelineWrapper(_FakePipe(), language="English")
    w.context = "Technical terms: hey atticus"
    w.warmup()
    assert calls["length"] == asr_onnx.SAMPLE_RATE // 2  # 0.5s buffer
    assert calls["context"] == "Technical terms: hey atticus"  # same bias seam
    assert calls["max_new_tokens"] == 4  # cheap, just prime kernels


def test_warmup_swallows_errors():
    from core import asr_onnx

    class _BadPipe:
        def transcribe(self, *a, **k):
            raise RuntimeError("boom")

    asr_onnx.QwenOnnxASRPipelineWrapper(_BadPipe()).warmup()  # must not raise


def test_onnx_transcribes_and_biases():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("librosa")
    import librosa
    import numpy as np

    from core import asr_onnx

    w = asr_onnx.load_asr_model()
    # Wrapper contract.
    assert hasattr(w, "context") and hasattr(w, "last_transcribe_seconds")

    samples = sorted((_MODEL_DIR / "test_audio").glob("*.wav"))
    if not samples:
        pytest.skip("no bundled test audio")
    wav, _ = librosa.load(str(samples[0]), sr=16000, mono=True)
    wav = wav.astype(np.float32)

    def gen(x):
        yield x

    out = list(w(gen(wav), batch_size=1, generate_kwargs={"max_new_tokens": 256}))
    assert out and out[0]["text"].strip()
    assert isinstance(w.last_transcribe_seconds, float)

    # Setting a context bias must not break transcription.
    w.context = "Technical terms: hey atticus, phoebe bridgers"
    out2 = list(w(gen(wav), generate_kwargs={"max_new_tokens": 256}))
    assert out2 and out2[0]["text"].strip()
