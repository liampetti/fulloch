"""Contract tests for the experimental Audio8 TTS adapter."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.tts_audio8 as tts  # noqa: E402


def test_load_tts_uses_cuda_bfloat16(monkeypatch):
    calls = {}

    class Model:
        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda"
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["processor"] = (model_id, kwargs)
            return cls()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return Model()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoModel=AutoModel, AutoProcessor=Processor)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16"),
    )

    tts.load_tts()

    assert calls["processor"] == (
        "Edge0/Audio8-TTS-Preview-0.6b",
        {"trust_remote_code": True, "revision": None},
    )
    assert calls["model"] == (
        "Edge0/Audio8-TTS-Preview-0.6b",
        {"trust_remote_code": True, "dtype": "bf16", "revision": None},
    )


def test_audio8_stream_decodes_and_chunks_pcm(monkeypatch):
    calls = {}

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to(self, _device):
            return self

        def __getitem__(self, item):
            return Tensor(self.values[item])

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Processor:
        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": Tensor([1])}

    class Model:
        device = "cuda"

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(codes="codes")

        def decode_audio(self, codes):
            assert codes == "codes"
            return Tensor([[0.1, 0.2, 0.3]]), [3]

    monkeypatch.setattr(tts, "_processor", Processor())
    monkeypatch.setattr(tts, "_model", Model())
    monkeypatch.setattr(tts, "_torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(tts, "_CHUNK_SAMPLES", 2)

    chunks = list(tts._stream("Hello", (Path("voice.wav"), "Reference text.")))

    assert calls["processor"]["reference_audio"] == ["voice.wav"]
    assert calls["processor"]["reference_text"] == ["Reference text."]
    assert calls["generate"]["max_new_tokens"] == 1024
    np.testing.assert_allclose(chunks[0], [0.1, 0.2])
    np.testing.assert_allclose(chunks[1], [0.3])
