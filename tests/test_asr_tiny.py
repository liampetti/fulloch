from unittest.mock import MagicMock, patch

import numpy as np

from core import asr_tiny


def test_moonshine_transcription_uses_inference_watchdog():
    pipeline = MagicMock(return_value={"text": "hello"})
    wrapper = asr_tiny.MoonshineASRPipelineWrapper(pipeline)
    watchdog = MagicMock()
    watchdog.__enter__.return_value = watchdog
    watchdog.__exit__.return_value = False

    with patch.object(asr_tiny, "InferenceWatchdog", return_value=watchdog) as watchdog_ctor:
        assert wrapper._transcribe(np.zeros(160, dtype=np.float32), {}) == "hello"

    watchdog_ctor.assert_called_once_with("Moonshine ASR transcription")


def test_moonshine_omits_empty_generation_kwargs():
    pipeline = MagicMock(return_value={"text": "hello"})
    wrapper = asr_tiny.MoonshineASRPipelineWrapper(pipeline)

    assert wrapper._transcribe(np.zeros(160, dtype=np.float32), {}) == "hello"
    pipeline.assert_called_once()
    assert pipeline.call_args.kwargs == {}
    assert pipeline.call_args.args[0]["sampling_rate"] == 16000


def test_moonshine_loader_uses_cpu_pipeline_device(monkeypatch):
    """The CPU image must use Transformers' portable CPU device sentinel."""
    model = MagicMock()
    processor = MagicMock()
    pipe = MagicMock()
    transformers = MagicMock(
        AutoProcessor=MagicMock(from_pretrained=MagicMock(return_value=processor)),
        MoonshineForConditionalGeneration=MagicMock(
            from_pretrained=MagicMock(return_value=model)
        ),
        pipeline=MagicMock(return_value=pipe),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setattr(asr_tiny, "DEVICE", "cpu")
    monkeypatch.setattr(asr_tiny, "DTYPE", asr_tiny.torch.float32)

    asr_tiny.load_asr_model("UsefulSensors/moonshine-tiny")

    transformers.pipeline.assert_called_once()
    assert transformers.pipeline.call_args.kwargs["device"] == -1
