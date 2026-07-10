"""`core.asr_onnx._create_session` — CoreML EP has a known bug loading
external-data ONNX models (small `.onnx` shell + companion `.onnx.data`
file): it throws `model_path.empty() was false` during init even though the
same file loads fine under CPUExecutionProvider alone. `_create_session`
retries CPU-only on exactly that failure instead of failing the whole
model load."""

from unittest.mock import MagicMock, patch

import pytest

from core import asr_onnx


class TestCreateSession:
    def test_returns_session_on_first_try(self):
        sentinel = MagicMock()
        with patch.object(asr_onnx.ort, "InferenceSession", return_value=sentinel) as ctor:
            result = asr_onnx._create_session(
                "model.onnx", MagicMock(), ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            )
        assert result is sentinel
        ctor.assert_called_once()

    def test_retries_cpu_only_on_coreml_model_path_bug(self):
        sentinel = MagicMock()
        coreml_error = RuntimeError(
            "RUNTIME_EXCEPTION : ... model_path.empty() was false. "
            "model_path must not be empty."
        )
        with patch.object(
            asr_onnx.ort, "InferenceSession", side_effect=[coreml_error, sentinel]
        ) as ctor:
            result = asr_onnx._create_session(
                "model.onnx", MagicMock(), ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            )
        assert result is sentinel
        assert ctor.call_count == 2
        # Second call must drop CoreML and go CPU-only.
        _, kwargs = ctor.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]

    def test_does_not_retry_when_coreml_not_requested(self):
        other_error = RuntimeError("model_path.empty() was false somehow")
        with patch.object(asr_onnx.ort, "InferenceSession", side_effect=other_error) as ctor:
            with pytest.raises(RuntimeError):
                asr_onnx._create_session("model.onnx", MagicMock(), ["CPUExecutionProvider"])
        ctor.assert_called_once()

    def test_reraises_unrelated_errors_even_with_coreml(self):
        other_error = RuntimeError("totally unrelated failure")
        with patch.object(asr_onnx.ort, "InferenceSession", side_effect=other_error) as ctor:
            with pytest.raises(RuntimeError, match="totally unrelated failure"):
                asr_onnx._create_session(
                    "model.onnx",
                    MagicMock(),
                    ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                )
        ctor.assert_called_once()
