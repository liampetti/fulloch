"""Optional openWakeWord streaming backend, loaded only when configured."""

import logging
from pathlib import Path

import numpy as np

from .wakeword import ScoreGate, WakewordResult

logger = logging.getLogger(__name__)


class OpenWakeWordBackend:
    """Owns a separate stateful openWakeWord model for every satellite."""

    def __init__(self, model_path: str, threshold: float, smoothing_frames: int, cooldown_ms: int):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Wakeword model not found: {path}")
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError("openwakeword is not installed") from exc
        self._model_type = Model
        self.model_path = str(path)
        suffix = path.suffix.lower()
        if suffix not in {".onnx", ".tflite"}:
            raise ValueError("Wakeword model must use an .onnx or .tflite extension")
        # openWakeWord 0.6 defaults to TFLite even when a custom ONNX model was
        # supplied. Select the matching runtime from the model artifact instead.
        self.inference_framework = suffix.removeprefix(".")
        self._models: dict[str, object] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._gate = ScoreGate(threshold, smoothing_frames, cooldown_ms)

    def _model(self, satellite_id: str):
        model = self._models.get(satellite_id)
        if model is None:
            model = self._model_type(
                wakeword_models=[self.model_path],
                inference_framework=self.inference_framework,
            )
            self._models[satellite_id] = model
        return model

    def feed_pcm(self, satellite_id: str, pcm: np.ndarray) -> WakewordResult:
        if pcm.ndim != 1:
            raise ValueError("wakeword PCM must be mono")
        # openWakeWord consumes 80 ms / 1280-sample S16LE frames. Buffer arbitrary
        # browser chunks so the feature extractor sees a continuous stream.
        samples = np.clip(pcm, -1.0, 1.0)
        pending = np.concatenate((self._buffers.get(satellite_id, np.empty(0, np.int16)),
                                  (samples * 32767).astype(np.int16)))
        result = WakewordResult(False, 0.0)
        while pending.size >= 1280:
            frame, pending = pending[:1280], pending[1280:]
            scores = self._model(satellite_id).predict(frame)
            score = max((float(value) for value in scores.values()), default=0.0)
            result = self._gate.feed(satellite_id, score)
            if result.matched:
                break
        self._buffers[satellite_id] = pending
        return result

    def reset(self, satellite_id: str) -> None:
        self._models.pop(satellite_id, None)
        self._buffers.pop(satellite_id, None)
        self._gate.reset(satellite_id)

    def self_test(self) -> None:
        # Exercise model construction and one valid streaming frame at startup.
        self.feed_pcm("__self_test__", np.zeros(1280, dtype=np.float32))
        self.reset("__self_test__")
