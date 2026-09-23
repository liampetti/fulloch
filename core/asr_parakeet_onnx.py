"""CPU-only NVIDIA Parakeet TDT v3 ASR through ONNX Runtime."""

import logging
import time
from typing import Generator, Optional, Union

import numpy as np

from .asr import AsrInput, stream_generator  # noqa: F401
from .inference_safety import InferenceWatchdog

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/parakeet-tdt-0.6b-v3-onnx"
MODEL_TYPE = "nemo-parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000


class ParakeetOnnxASRPipelineWrapper:
    """Pipeline-compatible wrapper around ``onnx-asr``'s TDT decoder."""

    def __init__(self, model):
        self.model = model
        self.context = ""
        self.supports_context_unbias = False
        self.last_transcribe_seconds: Optional[float] = None

    def set_context_phrases(self, _phrases: list[str]) -> None:
        """Keep the ASR context seam explicit; ONNX TDT has no phrase biasing."""
        logger.info("Parakeet ONNX does not support decoder phrase boosting")

    def _transcribe(self, audio) -> str:
        if isinstance(audio, AsrInput):
            audio = audio.pcm
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.monotonic()
        with InferenceWatchdog("Parakeet ONNX ASR transcription"):
            result = self.model.recognize(audio, sample_rate=SAMPLE_RATE)
        self.last_transcribe_seconds = time.monotonic() - started
        return getattr(result, "text", str(result)).strip()

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if isinstance(audio_input, Generator):
            return (
                {"text": self._transcribe(chunk)}
                for chunk in audio_input
                if chunk is not None
            )
        return [{"text": self._transcribe(audio_input)}]


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load the int8 Parakeet v3 ONNX model with CPUExecutionProvider only."""
    quantize = str(opts.get("quantize", "int8")).lower()
    if quantize not in ("int8", "none"):
        raise ValueError("Parakeet ONNX quantize must be 'int8' or 'none'")

    import onnx_asr
    import onnxruntime as ort

    model_dir = model_name or DEFAULT_MODEL_DIR
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    num_threads = int(opts.get("num_threads", 0))
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
    session_options.log_severity_level = 3

    logger.info("Loading Parakeet TDT v3 ONNX (%s) from %s", quantize, model_dir)
    if language:
        logger.info("Parakeet TDT v3 auto-detects its supported language; ignoring hint %r", language)
    model = onnx_asr.load_model(
        MODEL_TYPE,
        model_dir,
        quantization=None if quantize == "none" else quantize,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return ParakeetOnnxASRPipelineWrapper(model)
