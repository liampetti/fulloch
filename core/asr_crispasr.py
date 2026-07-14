"""Qwen3-ASR GGUF through an isolated CUDA CrispASR worker."""

import logging
import time
from typing import Generator, Optional, Union

import numpy as np

from .asr import SAMPLE_RATE, stream_generator  # noqa: F401
from .crispasr_worker import CrispASRWorker

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./data/models/qwen3-asr-1.7b-q4_k.gguf"
DEFAULT_LIB_DIR = "/opt/crispasr-python-cuda"


class CrispASRPipelineWrapper:
    """Pipeline-compatible Qwen3-ASR wrapper backed by the worker process."""

    def __init__(self, worker, language=None, worker_config=None):
        self.worker = worker
        self.language = language
        self._worker_config = worker_config
        self.context = ""
        self.last_transcribe_seconds = None

    def _restart_worker(self) -> bool:
        """Restore ASR after a native CUDA abort without killing the assistant."""
        if self.worker.alive or not self._worker_config:
            return False
        logger.warning("CrispASR ASR worker exited; starting a fresh worker")
        self.worker.close()
        self.worker = CrispASRWorker(**self._worker_config)
        return True

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        def transcribe(audio):
            started = time.monotonic()
            payload = {
                "audio": np.asarray(audio, dtype=np.float32),
                "language": self.language,
                "context": self.context,
            }
            try:
                text = self.worker.call("transcribe", **payload)
            except RuntimeError:
                if not self._restart_worker():
                    raise
                try:
                    text = self.worker.call("transcribe", **payload)
                except RuntimeError:
                    # The next utterance retries with a fresh worker. Returning
                    # no text keeps one failed barge-in from killing the loop.
                    logger.exception("CrispASR ASR remained unavailable after restart")
                    text = ""
            self.last_transcribe_seconds = time.monotonic() - started
            return {"text": text}

        if isinstance(audio_input, Generator):
            return (transcribe(chunk) for chunk in audio_input if chunk is not None)
        return [transcribe(audio_input)]


def load_asr_model(model_name=None, language=None, **opts):
    worker_config = {
        "model_path": model_name or DEFAULT_MODEL_PATH,
        "lib_dir": opts.get("lib_dir", DEFAULT_LIB_DIR),
        "backend": "qwen3",
        "num_threads": int(opts.get("num_threads", 4)),
    }
    worker = CrispASRWorker(**worker_config)
    return CrispASRPipelineWrapper(worker, language=language, worker_config=worker_config)
