"""Moonshine CPU ASR backend. It accepts but does not use context biasing."""

import logging
import time
from typing import Generator, Optional, Union

import numpy as np
import torch

# Generic queue drainer — backend-agnostic, re-exported so the assistant can
# pull it from whichever ASR module the registry selected.
from .asr import stream_generator  # noqa: F401
from .inference_safety import InferenceWatchdog

logger = logging.getLogger(__name__)

ASR_MODEL_NAME = "UsefulSensors/moonshine-base"
SAMPLE_RATE = 16000  # Moonshine expects 16 kHz mono
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def _to_array(chunk) -> np.ndarray:
    """Coerce a queue buffer (tensor / list / ndarray) to a float32 array."""
    if isinstance(chunk, torch.Tensor):
        return chunk.detach().cpu().numpy()
    if not isinstance(chunk, np.ndarray):
        return np.asarray(chunk, dtype=np.float32)
    return chunk


class MoonshineASRPipelineWrapper:
    """Mimics the streaming pipeline API on top of an HF Moonshine pipeline."""

    def __init__(self, pipe, language: Optional[str] = None):
        self.pipe = pipe
        # Accepted for parity with the Qwen wrapper. Moonshine-tiny is
        # English-only and has no language switch, so this is informational.
        self.language = language
        self.last_transcribe_seconds: Optional[float] = None
        # Accepted for parity; Moonshine can't bias the decoder on a context
        # prompt, so setting this is a no-op (left writable so the assistant's
        # asr_context_hint plumbing doesn't need a backend special-case).
        self.context: str = ""

    def _transcribe(self, arr: np.ndarray, generate_kwargs: dict) -> str:
        _t0 = time.monotonic()
        with InferenceWatchdog("Moonshine ASR transcription"):
            result = self.pipe(
                {"raw": arr, "sampling_rate": SAMPLE_RATE},
                generate_kwargs=generate_kwargs or None,
            )
        self.last_transcribe_seconds = time.monotonic() - _t0
        if isinstance(result, list):
            result = result[0] if result else {}
        return (result or {}).get("text", "") if isinstance(result, dict) else str(result)

    def _stream(self, audio_input: Generator, generate_kwargs: dict) -> Generator:
        for chunk in audio_input:
            if chunk is None:
                continue
            yield {"text": self._transcribe(_to_array(chunk), generate_kwargs)}

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # batch_size is accepted for HF-pipeline parity but ignored. Keeping
        # the streaming logic in `_stream` (not inlined here with a `yield`)
        # means `__call__` isn't itself a generator, so the non-streaming
        # branch can actually return its list.
        generate_kwargs = generate_kwargs or {}
        if isinstance(audio_input, Generator):
            return self._stream(audio_input, generate_kwargs)
        return [{"text": self._transcribe(_to_array(audio_input), generate_kwargs)}]


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load the Moonshine model and wrap it in the streaming pipeline.

    `model_name` defaults to the built-in `ASR_MODEL_NAME`; the registry passes
    the configured model id. `opts` absorbs any extra per-model config keys.
    """
    from transformers import (
        AutoProcessor,
        MoonshineForConditionalGeneration,
        pipeline,
    )

    model_name = model_name or ASR_MODEL_NAME
    logger.info(f"Loading {model_name} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = MoonshineForConditionalGeneration.from_pretrained(model_name).to(
        device=DEVICE, dtype=DTYPE
    )
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=DEVICE,
        dtype=DTYPE,
    )
    if language:
        logger.info(f"ASR language requested ({language!r}); Moonshine-tiny is English-only")
    return MoonshineASRPipelineWrapper(pipe, language=language)
