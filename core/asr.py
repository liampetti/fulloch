"""Qwen3-ASR streaming pipeline."""

import logging
import time
from typing import Generator, Optional, Union

import numpy as np
import torch

from .inference_safety import InferenceWatchdog

logger = logging.getLogger(__name__)


class AsrInput:
    """A queued audio buffer with an optional per-call ASR context override."""

    def __init__(self, pcm: np.ndarray, context: Optional[str] = None):
        self.pcm = pcm
        self.context = context

ASR_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
SAMPLE_RATE = 16000  # Qwen3-ASR standard sample rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class QwenASRPipelineWrapper:
    """Mimics the HF pipeline streaming API on top of Qwen3ASRModel."""

    def __init__(self, model, language: Optional[str] = None):
        self.model = model
        self.language = language
        # Wall time of the most recent transcribe() call — read by the
        # transcriber thread to seed a turn's ASR stat. Measured here (not at
        # the consumer loop) so it excludes idle waiting on the audio queue.
        self.last_transcribe_seconds = None
        # Optional system-prompt context injected into every transcribe() call.
        # Biases the decoder toward named terms (wakeword, domain vocab).
        # Empty string = no effect. Set once at model load; never mutated after.
        self.context: str = ""

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # batch_size / generate_kwargs are accepted for HF-pipeline parity but
        # ignored — Qwen3ASRModel.transcribe handles batching internally.
        if isinstance(audio_input, Generator):
            return self._stream(audio_input)

        # Non-streaming path: single array in, list of dicts out.
        lang_kwargs = {"language": self.language} if self.language else {}
        context = self.context if not isinstance(audio_input, AsrInput) or audio_input.context is None else audio_input.context
        if isinstance(audio_input, AsrInput):
            audio_input = audio_input.pcm
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        with InferenceWatchdog("Qwen3 ASR PyTorch transcription"):
            results = self.model.transcribe(
                audio=[(audio_input, SAMPLE_RATE)], context=context, **lang_kwargs
            )
        results = results if isinstance(results, list) else [results]
        return [{"text": getattr(r, "text", str(r))} for r in results]

    def _stream(self, audio_input: Generator) -> Generator:
        """Yield transcriptions without making ``__call__`` a generator itself."""
        lang_kwargs = {"language": self.language} if self.language else {}
        for chunk in audio_input:
            if chunk is None:
                continue
            context = self.context if not isinstance(chunk, AsrInput) or chunk.context is None else chunk.context
            if isinstance(chunk, AsrInput):
                chunk = chunk.pcm
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.cpu().numpy()
            elif not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk)
            _t0 = time.monotonic()
            with InferenceWatchdog("Qwen3 ASR PyTorch transcription"):
                results = self.model.transcribe(
                    audio=[(chunk, SAMPLE_RATE)],
                    context=context,
                    return_time_stamps=False,
                    **lang_kwargs,
                )
            self.last_transcribe_seconds = time.monotonic() - _t0
            for res in results if isinstance(results, list) else [results]:
                yield {"text": getattr(res, "text", str(res))}


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load the Qwen3-ASR model and wrap it in the streaming pipeline.

    `model_name` defaults to the built-in `ASR_MODEL_NAME`; the backend
    registry passes the configured model id. `opts` absorbs any extra
    per-model config keys forwarded by the registry.
    """
    # Imported here (not at module top) so core.asr — and the CPU backends that
    # re-export stream_generator from it — import without qwen_asr (CPU image).
    from qwen_asr import Qwen3ASRModel

    model_name = model_name or ASR_MODEL_NAME
    logger.info(f"Loading {model_name} on {DEVICE}...")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )
    if language:
        logger.info(f"ASR language locked to: {language!r}")
    return QwenASRPipelineWrapper(model, language=language)


def stream_generator(
    queue,
    onset_sink: Optional[dict] = None,
    loudness_sink: Optional[dict] = None,
    provisional_sink: Optional[dict] = None,
    audio_sink: Optional[dict] = None,
    satellite_id_sink: Optional[dict] = None,
    endpoint_wait_sink: Optional[dict] = None,
    wake_probe_sink: Optional[dict] = None,
    verification_context: Optional[str] = None,
    kws_candidate_sink: Optional[dict] = None,
    kws_wav_path_sink: Optional[dict] = None,
) -> Generator:
    """Yield audio buffers from a queue until a None sentinel.

    Queue items are
    `(buf, speech_onset_monotonic, loudness_dbfs[, provisional[, satellite_id[,
    endpoint_monotonic[, wake_probe[, kws_candidate[, kws_wav_path]]]]]])` tuples (everything past `buf`/`onset`
    is optional
    for backward compatibility). When `onset_sink` / `loudness_sink` /
    `provisional_sink` / `audio_sink` / `satellite_id_sink` / `endpoint_wait_sink`
    are given, each yielded buffer's onset time, dBFS volume, provisional flag
    (True for a soft-endpoint snapshot of an unfinished utterance), the buffer
    itself, the id of the satellite that recorded it, and the A2
    endpoint-wait (dequeue time here minus `endpoint_monotonic` — how long the
    utterance sat queued after the recorder detected its end, before ASR
    picked it up) are written to `onset_sink['t']` / `loudness_sink['db']` /
    `provisional_sink['flag']` / `audio_sink['buf']` / `satellite_id_sink['id']`
    / `endpoint_wait_sink['s']` / `wake_probe_sink['flag']` before it's yielded,
    so the consumer can
    correlate the resulting transcription with them (and re-transcribe the
    same audio if needed). Generators are lazy, so no further item is
    dequeued between yielding a buffer and the consumer receiving its
    transcription — the sinks stay correct for that result.
    """
    while (item := queue.get()) is not None:
        dequeue_t = time.monotonic()
        buf, onset_t = item[0], item[1]
        loudness_db = item[2] if len(item) > 2 else None
        provisional = item[3] if len(item) > 3 else False
        satellite_id = item[4] if len(item) > 4 else None
        endpoint_t = item[5] if len(item) > 5 else None
        wake_probe = item[6] if len(item) > 6 else False
        kws_candidate = item[7] if len(item) > 7 else False
        kws_wav_path = item[8] if len(item) > 8 else None
        if onset_sink is not None:
            onset_sink["t"] = onset_t
        if loudness_sink is not None:
            loudness_sink["db"] = loudness_db
        if provisional_sink is not None:
            provisional_sink["flag"] = provisional
        if audio_sink is not None:
            audio_sink["buf"] = buf
        if satellite_id_sink is not None:
            satellite_id_sink["id"] = satellite_id
        if endpoint_wait_sink is not None:
            endpoint_wait_sink["s"] = (dequeue_t - endpoint_t) if endpoint_t is not None else None
        if wake_probe_sink is not None:
            wake_probe_sink["flag"] = wake_probe
        if kws_candidate_sink is not None:
            kws_candidate_sink["flag"] = kws_candidate
        if kws_wav_path_sink is not None:
            kws_wav_path_sink["path"] = kws_wav_path
        yield AsrInput(buf, verification_context) if kws_candidate else buf
