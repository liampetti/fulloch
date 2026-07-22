"""Qwen3-ASR streaming pipeline."""

import logging
import time
from typing import Generator, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

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
        lang_kwargs = {"language": self.language} if self.language else {}
        if isinstance(audio_input, Generator):
            for chunk in audio_input:
                if chunk is None:
                    continue
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                elif not isinstance(chunk, np.ndarray):
                    chunk = np.array(chunk)
                _t0 = time.monotonic()
                results = self.model.transcribe(
                    audio=[(chunk, SAMPLE_RATE)],
                    context=self.context,
                    return_time_stamps=False,
                    **lang_kwargs,
                )
                self.last_transcribe_seconds = time.monotonic() - _t0
                for res in results if isinstance(results, list) else [results]:
                    yield {"text": getattr(res, "text", str(res))}
            return

        # Non-streaming path: single array in, list of dicts out.
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        results = self.model.transcribe(
            audio=[(audio_input, SAMPLE_RATE)], context=self.context, **lang_kwargs
        )
        results = results if isinstance(results, list) else [results]
        return [{"text": getattr(r, "text", str(r))} for r in results]


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
) -> Generator:
    """Yield audio buffers from a queue until a None sentinel.

    Queue items are
    `(buf, speech_onset_monotonic, loudness_dbfs[, provisional[, satellite_id[,
    endpoint_monotonic]]])` tuples (everything past `buf`/`onset` is optional
    for backward compatibility). When `onset_sink` / `loudness_sink` /
    `provisional_sink` / `audio_sink` / `satellite_id_sink` / `endpoint_wait_sink`
    are given, each yielded buffer's onset time, dBFS volume, provisional flag
    (True for a soft-endpoint snapshot of an unfinished utterance), the buffer
    itself, the id of the satellite that recorded it, and the A2
    endpoint-wait (dequeue time here minus `endpoint_monotonic` — how long the
    utterance sat queued after the recorder detected its end, before ASR
    picked it up) are written to `onset_sink['t']` / `loudness_sink['db']` /
    `provisional_sink['flag']` / `audio_sink['buf']` / `satellite_id_sink['id']`
    / `endpoint_wait_sink['s']` before it's yielded, so the consumer can
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
        yield buf
