"""NVIDIA Parakeet TDT ASR backend."""

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Generator, Optional, Union

import numpy as np

from .asr import SAMPLE_RATE, AsrInput, stream_generator  # noqa: F401
from .inference_safety import InferenceWatchdog

logger = logging.getLogger(__name__)

ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
# Parakeet's live path re-decodes a rolling buffer, so shorter first/refresh
# delays improve wake feedback and safe early commits without queueing long jobs.
PARTIAL_INITIAL_SECONDS = 0.75
PARTIAL_INTERVAL_SECONDS = 1.0
PARTIAL_WINDOW_SECONDS = 6.0


@dataclass
class LiveAsrResult:
    """A worker-produced transcript that must not be transcribed again."""

    text: str
    pcm: np.ndarray
    transcribe_seconds: Optional[float] = None


@dataclass
class LiveAsrEvent:
    result: LiveAsrResult
    item: tuple
    final: bool


@dataclass
class LiveAsrFailure:
    """A fatal live-ASR error delivered to the assistant event router."""

    error: Exception


class ParakeetLiveWorker:
    """Bounded live-frame worker for Parakeet's non-streaming NeMo API."""

    def __init__(self, model):
        self.model = model
        self._frames = queue.Queue(maxsize=64)
        self._events = queue.Queue(maxsize=16)
        self._buffers = defaultdict(deque)
        self._buffer_samples = defaultdict(int)
        self._last_partial = defaultdict(float)
        self._started_at = {}
        self._active = set()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="parakeet-live-asr", daemon=True)
        self._thread.start()

    def feed_frame(self, satellite_id, pcm) -> None:
        try:
            self._frames.put_nowait(("frame", satellite_id, np.asarray(pcm, dtype=np.float32)))
        except queue.Full:
            pass

    def start(self, satellite_id, pcm) -> None:
        """Begin a VAD-confirmed capture with its short speech pre-roll."""
        try:
            self._frames.put_nowait(("start", satellite_id, np.asarray(pcm, dtype=np.float32)))
        except queue.Full:
            pass

    def finish(self, item) -> bool:
        try:
            self._frames.put_nowait(("final", item[4], item))
            return True
        except queue.Full:
            # Preserve an endpoint over a stale partial frame.
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(("final", item[4], item))
                return True
            except queue.Empty:
                return False

    def discard(self, satellite_id) -> None:
        """Drop an abandoned capture without blocking the recorder."""
        try:
            self._frames.put_nowait(("discard", satellite_id, None))
        except queue.Full:
            # The next endpoint is more important than stale partial state.
            pass

    def get_event(self, timeout=0.5):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _transcribe(self, pcm) -> tuple[str, float]:
        started = time.monotonic()
        with InferenceWatchdog("Parakeet live ASR transcription"):
            results = self.model.transcribe(
                [pcm], batch_size=1, return_hypotheses=True, verbose=False
            )
        result = results[0] if results else ""
        return getattr(result, "text", str(result)), time.monotonic() - started

    def _emit(self, event) -> None:
        try:
            self._events.put_nowait(event)
        except queue.Full:
            if getattr(event, "final", False) or isinstance(event, LiveAsrFailure):
                try:
                    self._events.get_nowait()
                    self._events.put_nowait(event)
                except queue.Empty:
                    pass

    def _append(self, satellite_id, pcm) -> None:
        buffer = self._buffers[satellite_id]
        buffer.append(pcm)
        self._buffer_samples[satellite_id] += pcm.size
        limit = int(PARTIAL_WINDOW_SECONDS * SAMPLE_RATE)
        while self._buffer_samples[satellite_id] > limit and buffer:
            excess = self._buffer_samples[satellite_id] - limit
            oldest = buffer[0]
            if oldest.size <= excess:
                self._buffer_samples[satellite_id] -= oldest.size
                buffer.popleft()
            else:
                buffer[0] = oldest[excess:]
                self._buffer_samples[satellite_id] -= excess

    def _reset_capture(self, satellite_id) -> None:
        self._buffers.pop(satellite_id, None)
        self._buffer_samples.pop(satellite_id, None)
        self._last_partial.pop(satellite_id, None)
        self._started_at.pop(satellite_id, None)
        self._active.discard(satellite_id)

    def _run(self) -> None:
        while not self._closed.is_set():
            command = self._frames.get()
            if command is None:
                return
            kind, satellite_id, payload = command
            if kind == "start":
                self._reset_capture(satellite_id)
                self._active.add(satellite_id)
                self._started_at[satellite_id] = time.monotonic()
                self._append(satellite_id, payload)
            elif kind == "frame":
                if satellite_id not in self._active:
                    continue
                self._append(satellite_id, payload)
                now = time.monotonic()
                if self._last_partial[satellite_id] == 0:
                    if now - self._started_at[satellite_id] < PARTIAL_INITIAL_SECONDS:
                        continue
                elif now - self._last_partial[satellite_id] < PARTIAL_INTERVAL_SECONDS:
                    continue
                pcm = np.concatenate(self._buffers[satellite_id])
                self._last_partial[satellite_id] = now
                item = (
                    pcm,
                    time.monotonic() - pcm.size / SAMPLE_RATE,
                    None,
                    True,
                    satellite_id,
                    time.monotonic(),
                )
                try:
                    text, transcribe_seconds = self._transcribe(pcm)
                    self._emit(
                        LiveAsrEvent(
                            LiveAsrResult(text, pcm, transcribe_seconds), item, False
                        )
                    )
                except Exception as exc:
                    logger.exception("Parakeet live partial failed")
                    self._emit(LiveAsrFailure(exc))
                    self._closed.set()
                    return
            elif kind == "discard":
                self._reset_capture(satellite_id)
            else:
                item = payload
                pcm = np.asarray(item[0], dtype=np.float32)
                self._reset_capture(satellite_id)
                try:
                    text, transcribe_seconds = self._transcribe(pcm)
                    self._emit(
                        LiveAsrEvent(
                            LiveAsrResult(text, pcm, transcribe_seconds), item, True
                        )
                    )
                except Exception as exc:
                    logger.exception("Parakeet live final failed")
                    self._emit(LiveAsrFailure(exc))
                    self._closed.set()
                    return


class ParakeetASRPipelineWrapper:
    """Pipeline-compatible wrapper around NeMo's Parakeet TDT model."""

    def __init__(self, model):
        self.model = model
        self.live_worker = ParakeetLiveWorker(model)
        self.context = ""
        # NeMo's boosting tree changes the shared decoder, so temporarily
        # dropping the bias for bare-wakeword verification is not safe while
        # the live worker may be transcribing.
        self.supports_context_unbias = False
        self.last_transcribe_seconds = None

    def set_context_phrases(self, phrases: list[str]) -> None:
        """Bias NeMo's TDT decoder toward a small, fixed phrase list."""
        phrases = list(dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip()))
        if not phrases:
            return
        # NeMo 3.0 does not consume per-phrase alpha. All terms share the
        # global boosting-tree weight, including during normal command ASR.
        current_cfg = self.model.cfg.decoding
        try:
            from omegaconf import OmegaConf, open_dict

            decoding_cfg = OmegaConf.create(
                OmegaConf.to_container(current_cfg, resolve=True)
                if OmegaConf.is_config(current_cfg)
                else current_cfg
            )
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy_batch"
                if "greedy" not in decoding_cfg:
                    decoding_cfg.greedy = {}
                decoding_cfg.greedy.boosting_tree = {
                    "key_phrases_list": phrases,
                    "context_score": 2.0,
                    "depth_scaling": 2.0,
                }
                decoding_cfg.greedy.boosting_tree_alpha = 2.0
        except ModuleNotFoundError:
            # The lightweight unit seam does not install NeMo/OmegaConf.
            decoding_cfg = deepcopy(current_cfg)
            if isinstance(decoding_cfg, dict):
                decoding_cfg = SimpleNamespace(**decoding_cfg)
            decoding_cfg.strategy = "greedy_batch"
            if not hasattr(decoding_cfg, "greedy"):
                decoding_cfg.greedy = SimpleNamespace()
            decoding_cfg.greedy.boosting_tree = SimpleNamespace(
                key_phrases_list=phrases,
                context_score=2.0,
                depth_scaling=2.0,
            )
            decoding_cfg.greedy.boosting_tree_alpha = 2.0
        self.model.change_decoding_strategy(decoding_cfg, verbose=False)
        logger.info(
            "NeMo ASR phrase boosting enabled for %d terms (global alpha=2.0)",
            len(phrases),
        )

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        def transcribe(audio):
            if isinstance(audio, AsrInput):
                audio = audio.pcm
            if isinstance(audio, LiveAsrResult):
                self.last_transcribe_seconds = audio.transcribe_seconds
                return {"text": audio.text}
            started = time.monotonic()
            with InferenceWatchdog("Parakeet ASR transcription"):
                results = self.model.transcribe(
                    [np.asarray(audio, dtype=np.float32)],
                    batch_size=1,
                    return_hypotheses=True,
                    verbose=False,
                )
            self.last_transcribe_seconds = time.monotonic() - started
            result = results[0] if results else ""
            return {"text": getattr(result, "text", str(result))}

        if isinstance(audio_input, Generator):
            return (transcribe(chunk) for chunk in audio_input if chunk is not None)
        return [transcribe(audio_input)]


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load Parakeet on CUDA through NVIDIA NeMo.

    The model auto-detects one of its supported languages, so ``language`` is
    intentionally not forwarded as a decoder prompt.
    """
    import torch
    from nemo.collections.asr.models import ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet TDT requires CUDA in Fulloch's GPU image")
    model_name = model_name or ASR_MODEL_NAME
    logger.info("Loading %s on CUDA", model_name)
    model = ASRModel.from_pretrained(model_name=model_name)
    model.eval().freeze()
    model.cuda()
    return ParakeetASRPipelineWrapper(model)
