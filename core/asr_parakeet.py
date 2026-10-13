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
    generation: Optional[int] = None
    capture_id: Optional[int] = None


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
        self._capture_ids = {}
        self._capture_generations = {}
        # A rejection must take effect immediately in the recorder thread, not
        # after stale frame commands happen to drain from this worker's FIFO.
        # Commands and results carry this per-satellite generation so an
        # in-flight CUDA call cannot route obsolete speech after cancellation.
        self._generations = defaultdict(int)
        self._retired_captures = defaultdict(int)
        self._generation_lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="parakeet-live-asr", daemon=True)
        self._thread.start()

    def feed_frame(self, satellite_id, pcm, *, capture_id=None) -> None:
        generation = self._generation(satellite_id)
        if not self._is_current(satellite_id, generation, capture_id):
            return
        try:
            self._frames.put_nowait(
                ("frame", satellite_id, generation, capture_id, np.asarray(pcm, dtype=np.float32))
            )
        except queue.Full:
            pass

    def start(self, satellite_id, pcm, *, capture_id=None) -> None:
        """Begin a VAD-confirmed capture with its short speech pre-roll."""
        generation = self._generation(satellite_id)
        if not self._is_current(satellite_id, generation, capture_id):
            return
        try:
            self._frames.put_nowait(
                ("start", satellite_id, generation, capture_id, np.asarray(pcm, dtype=np.float32))
            )
        except queue.Full:
            pass

    def finish(self, item) -> bool:
        satellite_id = item[4]
        generation = self._generation(satellite_id)
        capture_id = item[12] if len(item) > 12 else None
        if not self._is_current(satellite_id, generation, capture_id):
            return True  # already accepted; no duplicate inference needed
        command = ("final", satellite_id, generation, capture_id, item)
        try:
            # Endpoint and wake-verification work is authoritative. Drop queued
            # rolling partials for this capture and run it next rather than
            # making it wait behind stale live snapshots.
            if not (len(item) > 10 and item[10]):
                self._discard_queued_frames(
                    satellite_id, generation, capture_id=capture_id, partials_only=True
                )
            with self._frames.mutex:
                if self._frames.maxsize > 0 and len(self._frames.queue) >= self._frames.maxsize:
                    return False
                self._frames.queue.appendleft(command)
                self._frames.not_empty.notify()
            return True
        except queue.Full:
            return False

    def discard(self, satellite_id) -> None:
        """Immediately invalidate pending live work for an abandoned capture."""
        with self._generation_lock:
            old_generation = self._generations[satellite_id]
            self._generations[satellite_id] += 1
            generation = self._generations[satellite_id]
        # A concurrently started new generation must not be removed.
        self._discard_queued_frames(satellite_id, old_generation)
        self._queue_reset(satellite_id, generation)

    def discard_capture(self, satellite_id, capture_id) -> None:
        """Retire a committed capture without invalidating newer speech."""
        with self._generation_lock:
            self._retired_captures[satellite_id] = max(
                self._retired_captures[satellite_id], capture_id
            )
            generation = self._generations[satellite_id]
        self._discard_queued_frames(satellite_id, capture_id=capture_id)
        self._queue_reset(satellite_id, generation, capture_id)

    def _queue_reset(self, satellite_id, generation, capture_id=None) -> None:
        # Buffer mutation belongs to the model worker. This control command is
        # non-blocking and never waits for an in-flight CUDA call. If the
        # queue is full, the next command's validity check performs the cleanup.
        with self._frames.mutex:
            if len(self._frames.queue) < self._frames.maxsize:
                self._frames.queue.appendleft(("reset", satellite_id, generation, capture_id, None))
                self._frames.not_empty.notify()

    def _generation(self, satellite_id) -> int:
        with self._generation_lock:
            return self._generations[satellite_id]

    def _is_current(self, satellite_id, generation, capture_id=None) -> bool:
        with self._generation_lock:
            return self._generations[satellite_id] == generation and (
                capture_id is None or capture_id > self._retired_captures[satellite_id]
            )

    def is_current_result(self, satellite_id, result) -> bool:
        """Recheck after emission/queueing as cancellation can happen at either boundary."""
        return result.generation is None or self._is_current(
            satellite_id, result.generation, result.capture_id
        )

    def _discard_queued_frames(
        self, satellite_id, generation=None, *, capture_id=None, partials_only=False
    ) -> None:
        """Remove stale partial commands without waiting for the worker loop.

        `queue.Queue` does not provide selective removal. Its queue is protected
        by `mutex`, so filtering it here is safe and lets cancellation free the
        backlog immediately. A currently executing NeMo call cannot be safely
        interrupted, but its generation check prevents it from emitting a stale
        result when it returns.
        """
        with self._frames.mutex:
            retained = deque(
                command
                for command in self._frames.queue
                if not (
                    command is not None
                    and command[1] == satellite_id
                    and (generation is None or command[2] <= generation)
                    and (not partials_only or command[0] in {"frame", "start"})
                    and (
                        capture_id is None
                        or (command[3] is not None and command[3] <= capture_id)
                    )
                )
            )
            if len(retained) != len(self._frames.queue):
                self._frames.queue.clear()
                self._frames.queue.extend(retained)
                self._frames.not_full.notify_all()

    def get_event(self, timeout=0.5):
        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self._events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            if not isinstance(event, LiveAsrEvent) or self.is_current_result(event.item[4], event.result):
                return event

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
        self._capture_ids.pop(satellite_id, None)
        self._capture_generations.pop(satellite_id, None)

    def _run(self) -> None:
        while not self._closed.is_set():
            command = self._frames.get()
            if command is None:
                return
            kind, satellite_id, generation, capture_id, payload = command
            if satellite_id in self._active and not self._is_current(
                satellite_id, self._capture_generations[satellite_id], self._capture_ids[satellite_id]
            ):
                self._reset_capture(satellite_id)
            if kind == "reset":
                # Cleanup above retires stale state but leaves a concurrently
                # started, current capture alone.
                continue
            if not self._is_current(satellite_id, generation, capture_id):
                continue
            if kind == "start":
                self._reset_capture(satellite_id)
                self._active.add(satellite_id)
                self._capture_ids[satellite_id] = capture_id
                self._capture_generations[satellite_id] = generation
                self._started_at[satellite_id] = time.monotonic()
                self._append(satellite_id, payload)
            elif kind == "frame":
                if satellite_id not in self._active or self._capture_ids[satellite_id] != capture_id:
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
                    False, False, None, None, False, None, capture_id,
                )
                try:
                    text, transcribe_seconds = self._transcribe(pcm)
                    if self._is_current(satellite_id, generation, capture_id):
                        self._emit(
                            LiveAsrEvent(
                                LiveAsrResult(text, pcm, transcribe_seconds, generation, capture_id),
                                item, False,
                            )
                        )
                except Exception as exc:
                    logger.exception("Parakeet live partial failed")
                    self._emit(LiveAsrFailure(exc))
                    self._closed.set()
                    return
            elif kind == "final":
                item = payload
                pcm = np.asarray(item[0], dtype=np.float32)
                early_verification = len(item) > 10 and item[10]
                if not early_verification and (
                    capture_id is None or self._capture_ids.get(satellite_id) == capture_id
                ):
                    self._reset_capture(satellite_id)
                try:
                    text, transcribe_seconds = self._transcribe(pcm)
                    if self._is_current(satellite_id, generation, capture_id):
                        self._emit(
                            LiveAsrEvent(
                                LiveAsrResult(text, pcm, transcribe_seconds, generation, capture_id),
                                item, True,
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
