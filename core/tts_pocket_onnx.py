"""Pocket TTS ONNX adapter for CPU one-shot voice cloning.

The model's maintained ONNX export supplies its own inference wrapper with the
model bundle.  We load that pinned bundle-local wrapper lazily, keeping setup
mode import-light and avoiding a second, divergent implementation of its
autoregressive state loop.
"""

import importlib.util
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .inference_safety import TTS_JOB_QUEUE_MAXSIZE, submit_tts_job
from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/pocket-tts-onnx"
SAMPLE_RATE = 24000
_CHUNK_SAMPLES = SAMPLE_RATE // 2
VOICES_DIR = Path("./data/voices")

_model = None
_voice: Optional[Path] = None


def load_tts(model_id: str = DEFAULT_MODEL_DIR, **opts):
    """Load the English INT8 Pocket TTS bundle on the CPU."""
    global _model
    root = Path(model_id)
    wrapper = root / "pocket_tts_onnx.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"Missing Pocket TTS wrapper at {wrapper}; re-run setup to fetch it")
    spec = importlib.util.spec_from_file_location("_fulloch_pocket_tts_onnx", wrapper)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Pocket TTS wrapper from {wrapper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _model = module.PocketTTSOnnx(
        models_dir=str(root / "onnx"),
        language="english_2026-04",
        precision="int8",
        device="cpu",
        temperature=float(opts.get("temperature", 0.7)),
        lsd_steps=int(opts.get("lsd_steps", 1)),
    )
    logger.info("Loaded Pocket TTS ONNX English INT8 bundle from %s", root)
    return _model


def set_voice(voice_name: Optional[str]):
    """Select a bundled voice reference; Pocket conditions on the WAV alone."""
    global _voice
    name = (voice_name or "").strip()
    wav_path = VOICES_DIR / f"{name}.wav"
    txt_path = VOICES_DIR / f"{name}.txt"
    if not txt_path.is_file():
        txt_path = VOICES_DIR / "default.txt"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Pocket TTS voice reference not found: {wav_path}")
    if not txt_path.is_file():
        raise FileNotFoundError(f"Pocket TTS transcript not found: {txt_path}")
    _voice = wav_path
    logger.info("Pocket TTS voice clone: %s (transcript %s validated; audio-only conditioning)", name, txt_path.name)
    return wav_path


def set_speed(speed) -> None:
    """Pocket TTS has no pitch-preserving rate control."""
    if speed not in (None, 1.0):
        logger.warning("Pocket TTS has no speed control; tts_speed=%s ignored", speed)


def _stream(text: str, voice: Path):
    if _model is None:
        raise RuntimeError("Pocket TTS is not loaded")
    for audio in _model.stream(text, voice=str(voice)):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        for start in range(0, len(audio), _CHUNK_SAMPLES):
            chunk = audio[start : start + _CHUNK_SAMPLES]
            if chunk.size:
                yield chunk


@dataclass
class _Job:
    text: str
    voice: Path
    out: "queue.Queue"
    session: TtsSession


_jobs: "queue.Queue[_Job]" = queue.Queue(maxsize=TTS_JOB_QUEUE_MAXSIZE)


def _worker_loop() -> None:
    while True:
        job = _jobs.get()
        try:
            for chunk in _stream(job.text, job.voice):
                if job.session.cancelled:
                    break
                # A cancelled consumer drains its queue.  Timeout rather than
                # block forever while it is racing that drain.
                while not job.session.cancelled:
                    try:
                        job.out.put((chunk, SAMPLE_RATE), timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except Exception as exc:  # noqa: BLE001 - must keep the shared worker alive
            logger.exception("Pocket TTS ONNX synth error: %s", exc)
        finally:
            while True:
                try:
                    job.out.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass


threading.Thread(target=_worker_loop, daemon=True, name="pocket-tts-onnx-worker").start()


def _submit(text: str, prompt: Optional[Path], session: TtsSession, maxsize: int = 8) -> "queue.Queue":
    voice = prompt or _voice
    if voice is None:
        raise RuntimeError("Pocket TTS voice is not configured")
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    submit_tts_job(_jobs, _Job(text=text, voice=voice, out=out, session=session))
    return out


def _drain(out: "queue.Queue") -> None:
    while out.get() is not None:
        pass


def _drain_nowait(out: "queue.Queue") -> None:
    while True:
        try:
            out.get_nowait()
        except queue.Empty:
            return


def force_cancel_playback(sink: Optional[queue.Queue] = None) -> None:
    if sink is not None:
        _drain_nowait(sink)
        sink.put(("cancel",))


def synthesize(text: str, prompt=None):
    out = _submit(text, prompt, TtsSession(), maxsize=0)
    chunks = []
    while (item := out.get()) is not None:
        chunks.append(item[0])
    return chunks, SAMPLE_RATE


def warmup_model(prompt=None):
    logger.info("Warming up Pocket TTS ONNX...")
    synthesize("The assistant is ready.", prompt)
    logger.info("Pocket TTS ONNX ready")


def play_chunks(chunks, sample_rate: int, session: Optional[TtsSession] = None,
                sink: Optional[queue.Queue] = None, tts_active_event: Optional[threading.Event] = None):
    """Deliver cached acknowledgements and phrase clips without synthesising."""
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    try:
        if not chunks or session.cancelled:
            return
        if tts_active_event is not None:
            tts_active_event.set()
        try:
            if sink is None:
                logger.warning("No satellite connected; TTS chunks dropped on the floor.")
                return
            sink.put(("start", sample_rate))
            for chunk in chunks:
                if session.cancelled:
                    _drain_nowait(sink)
                    sink.put(("cancel",))
                    return
                sink.put((chunk, None))
            sink.put(("end",))
        finally:
            if tts_active_event is not None:
                tts_active_event.clear()
    finally:
        session.active = False


def speak_stream(text: str, prompt=None, session: Optional[TtsSession] = None,
                 stats: Optional[TurnStats] = None, on_first_audio: Optional[Callable[[], None]] = None,
                 sink: Optional[queue.Queue] = None, tts_active_event: Optional[threading.Event] = None):
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    submitted = time.monotonic()
    out = _submit(text, prompt, session)
    try:
        first = out.get()
        if stats is not None:
            stats.tts_seconds = time.monotonic() - submitted
        if first is None or session.cancelled:
            if first is not None:
                _drain(out)
            return time.monotonic()
        if on_first_audio is not None:
            on_first_audio()
        if tts_active_event is not None:
            tts_active_event.set()
        start = time.monotonic()
        total_samples = 0
        try:
            if sink is None:
                _drain(out)
                return time.monotonic()
            sink.put(("start", SAMPLE_RATE))
            sink.put((first[0], None))
            total_samples += len(first[0])
            cancelled = False
            while (item := out.get()) is not None:
                if session.cancelled:
                    _drain(out)
                    cancelled = True
                    break
                sink.put((item[0], None))
                total_samples += len(item[0])
            if cancelled:
                _drain_nowait(sink)
                sink.put(("cancel",))
            else:
                sink.put(("end",))
            return start + total_samples / SAMPLE_RATE
        finally:
            if tts_active_event is not None:
                tts_active_event.clear()
    finally:
        session.active = False
