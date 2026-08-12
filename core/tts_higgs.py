"""Higgs TTS 3 GGUF backend through an isolated native server process."""

import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .higgs_controls import sanitize_for_higgs, split_leading_higgs_controls
from .higgs_worker import HiggsWorker
from .text_utils import split_clauses, split_sentences
from .tts_crispasr import VOICES_DIR
from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CHUNK_SAMPLES = SAMPLE_RATE // 2
DEFAULT_MODEL_DIR = "./data/models/higgs-tts-v3-q4"
DEFAULT_SERVER_PATH = "/opt/higgs-tts/higgs_server"
DEFAULT_RUNTIME_DIR = "/opt/higgs-tts"
MODEL_FILE = "higgs-v3-tts-q4_k.gguf"
TOKENIZER_FILE = "higgs_tts_v3_tokenizer.json"
# A 256-action Higgs request caps at about ten seconds of audio. Keep requests
# comfortably below that limit so a weather/calendar reply is never cut off.
MAX_SYNTHESIS_CHARS = 100
MAX_REQUESTS_PER_WORKER = 6

_worker: HiggsWorker | None = None
_settings: dict | None = None
_voice_prompt: tuple[str, str] | None = None
_request_count = 0
_lock = threading.Lock()


def load_tts(
    model_id: str = DEFAULT_MODEL_DIR,
    server_path: str = DEFAULT_SERVER_PATH,
    runtime_dir: str = DEFAULT_RUNTIME_DIR,
    max_actions: int = 256,
    **_opts,
):
    """Configure Higgs assets; the voice-selected process starts in set_voice."""
    root = Path(model_id)
    model_path = root / MODEL_FILE
    tokenizer_path = root / TOKENIZER_FILE
    for path in (model_path, tokenizer_path, Path(server_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Higgs asset not found: {path}")
    if int(max_actions) <= 0:
        raise ValueError("Higgs max_actions must be positive")
    global _settings, _worker, _voice_prompt
    if _worker is not None:
        _worker.close()
    _worker = None
    _voice_prompt = None
    _settings = {
        "server_path": server_path,
        "runtime_dir": runtime_dir,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "max_actions": int(max_actions),
    }


def _start_worker() -> HiggsWorker:
    if _settings is None or _voice_prompt is None:
        raise RuntimeError("Higgs TTS is not configured with a voice reference")
    global _worker
    if _worker is not None and _worker.alive:
        return _worker
    audio, text = _voice_prompt
    _worker = HiggsWorker(
        **_settings,
        reference_wav=audio,
        reference_text=text,
    )
    _worker.start()
    return _worker


def _recycle_worker_if_needed() -> None:
    global _worker, _request_count
    if _request_count < MAX_REQUESTS_PER_WORKER:
        return
    if _worker is not None:
        _worker.close()
    _worker = None
    _request_count = 0


def _synthesis_fragments(text: str):
    """Yield pause-delimited spans, splitting long clauses only as needed.

    Each request gets an independent autoregressive decode, so use sentences
    first. A comma only subdivides an over-limit sentence; short comma clauses
    such as a person's name otherwise stutter when synthesized independently.
    """
    for sentence in split_sentences(text) or [text]:
        clauses = list(split_clauses(sentence)) if len(sentence) > MAX_SYNTHESIS_CHARS else [sentence]
        for clause in clauses:
            remaining = clause.strip()
            while len(remaining) > MAX_SYNTHESIS_CHARS:
                cut = remaining.rfind(" ", 0, MAX_SYNTHESIS_CHARS + 1)
                if cut <= 0:
                    cut = MAX_SYNTHESIS_CHARS
                yield remaining[:cut].strip()
                remaining = remaining[cut:].strip()
            if remaining:
                yield remaining


def _synthesize_fragment(text: str) -> np.ndarray:
    global _request_count
    with _lock:
        _recycle_worker_if_needed()
        audio = _start_worker().synthesize(sanitize_for_higgs(text))
        _request_count += 1
    return audio


def set_voice(voice_name: str):
    """Set the clone reference and restart the warm server for that voice."""
    txt_path = Path(VOICES_DIR) / f"{voice_name}.txt"
    if not txt_path.is_file():
        txt_path = Path(VOICES_DIR) / "default.txt"
    ref_audio = Path(VOICES_DIR) / f"{voice_name}.wav"
    if not ref_audio.is_file() or not txt_path.is_file():
        raise FileNotFoundError(f"Missing Higgs voice reference for {voice_name}")
    global _voice_prompt, _worker
    with _lock:
        if _worker is not None:
            _worker.close()
            _worker = None
        _voice_prompt = (str(ref_audio), txt_path.read_text().strip())
        _start_worker()
    return str(ref_audio)


def set_speed(speed) -> None:
    if speed not in (None, 1.0):
        logger.warning("Higgs TTS speed is controlled by asking it to speak more slowly or quickly.")


def force_cancel_playback(sink: Optional["queue.Queue"] = None) -> None:
    global _worker
    if sink is not None:
        while True:
            try:
                sink.get_nowait()
            except queue.Empty:
                break
        sink.put(("cancel",))
    # The native TCP protocol has no request-cancel frame. Terminating this
    # isolated process bounds a cancelled generation; the next request starts a
    # fresh warm worker with the same configured voice reference.
    worker, _worker = _worker, None
    if worker is not None:
        worker.close()


def shutdown() -> None:
    """Stop the isolated Higgs server before application shutdown."""
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.close()


def synthesize(text: str, prompt=None):
    return [_synthesize_fragment(fragment) for fragment in _synthesis_fragments(text)], SAMPLE_RATE


def warmup_model(prompt=None):
    synthesize("Higgs voice synthesis is ready.")


def play_chunks(
    chunks,
    sample_rate: int,
    session: Optional[TtsSession] = None,
    sink: Optional["queue.Queue"] = None,
    tts_active_event: Optional[threading.Event] = None,
):
    """Play cached clips using the same sink contract as other TTS backends."""
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    try:
        if not chunks or session.cancelled or sink is None:
            return
        if tts_active_event is not None:
            tts_active_event.set()
        sink.put(("start", sample_rate))
        for chunk in chunks:
            if session.cancelled:
                force_cancel_playback(sink)
                return
            sink.put((chunk, None))
        sink.put(("end",))
    finally:
        if tts_active_event is not None:
            tts_active_event.clear()
        session.active = False


def speak_stream(
    text: str,
    prompt=None,
    session: Optional[TtsSession] = None,
    stats: Optional[TurnStats] = None,
    on_first_audio: Optional[Callable[[], None]] = None,
    sink: Optional["queue.Queue"] = None,
    tts_active_event: Optional[threading.Event] = None,
):
    """Forward native Higgs PCM windows as soon as their decoder lookahead is ready."""
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    started = time.monotonic()
    try:
        playback_start = None
        sent = 0
        leading_controls, speech_text = split_leading_higgs_controls(sanitize_for_higgs(text))
        # Delivery controls must survive each bounded native request. SFX are
        # intentionally excluded after the first fragment: repeating a cough
        # or laugh at every boundary changes the requested content.
        persistent_controls = re.sub(r"<\|sfx:[a-z_]+\|>", "", leading_controls)
        fragments = list(_synthesis_fragments(speech_text))
        for fragment_index, fragment in enumerate(fragments):
            if session.cancelled:
                force_cancel_playback(sink)
                return time.monotonic()
            with _lock:
                _recycle_worker_if_needed()
                controls = leading_controls if fragment_index == 0 else persistent_controls
                request_text = f"{controls}{fragment}"
                logger.debug(
                    "Higgs fragment %d/%d request: chars=%d text=%r",
                    fragment_index + 1,
                    len(fragments),
                    len(request_text),
                    request_text,
                )
                pcm_frames = _start_worker().synthesize_stream(request_text)
                fragment_samples = 0
                try:
                    for audio in pcm_frames:
                        if session.cancelled:
                            force_cancel_playback(sink)
                            return time.monotonic()
                        if sink is None:
                            return time.monotonic()
                        if playback_start is None:
                            if stats is not None:
                                stats.tts_seconds = time.monotonic() - started
                            if on_first_audio is not None:
                                on_first_audio()
                            if tts_active_event is not None:
                                tts_active_event.set()
                            playback_start = time.monotonic()
                            logger.info("Higgs native first PCM after %.3fs", playback_start - started)
                            sink.put(("start", SAMPLE_RATE))
                        # Frames are native decoder windows, not slices of a completed clip.
                        sink.put((audio, None))
                        sent += len(audio)
                        fragment_samples += len(audio)
                except ConnectionError:
                    # force_cancel_playback terminates the isolated native
                    # worker because its protocol has no cancellation frame.
                    # That intentionally breaks this generator's TCP read.
                    if session.cancelled:
                        return time.monotonic()
                    raise
                finally:
                    pcm_frames.close()
                global _request_count
                _request_count += 1
                logger.debug(
                    "Higgs fragment %d/%d complete: audio=%.2fs",
                    fragment_index + 1,
                    len(fragments),
                    fragment_samples / SAMPLE_RATE,
                )
        sink.put(("end",))
        return (playback_start or time.monotonic()) + sent / SAMPLE_RATE
    finally:
        if tts_active_event is not None:
            tts_active_event.clear()
        session.active = False
