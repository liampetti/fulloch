"""Official Kyutai Pocket TTS PyTorch adapter with CUDA PCM streaming.

Pocket's generator yields audio frames as Mimi decodes them. A single shared
worker keeps CUDA model access serial, matching Qwen's worker contract, while
the consumer sends each frame to the browser immediately.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .inference_safety import TTS_JOB_QUEUE_MAXSIZE, InferenceWatchdog, submit_tts_job
from .text_utils import split_sentences
from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
_CHUNK_SAMPLES = SAMPLE_RATE // 2
_MAX_SYNTHESIS_CHARS = 240
VOICES_DIR = Path("./data/voices")

_POCKET_TEXT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
)

_model = None
_torch = None
_voice_state = None


def load_tts(model_id=None, language: str = "english_2026-04", device: str = "cuda", **opts):
    """Load the official Pocket model and place its tensors on CUDA.

    The setup downloader pre-caches this language's weights. The model is
    gated by Kyutai, so Hugging Face must have an accepted license and token.
    """
    global _model, _torch
    import torch
    from pocket_tts import TTSModel

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Pocket TTS PyTorch requires an available CUDA device")
    _torch = torch
    temperature = opts.get("temperature")
    _model = TTSModel.load_model(
        language=language,
        temp=0.7 if temperature is None else float(temperature),
        lsd_decode_steps=int(opts.get("lsd_decode_steps", 1)),
    ).to(device)
    _model.eval()
    logger.info("Loaded official Pocket TTS PyTorch on %s", device)
    return _model


def set_voice(voice_name: Optional[str]):
    """Prepare and retain the voice-clone state for subsequent streamed calls."""
    if _model is None:
        raise RuntimeError("Pocket TTS PyTorch is not loaded")
    name = (voice_name or "").strip()
    wav_path = VOICES_DIR / f"{name}.wav"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Pocket TTS voice reference not found: {wav_path}")
    global _voice_state
    _voice_state = _model.get_state_for_audio_prompt(wav_path)
    logger.info("Pocket TTS PyTorch voice clone ready: %s", name)
    return _voice_state


def set_speed(speed) -> None:
    if speed not in (None, 1.0):
        logger.warning("Pocket TTS has no speed control; tts_speed=%s ignored", speed)


def _prepare_text(text: str) -> str:
    return " ".join(text.translate(_POCKET_TEXT_TRANSLATION).split())


def _split_bounded(text: str):
    remaining = text.strip()
    while len(remaining) > _MAX_SYNTHESIS_CHARS:
        cut = remaining.rfind(" ", 0, _MAX_SYNTHESIS_CHARS + 1)
        if cut <= 0:
            cut = _MAX_SYNTHESIS_CHARS
        yield remaining[:cut].strip()
        remaining = remaining[cut:].strip()
    if remaining:
        yield remaining


def _synthesis_fragments(text: str):
    """Start the first sentence promptly, then combine later requests."""
    sentences = list(split_sentences(_prepare_text(text))) or [text]
    pending = ""
    for sentence_index, sentence in enumerate(sentences):
        for fragment in _split_bounded(sentence):
            if sentence_index == 0 and not pending:
                yield fragment
            elif not pending:
                pending = fragment
            elif len(pending) + 1 + len(fragment) <= _MAX_SYNTHESIS_CHARS:
                pending = f"{pending} {fragment}"
            else:
                yield pending
                pending = fragment
    if pending:
        yield pending


def _stream(text: str, voice_state):
    if _model is None or _torch is None:
        raise RuntimeError("Pocket TTS PyTorch is not loaded")
    with InferenceWatchdog("Pocket TTS PyTorch generation"):
        for audio in _model.generate_audio_stream(model_state=voice_state, text_to_generate=text):
            pcm = audio.detach().to(device="cpu", dtype=_torch.float32).numpy().reshape(-1)
            for start in range(0, len(pcm), _CHUNK_SAMPLES):
                chunk = np.asarray(pcm[start : start + _CHUNK_SAMPLES], dtype=np.float32)
                if chunk.size:
                    yield chunk


@dataclass
class _Job:
    text: str
    voice_state: object
    out: "queue.Queue"
    session: TtsSession


_jobs: "queue.Queue[_Job]" = queue.Queue(maxsize=TTS_JOB_QUEUE_MAXSIZE)


def _put(out: "queue.Queue", item, session: TtsSession) -> bool:
    while not session.cancelled:
        try:
            out.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _worker_loop() -> None:
    while True:
        job = _jobs.get()
        try:
            for fragment in _synthesis_fragments(job.text):
                if job.session.cancelled:
                    break
                for chunk in _stream(fragment, job.voice_state):
                    if job.session.cancelled or not _put(
                        job.out, (chunk, SAMPLE_RATE), job.session
                    ):
                        break
                if job.session.cancelled:
                    break
        except Exception as exc:  # noqa: BLE001 - preserve shared worker after model errors
            logger.exception("Pocket TTS PyTorch synth error: %s", exc)
        finally:
            while True:
                try:
                    job.out.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass


threading.Thread(target=_worker_loop, daemon=True, name="pocket-tts-pytorch-worker").start()


def _submit(text: str, prompt, session: TtsSession, maxsize: int = 8) -> "queue.Queue":
    voice_state = prompt if prompt is not None else _voice_state
    if voice_state is None:
        raise RuntimeError("Pocket TTS PyTorch voice is not configured")
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    submit_tts_job(_jobs, _Job(text=text, voice_state=voice_state, out=out, session=session))
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


def force_cancel_playback(sink: Optional["queue.Queue"] = None) -> None:
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
    logger.info("Warming up Pocket TTS PyTorch...")
    synthesize("The assistant is ready.", prompt)
    logger.info("Pocket TTS PyTorch ready")


def play_chunks(
    chunks,
    sample_rate: int,
    session: Optional[TtsSession] = None,
    sink: Optional["queue.Queue"] = None,
    tts_active_event: Optional[threading.Event] = None,
):
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
            try:
                on_first_audio()
            except Exception as exc:  # noqa: BLE001 - optional latency hook
                logger.warning("on_first_audio hook failed: %s", exc)
        if sink is None:
            _drain(out)
            return time.monotonic()
        if tts_active_event is not None:
            tts_active_event.set()
        playback_start = time.monotonic()
        total_samples = len(first[0])
        sink.put(("start", SAMPLE_RATE))
        sink.put((first[0], None))
        cancelled = False
        while (item := out.get()) is not None:
            if session.cancelled:
                _drain(out)
                cancelled = True
                break
            sink.put((item[0], None))
            total_samples += len(item[0])
        if cancelled:
            force_cancel_playback(sink)
        else:
            sink.put(("end",))
        return playback_start + total_samples / SAMPLE_RATE
    finally:
        if tts_active_event is not None:
            tts_active_event.clear()
        session.active = False
