"""Audio8 TTS PyTorch adapter for CUDA voice cloning.

Audio8 currently returns a completed waveform per generation.  The worker keeps
that CUDA work serial and synthesizes the first sentence independently so the
browser receives audio as soon as that first fragment completes.
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

SAMPLE_RATE = 44100
_CHUNK_SAMPLES = SAMPLE_RATE // 2
_MAX_SYNTHESIS_CHARS = 240
VOICES_DIR = Path("./data/voices")

_model = None
_processor = None
_torch = None
_voice: Optional[tuple[Path, str]] = None


def load_tts(model_id: str = "Edge0/Audio8-TTS-Preview-0.6b", device: str = "cuda", **opts):
    """Load Audio8 on CUDA; the model's remote code owns codec decoding."""
    global _model, _processor, _torch
    import torch
    from transformers import AutoModel, AutoProcessor

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Audio8 TTS requires an available CUDA device")
    _torch = torch
    revision = opts.get("revision")
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    _model = (
        AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            revision=revision,
        )
        .eval()
        .to(device)
    )
    logger.info("Loaded Audio8 TTS on %s (revision %s)", device, revision or "latest")
    return _model


def set_voice(voice_name: Optional[str]):
    """Select the reference WAV and its required exact transcript."""
    global _voice
    name = (voice_name or "").strip()
    wav_path = VOICES_DIR / f"{name}.wav"
    txt_path = VOICES_DIR / f"{name}.txt"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio8 voice reference not found: {wav_path}")
    if not txt_path.is_file():
        raise FileNotFoundError(f"Audio8 voice transcript not found: {txt_path}")
    transcript = txt_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError(f"Audio8 voice transcript is empty: {txt_path}")
    _voice = (wav_path, transcript)
    logger.info("Audio8 voice clone ready: %s", name)
    return _voice


def set_speed(speed) -> None:
    if speed not in (None, 1.0):
        logger.warning("Audio8 TTS has no speed control; tts_speed=%s ignored", speed)


def _synthesis_fragments(text: str):
    """Keep first-audio latency low without making every clause a model call."""
    sentences = list(split_sentences(text)) or [text]
    pending = ""
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        if index == 0:
            yield sentence[:_MAX_SYNTHESIS_CHARS]
            pending = sentence[_MAX_SYNTHESIS_CHARS:].strip()
        elif not pending:
            pending = sentence
        elif len(pending) + len(sentence) + 1 <= _MAX_SYNTHESIS_CHARS:
            pending = f"{pending} {sentence}"
        else:
            yield pending
            pending = sentence
    if pending:
        yield pending


def _stream(text: str, voice: tuple[Path, str]):
    if _model is None or _processor is None or _torch is None:
        raise RuntimeError("Audio8 TTS is not loaded")
    wav_path, transcript = voice
    inputs = _processor(
        text=[text],
        reference_audio=[str(wav_path)],
        reference_text=[transcript],
        return_tensors="pt",
    )
    inputs = {
        name: value.to(_model.device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    with InferenceWatchdog("Audio8 TTS generation"), _torch.inference_mode():
        output = _model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, lengths = _model.decode_audio(output.codes)
    length = int(lengths[0])
    pcm = waveforms[0, :length].float().cpu().numpy().reshape(-1)
    for start in range(0, len(pcm), _CHUNK_SAMPLES):
        chunk = np.asarray(pcm[start : start + _CHUNK_SAMPLES], dtype=np.float32)
        if chunk.size:
            yield chunk


@dataclass
class _Job:
    text: str
    voice: tuple[Path, str]
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
                for chunk in _stream(fragment, job.voice):
                    if job.session.cancelled or not _put(
                        job.out, (chunk, SAMPLE_RATE), job.session
                    ):
                        break
        except Exception as exc:  # noqa: BLE001 - preserve the shared worker after model errors
            logger.exception("Audio8 TTS synth error: %s", exc)
        finally:
            while True:
                try:
                    job.out.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass


threading.Thread(target=_worker_loop, daemon=True, name="audio8-tts-worker").start()


def _submit(text: str, prompt, session: TtsSession, maxsize: int = 8) -> "queue.Queue":
    voice = prompt if prompt is not None else _voice
    if voice is None:
        raise RuntimeError("Audio8 TTS voice is not configured")
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
    logger.info("Warming up Audio8 TTS...")
    synthesize("The assistant is ready.", prompt)
    logger.info("Audio8 TTS ready")


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
            on_first_audio()
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
