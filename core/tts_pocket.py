"""Official Kyutai Pocket TTS PyTorch adapter with CPU/GPU PCM streaming.

Pocket's generator yields audio frames as Mimi decodes them. A single shared
worker keeps model access serial, matching Qwen's worker contract, while
the consumer sends each frame to the browser immediately.
"""

import importlib.resources
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
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
_voice_binding = None
_model_lock = threading.RLock()
_default_language = "english_2026-04"
_load_options = {}
# Alternates stay on CPU: Premium's default GPU model remains the only Pocket
# model occupying VRAM. The serial synthesis worker owns this single slot.
_alternate = None
_download_lock = threading.Lock()
_download_active = None
_download_failures = {}
_load_failures = {}
_RETRY_SECONDS = 300
_ALTERNATE_RAM_HEADROOM = 2 * 1024 ** 3

MIRROR_REPO = "lunahr/pocket-tts-ungated"
MIRROR_REVISION = "d03cd73415a8d46d8eb115c7b524aebb0a729f4a"


def _load_model(language: str, device: str, opts: dict):
    """Load one model using the pinned cloning weights and tokenizer."""
    from pocket_tts import TTSModel

    from .backends import pocket_pytorch_snapshots

    pocket_pytorch_snapshots(language)
    temperature = opts.get("temperature")
    config = importlib.resources.files("pocket_tts").joinpath(f"config/{language}.yaml").read_text(encoding="utf-8")
    base = f"hf://{MIRROR_REPO}/languages/{language}"
    config, weights_count = re.subn(r"(?m)^weights_path: hf://kyutai/pocket-tts/\S+[ \t]*$", f"weights_path: {base}/model.safetensors@{MIRROR_REVISION}", config)
    config, tokenizer_count = re.subn(r"(?m)^([ \t]*tokenizer_path:) hf://kyutai/pocket-tts-without-voice-cloning/[^\s]+", rf"\1 {base}/tokenizer.model@{MIRROR_REVISION}", config)
    if weights_count != 1 or tokenizer_count != 1:
        raise RuntimeError("Unsupported Pocket TTS config: expected Kyutai weights and tokenizer URIs")
    # Do not allow an inaccessible mirror to silently switch to non-cloning
    # weights: setup should report the actual download/load failure instead.
    config = "\n".join(line for line in config.splitlines() if not line.startswith("weights_path_without_voice_cloning:")) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as file:
        file.write(config)
        file.flush()
        model = TTSModel.load_model(
            config=file.name,
            temp=0.7 if temperature is None else float(temperature),
            lsd_decode_steps=int(opts.get("lsd_decode_steps", 1)),
        ).to(device)
    model.eval()
    return model


def load_tts(model_id=None, language: str = "english_2026-04", device: Optional[str] = None, **opts):
    """Load the configured default model on CPU or CUDA."""
    global _model, _torch, _default_language, _load_options, _alternate
    import torch

    from .backends import variant

    language = "english_2026-04" if language == "english" else language
    if device is None:
        device = "cuda" if variant() != "cpu" and torch.cuda.is_available() else "cpu"
    if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError(f"Pocket TTS PyTorch device unavailable: {device}")
    _torch = torch
    _model = _load_model(language, device, opts)
    _default_language = language
    _load_options = dict(opts)
    _alternate = None
    logger.info("Loaded official Pocket TTS PyTorch on %s", device)
    return _model


def set_voice(voice_name: Optional[str]):
    """Prepare and retain the voice-clone state for subsequent streamed calls."""
    if _model is None:
        raise RuntimeError("Pocket TTS PyTorch is not loaded")
    name = (voice_name or "").strip()
    if name.lower().endswith(".wav"):
        from .voice_reference import voice_reference

        wav_path, _ = voice_reference(name, VOICES_DIR)
    else:
        wav_path = VOICES_DIR / f"{name}.wav"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Pocket TTS voice reference not found: {wav_path}")
    global _voice_state, _voice_binding
    with _model_lock:
        state = _model.get_state_for_audio_prompt(wav_path)
        _voice_state = state
        _voice_binding = (state, wav_path)
    logger.info("Pocket TTS PyTorch voice clone ready: %s", name)
    return state


def set_speed(speed) -> None:
    if speed not in (None, 1.0):
        logger.warning("Pocket TTS has no speed control; tts_speed=%s ignored", speed)


def _language_model(language: Optional[str]) -> Optional[str]:
    """Map a turn's BCP-47 language to a supported Pocket model."""
    from .backends import TTS_MODEL_LANGUAGES

    base = (language or "").replace("_", "-").split("-", 1)[0].lower()
    name = {"en": "english", "fr": "french", "de": "german", "it": "italian",
            "pt": "portuguese", "es": "spanish"}.get(base)
    return TTS_MODEL_LANGUAGES["pocket-tts-pytorch"].get(name)


def _speech_language(text: str, language: Optional[str]) -> Optional[str]:
    """Retain non-English turn metadata; detect clear foreign-language replies."""
    if (language or "").replace("_", "-").split("-", 1)[0].lower() != "en":
        return language
    if len(text) < 48 or len(text.split()) < 8:
        return language
    from utils.response_language import detect_response_language

    detected = detect_response_language(text)
    return detected if detected != "en" and _language_model(detected) else language


def _weights_cached(language: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    from .backends import pocket_pytorch_snapshots

    for repo, files, revision in pocket_pytorch_snapshots(language):
        for filename in files:
            path = try_to_load_from_cache(repo, filename, revision=revision)
            if not isinstance(path, str) or not Path(path).is_file():
                return False
    return True


def _download_language(language: str) -> None:
    global _download_active
    try:
        from .backends import pocket_pytorch_snapshots

        # Runtime model loading is deliberately offline. Use a separate process
        # for this explicit download rather than changing Hugging Face's global
        # offline flag (and HTTP sessions) for unrelated inference threads.
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "0"
        for repo, files, revision in pocket_pytorch_snapshots(language):
            subprocess.run(
                [sys.executable, "-c",
                 "import sys; from huggingface_hub import snapshot_download; "
                 "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], allow_patterns=sys.argv[3:])",
                 repo, revision, *files],
                env=env, check=True,
            )
        if not _weights_cached(language):
            raise RuntimeError("Pocket TTS language download is incomplete")
        with _download_lock:
            _download_failures.pop(language, None)
        logger.info("Pocket TTS language %s is downloaded and ready to load", language)
    except Exception:  # noqa: BLE001 — keep default speech available offline or on download failure
        with _download_lock:
            _download_failures[language] = time.monotonic()
        logger.exception("Pocket TTS language %s download failed; retaining default", language)
    finally:
        with _download_lock:
            _download_active = None


def _request_language_download(language: str) -> None:
    """Start at most one pinned download, never blocking the synthesis worker."""
    global _download_active
    with _download_lock:
        if _download_active is not None or time.monotonic() - _download_failures.get(language, -_RETRY_SECONDS) < _RETRY_SECONDS:
            return
        _download_active = language
    try:
        threading.Thread(target=_download_language, args=(language,), daemon=True,
                         name="pocket-tts-language-download").start()
    except Exception:
        with _download_lock:
            _download_active = None
            _download_failures[language] = time.monotonic()
        logger.exception("Could not start Pocket TTS language download")


def _enough_alternate_ram() -> bool:
    """Require headroom for CPU weights and transient voice preparation."""
    from .memory_guard import snapshot

    memory = snapshot()
    available = []
    if memory.available is not None:
        available.append(memory.available)
    if memory.limit is not None and memory.usage is not None:
        available.append(memory.limit - memory.usage)
    return bool(available) and min(available) >= _ALTERNATE_RAM_HEADROOM


def _select_job_model(job):
    """Resolve a queued utterance without allocating an alternate on CUDA."""
    global _alternate
    language = _language_model(job.language)
    if not language or language == _default_language or job.voice_path is None:
        return _model, job.voice_state
    if _alternate is not None and _alternate[:2] == (language, job.voice_path):
        return _alternate[2], _alternate[3]
    if time.monotonic() - _load_failures.get(language, -_RETRY_SECONDS) < _RETRY_SECONDS:
        return _model, job.voice_state
    try:
        if not _weights_cached(language):
            _request_language_download(language)
            return _model, job.voice_state
        if not _enough_alternate_ram():
            logger.warning("Insufficient system RAM for Pocket TTS %s; retaining default", language)
            return _model, job.voice_state
        # Release the previous CPU alternate before allocating the next one.
        _alternate = None
        model = _load_model(language, "cpu", _load_options)
        state = model.get_state_for_audio_prompt(job.voice_path)
        _alternate = (language, job.voice_path, model, state)
        _load_failures.pop(language, None)
        logger.info("Pocket TTS language %s loaded on CPU", language)
        return model, state
    except Exception:  # noqa: BLE001 — an optional language must never suppress speech
        _load_failures[language] = time.monotonic()
        logger.exception("Pocket TTS language %s failed to load; retaining default", language)
        return _model, job.voice_state


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


def _stream(text: str, voice_state, model=None):
    model = model if model is not None else _model
    if model is None or _torch is None:
        raise RuntimeError("Pocket TTS PyTorch is not loaded")
    with InferenceWatchdog("Pocket TTS PyTorch generation"):
        for audio in model.generate_audio_stream(model_state=voice_state, text_to_generate=text):
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
    language: Optional[str] = None
    voice_path: Optional[Path] = None


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
            # Voice changes and model selection cannot race active synthesis.
            with _model_lock:
                if job.session.cancelled:
                    continue
                model, voice_state = _select_job_model(job)
                for fragment in _synthesis_fragments(job.text):
                    if job.session.cancelled:
                        break
                    for chunk in _stream(fragment, voice_state, model):
                        if job.session.cancelled or not _put(
                            job.out, (chunk, SAMPLE_RATE), job.session
                        ):
                            break
                    if job.session.cancelled:
                        break
        except Exception as exc:  # noqa: BLE001 - preserve shared worker after model errors
            logger.exception("Pocket TTS PyTorch synth error: %s", exc)
        finally:
            # Do not retain a prior alternate in the worker while loading its
            # replacement on the next job.
            model = voice_state = None
            while True:
                try:
                    job.out.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass


threading.Thread(target=_worker_loop, daemon=True, name="pocket-tts-pytorch-worker").start()


def _submit(text: str, prompt, session: TtsSession, maxsize: int = 8, language: Optional[str] = None) -> "queue.Queue":
    binding = _voice_binding
    voice_state = prompt if prompt is not None else (binding[0] if binding else _voice_state)
    if voice_state is None:
        raise RuntimeError("Pocket TTS PyTorch voice is not configured")
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    voice_path = binding[1] if binding and (prompt is None or prompt is binding[0]) else None
    submit_tts_job(_jobs, _Job(text=text, voice_state=voice_state, out=out, session=session,
                               language=language, voice_path=voice_path))
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
    language: Optional[str] = None,
):
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    submitted = time.monotonic()
    out = _submit(text, prompt, session, language=_speech_language(text, language))
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
