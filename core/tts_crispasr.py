"""CrispASR-hosted Qwen3 or Pocket TTS GGUF voice-clone TTS via ggml.

Uses CrispASR's direct Qwen3-TTS C ABI in a persistent worker. Unlike the
unified `crispasr.Session.synthesize()` API, this ABI emits PCM while codec
frames are generated, so browser playback starts before the full clause is
complete.

GPU runs in a persistent isolated subprocess. This prevents CrispASR's bundled
ggml libraries from colliding with other CUDA model runtimes
inside the main assistant process, while retaining a warm session between
clauses.

The GPU image bakes the pinned CUDA runtime into `/opt`; the setup wizard
downloads the talker and codec GGUF files. The legacy fetch script remains
useful for native development.

Consent/disclaimer note: the CrispASR *CLI*'s `--i-have-rights` gate and
`--no-spoken-disclaimer`/watermark handling live in the CLI application layer,
not in `libcrispasr`'s public C ABI — `crispasr.Session.synthesize()` has no
equivalent flag or check at all. Same posture as core/tts.py's existing Qwen
voice clone (no separate consent friction beyond the operator supplying their
own data/voices/<name>.wav reference).
"""

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .crispasr_worker import CrispASRWorker
from .text_utils import split_clauses
from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/qwen3-tts-crispasr-gguf"
DEFAULT_MODEL_DIR_0_6B = "./data/models/qwen3-tts-crispasr-0.6b-gguf"
DEFAULT_MODEL_DIR_POCKET = "./data/models/pocket-tts-gguf"
DEFAULT_LIB_DIR = "./data/models/crispasr-python"
DEFAULT_LIB_DIR_GPU = "/opt/crispasr-python-cuda"
CODEC_FILE = "qwen3-tts-tokenizer-12hz.gguf"

# Talker filenames accepted by the direct Qwen3-TTS ABI.
# The codec (12Hz tokenizer) is shared, fixed at f16, across both sizes —
# quantizing it hurts more than the talker does. Confirmed both
# open+synthesize correctly with real audio.
_TALKER_BACKENDS = {
    "qwen3-tts-12hz-1.7b-base-f16.gguf": "qwen3-tts-1.7b-base",
    "qwen3-tts-12hz-1.7b-base-q8_0.gguf": "qwen3-tts-1.7b-base",
    "qwen3-tts-12hz-0.6b-base-q8_0.gguf": "qwen3-tts",  # registry's canonical 0.6B-base key
}

VOICES_DIR = "./data/voices"

SAMPLE_RATE = 24000
_CHUNK_SAMPLES = SAMPLE_RATE // 2  # ~500ms barge-in granularity, matching core/tts_onnx.py

_session = None  # Warm CrispASR session/worker, set by load_tts().
_worker_config = None
_voice_prompt = None
_worker_stream_count = 0
_pocket_tts = False

# CrispASR's native Qwen TTS graph allocation grows with a single input span.
# Keep long note/news replies bounded so a full 16 GB GPU stack retains room for
# the decoder graph rather than aborting the worker on one oversized clause.
_MAX_SYNTHESIS_CHARS = 240
# CrispASR's direct Qwen TTS ABI retains ggml scheduler buffers between stream
# calls. Recycle before its allocation growth can consume the Full stack's
# remaining 16 GB headroom. Pocket TTS uses its separate complete-buffer API,
# so keeping it warm avoids an unnecessary model reload between turns.
_MAX_STREAMS_PER_WORKER = 8

# Pocket TTS's embedded English SentencePiece tokenizer can return no audio for
# typographic punctuation emitted by the SLM. Keep this at its backend boundary
# so other TTS backends retain their native Unicode handling.
_POCKET_TEXT_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
})


def force_cancel_playback(sink: Optional["queue.Queue"] = None) -> None:
    """Tell the browser to stop already-scheduled audio right now.

    See core/tts.py's `force_cancel_playback` docstring: on fast hardware
    generation can outrun playback and finish before a stop/barge-in ever
    flips the session's cancelled flag, so nothing is left to emit the
    mid-stream "cancel" message. Callers stopping a turn must call this
    unconditionally rather than relying on the session flag alone.
    """
    if sink is None:
        return
    _drain_queue_nowait(sink)
    sink.put(("cancel",))


def load_tts(
    model_id: str = DEFAULT_MODEL_DIR,
    lib_dir: Optional[str] = None,
    num_threads: int = 4,
    gpu: bool = False,
    backend: Optional[str] = None,
    **_opts,
):
    """Load a CrispASR Qwen3 or Pocket TTS GGUF model, warm.

    Both `lib_dir` (the extracted `crispasr` Python package + its .so's) and
    `model_id` (a directory holding one known talker GGUF + the codec GGUF)
    must already exist on disk — see the module docstring. The talker size
    (1.7B vs 0.6B) is auto-detected from whichever file in `model_id` matches
    `_TALKER_BACKENDS`, so the same loader serves either size — only
    `model_id` (i.e. `default_model` on the backend registration) differs.

    `gpu=True` starts the CUDA session in an isolated worker process.
    """
    global _session, _worker_config, _voice_prompt, _worker_stream_count, _pocket_tts

    if lib_dir is None:
        lib_dir = DEFAULT_LIB_DIR_GPU if gpu else DEFAULT_LIB_DIR
        if gpu and not Path(lib_dir).is_dir():
            # Native development installs created by the legacy fetch script.
            lib_dir = "./data/models/crispasr-python-cuda"
    lib_root = Path(lib_dir)
    if not (lib_root / "crispasr" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"CrispASR Python runtime not found at {lib_root} — run "
            f"scripts/fetch_crispasr_tts.py{' --gpu' if gpu else ''} first."
        )
    root = Path(model_id)
    _pocket_tts = backend == "pocket-tts"
    if _pocket_tts:
        model_path = root / "pocket-tts-english-q8_0.gguf" if root.is_dir() else root
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pocket TTS GGUF not found at {model_path} — rerun setup to download it."
            )
        logger.info("Loading CrispASR Pocket TTS GGUF (%s) …", model_path.name)
        t0 = time.monotonic()
        if gpu:
            _worker_config = {
                "model_path": model_path,
                "lib_dir": lib_root,
                "backend": backend,
                "num_threads": num_threads,
            }
            _voice_prompt = None
            _worker_stream_count = 0
            _session = CrispASRWorker(**_worker_config)
        else:
            if str(lib_root) not in sys.path:
                sys.path.insert(0, str(lib_root))
            import crispasr  # heavy ctypes-backed import, deferred to load time

            _session = crispasr.Session(str(model_path), backend=backend, n_threads=int(num_threads))
        logger.info("CrispASR Pocket TTS ready in %.1fs", time.monotonic() - t0)
        return _session

    talker_path = backend_name = None
    for filename, be in _TALKER_BACKENDS.items():
        candidate = root / filename
        if candidate.is_file():
            talker_path, backend_name = candidate, be
            break
    codec_path = root / CODEC_FILE
    if talker_path is None or not codec_path.is_file():
        raise FileNotFoundError(
            f"Qwen3-TTS GGUF pair not found in {root} (known talkers: "
            f"{sorted(_TALKER_BACKENDS)}) — run scripts/fetch_crispasr_tts.py first."
        )

    logger.info("Loading CrispASR Qwen3-TTS GGUF (%s) …", talker_path.name)
    t0 = time.monotonic()
    if gpu:
        _worker_config = {
            "model_path": talker_path,
            "lib_dir": lib_root,
            "backend": backend_name,
            "codec_path": codec_path,
            "num_threads": num_threads,
            "direct_tts": True,
        }
        _voice_prompt = None
        _worker_stream_count = 0
        _session = CrispASRWorker(**_worker_config)
    else:
        if str(lib_root) not in sys.path:
            sys.path.insert(0, str(lib_root))
        import crispasr  # heavy ctypes-backed import, deferred to load time

        _session = crispasr.Session(str(talker_path), backend=backend_name, n_threads=int(num_threads))
        _session.set_codec_path(str(codec_path))
    logger.info("CrispASR Qwen3-TTS ready in %.1fs", time.monotonic() - t0)
    return _session


def set_voice(voice_name: str):
    """Point the warm session at data/voices/<voice_name>.{wav,txt}.

    Same convention as core/tts.py: the transcript falls back to
    data/voices/default.txt when <voice_name>.txt doesn't exist.
    """
    ref_audio = f"{VOICES_DIR}/{voice_name}.wav"
    txt_path = Path(f"{VOICES_DIR}/{voice_name}.txt")
    if not txt_path.is_file():
        txt_path = Path(f"{VOICES_DIR}/default.txt")
    ref_text = txt_path.read_text().strip()
    logger.info("Setting voice clone to: %s", voice_name)
    global _voice_prompt
    _voice_prompt = (ref_audio, ref_text)
    if isinstance(_session, CrispASRWorker):
        _session.call("set_voice", audio=ref_audio, text=ref_text)
    else:
        _session.set_voice(ref_audio, ref_text)
    return ref_audio


_speed_warned = False


def set_speed(speed) -> None:
    """Accept the speed-control contract, but no-op — same as core/tts.py's
    Qwen voice clone: CrispASR's qwen3-tts backend has no rate knob, the
    clone's pace is baked into the voice-clone reference recording.
    """
    global _speed_warned
    if speed in (None, 1.0) or _speed_warned:
        return
    logger.warning(
        "CrispASR Qwen3-TTS has no speed control — tts_speed=%s ignored.", speed
    )
    _speed_warned = True


def _synth(text: str) -> np.ndarray:
    """Synthesise one (short) text span to float32 PCM at 24 kHz on the warm session."""
    audio = _session.call("synthesize", text=text) if isinstance(_session, CrispASRWorker) else _session.synthesize(text)
    # PLACEHOLDER — watermarking (not implemented yet). crispasr's Python
    # package watermark_embed()/watermark_load_model() (v0.4.9, the release
    # this module targets) call an undefined `_get_lib()` helper — a real
    # packaging bug upstream, not a design limitation — so they crash as
    # shipped. Workaround once needed: bypass the free function and call
    # `_session._lib.crispasr_watermark_embed(...)` directly (the ctypes
    # handle is already loaded on the warm session). Re-check latency on a
    # realistic clause-length clip before enabling by default — that's what
    # this comment is standing in for.
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _synth_stream(text: str):
    """Yield PCM as CrispASR decodes each generated Qwen codec window."""
    if _pocket_tts:
        # Pocket's CrispASR API returns a complete PCM buffer; it has no native
        # streaming callback like Qwen3-TTS.
        yield _synth(text.translate(_POCKET_TEXT_TRANSLATION))
        return
    if isinstance(_session, CrispASRWorker):
        yield from _session.stream("synthesize_stream", text=text)
        return
    yield _synth(text)


def _synthesis_fragments(text: str):
    """Yield clause fragments small enough for CrispASR's native CUDA graph."""
    for clause in list(split_clauses(text)) or [text]:
        remaining = clause.strip()
        while len(remaining) > _MAX_SYNTHESIS_CHARS:
            cut = remaining.rfind(" ", 0, _MAX_SYNTHESIS_CHARS + 1)
            if cut <= 0:
                cut = _MAX_SYNTHESIS_CHARS
            yield remaining[:cut].strip()
            remaining = remaining[cut:].strip()
        if remaining:
            yield remaining


def _restart_dead_worker(force: bool = False) -> None:
    """Restore or recycle the native worker without replaying partial audio."""
    global _session, _worker_stream_count
    if not isinstance(_session, CrispASRWorker) or (_session.alive and not force):
        return
    if not _worker_config:
        return
    if force:
        logger.info("Recycling CrispASR TTS worker to release retained CUDA buffers")
    else:
        logger.warning("CrispASR TTS worker exited; starting a fresh worker for the next turn")
    _session.close()
    _session = CrispASRWorker(**_worker_config)
    _worker_stream_count = 0
    if _voice_prompt is not None:
        audio, text = _voice_prompt
        _session.call("set_voice", audio=audio, text=text)


def _recycle_worker_if_needed() -> None:
    """Bound retained direct-Qwen CUDA scheduler buffers between streams."""
    if (
        not _pocket_tts
        and isinstance(_session, CrispASRWorker)
        and _worker_stream_count >= _MAX_STREAMS_PER_WORKER
    ):
        _restart_dead_worker(force=True)


def _to_chunks(audio: np.ndarray) -> list:
    return [audio[i : i + _CHUNK_SAMPLES] for i in range(0, len(audio), _CHUNK_SAMPLES)]


@dataclass
class _Job:
    text: str
    out: "queue.Queue"
    session: TtsSession


_job_queue: "queue.Queue[_Job]" = queue.Queue()


def _worker_loop() -> None:
    """Synthesise jobs fragment-by-fragment on the warm session, pushing
    ~500ms chunks to the job's output queue — one worker thread so the next
    fragment synthesises while the consumer plays the current one (overlap).
    """
    global _worker_stream_count
    while True:
        job = _job_queue.get()
        try:
            for fragment in _synthesis_fragments(job.text):
                if job.session.cancelled:
                    break
                _recycle_worker_if_needed()
                stream_started = time.monotonic()
                first_chunk = True
                for audio in _synth_stream(fragment):
                    # The native worker keeps sending PCM until it reaches its
                    # "done" frame. Drain it after cancellation so those frames
                    # cannot become the next turn's first audio packet.
                    if job.session.cancelled:
                        continue
                    for chunk in _to_chunks(np.asarray(audio, dtype=np.float32)):
                        if job.session.cancelled:
                            break
                        if chunk.size:
                            job.out.put((chunk, SAMPLE_RATE))
                            if first_chunk:
                                first_chunk = False
                                logger.debug(
                                    "CrispASR TTS queue first PCM after %.3fs", time.monotonic() - stream_started
                                )
                if isinstance(_session, CrispASRWorker):
                    _worker_stream_count += 1
                if job.session.cancelled:
                    break
        except Exception as e:
            logger.exception("CrispASR TTS synth error: %s: %s", type(e).__name__, e)
            try:
                _restart_dead_worker()
            except Exception as restart_error:  # noqa: BLE001 - native worker boundary
                logger.exception("CrispASR TTS worker recovery failed: %s", restart_error)
        finally:
            job.out.put(None)


_worker = threading.Thread(target=_worker_loop, daemon=True, name="crispasr-tts-worker")
_worker.start()


def _submit(text: str, session: TtsSession, maxsize: int = 8) -> "queue.Queue":
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    _job_queue.put(_Job(text=text, out=out, session=session))
    return out


def _drain(out: "queue.Queue") -> None:
    while out.get() is not None:
        pass


def _drain_queue_nowait(q: "queue.Queue") -> None:
    """Discard everything currently sitting in `q` without blocking — see
    core/tts_onnx.py's identical helper for why (barge-in cancel ordering).
    """
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def synthesize(text: str, prompt=None):
    """Run TTS to completion; return (chunks, sample_rate)."""
    out = _submit(text, TtsSession(), maxsize=0)
    chunks = []
    while True:
        item = out.get()
        if item is None:
            break
        chunks.append(item[0])
    return chunks, SAMPLE_RATE


def warmup_model(prompt=None):
    logger.info("Warming up CrispASR TTS …")
    synthesize("A rainbow is caused by reflection and refraction of light.")
    logger.info("CrispASR TTS ready")


def play_chunks(
    chunks,
    sample_rate: int,
    session: Optional[TtsSession] = None,
    sink: Optional["queue.Queue"] = None,
    tts_active_event: Optional[threading.Event] = None,
):
    """Play pre-rendered chunks (e.g. a cached greeting) with no fresh synthesis."""
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
            if sink is not None:
                sink.put(("start", sample_rate))
                for chunk in chunks:
                    if session.cancelled:
                        break
                    sink.put((chunk, None))
                sink.put(("end",))
            else:
                logger.warning("No satellite connected; TTS chunks dropped on the floor.")
        finally:
            if tts_active_event is not None:
                tts_active_event.clear()
    finally:
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
    """Synthesise on the worker (overlapped) and play back here, cancellable.

    Returns the estimated monotonic-clock time playback will finish on the
    browser (playback start + total audio duration) — see core/tts.py's
    speak_stream docstring for why callers arming the follow-up window need
    this instead of the time this function itself returns.
    """
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True

    t_submit = time.monotonic()
    out = _submit(text, session, maxsize=8)
    try:
        first = out.get()
        if stats is not None:
            stats.tts_seconds = time.monotonic() - t_submit
        if first is None:
            return time.monotonic()
        if session.cancelled:
            _drain(out)
            return time.monotonic()
        if on_first_audio is not None:
            try:
                on_first_audio()
            except Exception as e:
                logger.warning(f"on_first_audio hook failed: {e}")
        first_chunk, sr = first
        if tts_active_event is not None:
            tts_active_event.set()
        t_playback_start = time.monotonic()
        total_samples = 0
        try:
            if sink is not None:
                sink.put(("start", sr))
                sink.put((first_chunk, None))
                total_samples += len(first_chunk)
                cancelled_mid_stream = False
                while True:
                    item = out.get()
                    if item is None:
                        break
                    if session.cancelled:
                        _drain(out)
                        cancelled_mid_stream = True
                        break
                    sink.put((item[0], None))
                    total_samples += len(item[0])
                if cancelled_mid_stream:
                    _drain_queue_nowait(sink)
                    sink.put(("cancel",))
                else:
                    sink.put(("end",))
            else:
                logger.warning("No satellite connected; TTS chunks dropped on the floor.")
                _drain(out)
        finally:
            if tts_active_event is not None:
                tts_active_event.clear()
        if sink is None or not sr:
            return time.monotonic()
        return t_playback_start + total_samples / sr
    finally:
        session.active = False
