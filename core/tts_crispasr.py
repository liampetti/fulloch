"""CrispASR-hosted Qwen3-TTS-Base GGUF (1.7B or 0.6B) — voice-clone TTS via ggml,
CPU by default, optionally CUDA.

Uses CrispASR's Python ctypes binding (`crispasr.Session`), not the `crispasr`
CLI binary. That distinction matters: the CLI spawns a fresh process and reloads the
whole model on every call (fine for a one-shot benchmark, bad for a live
assistant), while `crispasr.Session` loads the talker + codec GGUFs once and
stays resident in-process — `synthesize(text) -> np.ndarray` calls on an
already-open session pay only the forward-pass cost. That's the same
load-once-then-synth-per-clause contract as core/tts.py and core/tts_onnx.py,
so this module mirrors their worker-thread/clause-split structure directly.

GPU (`models.tts.gpu: true`) is currently DISABLED — see _GPU_DISABLED_REASON
below. In isolation (a bare Python process with nothing else loaded) the
hybrid CPU-glue + CUDA-.so package assembled by
`scripts/fetch_crispasr_tts.py --gpu` genuinely works: ~3.3GB VRAM for the
1.7B q8_0 talker, RTF ~0.6x vs the CPU build's ~1.5-2.5x (verified by hand,
2026-07-05, RTX 5060 Ti). But inside the real assistant process — which also
loads llama-cpp-python for the 9B SLM — it crashes with `undefined symbol:
ggml_col2im_1d`. Root cause: both CrispASR and llama-cpp-python bundle their
own copy of ggml under the same sonames (libggml-base.so.0 etc, upstream
ggml's own naming convention); whichever loads first into the process claims
that soname process-wide, and the dynamic linker satisfies the other
library's same-named dependency by reusing the first one rather than the
co-located copy sitting next to it. The SLM loads before TTS in
`_load_models()`, so llama-cpp-python's (older) ggml wins, and CrispASR's
`libcrispasr.so` ends up resolving a symbol its own bundled ggml has but
llama-cpp-python's older one doesn't. Fixing this needs either soname
isolation (patchelf-renaming CrispASR's bundled ggml libs so they can't
collide) or running CrispASR out-of-process (it has a built-in HTTP TTS
server mode) — both real efforts, shelved for now rather than attempted
blind. See git history around 2026-07-05 for the investigation.

The runtime library and the two GGUFs (talker + codec) are NOT fetched here.
This project's assistant runs with `HF_HUB_OFFLINE=1` after setup — every
other backend's weights are fetched by the setup wizard, not lazily at
`load_tts()` time — so run `scripts/fetch_crispasr_tts.py` (add `--gpu` for
the CUDA runtime too) once beforehand (manual for now; this backend is
experimental/selectable, not yet wired into the wizard's download flow).

Consent/disclaimer note: the CrispASR *CLI*'s `--i-have-rights` gate and
`--no-spoken-disclaimer`/watermark handling live in the CLI application layer,
not in `libcrispasr`'s public C ABI — `crispasr.Session.synthesize()` has no
equivalent flag or check at all. Same posture as core/tts.py's existing Qwen
voice clone (no separate consent friction beyond the operator supplying their
own data/voices/<name>.wav reference).
"""

import logging
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/qwen3-tts-crispasr-gguf"
DEFAULT_MODEL_DIR_0_6B = "./data/models/qwen3-tts-crispasr-0.6b-gguf"
DEFAULT_LIB_DIR = "./data/models/crispasr-python"
DEFAULT_LIB_DIR_GPU = "./data/models/crispasr-python-cuda"  # unused while gpu=True is disabled; kept for the writeup
CODEC_FILE = "qwen3-tts-tokenizer-12hz.gguf"

# Talker filename -> CrispASR backend id (crispasr.Session(..., backend=...)).
# The codec (12Hz tokenizer) is shared, fixed at f16, across both sizes —
# quantizing it hurts more than the talker does. Confirmed both
# open+synthesize correctly with real audio.
_TALKER_BACKENDS = {
    "qwen3-tts-12hz-1.7b-base-q8_0.gguf": "qwen3-tts-1.7b-base",
    "qwen3-tts-12hz-0.6b-base-q8_0.gguf": "qwen3-tts",  # registry's canonical 0.6B-base key
}

VOICES_DIR = "./data/voices"

# See the module docstring: the CUDA hybrid genuinely works in isolation but
# crashes (undefined symbol: ggml_col2im_1d) inside the real assistant
# process, which also loads llama-cpp-python's own bundled ggml under the
# same sonames. Guarded here so `gpu: true` fails fast with an explanation
# instead of crashing _load_models with a cryptic ctypes OSError.
_GPU_DISABLED_REASON = (
    "crispasr-qwen3-tts* gpu=true is disabled: CrispASR's bundled ggml and "
    "llama-cpp-python's bundled ggml collide (same sonames, first-loaded "
    "wins process-wide) when both are loaded in the same process — the SLM "
    "loads first, so CrispASR's libcrispasr.so resolves a symbol "
    "(ggml_col2im_1d) against llama-cpp-python's older ggml, which lacks it. "
    "See core/tts_crispasr.py's module docstring for the full writeup."
)

SAMPLE_RATE = 24000
_CHUNK_SAMPLES = SAMPLE_RATE // 2  # ~500ms barge-in granularity, matching core/tts_onnx.py

# Break at sentence + clause punctuation, keeping the delimiter on the left part —
# same split as core/tts_onnx.py so the first clause plays while later ones render.
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+")

_session = None  # crispasr.Session, warm — set by load_tts()
_satellite_sink: Optional["queue.Queue"] = None
_TTS_ACTIVE_EVENT = None


def set_satellite_sink(q: Optional["queue.Queue"]) -> None:
    global _satellite_sink
    _satellite_sink = q


def set_tts_active_event(event) -> None:
    global _TTS_ACTIVE_EVENT
    _TTS_ACTIVE_EVENT = event


def load_tts(
    model_id: str = DEFAULT_MODEL_DIR,
    lib_dir: Optional[str] = None,
    num_threads: int = 4,
    gpu: bool = False,
    **_opts,
):
    """Load the CrispASR Python runtime + Qwen3-TTS GGUF talker/codec, warm.

    Both `lib_dir` (the extracted `crispasr` Python package + its .so's) and
    `model_id` (a directory holding one known talker GGUF + the codec GGUF)
    must already exist on disk — see the module docstring. The talker size
    (1.7B vs 0.6B) is auto-detected from whichever file in `model_id` matches
    `_TALKER_BACKENDS`, so the same loader serves either size — only
    `model_id` (i.e. `default_model` on the backend registration) differs.

    `gpu=True` (set via `models.tts.gpu: true`) is currently disabled — see
    _GPU_DISABLED_REASON / the module docstring.
    """
    global _session

    if gpu:
        raise RuntimeError(_GPU_DISABLED_REASON)
    if lib_dir is None:
        lib_dir = DEFAULT_LIB_DIR
    lib_root = Path(lib_dir)
    if not (lib_root / "crispasr" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"CrispASR Python runtime not found at {lib_root} — run "
            f"scripts/fetch_crispasr_tts.py{' --gpu' if gpu else ''} first."
        )
    if str(lib_root) not in sys.path:
        sys.path.insert(0, str(lib_root))
    import crispasr  # heavy ctypes-backed import, deferred to load time

    root = Path(model_id)
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
    audio = _session.synthesize(text)
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


def _to_chunks(audio: np.ndarray) -> list:
    return [audio[i : i + _CHUNK_SAMPLES] for i in range(0, len(audio), _CHUNK_SAMPLES)]


def _iter_fragments(text: str):
    """Yield synthesis fragments split on clause/sentence punctuation.

    One fragment per clause so the first plays while the rest renders.
    """
    for fragment in _CLAUSE_SPLIT.split(text.strip()):
        fragment = fragment.strip()
        if fragment:
            yield fragment


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
    while True:
        job = _job_queue.get()
        try:
            for fragment in list(_iter_fragments(job.text)) or [job.text]:
                if job.session.cancelled:
                    break
                for chunk in _to_chunks(_synth(fragment)):
                    if job.session.cancelled:
                        break
                    if chunk.size:
                        job.out.put((chunk, SAMPLE_RATE))
        except Exception as e:
            logger.exception("CrispASR TTS synth error: %s: %s", type(e).__name__, e)
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


def play_chunks(chunks, sample_rate: int, session: Optional[TtsSession] = None):
    """Play pre-rendered chunks (e.g. a cached greeting) with no fresh synthesis."""
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True
    try:
        if not chunks or session.cancelled:
            return
        sink = _satellite_sink
        if _TTS_ACTIVE_EVENT is not None:
            _TTS_ACTIVE_EVENT.set()
        try:
            if sink is not None:
                sink.put(("start", sample_rate))
                for chunk in chunks:
                    if session.cancelled:
                        break
                    sink.put((chunk, None))
                sink.put(("end",))
            else:
                logger.debug("No satellite connected; TTS chunks dropped on the floor.")
        finally:
            if _TTS_ACTIVE_EVENT is not None:
                _TTS_ACTIVE_EVENT.clear()
    finally:
        session.active = False


def speak_stream(
    text: str,
    prompt=None,
    session: Optional[TtsSession] = None,
    stats: Optional[TurnStats] = None,
    on_first_audio: Optional[Callable[[], None]] = None,
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
        sink = _satellite_sink
        if _TTS_ACTIVE_EVENT is not None:
            _TTS_ACTIVE_EVENT.set()
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
                logger.debug("No satellite connected; TTS chunks dropped on the floor.")
                _drain(out)
        finally:
            if _TTS_ACTIVE_EVENT is not None:
                _TTS_ACTIVE_EVENT.clear()
        if sink is None or not sr:
            return time.monotonic()
        return t_playback_start + total_samples / sr
    finally:
        session.active = False
