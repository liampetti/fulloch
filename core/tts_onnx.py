"""Kokoro 82M ONNX TTS — fast CPU text-to-speech via onnxruntime (no torch).

Uses onnx-community/Kokoro-82M-v1.0-ONNX (`model.onnx`, fp32, by default) plus
misaki G2P. Built-in named voices (no cloning). The input is split on
clause/sentence punctuation and
synthesised one fragment at a time on a single long-lived worker thread, so the
first clause plays while the rest renders (overlap), mirroring core/tts.py;
barge-in is honoured per ~500ms chunk.

Inference recipe (from the model card):
  phonemes = misaki(text); tokens = [vocab[p] for p in phonemes][:510]
  ref_s = voices[len(tokens)]              # voice .bin is (-1, 1, 256) float32
  audio = sess.run(input_ids=[[0,*tokens,0]], style=ref_s, speed=[s])  # 24 kHz

Heavy imports (onnxruntime, misaki) are deferred to load_tts, so importing this
module costs nothing.
"""

import logging
import queue
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .text_utils import split_clauses
from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/kokoro-82m-onnx"
DEFAULT_ONNX_FILE = "model.onnx"
SAMPLE_RATE = 24000
MAX_TOKENS = 510
_CHUNK_SAMPLES = SAMPLE_RATE // 2  # ~500ms barge-in granularity
DEFAULT_VOICE = "af_heart"  # recommended default; Kokoro's highest-graded voice

# Kokoro is non-autoregressive: one forward pass renders the whole input span,
# so time-to-first-audio == synth time of the first span. We split the input on
# clause/sentence punctuation (core.text_utils.split_clauses) and synthesise one
# fragment at a time, so the first clause plays while the rest renders (the
# consumer overlaps playback). Synth is fast enough that this simple clause
# split needs no word-budget tuning; the earlier small-first-fragment ramp only
# added audible mid-sentence gaps.

# Kokoro v1.0 phoneme -> token id map (fixed; from hexgrad/Kokoro-82M config.json).
# Embedded so the backend needs only the ONNX model + voice files at runtime.
VOCAB = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    '"': 11,
    "(": 12,
    ")": 13,
    "“": 14,
    "”": 15,
    " ": 16,
    "̃": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "ꭧ": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177,
}

# Built-in Kokoro v1.0 English voices (voices/<name>.bin in the repo). The repo
# also ships non-English packs (ef_/em_/ff_/hf_/jf_/...) which we don't list —
# they'd mispronounce English — but set_voice still loads any present .bin.
# All 28 English voices: under the fp32 model every one renders NaN-free and
# transcribes back at the same ~7% ASR WER (verified). The 11 that emitted NaN
# under fp16 are back.
KOKORO_VOICES = frozenset(
    {
        # American female / male
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        # British female / male
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    }
)

# Only long, purely-alphabetic tokens are considered possible run-ons. Ordinary
# long words ("temperature", "information") are returned unchanged by wordninja,
# so the >=2-real-pieces gate in _desegment excludes them; this length floor just
# avoids touching short tokens at all.
_MIN_DESEGMENT_LEN = 9

# Populated by load_tts().
_session = None
_vocab: Optional[dict] = None
_g2p = None
_wordninja = None  # module handle if installed; None disables desegmentation
_voices_dir: Optional[Path] = None
_voice = DEFAULT_VOICE
_voice_style: Optional[np.ndarray] = None  # (-1, 1, 256) for the active voice
_speed = 1.2

def force_cancel_playback(sink: Optional[queue.Queue] = None) -> None:
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


def _build_g2p(en_module):
    """misaki English G2P, with an espeak fallback for OOV words when available.

    Without a fallback, misaki *drops* words it can't look up — they phonemise to
    nothing and vanish from speech (the original "partlycloudy" symptom). An
    EspeakFallback letter-to-sounds them so they are at least spoken. It needs
    espeak-ng + misaki's espeak extras (phonemizer-fork / espeakng-loader); if any
    are missing we log and proceed with G2P only — the desegmenter and drop
    logging still apply, so the failure is visible and mostly recovered anyway.
    """
    fallback = None
    try:
        from misaki.espeak import EspeakFallback

        fallback = EspeakFallback(british=False)
        logger.info("misaki espeak fallback enabled (OOV words letter-to-sounded)")
    except Exception as e:  # noqa: BLE001 — any import/runtime gap = no fallback
        logger.info(
            "misaki espeak fallback unavailable (%s); relying on desegmentation for OOV words", e
        )
    try:
        return en_module.G2P(british=False, fallback=fallback)
    except TypeError:
        # Older misaki without the `fallback` kwarg.
        return en_module.G2P(british=False)


def load_tts(
    model_id: str = DEFAULT_MODEL_DIR,
    onnx_file: str = DEFAULT_ONNX_FILE,
    speed: Optional[float] = None,
    **opts,
):
    """Load the Kokoro ONNX session, vocab, and G2P. Sets module globals.

    `opts` (from the models.tts config block) may set ORT threads (num_threads);
    `onnx_file` picks the quant variant (e.g. model_fp16.onnx) if a different one
    is on disk — fp32 model.onnx is the default and the only one we ship.
    """
    global _session, _vocab, _g2p, _voices_dir, _speed, _wordninja
    import onnxruntime as ort
    from misaki import en

    root = Path(model_id)
    _vocab = VOCAB
    _voices_dir = root / "voices"
    try:
        import wordninja

        _wordninja = wordninja
    except ImportError:
        _wordninja = None
        logger.info("wordninja not installed; OOV word desegmentation disabled")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if opts.get("num_threads"):
        so.intra_op_num_threads = int(opts["num_threads"])
    onnx_path = root / "onnx" / onnx_file
    if not onnx_path.is_file():
        # Tolerate a different variant on disk (e.g. an older install that
        # fetched model_quantized.onnx). model.onnx (fp32) sorts first — a safe,
        # fast CPU fallback.
        alts = sorted((root / "onnx").glob("model*.onnx"))
        if not alts:
            raise FileNotFoundError(
                f"No Kokoro ONNX model in {root / 'onnx'} — re-run setup to fetch it"
            )
        logger.warning("Kokoro ONNX %r not found; using %s", onnx_file, alts[0].name)
        onnx_path = alts[0]
    logger.info("Loading Kokoro ONNX (%s) from %s", onnx_path.name, root)
    _session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    _g2p = _build_g2p(en)
    if speed:
        set_speed(speed)
    return _session


def set_speed(speed) -> None:
    """Set the speech-rate multiplier (>1 faster, <1 slower).

    Kokoro takes a `speed` input natively, so this is pitch-preserving. Clamped
    to a sane 0.5–2.0; a bad value is ignored rather than raised.
    """
    global _speed
    try:
        s = float(speed)
    except (TypeError, ValueError):
        return
    _speed = min(max(s, 0.5), 2.0)
    logger.info("Kokoro TTS speed: %.2fx", _speed)


def set_voice(voice_name: Optional[str]):
    """Select a built-in Kokoro voice; load its style matrix. Returns the name.

    Falls back gracefully if the requested voice isn't a known name OR its .bin
    isn't on disk (e.g. a partial/stale download): tries DEFAULT_VOICE, then any
    voice file present, so TTS works instead of crashing the transcriber thread.
    """
    global _voice, _voice_style
    name = voice_name if voice_name in KOKORO_VOICES else DEFAULT_VOICE
    if voice_name and name != voice_name:
        logger.warning("%r is not a known Kokoro voice; using %r", voice_name, name)
    path = _voices_dir / f"{name}.bin"
    if not path.is_file():
        default_path = _voices_dir / f"{DEFAULT_VOICE}.bin"
        available = sorted(_voices_dir.glob("*.bin"))
        if default_path.is_file():
            path, name = default_path, DEFAULT_VOICE
        elif available:
            path, name = available[0], available[0].stem
        else:
            raise FileNotFoundError(
                f"No Kokoro voice files in {_voices_dir} — re-run setup to download voices/*.bin"
            )
        logger.warning("Kokoro voice %r not on disk; falling back to %r", voice_name, name)
    _voice_style = np.fromfile(str(path), dtype=np.float32).reshape(-1, 1, 256)
    _voice = name
    logger.info("Kokoro voice: %s", name)
    return name


def _desegment(text: str) -> str:
    """Split likely OOV concatenations ("partlycloudy" -> "partly cloudy").

    misaki has no entry for run-on tokens like HA's `partlycloudy` slug, so they
    phonemise to nothing and the word is dropped. wordninja recovers the intended
    words. Gated tight — long, alphabetic, and splitting into >=2 pieces each
    >=3 chars — so ordinary words (which wordninja returns unchanged) and
    alphanumerics ("v2.2") are never touched. No-op without wordninja.
    """
    if _wordninja is None:
        return text
    out = []
    for tok in text.split(" "):
        core = tok.strip(string.punctuation)
        if len(core) >= _MIN_DESEGMENT_LEN and core.isalpha():
            pieces = _wordninja.split(core)
            if len(pieces) >= 2 and all(len(p) >= 3 for p in pieces):
                start = tok.index(core)
                lead, trail = tok[:start], tok[start + len(core) :]
                out.append(lead + " ".join(pieces) + trail)
                logger.debug("Desegmented run-on %r -> %r", core, " ".join(pieces))
                continue
        out.append(tok)
    return " ".join(out)


def _phoneme_tokens(text: str) -> list:
    out = _g2p(_desegment(text))
    phonemes = out[0] if isinstance(out, (tuple, list)) else out
    toks, dropped = [], []
    for c in phonemes:
        tid = _vocab.get(c)
        if tid is not None:
            toks.append(tid)
        elif not c.isspace():
            dropped.append(c)
    if dropped:
        # A word that loses phonemes here gets mispronounced or silently dropped;
        # surfacing it makes the otherwise-invisible failure debuggable.
        logger.debug(
            "Kokoro G2P: %d out-of-vocab phoneme(s) dropped (%r) in %r",
            len(dropped),
            "".join(dropped),
            text,
        )
    if len(toks) > MAX_TOKENS:
        logger.warning(
            "Kokoro phonemes truncated %d -> %d (clause too long): %r", len(toks), MAX_TOKENS, text
        )
    return toks[:MAX_TOKENS]


def _synth(text: str) -> np.ndarray:
    """Synthesise one (short) text span to a float32 waveform at 24 kHz."""
    toks = _phoneme_tokens(text)
    if not toks:
        return np.zeros(0, dtype=np.float32)
    idx = min(len(toks), _voice_style.shape[0] - 1)
    ref_s = _voice_style[idx]  # (1, 256)
    input_ids = np.array([[0, *toks, 0]], dtype=np.int64)
    audio = _session.run(
        None,
        {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([_speed], dtype=np.float32),
        },
    )[0]
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _to_chunks(audio: np.ndarray) -> list:
    return [audio[i : i + _CHUNK_SAMPLES] for i in range(0, len(audio), _CHUNK_SAMPLES)]


@dataclass
class _Job:
    text: str
    out: "queue.Queue"
    session: TtsSession


_job_queue: "queue.Queue[_Job]" = queue.Queue()


def _worker_loop() -> None:
    """Synthesise jobs fragment-by-fragment, pushing ~500ms chunks to the job's
    output queue. The small first fragment minimises time-to-first-audio; running
    on one thread lets the next fragment synthesise while the consumer plays the
    current one (overlap)."""
    while True:
        job = _job_queue.get()
        try:
            for fragment in list(split_clauses(job.text)) or [job.text]:
                if job.session.cancelled:
                    break
                for chunk in _to_chunks(_synth(fragment)):
                    if job.session.cancelled:
                        break
                    if chunk.size:
                        job.out.put((chunk, SAMPLE_RATE))
        except Exception as e:
            logger.exception("Kokoro ONNX synth error: %s: %s", type(e).__name__, e)
        finally:
            job.out.put(None)


_worker = threading.Thread(target=_worker_loop, daemon=True, name="kokoro-onnx-worker")
_worker.start()


def _submit(text: str, session: TtsSession, maxsize: int = 8) -> "queue.Queue":
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    _job_queue.put(_Job(text=text, out=out, session=session))
    return out


def _drain(out: "queue.Queue") -> None:
    while out.get() is not None:
        pass


def _drain_queue_nowait(q: "queue.Queue") -> None:
    """Discard everything currently sitting in `q` without blocking.

    Used on barge-in to drop audio chunks this same producer already queued
    for the browser but that `_send()` (server/dashboard.py) hasn't sent
    yet, so a "cancel" message isn't stuck behind stale, about-to-be-cut
    audio.
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
    logger.info("Warming up TTS model...")
    synthesize("A rainbow is caused by reflection and refraction of light.", prompt)
    logger.info("TTS model ready")


def play_chunks(
    chunks,
    sample_rate: int,
    session: Optional[TtsSession] = None,
    sink: Optional[queue.Queue] = None,
    tts_active_event: Optional[threading.Event] = None,
):
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
    sink: Optional[queue.Queue] = None,
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
                    # Chunks are sent to the browser as fast as they're
                    # generated with no flow control, so audio already
                    # queued here is likely already playing client-side by
                    # the time a barge-in is detected. "end" is a no-op on
                    # the client; "cancel" tells it to actually stop
                    # already-scheduled playback. Drop anything still
                    # sitting unsent first so the cancel isn't stuck behind
                    # stale audio.
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
