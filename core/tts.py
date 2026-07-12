"""Qwen3 TTS with voice cloning, streaming via the Qwen3-TTS-streaming fork
(installed from liampetti/Qwen3-TTS-streaming@97da215, a pinned org mirror of
rekuenkdr/Qwen3-TTS-streaming so a deleted upstream fork can't brick installs).

All generation runs on a single long-lived worker thread (`_worker_thread`).
Two things this buys us:

  - `torch.compile` with `compile_mode="reduce-overhead"` uses CUDA graphs,
    and Inductor's tree manager is stored in thread-local storage. The
    previous per-call producer thread design hit the
    `assert torch._C._is_key_in_tls(attr_name)` in
    `torch/_inductor/cudagraph_trees.py` because every turn ran on a fresh
    thread that had never registered a tree manager. One worker = one TLS
    context = CUDA graphs work.
  - CUDA graphs pre-allocate + reuse the activation buffers across every
    decode step. Without them, Inductor re-allocates intermediates each
    forward, costing ~4-5 GB more VRAM at steady state. On the 16 GB
    card with the 9B SLM behind us that's the difference between fitting
    and OOM-ing.

Callers (`synthesize`, `speak_stream`, `warmup_model`) post a `_TtsJob`
to `_job_queue` and consume chunks via the job's own output queue. The
worker generates chunks sequentially and pumps them into that queue,
respecting the job's `TtsSession.cancelled` flag for barge-in.
"""

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from qwen_tts import Qwen3TTSModel

from .tts_session import TtsSession
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

VOICES_DIR = "./data/voices"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared kwargs for every `stream_generate_voice_clone` call — warmup,
# pre-render, and live streaming all use the same generation cadence so the
# voice clone sounds consistent across them.
_STREAM_KWARGS = {
    "language": "english",
    "overlap_samples": 512,
    # The codec runs at 12 Hz, so each frame encodes ~83ms of audio. At
    # emit_every_frames=6 (~500ms chunks) the worker checks
    # `session.cancelled` twice per second, so a mid-TTS barge-in waits
    # at most ~500ms for the current chunk to finish.
    "emit_every_frames": 6,
    "decode_window_frames": 80,
    "first_chunk_emit_every": 5,
    "first_chunk_decode_window": 48,
    "first_chunk_frames": 48,
}


# Populated by `load_tts()` once a backend is chosen — `import core.tts`
# itself loads nothing, so the module can be imported in setup mode (before
# any model is selected) without touching the GPU. The worker thread reads
# this global lazily inside `_worker_loop`, so it only needs to be set before
# the first generation job is submitted (warmup).
model = None

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def load_tts(model_id: str = DEFAULT_MODEL_ID, decode_window_frames: int = 80, **opts):
    """Load the Qwen3-TTS model and enable streaming optimizations.

    Sets the module-global `model` and returns it. Called from
    `Assistant._load_models` (via the backend registry) so loading is
    deferred until a TTS backend has actually been selected. `opts` absorbs
    any extra per-model config keys forwarded by the registry.
    """
    global model
    logger.info(f"Loading TTS model {model_id} on {DEVICE}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
    model.enable_streaming_optimizations(
        decode_window_frames=decode_window_frames,
        use_compile=True,
        compile_mode="reduce-overhead",
    )
    return model


@dataclass
class _TtsJob:
    """A single generation job processed by the TTS worker.

    `out` carries `(chunk, sample_rate)` tuples; the worker pushes `None`
    when the job is finished (either completed or cancelled mid-stream).
    """

    text: str
    voice_clone_prompt: Any
    out: "queue.Queue"
    session: TtsSession


_job_queue: "queue.Queue[_TtsJob]" = queue.Queue()


def _worker_loop() -> None:
    """Consume `_TtsJob`s and run generation on this thread only.

    Pinning every `stream_generate_voice_clone` call to this thread is
    what lets `compile_mode="reduce-overhead"` work — the CUDA graph
    tree manager registers in this thread's TLS on the first call and
    every subsequent call reuses it.
    """
    while True:
        job = _job_queue.get()
        try:
            for chunk, sample_rate in model.stream_generate_voice_clone(
                text=job.text,
                voice_clone_prompt=job.voice_clone_prompt,
                **_STREAM_KWARGS,
            ):
                if job.session.cancelled:
                    break
                job.out.put((chunk, sample_rate))
        except Exception as e:
            logger.exception(
                f"TTS generation error for text '{job.text[:50]}': {type(e).__name__}: {e}"
            )
        finally:
            job.out.put(None)


# Daemon so the worker dies with the main process. The first job that
# lands (warmup) is what triggers torch.compile — anchored on this
# thread, so reduce-overhead's CUDA graphs register against the right
# TLS context and every following job hits the cached graph.
_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="tts-worker")
_worker_thread.start()


def _submit(text: str, prompt, session: TtsSession, maxsize: int = 0) -> "queue.Queue":
    """Post a generation job and return the output queue to read chunks from."""
    out: "queue.Queue" = queue.Queue(maxsize=maxsize)
    _job_queue.put(
        _TtsJob(
            text=text,
            voice_clone_prompt=prompt,
            out=out,
            session=session,
        )
    )
    return out


def _drain_job(out: "queue.Queue") -> None:
    """Read and discard until the worker's `None` sentinel.

    A cancelled consumer MUST keep draining rather than walk away. The TTS
    worker is a single thread shared by every job, and it blocks on
    `job.out.put(...)` whenever a bounded output queue fills. If the consumer
    stopped reading mid-stream the worker would wedge on its next put — most
    insidiously the final `None` in its `finally` — and since that one worker
    serves all future speech, the assistant would go permanently mute after
    the first barge-in. Draining lets the worker break out of its generation
    loop and return to `_job_queue`. Bounded by one generation step: the
    worker checks `session.cancelled` before each put, so it emits `None`
    almost immediately once cancelled.
    """
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


def force_cancel_playback(sink: Optional[queue.Queue] = None) -> None:
    """Tell the browser to stop already-scheduled audio right now.

    On fast hardware, generation can outrun playback enough that
    `speak_stream`'s send loop finishes and exits *before* a barge-in/stop
    ever flips `session.cancelled` — at that point nothing is left polling
    the session, so the mid-stream "cancel" path in `speak_stream` never
    fires and the browser just keeps playing everything already queued.
    Callers that stop a turn (`request_stop`, barge-in) must call this
    unconditionally, not rely on `session.cancelled` alone, since by the
    time they run there may be no generation loop left to catch it.
    """
    if sink is None:
        return
    _drain_queue_nowait(sink)
    sink.put(("cancel",))


def set_voice(voice_name: str):
    """Build a voice-clone prompt from data/voices/<voice_name>.wav + transcript.

    The transcript is `<voice_name>.txt`, falling back to `data/voices/default.txt`
    — the shared transcript every bundled sample clip is recorded with — so a clip
    needs no per-voice .txt of its own.
    """
    ref_audio = f"{VOICES_DIR}/{voice_name}.wav"
    txt_path = f"{VOICES_DIR}/{voice_name}.txt"
    if not os.path.isfile(txt_path):
        txt_path = f"{VOICES_DIR}/default.txt"
    with open(txt_path) as f:
        ref_text = f.read()
    logger.info(f"Setting voice clone to: {voice_name}")
    return model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)


_speed_warned = False


def set_speed(speed) -> None:
    """Accept the speed-control contract, but no-op — Qwen3 TTS has no rate knob.

    The clone's pace is baked in by the voice-clone reference recording, not a
    generation parameter. Warn once so a configured `tts_speed` isn't silently
    ignored; the value otherwise has no effect here.
    """
    global _speed_warned
    if speed in (None, 1.0) or _speed_warned:
        return
    logger.warning(
        "Qwen TTS has no speed control — tts_speed=%s ignored. Set the pace via "
        "the voice-clone reference recording instead.",
        speed,
    )
    _speed_warned = True


def warmup_model(prompt):
    """Run one dummy generation so torch.compile + CUDA graphs are ready.

    The reference text must be long enough that the clone settles before
    live use — short warmup phrases leave the first real utterance sounding
    generic. Pairs with the greeting pre-render in `Assistant._warm_and_announce`
    which absorbs the still-locking-in first buffered pass. This is also
    the call that anchors the CUDA-graph tree manager on the worker
    thread; later jobs inherit it.
    """
    logger.info("Warming up TTS model...")
    out = _submit(
        text=(
            "A rainbow is a meteorological phenomenon that is caused by "
            "reflection, refraction and dispersion of light in water droplets."
        ),
        prompt=prompt,
        session=TtsSession(),
    )
    while out.get() is not None:
        pass
    logger.info("TTS model ready")


def synthesize(text: str, prompt):
    """Run TTS to completion and return (chunks, sample_rate).

    For phrases pre-rendered once and replayed via `play_chunks` — the stall
    phrases and the startup greeting. Pre-rendering hides the "clone hasn't
    locked in" generic-sounding output that short live phrases get.
    """
    out = _submit(text=text, prompt=prompt, session=TtsSession())
    chunks = []
    sample_rate = None
    while True:
        item = out.get()
        if item is None:
            break
        chunk, sr = item
        chunks.append(chunk)
        sample_rate = sr
    return chunks, sample_rate


def play_chunks(
    chunks,
    sample_rate: int,
    session: Optional[TtsSession] = None,
    sink: Optional[queue.Queue] = None,
    tts_active_event: Optional[threading.Event] = None,
):
    """Route a pre-synthesized chunk list to the WebSocket satellite.

    TTS is a browser-only I/O channel: chunks are pushed to `sink` when a
    browser is connected (sink is non-None). With no browser open, the
    chunks are dropped on the floor and the user simply doesn't hear this
    turn — the generation cost is wasted but the pipeline stays consistent.

    The TtsSession.cancelled path is honoured: a barge-in stops enqueuing
    further chunks and emits the "end" sentinel so the browser flushes.
    """
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
    prompt,
    session: Optional[TtsSession] = None,
    stats: Optional[TurnStats] = None,
    on_first_audio: Optional[Callable[[], None]] = None,
    sink: Optional[queue.Queue] = None,
    tts_active_event: Optional[threading.Event] = None,
):
    """Synthesise `text` with the given voice-clone prompt and play it back.

    Generation runs on the worker thread; this caller thread reads chunks
    off the worker's output queue and writes them to the output device.
    The two run concurrently bounded by the queue's maxsize. If `session`
    is supplied, another thread can call `session.stop()` to abort within
    ~one chunk (~500ms) — the worker breaks out of its generation loop
    and this loop calls `stream.abort()` to drop buffered audio.

    `on_first_audio` fires once, right after the first chunk is ready (and
    `stats.tts_seconds` is set) — before playback runs to completion — so the
    dashboard can show the TTS time without waiting for the voice to finish.

    Returns the estimated monotonic-clock time playback will finish on the
    browser (playback start + total audio duration). Chunks are handed to
    the satellite sink as fast as they're generated with no flow control, so
    generation can finish well before the audio has actually played out —
    callers that need to know "when did the user actually hear the reply
    end" (e.g. arming the follow-up window) should use this return value
    instead of the time `speak_stream` itself returns.
    """
    if session is None:
        session = TtsSession()
    session.stop_event.clear()
    session.active = True

    t_submit = time.monotonic()
    out = _submit(text=text, prompt=prompt, session=session, maxsize=5)
    try:
        first_item = out.get()
        if stats is not None:
            stats.tts_seconds = time.monotonic() - t_submit  # time to first audio
        if first_item is None:
            return time.monotonic()
        if session.cancelled:
            _drain_job(out)  # let the shared worker finish; don't wedge it
            return time.monotonic()
        if on_first_audio is not None:
            try:
                on_first_audio()
            except Exception as e:
                logger.warning(f"on_first_audio hook failed: {e}")
        first_chunk, sample_rate = first_item

        if tts_active_event is not None:
            tts_active_event.set()
        t_playback_start = time.monotonic()
        total_samples = 0
        try:
            if sink is not None:
                sink.put(("start", sample_rate))
                sink.put((first_chunk, None))
                total_samples += len(first_chunk)
                cancelled_mid_stream = False
                while True:
                    item = out.get()
                    if item is None:
                        break
                    if session.cancelled:
                        _drain_job(out)
                        cancelled_mid_stream = True
                        break
                    sink.put((item[0], None))
                    total_samples += len(item[0])
                if cancelled_mid_stream:
                    # Chunks are sent to the browser as fast as they're
                    # generated with no flow control, so audio already queued
                    # here is likely already playing client-side by the time
                    # a barge-in is detected. "end" is a no-op on the client;
                    # "cancel" tells it to actually stop already-scheduled
                    # playback. Drop anything still sitting unsent first so
                    # the cancel isn't stuck behind stale audio.
                    _drain_queue_nowait(sink)
                    sink.put(("cancel",))
                else:
                    sink.put(("end",))
            else:
                logger.warning("No satellite connected; TTS chunks dropped on the floor.")
                _drain_job(out)
        finally:
            if tts_active_event is not None:
                tts_active_event.clear()
        if sink is None or not sample_rate:
            return time.monotonic()
        return t_playback_start + total_samples / sample_rate
    finally:
        session.active = False
