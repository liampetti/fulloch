"""Mic capture, endpointing (VAD or RMS), and utterance buffering for ASR."""

import logging
import math
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)


def resolve_device(name: Optional[str], want_input: bool) -> Optional[object]:
    """Resolve a device-name substring to a PortAudio index, falling back to
    the system default (None) when nothing matches.

    sounddevice raises if a `device=` substring matches no device, so passing
    a configured device name straight through crashes the stream whenever that
    device is absent. Pre-resolving here lets a configured mic/speaker fall
    back to the system default instead of taking down startup — the named
    device is used when present, the default when it isn't.
    """
    if not name:
        return None
    key = "max_input_channels" if want_input else "max_output_channels"
    kind = "input" if want_input else "output"
    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - host audio quirks
        logger.warning(f"Could not query audio devices ({exc}); using system default {kind}")
        return None
    lowered = name.lower()
    for idx, dev in enumerate(devices):
        if lowered in dev["name"].lower() and dev[key] > 0:
            logger.info(f"Resolved {kind} device {name!r} -> [{idx}] {dev['name']}")
            return idx
    logger.warning(f"No {kind} device matching {name!r}; falling back to system default")
    return None


# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
SILENCE_DURATION_MS = 1500
MIN_UTTERANCE_MS = 1500
MAX_UTTERANCE_MS = 30000
SILENCE_THRESHOLD = 0.001
# Barge-in capture floor: the mic-silence threshold used *only* while the
# assistant is speaking. Its sole job is barge-in sensitivity — an interrupting
# voice must exceed it to be captured during playback. (The assistant's own
# voice leaks back as residue above SILENCE_THRESHOLD; without a stricter floor
# an utterance captured during TTS never reaches silence and grows until the
# force-flush.) Expressed in dBFS — the same unit each transcription's volume is
# logged in — so it can be read off the logs and tuned directly: lower = easier
# to barge in; raise it if the assistant's own voice self-interrupts. Converted
# to linear RMS once, at the one comparison site. Override per-mic via
# general.barge_in_threshold_dbfs (e.g. a speakerphone that ducks its mic during
# playback makes interrupts arrive quiet, so it wants a lower floor).
BARGE_IN_THRESHOLD_DBFS = -48.0
# Force-flush the buffer this often while TTS is playing, even if silence
# never triggers. Without this, a long uninterrupted TTS response (e.g.
# 20s of narration) holds the buffer until playback ends, so the
# wakeword the user spoke mid-TTS only surfaces as ASR text after the
# assistant has already finished speaking — too late to barge in.
TTS_MAX_UTTERANCE_MS = 2000
# When force-flushing during TTS, retain this much of the tail as the
# start of the next buffer so the wakeword can't fall on a chunk
# boundary and get split between two ASR results.
TTS_OVERLAP_MS = 1000
# Shorter min during TTS so a brief wakeword utterance ("Atticus.") is
# still long enough to enqueue when it's force-flushed.
TTS_MIN_UTTERANCE_MS = 500
# Shorter min while the wakeword-free follow-up window is open: replies to the
# assistant are often one or two words ("yes", "stop", "the kitchen one") that
# fall under MIN_UTTERANCE_MS and would otherwise be dropped before ASR. The
# follow-up window only accepts short utterances anyway, and VAD already guards
# against noise blips (speech must have been detected), so the long floor isn't
# needed here.
FOLLOW_UP_MIN_UTTERANCE_MS = 500
# VAD endpointing: speech probability threshold (Silero outputs 0..1 per
# window; higher = stricter about what counts as voice) and how long the
# probability must stay low before the speaker is judged to have finished.
VAD_THRESHOLD = 0.6
VAD_ENDPOINT_SILENCE_MS = SILENCE_DURATION_MS
# Soft (early) endpoint: a second, shorter silence after which the recorder
# emits a *provisional* snapshot of the utterance-so-far for the transcriber to
# probe (ASR + completeness/safe-intent), letting a clearly-finished command
# commit before the full 1.5s hard endpoint elapses. See
# docs/speculative-early-action.md. 0 disables the early-commit path entirely.
VAD_SOFT_ENDPOINT_SILENCE_MS = 500
# While VAD has not yet detected any speech, discard the buffer once it grows
# past this so a noisy room doesn't accumulate seconds of pre-speech audio
# (which would both inflate onset latency and hand ASR a long noise clip).
VAD_IDLE_RESET_MS = 3000
# Minimum *voiced* span (VAD end - start) for a silence-endpointed segment to
# be enqueued, outside the follow-up window. Silero scores a brief burst (a
# cough, a tap) above the speech threshold, so without this floor a single
# cough is enqueued and ASR hallucinates the wakeword from it. The wakeword
# phrase is ~800ms+ so 300ms is safe; the follow-up window stays exempt (a
# cough and a one-word reply like "no" are acoustically identical there).
VAD_MIN_SPEECH_MS = 300


def is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    """True if `chunk`'s RMS energy is below `threshold`."""
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk**2))
    return rms < threshold


# dBFS reported for digital silence (RMS ≈ 0), avoiding log(0). Real speech at
# this mic sits well above it; this is just the floor sentinel.
DBFS_SILENCE = -90.0


def _buf_rms(buf: np.ndarray) -> float:
    """Linear RMS of a buffer (0.0 for empty)."""
    if buf.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(buf**2)))


def rms_to_dbfs(rms: float) -> float:
    """Convert a linear RMS (0..1 for float32 PCM) to dBFS.

    dB is the meaningful unit for comparing loudness ("6 dB louder than the
    background") — linear RMS at this floor is tiny and not perceptually
    linear. Sub-floor / zero RMS clamps to `DBFS_SILENCE`.
    """
    if rms <= 1e-9:
        return DBFS_SILENCE
    return 20.0 * math.log10(rms)


def dbfs_to_rms(dbfs: float) -> float:
    """Inverse of `rms_to_dbfs`: dBFS back to linear RMS (0..1 for float32).

    Lets config express a threshold in the same dBFS unit the transcription
    volume is logged in, so it can be read off the logs directly.
    """
    return 10.0 ** (dbfs / 20.0)


def _contains_speech(buf: np.ndarray, vad_model, get_timestamps, sample_rate: int) -> bool:
    """Return True if Silero VAD detects at least one speech frame in `buf`."""
    tensor = torch.from_numpy(buf).float()
    timestamps = get_timestamps(tensor, vad_model, sampling_rate=sample_rate)
    return len(timestamps) > 0


class AudioCapture:
    """Mic capture with RMS silence detection.

    Recorder thread fills `audio_buffer`; once an utterance is silence- or
    max-length-bounded it's pushed to `audio_queue` for the transcriber to
    consume. `transcribing` gates processing; `tts_active` flips the silence
    threshold while the assistant is speaking; `_flush_pending` is the
    cross-thread signal `flush()` uses to drop the in-progress buffer.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_ms: int = CHUNK_DURATION_MS,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
        max_utterance_ms: int = MAX_UTTERANCE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
        barge_in_threshold_dbfs: Optional[float] = None,
        tts_max_utterance_ms: int = TTS_MAX_UTTERANCE_MS,
        tts_overlap_ms: int = TTS_OVERLAP_MS,
        tts_min_utterance_ms: int = TTS_MIN_UTTERANCE_MS,
        follow_up_min_utterance_ms: int = FOLLOW_UP_MIN_UTTERANCE_MS,
        input_device: Optional[str] = None,
        use_vad: bool = True,
        vad_threshold: Optional[float] = None,
        vad_endpoint_silence_ms: Optional[int] = None,
        vad_min_speech_ms: Optional[int] = None,
        vad_soft_endpoint_silence_ms: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold
        # Barge-in capture floor while TTS plays (see BARGE_IN_THRESHOLD_DBFS).
        # Kept in dBFS for clarity and log-parity; converted once here to the
        # linear RMS the per-chunk silence check actually compares against.
        self.barge_in_threshold_dbfs = (
            BARGE_IN_THRESHOLD_DBFS if barge_in_threshold_dbfs is None else barge_in_threshold_dbfs
        )
        self._barge_in_rms = dbfs_to_rms(self.barge_in_threshold_dbfs)
        self.input_device = resolve_device(input_device, want_input=True)

        self._vad_model = None
        self._vad_get_timestamps = None
        # Streaming endpointer (None when VAD is off/unavailable). When present
        # it — not RMS — decides end-of-speech outside TTS; RMS remains the
        # endpoint mechanism on the fallback path and while TTS is playing.
        # `_endpointer` is the *live* handle the recorder reads; `_endpointer_built`
        # retains the constructed object so `set_use_vad` can toggle VAD off and
        # back on without a model reload (None on _built means VAD never loaded).
        self._endpointer = None
        self._endpointer_built = None
        if use_vad:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                from .vad import VadEndpointer

                self._vad_model = load_silero_vad()
                self._vad_get_timestamps = get_speech_timestamps
                soft_ms = (
                    VAD_SOFT_ENDPOINT_SILENCE_MS
                    if vad_soft_endpoint_silence_ms is None
                    else vad_soft_endpoint_silence_ms
                )
                # The soft endpoint needs its own Silero handle so it can't
                # corrupt the hard iterator's LSTM state (see VadEndpointer).
                soft_model = load_silero_vad() if soft_ms else None
                self._endpointer_built = VadEndpointer(
                    self._vad_model,
                    sample_rate=sample_rate,
                    threshold=VAD_THRESHOLD if vad_threshold is None else vad_threshold,
                    endpoint_silence_ms=(
                        vad_endpoint_silence_ms
                        if vad_endpoint_silence_ms is not None
                        else silence_duration_ms
                    ),
                    soft_model=soft_model,
                    soft_endpoint_silence_ms=soft_ms,
                )
                self._endpointer = self._endpointer_built
                logger.info(
                    "Silero VAD loaded — speech-based endpointing enabled"
                    + (f" (soft endpoint {soft_ms}ms)" if soft_ms else "")
                )
            except Exception as e:
                logger.warning(
                    f"Silero VAD requested but failed to load ({e}); running without VAD"
                )

        # Derived values
        self.frames_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_chunks_needed = max(1, int(silence_duration_ms / chunk_duration_ms))
        self.min_utterance_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_ms / 1000)
        self.tts_max_utterance_samples = int(sample_rate * tts_max_utterance_ms / 1000)
        self.tts_overlap_samples = int(sample_rate * tts_overlap_ms / 1000)
        self.tts_min_utterance_samples = int(sample_rate * tts_min_utterance_ms / 1000)
        self.follow_up_min_utterance_samples = int(sample_rate * follow_up_min_utterance_ms / 1000)
        self.vad_idle_reset_samples = int(sample_rate * VAD_IDLE_RESET_MS / 1000)
        self.vad_min_speech_samples = int(
            sample_rate
            * (VAD_MIN_SPEECH_MS if vad_min_speech_ms is None else vad_min_speech_ms)
            / 1000
        )
        # Slack added to the follow-up deadline so a reply that *starts* just
        # inside the window still clears the recorder's shorter-min gate when
        # it's endpointed ~silence_duration later (the assistant gauges the
        # window from speech onset, the recorder acts at endpoint time).
        self._follow_up_slack_s = silence_duration_ms / 1000.0 + 1.5

        # State
        self.audio_buffer: deque = deque()
        # Each item is `(buf, speech_onset_monotonic, loudness_dbfs)` — the
        # onset lets the transcriber measure the follow-up window from when the
        # speaker began, and the loudness tags the utterance with its dBFS
        # volume (voiced-window RMS where VAD is active) for noise-baseline
        # logging. None is the stop pill.
        self.audio_queue: "queue.Queue[Optional[tuple[np.ndarray, float, float]]]" = queue.Queue()
        self.running = True
        self.transcribing = True
        # Set by Assistant while any TTS audio is playing. The recorder
        # switches to the barge-in floor (`_barge_in_rms`) so the assistant's
        # own residue is treated as silence and utterances end on time.
        self.tts_active = threading.Event()
        # Set by Assistant while the wakeword-free follow-up window is open.
        # The recorder then accepts shorter utterances (a brief reply to the
        # assistant) instead of holding them to the full min length. Paired
        # with `_follow_up_until` so the window auto-expires even if the
        # Assistant never gets a chance to clear it.
        self.follow_up_active = threading.Event()
        self._follow_up_until = 0.0
        # Cross-thread signal: when set, the recorder drops its in-progress
        # buffer at the top of the next iteration. Set by `flush()`; only
        # the recorder thread mutates `audio_buffer` to avoid races with
        # the InputStream callback.
        self._flush_pending = False
        # Debounce for the soft-endpoint provisional probe: set once a provisional
        # has been emitted for the current pause, cleared when speech resumes
        # (the endpointer re-arms `soft_endpointed`), so each pause yields at most
        # one provisional snapshot rather than one per chunk.
        self._soft_probe_emitted = False

    # --- Live config setters (settings console hot-apply) ------------------
    # Each mutates a single derived value the recorder reads on its next loop
    # iteration; rebinding a scalar/handle is atomic, so no lock is needed (the
    # recorder thread sees either the old or new value, never a torn one).

    def set_use_vad(self, enabled: bool) -> bool:
        """Toggle VAD endpointing live. Returns False if it can't (VAD model
        was never loaded, so enabling needs a restart)."""
        if enabled and self._endpointer_built is None:
            return False
        self._endpointer = self._endpointer_built if enabled else None
        return True

    def set_barge_in_threshold_dbfs(self, dbfs: float) -> None:
        """Update the TTS-path silence floor (dBFS) and its linear-RMS cache."""
        self.barge_in_threshold_dbfs = float(dbfs)
        self._barge_in_rms = dbfs_to_rms(self.barge_in_threshold_dbfs)

    def set_vad_min_speech_ms(self, ms: int) -> None:
        """Update the minimum voiced duration sent to ASR."""
        self.vad_min_speech_samples = int(self.sample_rate * int(ms) / 1000)

    def set_vad_params(
        self, threshold=None, endpoint_silence_ms=None, soft_endpoint_silence_ms=None
    ) -> None:
        """Live-tune the endpointer thresholds/silences (no-op without VAD)."""
        if self._endpointer_built is not None:
            self._endpointer_built.update_params(
                threshold=threshold,
                endpoint_silence_ms=endpoint_silence_ms,
                soft_endpoint_silence_ms=soft_endpoint_silence_ms,
            )

    def arm_follow_up(self, window_seconds: float) -> None:
        """Open the follow-up window for `window_seconds` (plus capture slack).

        While open the recorder uses `follow_up_min_utterance_samples`, so a
        short reply isn't dropped before it reaches ASR.
        """
        self._follow_up_until = time.monotonic() + window_seconds + self._follow_up_slack_s
        self.follow_up_active.set()

    def clear_follow_up(self) -> None:
        """Close the follow-up window (e.g. when a fresh turn begins)."""
        self.follow_up_active.clear()
        self._follow_up_until = 0.0

    def _follow_up_open(self) -> bool:
        """True while the follow-up window is armed and not yet expired."""
        if not self.follow_up_active.is_set():
            return False
        if time.monotonic() >= self._follow_up_until:
            self.follow_up_active.clear()
            return False
        return True

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream. Must be fast — no resampling here."""
        if status:
            logger.info(status)
        self.audio_buffer.append(indata[:, 0].copy())

    def flush(self) -> None:
        """Discard the in-progress utterance and any queued utterances.

        Drains `audio_queue` directly (Queue is thread-safe) and signals the
        recorder thread to clear `audio_buffer` on its next iteration —
        keeping all mutations of `audio_buffer` on a single thread. Called
        after a barge-in cancel so contaminated TTS-bleed audio captured
        during the cancelled turn doesn't pollute the next user utterance.
        """
        drained = 0
        while True:
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        self._flush_pending = True
        if drained:
            logger.debug(f"Flushed {drained} queued utterances; buffer flush pending")

    def recorder_thread(self):
        """Capture audio chunks; enqueue each utterance once silence-bounded."""
        if self.input_device:
            logger.info(f"Starting microphone stream on device: {self.input_device}")
        else:
            logger.info("Starting microphone stream on system default device")
        silence_counter = 0
        # Monotonic time the current utterance's speech began — the first
        # non-silent chunk since the last buffer clear. Enqueued with the
        # audio so the follow-up window measures from speech onset.
        speech_onset_t: Optional[float] = None

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.frames_per_chunk,
            latency="low",
            callback=self._audio_callback,
            device=self.input_device,
        ):
            while self.running:
                time.sleep(self.chunk_duration_ms / 1000.0)

                if self._flush_pending:
                    self.audio_buffer.clear()
                    silence_counter = 0
                    speech_onset_t = None
                    self._soft_probe_emitted = False
                    if self._endpointer is not None:
                        self._endpointer.reset()
                    self._flush_pending = False
                    continue

                if self.transcribing:
                    if not self.audio_buffer:
                        continue

                    tts_active = self.tts_active.is_set()

                    # VAD-driven endpointing (outside TTS): speech probability,
                    # not RMS energy, decides when the speaker has finished —
                    # robust in a noisy room where energy never drops to a
                    # silence floor. RMS still governs the TTS/barge-in path
                    # below (a latency mechanism, not a noise problem).
                    if self._endpointer is not None and not tts_active:
                        self._endpointer.process(self.audio_buffer[-1])
                        buffer_samples = sum(c.size for c in self.audio_buffer)

                        # Discard accumulating noise before any speech is
                        # detected so a noisy room neither inflates onset
                        # latency nor hands ASR a long noise clip.
                        if (
                            not self._endpointer.speech_started
                            and buffer_samples >= self.vad_idle_reset_samples
                        ):
                            self.audio_buffer.clear()
                            self._endpointer.reset()
                            continue

                        # Soft (early) endpoint: the speaker has briefly paused
                        # but the hard endpoint hasn't fired. Emit one provisional
                        # snapshot per pause for the transcriber to probe — it
                        # commits the turn early if the partial is a complete/safe
                        # command, else drops it and waits for the hard endpoint.
                        # Nothing is cleared/reset here, so the buffer keeps
                        # growing toward the real endpoint regardless.
                        if (
                            self._endpointer.soft_endpointed
                            and not self._endpointer.endpointed
                            and self._endpointer.speech_started
                        ):
                            if not self._soft_probe_emitted:
                                min_required = (
                                    self.follow_up_min_utterance_samples
                                    if self._follow_up_open()
                                    else self.min_utterance_samples
                                )
                                if buffer_samples >= min_required:
                                    buf = np.concatenate(list(self.audio_buffer), axis=0)
                                    onset = self._endpointer.speech_onset or time.monotonic()
                                    rms = self._endpointer.voiced_rms
                                    if rms is None:
                                        rms = _buf_rms(buf)
                                    loudness_db = rms_to_dbfs(rms)
                                    # 4-tuple: the trailing True marks it provisional.
                                    self.audio_queue.put((buf, onset, loudness_db, True))
                                    self._soft_probe_emitted = True
                                    secs = buf.size / self.sample_rate
                                    logger.debug(
                                        f"VAD soft endpoint: provisional {secs:.2f}s enqueued"
                                    )
                        elif not self._endpointer.soft_endpointed:
                            # Speech resumed (or never paused) — re-arm the probe.
                            self._soft_probe_emitted = False

                        hit_silence = self._endpointer.endpointed
                        hit_max = buffer_samples >= self.max_utterance_samples
                        if not (hit_silence or hit_max):
                            continue

                        # Speech-duration floor (silence-endpointed segments
                        # only — a hit_max segment is long genuine speech).
                        # Drop a too-brief voiced burst — a cough Silero scored
                        # as speech — before it reaches ASR and gets
                        # hallucinated into the wakeword. Exempt while the
                        # follow-up window is open: a cough there is
                        # indistinguishable from a one-word reply.
                        if (
                            hit_silence
                            and not self._follow_up_open()
                            and self._endpointer.last_speech_samples < self.vad_min_speech_samples
                        ):
                            secs = self._endpointer.last_speech_samples / self.sample_rate
                            logger.debug(f"VAD: speech span {secs:.2f}s < min — dropped as noise")
                            self.audio_buffer.clear()
                            self._endpointer.reset()
                            continue

                        # A short reply during the follow-up window ("yes",
                        # "stop") would fall under the normal min; accept the
                        # shorter floor while it's open.
                        min_required = (
                            self.follow_up_min_utterance_samples
                            if self._follow_up_open()
                            else self.min_utterance_samples
                        )
                        buf = np.concatenate(list(self.audio_buffer), axis=0)
                        if buf.size >= min_required and self._endpointer.speech_started:
                            onset = self._endpointer.speech_onset or time.monotonic()
                            # Tag with the voiced-window loudness; fall back to
                            # whole-buffer RMS if no segment finalised (hit_max
                            # before an endpoint).
                            rms = self._endpointer.voiced_rms
                            if rms is None:
                                rms = _buf_rms(buf)
                            loudness_db = rms_to_dbfs(rms)
                            self.audio_queue.put((buf, onset, loudness_db))
                            secs = buf.size / self.sample_rate
                            logger.debug(f"VAD endpoint: enqueued {secs:.2f}s for transcription")
                        self.audio_buffer.clear()
                        self._endpointer.reset()
                        continue

                    # The assistant's own residue during TTS sits above the
                    # normal threshold, so utterances captured during playback
                    # would never reach silence. Use the barge-in floor while
                    # TTS is active — only a louder interrupt counts as speech.
                    threshold = self._barge_in_rms if tts_active else self.silence_threshold
                    last_chunk = self.audio_buffer[-1]
                    if is_silent(last_chunk, threshold):
                        silence_counter += 1
                    else:
                        silence_counter = 0
                        if speech_onset_t is None:
                            speech_onset_t = time.monotonic()

                    # During TTS we cap the buffer much shorter so the
                    # wakeword surfaces while we're still speaking; outside
                    # TTS we keep the long max for natural utterances.
                    max_samples = (
                        self.tts_max_utterance_samples if tts_active else self.max_utterance_samples
                    )
                    min_samples = (
                        self.tts_min_utterance_samples
                        if tts_active
                        else self.follow_up_min_utterance_samples
                        if self._follow_up_open()
                        else self.min_utterance_samples
                    )

                    buffer_samples = sum(c.size for c in self.audio_buffer)
                    hit_silence = silence_counter >= self.silence_chunks_needed
                    hit_max = buffer_samples >= max_samples
                    if not (hit_silence or hit_max):
                        continue

                    buf = np.concatenate(list(self.audio_buffer), axis=0)

                    # Post-hoc VAD gate: enqueue only a long-enough buffer that
                    # contains speech. A no-speech buffer is dropped silently —
                    # on the TTS/RMS path this fires constantly on AEC residue.
                    # `and` short-circuits, so VAD only runs once the size gate
                    # passes.
                    if buf.size >= min_samples and (
                        self._vad_model is None
                        or _contains_speech(
                            buf, self._vad_model, self._vad_get_timestamps, self.sample_rate
                        )
                    ):
                        onset = speech_onset_t if speech_onset_t is not None else time.monotonic()
                        # RMS path has no per-window voiced measure; tag
                        # with whole-buffer RMS.
                        loudness_db = rms_to_dbfs(_buf_rms(buf))
                        self.audio_queue.put((buf, onset, loudness_db))
                        secs = buf.size / self.sample_rate
                        logger.debug(f"Enqueued {secs:.2f}s for transcription")

                    if tts_active and hit_max and not hit_silence:
                        # Force-flush mid-stream — retain the last ~1s as the
                        # seed of the next chunk so the wakeword can't fall
                        # across a boundary and get split between two ASR
                        # results. Pop from the left in-place (rather than
                        # reassigning self.audio_buffer) so a concurrent
                        # callback append can't be lost to the orphaned
                        # deque. silence_counter intentionally preserved.
                        total = sum(c.size for c in self.audio_buffer)
                        while (
                            len(self.audio_buffer) > 1
                            and total - self.audio_buffer[0].size >= self.tts_overlap_samples
                        ):
                            total -= self.audio_buffer.popleft().size
                        continue

                self.audio_buffer.clear()
                silence_counter = 0
                speech_onset_t = None

    def stop(self):
        """Signal the recorder to stop and inject poison pill."""
        self.running = False
        self.audio_queue.put(None)
