"""
Audio capture and silence detection module.

Handles microphone input, silence detection via RMS threshold,
and audio buffering for the speech recognition pipeline.
"""

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
SILENCE_DURATION_MS = 1500
MIN_UTTERANCE_MS = 1500
MAX_UTTERANCE_MS = 30000
SILENCE_THRESHOLD = 0.001
# Stricter silence threshold used while TTS is playing. PulseAudio AEC
# attenuates the assistant's own voice but leaves a low-amplitude residue
# above SILENCE_THRESHOLD — without raising the bar, utterances captured
# during TTS never reach silence and grow until MAX_UTTERANCE_MS (30s),
# which produces the "assistant hangs after barge-in" symptom. Real user
# speech is loud enough at the mic to break through this stricter floor.
TTS_SILENCE_THRESHOLD = 0.01
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
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms < threshold


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
        tts_silence_threshold: float = TTS_SILENCE_THRESHOLD,
        tts_max_utterance_ms: int = TTS_MAX_UTTERANCE_MS,
        tts_overlap_ms: int = TTS_OVERLAP_MS,
        tts_min_utterance_ms: int = TTS_MIN_UTTERANCE_MS,
        follow_up_min_utterance_ms: int = FOLLOW_UP_MIN_UTTERANCE_MS,
        input_device: Optional[str] = None,
        use_vad: bool = True,
        vad_threshold: Optional[float] = None,
        vad_endpoint_silence_ms: Optional[int] = None,
        vad_min_speech_ms: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold
        self.tts_silence_threshold = tts_silence_threshold
        self.input_device = input_device

        self._vad_model = None
        self._vad_get_timestamps = None
        # Streaming endpointer (None when VAD is off/unavailable). When present
        # it — not RMS — decides end-of-speech outside TTS; RMS remains the
        # endpoint mechanism on the fallback path and while TTS is playing.
        self._endpointer = None
        if use_vad:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                from .vad import VadEndpointer
                self._vad_model = load_silero_vad()
                self._vad_get_timestamps = get_speech_timestamps
                self._endpointer = VadEndpointer(
                    self._vad_model,
                    sample_rate=sample_rate,
                    threshold=VAD_THRESHOLD if vad_threshold is None else vad_threshold,
                    endpoint_silence_ms=(
                        vad_endpoint_silence_ms
                        if vad_endpoint_silence_ms is not None
                        else silence_duration_ms
                    ),
                )
                logger.info("Silero VAD loaded — speech-based endpointing enabled")
            except Exception as e:
                logger.warning(f"Silero VAD requested but failed to load ({e}); running without VAD")

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
        # Each item is `(buf, speech_onset_monotonic)` — the onset lets the
        # transcriber measure the follow-up window from when the speaker
        # began, not from when the transcription lands. None is the stop pill.
        self.audio_queue: "queue.Queue[Optional[tuple[np.ndarray, float]]]" = queue.Queue()
        self.running = True
        self.transcribing = True
        # Set by Assistant while any TTS audio is playing. The recorder
        # switches to `tts_silence_threshold` so AEC residue is treated as
        # silence and utterances end on time.
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
                        if (not self._endpointer.speech_started
                                and buffer_samples >= self.vad_idle_reset_samples):
                            self.audio_buffer.clear()
                            self._endpointer.reset()
                            continue

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
                        if (hit_silence and not self._follow_up_open()
                                and self._endpointer.last_speech_samples
                                    < self.vad_min_speech_samples):
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
                        if (buf.size >= min_required
                                and self._endpointer.speech_started):
                            onset = self._endpointer.speech_onset or time.monotonic()
                            self.audio_queue.put((buf, onset))
                            secs = buf.size / self.sample_rate
                            logger.debug(f"VAD endpoint: enqueued {secs:.2f}s for transcription")
                        self.audio_buffer.clear()
                        self._endpointer.reset()
                        continue

                    # AEC residue during TTS sits above the normal threshold,
                    # so utterances captured during playback would never reach
                    # silence. Use the stricter floor while TTS is active.
                    threshold = (
                        self.tts_silence_threshold
                        if tts_active
                        else self.silence_threshold
                    )
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
                        self.tts_max_utterance_samples if tts_active
                        else self.max_utterance_samples
                    )
                    min_samples = (
                        self.tts_min_utterance_samples if tts_active
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

                    if buf.size >= min_samples:
                        if self._vad_model is not None and not _contains_speech(
                            buf, self._vad_model, self._vad_get_timestamps, self.sample_rate
                        ):
                            logger.debug("VAD: no speech detected — buffer dropped")
                        else:
                            onset = (
                                speech_onset_t if speech_onset_t is not None
                                else time.monotonic()
                            )
                            self.audio_queue.put((buf, onset))
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
                            and total - self.audio_buffer[0].size
                                >= self.tts_overlap_samples
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
