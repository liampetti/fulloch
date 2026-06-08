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
# Shorter min during TTS so a brief wakeword utterance ("Connery.") is
# still long enough to enqueue when it's force-flushed.
TTS_MIN_UTTERANCE_MS = 500


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
        input_device: Optional[str] = None,
        use_vad: bool = True,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold
        self.tts_silence_threshold = tts_silence_threshold
        self.input_device = input_device

        self._vad_model = None
        self._vad_get_timestamps = None
        if use_vad:
            try:
                from silero_vad import load_silero_vad, get_speech_timestamps
                self._vad_model = load_silero_vad()
                self._vad_get_timestamps = get_speech_timestamps
                logger.info("Silero VAD loaded — non-speech buffers will be dropped")
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
        # Cross-thread signal: when set, the recorder drops its in-progress
        # buffer at the top of the next iteration. Set by `flush()`; only
        # the recorder thread mutates `audio_buffer` to avoid races with
        # the InputStream callback.
        self._flush_pending = False

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
                    self._flush_pending = False
                    continue

                if self.transcribing:
                    if not self.audio_buffer:
                        continue

                    tts_active = self.tts_active.is_set()
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
