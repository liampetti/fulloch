"""
Audio capture and silence detection module.

Handles microphone input, silence detection via RMS threshold,
and audio buffering for the speech recognition pipeline.
"""

import logging
import queue
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
SILENCE_DURATION_MS = 1000
MIN_UTTERANCE_MS = 1500
MAX_UTTERANCE_MS = 10000
SILENCE_THRESHOLD = 0.001

# Derived values
frames_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
silence_chunks_needed = max(1, int(SILENCE_DURATION_MS / CHUNK_DURATION_MS))
min_utterance_samples = int(SAMPLE_RATE * MIN_UTTERANCE_MS / 1000)
max_utterance_samples = int(SAMPLE_RATE * MAX_UTTERANCE_MS / 1000)


def is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    """
    Check if an audio chunk is silent based on RMS energy.

    Args:
        chunk: Audio samples as numpy array
        threshold: RMS threshold below which audio is considered silent

    Returns:
        True if the chunk is silent, False otherwise
    """
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms < threshold


class AudioCapture:
    """
    Manages audio capture from microphone with silence detection.

    Attributes:
        audio_buffer: Deque holding audio chunks
        audio_queue: Queue for complete utterances ready for transcription
        running: Flag to control capture loop
        transcribing: Flag to control whether to process audio
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_ms: int = CHUNK_DURATION_MS,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
        max_utterance_ms: int = MAX_UTTERANCE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold

        # Derived values
        self.frames_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_chunks_needed = max(1, int(silence_duration_ms / chunk_duration_ms))
        self.min_utterance_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_ms / 1000)

        # State
        self.audio_buffer: deque = deque()
        self.audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self.running = True
        self.transcribing = True

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream. Must be fast — no resampling here."""
        if status:
            logger.info(status)
        self.audio_buffer.append(indata[:, 0].copy())

    def recorder_thread(self):
        """
        Main recording thread that captures audio and detects utterances.

        Runs continuously, accumulating audio chunks and detecting
        end-of-utterance based on silence duration.
        """
        logger.info("Starting microphone stream...")
        silence_counter = 0

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.frames_per_chunk,
            latency="low",
            callback=self._audio_callback,
        ):
            while self.running:
                time.sleep(self.chunk_duration_ms / 1000.0)

                if self.transcribing:
                    if not self.audio_buffer:
                        continue

                    # Check silence on the most recent chunk
                    last_chunk = self.audio_buffer[-1]
                    if is_silent(last_chunk, self.silence_threshold):
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    # Check if we've hit silence threshold or max length
                    buffer_samples = sum(c.size for c in self.audio_buffer)
                    if (silence_counter < self.silence_chunks_needed and
                        buffer_samples <= self.max_utterance_samples):
                        continue

                    # End of utterance - concatenate buffer
                    buf = np.concatenate(list(self.audio_buffer), axis=0)

                    # Only enqueue if minimum length met
                    if buf.size >= self.min_utterance_samples:
                        self.audio_queue.put(buf)
                        secs = buf.size / self.sample_rate
                        logger.debug(f"Enqueued {secs:.2f}s for transcription")

                # Reset state
                self.audio_buffer.clear()
                silence_counter = 0

    def stop(self):
        """Signal the recorder to stop and inject poison pill."""
        self.running = False
        self.audio_queue.put(None)
