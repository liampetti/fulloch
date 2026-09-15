"""PCM channel selection, loudness, and batch speech checks."""

import math
from collections import deque

import numpy as np
import torch

SILENCE_THRESHOLD = 0.001


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


def _endpoint_mono(chunk: np.ndarray) -> np.ndarray:
    """Return the loudest channel for endpointing a mono/stereo chunk."""
    if chunk.ndim != 2:
        return chunk
    channel = int(np.argmax(np.mean(chunk**2, axis=0)))
    return chunk[:, channel]


def _utterance_mono(chunks: deque) -> np.ndarray:
    """Materialize an utterance, selecting its strongest persisted channel once."""
    buf = np.concatenate(list(chunks), axis=0)
    if buf.ndim == 1:
        return buf
    channel = int(np.argmax(np.mean(buf**2, axis=0)))
    return buf[:, channel]


def _utterance_pcm(chunks: deque) -> np.ndarray:
    """Materialize an utterance without discarding its satellite channels."""
    return np.concatenate(list(chunks), axis=0)


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
