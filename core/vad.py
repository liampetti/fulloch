"""Streaming voice-activity endpointer built on Silero VAD.

End-of-speech detection driven by *speech probability* rather than RMS
energy. RMS endpointing fails in a noisy room: background noise (fans, TV,
keyboard) keeps the energy above any silence floor, so the trailing-silence
timer never accumulates and the utterance grows until the max-length cap.
Silero scores each 32 ms window for the presence of *voice* specifically, so
non-speech noise leaves a low probability and the trailing-silence timer keeps
counting — the speaker's pause is detected even over a loud background.

`VadEndpointer` wraps Silero's `VADIterator` (the streaming wrapper) and feeds
it exact 512-sample windows, surfacing two transitions for the recorder:
`speech_started` (the speaker began) and `endpointed` (the speaker has been
silent for `endpoint_silence_ms` after speaking). Kept in its own module so
tests can exercise the windowing/state logic without importing the audio
stack.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD operates on fixed 512-sample windows at 16 kHz (32 ms). Input
# arriving in arbitrary chunk sizes (the recorder's 200 ms = 3200 samples is
# not a multiple of 512) is sliced into windows; the leftover tail is carried
# in `_residual` and prepended to the next batch.
VAD_WINDOW_SAMPLES = 512


class VadEndpointer:
    """Streaming speech endpointer over a preloaded Silero VAD model.

    The model is loaded by the caller (so the same handle can back both this
    streaming path and any batch `get_speech_timestamps` use) and passed in.
    `VADIterator` carries its own hysteresis: `threshold` gates speech, and
    `min_silence_duration_ms` is how long the probability must stay low before
    it emits an `{'end'}` — i.e. our end-of-speech criterion.
    """

    def __init__(
        self,
        model,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        endpoint_silence_ms: int = 1500,
        soft_model=None,
        soft_endpoint_silence_ms: Optional[int] = None,
    ):
        from silero_vad import VADIterator

        self.sample_rate = sample_rate
        self._iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=endpoint_silence_ms,
        )
        # Optional *soft* endpoint: a second VADIterator with a shorter silence
        # requirement, giving an early "speaker has briefly paused" signal for
        # the speculative early-commit path.
        # It needs its OWN Silero handle — VADIterator advances the model's LSTM
        # state on every window, so sharing the hard iterator's model would
        # corrupt the proven endpoint. Disabled (None) unless both a soft model
        # and a positive silence are supplied; the hard path is then unchanged.
        self._soft_iterator = None
        if soft_model is not None and soft_endpoint_silence_ms:
            self._soft_iterator = VADIterator(
                soft_model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_silence_duration_ms=soft_endpoint_silence_ms,
            )
        # Latched by the soft iterator's `{'end'}` and re-armed (cleared) on its
        # next `{'start'}`, so each pause yields one soft endpoint. Always False
        # when the soft iterator is disabled.
        self.soft_endpointed = False
        self._residual = np.empty(0, dtype=np.float32)
        # True once VADIterator has emitted a `{'start'}` for the current
        # utterance; `endpointed` only after a subsequent `{'end'}`.
        self.speech_started = False
        self.endpointed = False
        # Monotonic time speech began, captured on the `{'start'}` event so
        # the follow-up window measures from speech onset.
        self.speech_onset: Optional[float] = None
        # Sample index of the current segment's `{'start'}`, and the voiced
        # span (end - start) of the segment that just endpointed. The recorder
        # gates on `last_speech_samples` outside the follow-up window so a
        # brief burst Silero scores as speech (a cough/tap) is dropped instead
        # of enqueued — otherwise ASR hallucinates the wakeword (or echoes its
        # context prompt) from the noise.
        self._seg_start_sample: Optional[int] = None
        self.last_speech_samples = 0
        # Loudness (linear RMS) of the segment that just endpointed, measured
        # over the *voiced* 512-sample windows only (the 80th-percentile window
        # RMS — robust to a soft onset/offset and to the leading/trailing
        # near-silence a whole-buffer RMS would dilute). None until a segment
        # has endpointed. Read by the recorder to tag the enqueued utterance
        # with a dBFS volume; nothing acts on it yet.
        self.voiced_rms: Optional[float] = None
        self._in_speech = False
        self._voiced_rms_vals: list[float] = []

    def process(self, samples: np.ndarray) -> None:
        """Feed newly captured samples, advancing the endpointer state.

        Slices `samples` (prepended with any carried residual) into 512-sample
        windows and runs each through `VADIterator`, updating `speech_started`
        / `endpointed` / `speech_onset` from the emitted events. Sub-window
        leftovers are retained for the next call.
        """
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if self._residual.size:
            samples = np.concatenate([self._residual, samples])

        offset = 0
        n = samples.size
        while offset + VAD_WINDOW_SAMPLES <= n:
            window = samples[offset : offset + VAD_WINDOW_SAMPLES]
            offset += VAD_WINDOW_SAMPLES
            rms = float(np.sqrt(np.mean(window**2)))
            # Soft endpoint: re-arm on a fresh speech start, latch on its (early)
            # end. Fed the same window as the hard iterator but with its own
            # model state, so it never disturbs the hard endpoint below.
            if self._soft_iterator is not None:
                soft_event = self._soft_iterator(window)
                if soft_event:
                    if "start" in soft_event:
                        self.soft_endpointed = False
                    if "end" in soft_event:
                        self.soft_endpointed = True
            event = self._iterator(window)
            if not event:
                # Accumulate the RMS of windows inside the current speech
                # segment so `voiced_rms` reflects only voiced audio.
                if self._in_speech:
                    self._voiced_rms_vals.append(rms)
                continue
            if "start" in event:
                self.speech_started = True
                self._seg_start_sample = event["start"]
                if self.speech_onset is None:
                    self.speech_onset = time.monotonic()
                self._in_speech = True
                self._voiced_rms_vals = [rms]
            if "end" in event:
                self.endpointed = True
                start = (
                    self._seg_start_sample if self._seg_start_sample is not None else event["end"]
                )
                self.last_speech_samples = max(0, event["end"] - start)
                if self._voiced_rms_vals:
                    self.voiced_rms = float(np.percentile(self._voiced_rms_vals, 80))
                self._in_speech = False

        self._residual = samples[offset:].copy()

    def update_params(
        self,
        threshold: Optional[float] = None,
        endpoint_silence_ms: Optional[int] = None,
        soft_endpoint_silence_ms: Optional[int] = None,
    ) -> None:
        """Live-tune the endpointer without rebuilding it (settings console).

        `VADIterator` keeps `threshold` and `min_silence_samples` as plain
        attributes (the latter = sampling_rate * ms / 1000, matching its own
        constructor), so a settings change can adjust them in place — no model
        reload, and any in-flight `process()` simply reads the new value on its
        next window. Rebinding the soft silence only applies when a soft
        iterator exists; turning the soft endpoint on from off needs a restart
        (no second Silero handle was loaded).
        """
        if threshold is not None:
            t = float(threshold)
            self._iterator.threshold = t
            if self._soft_iterator is not None:
                self._soft_iterator.threshold = t
        if endpoint_silence_ms is not None:
            self._iterator.min_silence_samples = self.sample_rate * int(endpoint_silence_ms) / 1000
        if soft_endpoint_silence_ms is not None and self._soft_iterator is not None:
            self._soft_iterator.min_silence_samples = (
                self.sample_rate * int(soft_endpoint_silence_ms) / 1000
            )

    def reset(self) -> None:
        """Clear all per-utterance state, ready for the next utterance."""
        self._iterator.reset_states()
        if self._soft_iterator is not None:
            self._soft_iterator.reset_states()
        self.soft_endpointed = False
        self._residual = np.empty(0, dtype=np.float32)
        self.speech_started = False
        self.endpointed = False
        self.speech_onset = None
        self._seg_start_sample = None
        self.last_speech_samples = 0
        self.voiced_rms = None
        self._in_speech = False
        self._voiced_rms_vals = []
