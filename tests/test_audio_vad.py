"""Tests for the VAD streaming endpointer (`core/vad.py`).

`VadEndpointer` is the unit that replaces RMS energy with Silero speech
probability for end-of-speech detection. The Silero model / `VADIterator` is
a heavy dependency, so here we drive the endpointer's windowing and state
machine with a scripted fake iterator — the real model only matters at
runtime, the logic under test is the framing of arbitrary-length input into
512-sample windows and the start/end transitions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vad import VAD_WINDOW_SAMPLES, VadEndpointer


class FakeIterator:
    """Scripted stand-in for Silero's `VADIterator`.

    Returns the next event from `events` on each window call (one call per
    512-sample window), `None` once exhausted, and counts calls/resets so
    tests can assert how many windows were processed.
    """

    def __init__(self, events):
        self._events = list(events)
        self.calls = 0
        self.reset_calls = 0

    def __call__(self, window):
        self.calls += 1
        return self._events.pop(0) if self._events else None

    def reset_states(self):
        self.reset_calls += 1


def make_endpointer(events):
    """Build an endpointer whose iterator is the scripted fake.

    The real `VADIterator` is constructed (harmlessly, against a mock model)
    in __init__ and then replaced, so the test never touches the model.
    """
    ep = VadEndpointer(MagicMock(), sample_rate=16000)
    ep._iterator = FakeIterator(events)
    return ep


def make_endpointer_with_soft(hard_events, soft_events):
    """Endpointer with both the hard and a scripted *soft* iterator installed."""
    ep = VadEndpointer(MagicMock(), sample_rate=16000)
    ep._iterator = FakeIterator(hard_events)
    ep._soft_iterator = FakeIterator(soft_events)
    return ep


def _windows(n):
    """`n` windows' worth of (silent) float32 samples."""
    return np.zeros(n * VAD_WINDOW_SAMPLES, dtype=np.float32)


def test_start_then_end_sets_transitions():
    ep = make_endpointer([{"start": 0}, None, {"end": 100}])
    ep.process(_windows(3))
    assert ep.speech_started is True
    assert ep.endpointed is True
    assert ep.speech_onset is not None


def test_no_speech_never_endpoints():
    ep = make_endpointer([None, None, None])
    ep.process(_windows(3))
    assert ep.speech_started is False
    assert ep.endpointed is False
    assert ep.speech_onset is None


def test_speech_started_without_end_does_not_endpoint():
    ep = make_endpointer([{"start": 0}, None])
    ep.process(_windows(2))
    assert ep.speech_started is True
    assert ep.endpointed is False


def test_onset_recorded_only_once():
    ep = make_endpointer([{"start": 0}, {"start": 50}])
    ep.process(_windows(2))
    first = ep.speech_onset
    # A second {'start'} (e.g. after a brief dip) must not move the onset.
    assert ep.speech_onset == first


def test_residual_carried_across_calls():
    """Sub-window leftovers are buffered, not dropped."""
    ep = make_endpointer([{"start": 0}, {"end": 1}])
    # 512 + 100 samples: one full window now, 100 carried.
    ep.process(np.zeros(VAD_WINDOW_SAMPLES + 100, dtype=np.float32))
    assert ep._iterator.calls == 1
    assert ep._residual.size == 100
    # 412 more completes the second window (100 + 412 == 512).
    ep.process(np.zeros(412, dtype=np.float32))
    assert ep._iterator.calls == 2
    assert ep._residual.size == 0
    assert ep.endpointed is True


def test_partial_window_alone_processes_nothing():
    ep = make_endpointer([{"start": 0}])
    ep.process(np.zeros(VAD_WINDOW_SAMPLES - 1, dtype=np.float32))
    assert ep._iterator.calls == 0
    assert ep.speech_started is False
    assert ep._residual.size == VAD_WINDOW_SAMPLES - 1


def test_reset_clears_all_state():
    ep = make_endpointer([{"start": 0}, {"end": 1}])
    ep.process(_windows(2))
    ep.process(np.zeros(50, dtype=np.float32))  # leave a residual
    assert ep._residual.size == 50
    ep.reset()
    assert ep.speech_started is False
    assert ep.endpointed is False
    assert ep.speech_onset is None
    assert ep._residual.size == 0
    assert ep._iterator.reset_calls == 1


def test_non_float32_input_is_accepted():
    ep = make_endpointer([{"start": 0}])
    ep.process(np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float64))
    assert ep._iterator.calls == 1
    assert ep.speech_started is True


def test_last_speech_samples_is_voiced_span():
    # The recorder gates on this (end - start) to drop brief noise bursts.
    ep = make_endpointer([{"start": 1000}, {"end": 9000}])
    ep.process(_windows(2))
    assert ep.last_speech_samples == 8000


def test_reset_clears_last_speech_samples():
    ep = make_endpointer([{"start": 1000}, {"end": 9000}])
    ep.process(_windows(2))
    ep.reset()
    assert ep.last_speech_samples == 0
    assert ep._seg_start_sample is None


# --- soft endpoint (early-commit signal) ----------------------------------

def test_soft_endpoint_disabled_by_default():
    # No soft iterator → flag never sets, even across a full start/end.
    ep = make_endpointer([{"start": 0}, {"end": 100}])
    ep.process(_windows(2))
    assert ep.soft_endpointed is False


def test_soft_endpoint_latches_on_soft_end():
    # Soft iterator ends earlier than the hard one (still mid-silence here).
    ep = make_endpointer_with_soft(
        hard_events=[{"start": 0}, None, None],
        soft_events=[{"start": 0}, {"end": 50}, None],
    )
    ep.process(_windows(3))
    assert ep.soft_endpointed is True
    assert ep.endpointed is False  # hard endpoint hasn't fired yet


def test_soft_endpoint_rearms_on_resumed_speech():
    # Pause (soft end) → speaker resumes (soft start) → flag clears again.
    ep = make_endpointer_with_soft(
        hard_events=[{"start": 0}, None, None, None],
        soft_events=[{"start": 0}, {"end": 50}, {"start": 60}, None],
    )
    ep.process(_windows(4))
    assert ep.soft_endpointed is False


def test_reset_clears_soft_endpointed():
    ep = make_endpointer_with_soft(
        hard_events=[{"start": 0}, None],
        soft_events=[{"start": 0}, {"end": 50}],
    )
    ep.process(_windows(2))
    assert ep.soft_endpointed is True
    ep.reset()
    assert ep.soft_endpointed is False
    assert ep._soft_iterator.reset_calls == 1
