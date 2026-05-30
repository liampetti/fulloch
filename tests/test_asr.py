"""Tests for the ASR stream generator's queue contract.

`stream_generator` is the bridge between the recorder's audio queue and the
ASR pipeline. It must unpack `(buf, onset)` tuples, surface each buffer's
speech-onset time via the sink so the follow-up window can measure from when
the speaker began, and stop on the None sentinel.
"""

import queue
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.asr import stream_generator  # noqa: E402


def _buf(n=4):
    return np.zeros(n, dtype=np.float32)


def test_yields_buffers_until_none_sentinel():
    q = queue.Queue()
    q.put((_buf(), 1.0))
    q.put((_buf(), 2.0))
    q.put(None)
    out = list(stream_generator(q))
    assert len(out) == 2
    assert all(isinstance(b, np.ndarray) for b in out)


def test_sink_holds_onset_of_the_yielded_buffer():
    q = queue.Queue()
    q.put((_buf(), 11.0))
    q.put((_buf(), 22.0))
    q.put(None)
    sink = {"t": 0.0}
    gen = stream_generator(q, sink)

    next(gen)
    assert sink["t"] == 11.0  # onset of first buffer, before its transcription
    next(gen)
    assert sink["t"] == 22.0  # advanced only when the next buffer is pulled


def test_no_sink_is_tolerated():
    q = queue.Queue()
    q.put((_buf(), 5.0))
    q.put(None)
    assert len(list(stream_generator(q))) == 1


def test_empty_queue_stops_immediately_on_sentinel():
    q = queue.Queue()
    q.put(None)
    assert list(stream_generator(q)) == []
