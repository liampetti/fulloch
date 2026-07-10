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


def test_loudness_sink_holds_dbfs_of_the_yielded_buffer():
    q = queue.Queue()
    q.put((_buf(), 11.0, -42.0))
    q.put((_buf(), 22.0, -18.5))
    q.put(None)
    onset = {"t": 0.0}
    loud = {"db": None}
    gen = stream_generator(q, onset, loud)

    next(gen)
    assert loud["db"] == -42.0
    next(gen)
    assert loud["db"] == -18.5


def test_audio_sink_holds_the_yielded_buffer():
    # The audio sink stashes the raw buffer so a bare wakeword can be
    # re-transcribed without the context bias to confirm it.
    q = queue.Queue()
    b1, b2 = _buf(), _buf()
    q.put((b1, 11.0, -42.0, False))
    q.put((b2, 22.0, -18.5, False))
    q.put(None)
    audio = {"buf": None}
    gen = stream_generator(q, None, None, None, audio)

    next(gen)
    assert audio["buf"] is b1
    next(gen)
    assert audio["buf"] is b2


def test_missing_loudness_is_tolerated():
    # 2-tuples (no loudness) still unpack; sink reports None.
    q = queue.Queue()
    q.put((_buf(), 5.0))
    q.put(None)
    loud = {"db": 1.0}
    assert len(list(stream_generator(q, None, loud))) == 1
    assert loud["db"] is None


def test_no_sink_is_tolerated():
    q = queue.Queue()
    q.put((_buf(), 5.0))
    q.put(None)
    assert len(list(stream_generator(q))) == 1


def test_empty_queue_stops_immediately_on_sentinel():
    q = queue.Queue()
    q.put(None)
    assert list(stream_generator(q)) == []


def test_endpoint_wait_sink_holds_dequeue_minus_endpoint_time():
    import time

    q = queue.Queue()
    endpoint_t = time.monotonic() - 0.05  # queued 50ms ago
    q.put((_buf(), 11.0, -42.0, False, "sat-a", endpoint_t))
    q.put(None)
    sink = {"s": None}
    gen = stream_generator(q, None, None, None, None, None, sink)

    next(gen)
    assert sink["s"] is not None
    assert sink["s"] >= 0.05


def test_endpoint_wait_sink_none_without_endpoint_time():
    # 5-tuple (no endpoint_monotonic) still unpacks; sink reports None.
    q = queue.Queue()
    q.put((_buf(), 11.0, -42.0, False, "sat-a"))
    q.put(None)
    sink = {"s": "unset"}
    gen = stream_generator(q, None, None, None, None, None, sink)

    next(gen)
    assert sink["s"] is None
