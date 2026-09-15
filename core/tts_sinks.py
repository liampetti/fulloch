"""PCM gain, bounded transport queues, and multi-satellite TTS delivery."""

import queue
import threading
from typing import Optional

import numpy as np


class GainSink:
    """Apply a per-turn gain without changing the shared TTS sink protocol."""

    def __init__(self, sink, gain: float):
        self._sink = sink
        self._gain = gain

    def put(self, item, *args, **kwargs) -> None:
        if isinstance(item, tuple) and item and isinstance(item[0], np.ndarray):
            item = (item[0] * self._gain, *item[1:])
        self._sink.put(item, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sink, name)


class NonBlockingSink:
    """Pace PCM producers while letting start/cancel/stop replace queued audio."""

    def __init__(self, sink: queue.Queue):
        self._sink = sink

    def put(self, item, *args, **kwargs) -> None:
        kind = item[0] if isinstance(item, tuple) and item else None
        if not isinstance(kind, str):
            self._sink.put(item, *args, **kwargs)
            return
        try:
            self._sink.put_nowait(item)
            return
        except queue.Full:
            pass
        if kind in {"start", "cancel", "stop"}:
            self._drain()
        else:  # Completion markers must not displace PCM.
            self._sink.put(item, *args, **kwargs)
            return
        try:
            self._sink.put_nowait(item)
        except queue.Full:
            pass

    def put_nowait(self, item) -> None:
        self.put(item)

    def get_nowait(self):
        return self._sink.get_nowait()

    def _drain(self) -> None:
        while True:
            try:
                self._sink.get_nowait()
            except queue.Empty:
                return


class BroadcastSink:
    """Fan out one TTS stream to every active proactive-speech recipient."""

    def __init__(self, sinks: list):
        self._sinks = sinks

    def put(self, item, *args, **kwargs) -> None:
        for sink in self._sinks:
            sink.put(item, *args, **kwargs)


class BroadcastEvent:
    """Mirror a TTS backend's active/inactive transition across satellites."""

    def __init__(self, events: list[threading.Event]):
        self._events = events

    def set(self) -> None:
        for event in self._events:
            event.set()

    def clear(self) -> None:
        for event in self._events:
            event.clear()


def safe_sink(sink: Optional[queue.Queue]):
    """Let bounded browser/native send queues pace the producer without PCM loss."""
    return NonBlockingSink(sink) if sink is not None and sink.maxsize > 0 else sink
