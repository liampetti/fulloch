"""Periodic 'still thinking' stall during long SLM thinking-mode calls.

Used as a context manager around a thinking-mode `generate_slm` call. While
the call runs, a background thread plays a pre-rendered stall phrase every
`interval` seconds. Exits cleanly on normal completion (the `with` block
finishes) or on user barge-in (`session.cancelled` flips to True).

Pre-rendering the stall phrases — same pattern as the chat stall cache —
keeps the cloned voice consistent and avoids stalling-the-staller (we
can't reasonably synthesise live while the SLM is the GPU-bound task).
"""

from __future__ import annotations

import logging
import queue
import random
import threading
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 6.0


class ThinkingWatchdog:
    """Plays a stall phrase every `interval` seconds while the body runs.

    Args:
        stall_cache: list of `(audio_chunks, sample_rate)` tuples produced
            by `tts.synthesize`. Picked at random each tick so the user
            doesn't hear the same phrase twice in a row.
        play_chunks: the assistant's bound `tts.play_chunks` function.
        session: the active `TtsSession` — playback honours `.cancelled`
            so a barge-in stops the stall mid-phrase, and the watchdog
            itself exits early when the session flips to cancelled.
        sink / tts_active_event: forwarded from the call site's own thread
            (`_turn_local` doesn't reach the watchdog's daemon thread).
        interval: seconds between stalls. First stall fires after one
            interval (not immediately) — the user just asked for thinking
            and expects a brief delay before any vocal feedback.
        max_stalls: optional cap on phrases played. `None` repeats until the
            guarded operation completes or is cancelled.
    """

    def __init__(
        self,
        stall_cache: List[Tuple],
        play_chunks: Callable,
        session,
        sink: Optional["queue.Queue"] = None,
        tts_active_event: Optional[threading.Event] = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        max_stalls: Optional[int] = None,
    ):
        self._stall_cache = stall_cache or []
        self._play = play_chunks
        self._session = session
        self._sink = sink
        self._tts_active_event = tts_active_event
        self._interval = interval
        self._max_stalls = max_stalls
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ThinkingWatchdog":
        # If no stall phrases are cached (warmup hasn't run or failed),
        # the watchdog still acts as a no-op context — keeps the call
        # site simple regardless of cache state.
        if self._stall_cache and self._max_stalls != 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        # wait() returns False after a timeout, True when stop is set.
        # Looping until stop fires means we stall every `interval` seconds.
        stalls = 0
        while not self._stop.wait(self._interval):
            if self._max_stalls is not None and stalls >= self._max_stalls:
                return
            if self._session is not None and getattr(self._session, "cancelled", False):
                return
            try:
                chunks, sample_rate = random.choice(self._stall_cache)
                self._play(chunks, sample_rate, session=self._session, sink=self._sink, tts_active_event=self._tts_active_event)
                stalls += 1
            except Exception as e:
                # A failed stall must not break the SLM call — log and let
                # the watchdog tick again on the next interval.
                logger.error(f"Thinking watchdog play failed: {e}")
