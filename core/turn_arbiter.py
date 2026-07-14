"""`TurnArbiter` — exclusive ownership of "whose turn is it" across every
turn source (every connected satellite and the dashboard's own
`"dashboard-text"` pseudo-satellite).

Wraps (does not replace) `Assistant._turn_lock`, which only ever serialises
the SLM call itself (the local model server is shared). The arbiter spans
the *whole* turn — from the moment a wakeword/text prompt is about to be
dispatched through to the reply finishing (or being cancelled) — so a second
source can never interleave `_history` writes or steal another satellite's
turn bookkeeping mid-turn.

Because the transcriber loop is single-threaded over one shared
`audio_queue`, two satellites' wakeword events can't race each other into
`_start_turn` simultaneously in the first place. The arbiter mainly protects
against a turn that's still running (its worker thread hasn't finished, or a
dashboard text turn is mid-SLM-call on the FastAPI thread) when a *different*
source's turn attempt arrives while the transcriber loop or another request
handler is live at the same time.
"""

import threading
from typing import Optional


class TurnArbiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Optional[str] = None

    def try_acquire(self, owner: str) -> bool:
        """Non-blocking: a loser must not stall its caller (the transcriber
        loop, or a dashboard request thread) waiting for the winner's turn
        to finish."""
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._owner = owner
        return acquired

    def release(self, owner: str) -> None:
        """No-op if `owner` doesn't currently hold the arbiter — guards
        against a stray/late release (e.g. a turn that already lost and
        never acquired) accidentally freeing someone else's turn."""
        if self._owner != owner:
            return
        self._owner = None
        self._lock.release()

    @property
    def owner(self) -> Optional[str]:
        return self._owner
