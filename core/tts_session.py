"""Cancellation primitive for an in-progress TTS playback.

`speak_stream` accepts an optional `TtsSession`. Other threads call `.stop()`
to abort playback; `speak_stream` checks `.cancelled` in its producer and
consumer loops to stop enqueuing further chunks to the satellite sink.
"""

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TtsSession:
    stop_event: threading.Event = field(default_factory=threading.Event)
    active: bool = False

    def stop(self) -> None:
        """Signal active TTS playback to abort. Idempotent; no-op when idle."""
        self.stop_event.set()

    @property
    def cancelled(self) -> bool:
        return self.stop_event.is_set()


def parse_barge_time(value) -> float:
    """Parse a duration config value ("0s", "<N>s", "always") into seconds.

    Used for `follow_up_time`. Falls back to 0.0 on anything unrecognised.
    """
    if value is None:
        return 0.0
    s = str(value).strip().lower()
    if s == "always":
        return float("inf")
    if s.endswith("s"):
        s = s[:-1]
    try:
        return max(0.0, float(s))
    except ValueError:
        logger.warning(f"Invalid duration value: {value!r}; defaulting to 0s")
        return 0.0
