"""Failure containment for in-process inference workers."""

import logging
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

INFERENCE_TIMEOUT_SECONDS = 60
TTS_JOB_QUEUE_MAXSIZE = 4
_DIAGNOSTIC_PATH = Path("data/logs/inference_watchdog.log")


class TtsQueueFullError(RuntimeError):
    """Raised when the bounded shared TTS worker queue cannot admit a job."""


def submit_tts_job(job_queue: queue.Queue, job) -> None:
    """Admit a TTS job without making a caller wait behind an overloaded worker."""
    try:
        job_queue.put_nowait(job)
    except queue.Full as exc:
        raise TtsQueueFullError("TTS is busy; queued speech limit reached") from exc


def _record_timeout(reason: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    message = f"{timestamp} inference watchdog timeout: {reason}"
    logger.critical(message)
    if os.environ.get("FULLOCH_PERSISTENT_LOGGING_ENABLED") != "1":
        return
    try:
        _DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DIAGNOSTIC_PATH.open("a", encoding="ascii") as diagnostic_file:
            diagnostic_file.write(message + "\n")
    except OSError:
        logger.exception("Could not write inference watchdog diagnostic")


class InferenceWatchdog:
    """Force a container restart if one in-process inference call stalls."""

    def __init__(
        self,
        reason: str,
        timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS,
        *,
        exit_fn=os._exit,
    ):
        self.reason = reason
        self.timeout_seconds = timeout_seconds
        self.exit_fn = exit_fn
        self._timer: threading.Timer | None = None

    def _expire(self) -> None:
        _record_timeout(self.reason)
        self.exit_fn(1)

    def __enter__(self):
        self._timer = threading.Timer(self.timeout_seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._timer is not None:
            self._timer.cancel()
