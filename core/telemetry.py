"""Small structured event stream for correlating assistant work with VM telemetry."""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("fulloch.telemetry")
logger.addHandler(logging.NullHandler())
logger.propagate = False


def event(kind: str, **fields) -> None:
    """Write one JSON event without retaining audio or transcript content."""
    logger.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}, separators=(",", ":")))
