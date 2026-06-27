"""Runtime lifecycle for the two-phase startup (v2.2 Step 3).

Phase A (this object exists from the very first line of `main`) parses config,
starts the web server, and decides whether first-run setup is needed. Phase B
loads the models and runs the assistant. `/status` exposes the phase so the
browser routes between the setup wizard and the dashboard and drives the
download/loading screens.

    NEEDS_SETUP ──► DOWNLOADING ──► LOADING ──► READY
         │                                        ▲
         └────────────── (existing install) ──────┘
    any phase ──► ERROR  (config update needed, fatal load failure)

Import-light (stdlib only) so it can be created before — and independently of —
torch / llama / the model stack.
"""

import collections
import itertools
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class LogBuffer:
    """Thread-safe ring buffer of recent log lines for the loading screen.

    The loading screen polls `/status` while models load; surfacing the tail of
    the project's own INFO logs there turns the bare spinner into a live
    "here's what's happening" terminal. Each line carries a monotonic `seq` so
    the browser can tell which lines it hasn't shown yet (and animate just
    those) without the server tracking per-client state.
    """

    def __init__(self, maxlen: int = 80):
        self._lock = threading.Lock()
        self._items: collections.deque = collections.deque(maxlen=maxlen)
        self._seq = itertools.count(1)

    def add(self, text: str, level: str = "INFO") -> None:
        with self._lock:
            self._items.append({"seq": next(self._seq), "text": text, "level": level})

    def tail(self, limit: int = 40) -> list:
        with self._lock:
            items = list(self._items)
        return items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# Module-level singleton the capture handler writes to and `/status` reads from.
LOG_BUFFER = LogBuffer()


class _BufferLogHandler(logging.Handler):
    """Routes project-side log records into `LOG_BUFFER`.

    Only first-party loggers are captured (third-party startup chatter is noise
    on a user-facing screen, and is already capped at WARNING in app.py). The
    bare message — no timestamp/name/level prefix — is stored so it reads as a
    clean terminal line; `level` rides alongside for colouring warnings/errors.
    """

    _PROJECT_ROOTS = ("core", "utils", "tools", "audio", "server", "app", "__main__")

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.split(".", 1)[0] not in self._PROJECT_ROOTS:
            return
        try:
            LOG_BUFFER.add(record.getMessage(), record.levelname)
        except Exception:  # never let logging blow up the app
            pass


def install_log_capture(level: int = logging.INFO) -> _BufferLogHandler:
    """Attach the buffering handler to the root logger. Call once at startup."""
    handler = _BufferLogHandler()
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler

NEEDS_SETUP = "NEEDS_SETUP"
DOWNLOADING = "DOWNLOADING"
LOADING = "LOADING"
READY = "READY"
ERROR = "ERROR"

PHASES = (NEEDS_SETUP, DOWNLOADING, LOADING, READY, ERROR)


class Lifecycle:
    """Thread-safe holder for the current startup phase + a human detail string.

    `proceed` is set when first-run setup finishes, so the blocked `main` thread
    advances from the setup-mode server into Phase B (wired by the wizard's
    install endpoint in a later step).
    """

    def __init__(self, phase: str = LOADING, detail: str = "", **extra):
        self._lock = threading.Lock()
        self._phase = phase
        self._detail = detail
        self._extra = dict(extra)
        self.proceed = threading.Event()

    def set(self, phase: str, detail: str = "", **extra) -> None:
        if phase not in PHASES:
            logger.warning("Unknown lifecycle phase %r", phase)
        with self._lock:
            self._phase = phase
            self._detail = detail
            self._extra.update(extra)
        logger.info("Lifecycle -> %s%s", phase, f": {detail}" if detail else "")

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def is_ready(self) -> bool:
        return self.phase == READY

    def snapshot(self) -> dict:
        """`{phase, detail, **extra}` — merged into the `/status` payload."""
        with self._lock:
            return {"phase": self._phase, "detail": self._detail, **self._extra}

    def signal_proceed(self) -> None:
        """Release the setup-mode block so `main` enters Phase B."""
        self.proceed.set()


class AppContext:
    """Shared mutable handle the dashboard reads live.

    Lets a single dashboard server start in setup mode (assistant=None) and
    later have the assistant attached after Phase B, without a second server.
    `on_attach` callbacks (e.g. registering the SSE turn listener) fire as soon
    as the assistant is available — immediately if already set, else on
    `set_assistant`.
    """

    def __init__(self, lifecycle: Lifecycle, assistant=None,
                 downloader=None, config_path: str = "./data/config.yml"):
        self.lifecycle = lifecycle
        self.assistant = assistant
        self.downloader = downloader
        self.config_path = config_path
        # Live dashboard bearer token. None until create_app seeds it from the
        # env; the post-setup token step sets it so the console is gated without
        # a restart. Empty/None = no auth (zero-config local-only).
        self.dashboard_token: Optional[str] = None
        self._on_attach: list = []

    def on_attach(self, callback) -> None:
        self._on_attach.append(callback)
        if self.assistant is not None:
            callback(self.assistant)

    def set_assistant(self, assistant) -> None:
        self.assistant = assistant
        for cb in self._on_attach:
            try:
                cb(assistant)
            except Exception:
                logger.exception("on_attach callback failed")
