"""Runtime lifecycle for two-phase startup.

Phase A (this object exists from the very first line of `main`) parses config,
starts the web server, and decides whether first-run setup is needed. Phase B
loads the models and runs the assistant. `/status` exposes the phase so the
browser routes between the setup wizard and the dashboard and drives the
download/loading screens.

    NEEDS_SETUP ──► DOWNLOADING ──► LOADING ──► READY
         │                                        ▲
         └────────────── (existing install) ──────┘
    any phase ──► ERROR  (config update needed, fatal load failure)

Import-light so it can be created before model loading.
"""

import collections
import itertools
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LogBuffer:
    """Thread-safe recent-log buffer for the loading screen."""

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

    def __init__(
        self,
        lifecycle: Lifecycle,
        assistant=None,
        downloader=None,
        config_path: str = "./data/config.yml",
    ):
        self.lifecycle = lifecycle
        self.assistant = assistant
        self.downloader = downloader
        self.config_path = config_path
        # PBKDF2 hash loaded from credentials.json at startup. None = no auth
        # (zero-config local-only). Set by /setup/password without restart.
        self.dashboard_password_hash: Optional[str] = None
        # Persisted dashboard sessions.
        from .auth import load_sessions
        self.sessions: dict = load_sessions()
        self._on_attach: list = []
        # Obsidian plugin command queue — set when plugin connects, cleared on disconnect.
        # Notes tool pushes {"type": "open_file", "path": "..."} here after writes.
        self.obsidian_cmd_q: Optional[Any] = None
        # Obsidian plugin + vault state — set when the plugin WebSocket connects,
        # cleared on disconnect. `indexing_progress` is 0.0–1.0 during a rebuild
        # triggered by a vault-path change; None when idle.
        self.obsidian_vault_state: dict = {
            "connected": False,
            "vault_path": None,
            "vault_resolved_path": None,
            "plugin_vault_path": None,
            "path_navigation_mismatch": False,
            "last_connected_at": None,
            "last_error": None,
            "indexing_progress": None,
        }

    def on_attach(self, callback) -> None:
        self._on_attach.append(callback)
        if self.assistant is not None:
            callback(self.assistant)

    def set_assistant(self, assistant) -> None:
        self.assistant = assistant
        # Expose the live assistant through the lightweight satellite context.
        from core.satellite_context import set_current_assistant

        set_current_assistant(assistant)
        for cb in self._on_attach:
            try:
                cb(assistant)
            except Exception:
                logger.exception("on_attach callback failed")
