"""Dashboard app composition, shared middleware and uvicorn startup.

Route families receive the same late-attachable AppContext. Their implementation
and state live in server.routes_*; startup entry points remain available here.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .lifecycle import READY, AppContext, Lifecycle
from .routes_auth import register_auth_routes
from .routes_chat import (
    HISTORY_LIMIT,
    PROACTIVE_REQUEST_LIMIT,
    SSE_SUBSCRIBER_QUEUE_SIZE,
    SUBSCRIBER_IDLE_KEEPALIVE_S,
    ChatRequest,
    EntityDenyRequest,
    FactRequest,
    MicRequest,
    NoteRequest,
    SatelliteSettingRequest,
    SpeakRequest,
    StopRequest,
    ThinkingTaskRequest,
    register_chat_routes,
)
from .routes_obsidian import register_obsidian_routes
from .routes_setup import (
    ConfigUpdateRequest,
    HaTestRequest,
    LlmModelsRequest,
    LlmSwitchRequest,
    LlmTestRequest,
    ModelsRequest,
    PathTestRequest,
    TimezoneRequest,
    VoiceRequest,
    VoiceSaveRequest,
    register_setup_routes,
    start_auto_download,
)
from .satellite_protocols import register_satellite_routes
from .tls_dispatcher import _pick_free_local_port, start_tls_dispatcher

# Preserve public imports while implementations belong to their route modules.
__all__ = [
    "AppContext", "Lifecycle", "create_app", "start_dashboard", "start_auto_download", "start_tls_dispatcher",
    "HISTORY_LIMIT", "PROACTIVE_REQUEST_LIMIT", "SSE_SUBSCRIBER_QUEUE_SIZE", "SUBSCRIBER_IDLE_KEEPALIVE_S",
    "ChatRequest", "EntityDenyRequest", "FactRequest", "MicRequest", "NoteRequest", "SatelliteSettingRequest",
    "SpeakRequest", "StopRequest", "ThinkingTaskRequest", "ConfigUpdateRequest", "HaTestRequest",
    "LlmModelsRequest", "LlmSwitchRequest", "LlmTestRequest", "ModelsRequest", "PathTestRequest",
    "TimezoneRequest", "VoiceRequest", "VoiceSaveRequest",
]

logger = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# The same session gate protects every HTTP route family. WebSocket protocols
# retain their own authentication checks.
_AUTH_EXEMPT_PATHS = frozenset({
    "/login", "/auth/login", "/auth/logout",
    "/logo.png", "/parloch.png", "/favicon.ico",
    "/ready",
})


def create_app(
    assistant=None,
    lifecycle: Optional[Lifecycle] = None,
    context: Optional[AppContext] = None,
) -> FastAPI:
    """Build the dashboard around a shared context or a legacy bare assistant.

    A supplied context can attach its assistant after first-run setup. Until it
    reaches READY, assistant-backed routes return 503 and the root serves setup.
    """
    if context is None:
        if lifecycle is None:
            lifecycle = Lifecycle(phase=READY)
        context = AppContext(lifecycle=lifecycle, assistant=assistant)
    if context.downloader is None:
        from .downloader import DownloadManager

        context.downloader = DownloadManager()

    lifecycle = context.lifecycle
    app = FastAPI(title="Fulloch Dashboard")
    register_satellite_routes(app, context, lifecycle)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), check_dir=False), name="static")

    def _require_ready() -> None:
        if context.assistant is None or not lifecycle.is_ready():
            raise HTTPException(
                status_code=503,
                detail="assistant not ready (setup or model load in progress)",
            )

    @app.middleware("http")
    async def _no_cache_static(request, call_next):
        """Revalidate static assets so protocol changes reach existing clients."""
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # Seed from credentials injected into the environment at startup. Password
    # setup updates this same context in place, making auth hot-applicable.
    if context.dashboard_password_hash is None:
        pw_hash = os.environ.get("DASHBOARD_PASSWORD", "").strip()
        if pw_hash:
            context.dashboard_password_hash = pw_hash
            logger.info("Dashboard password auth enabled")

    @app.middleware("http")
    async def _require_auth(request: Request, call_next):
        from .auth import SESSION_COOKIE

        path = request.url.path
        if path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        pw_hash = context.dashboard_password_hash
        if pw_hash:
            sid = request.cookies.get(SESSION_COOKIE, "")
            if sid and sid in context.sessions:
                return await call_next(request)
            upgrade = request.headers.get("upgrade", "").lower()
            accept = request.headers.get("accept", "")
            if upgrade != "websocket" and "text/html" in accept:
                return RedirectResponse("/login", status_code=303)
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    register_chat_routes(app, context, require_ready=_require_ready, static_dir=_STATIC_DIR)
    register_setup_routes(app, context, require_ready=_require_ready, static_dir=_STATIC_DIR)
    register_auth_routes(app, context)
    register_obsidian_routes(app, context)
    return app


def start_dashboard(
    assistant=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    lifecycle: Optional[Lifecycle] = None,
    context: Optional[AppContext] = None,
) -> threading.Thread:
    """Launch the dashboard on a daemon thread, returning without blocking.

    With an existing cert/key pair, uvicorn terminates TLS on an ephemeral
    localhost port and a same-port public dispatcher redirects HTTP or relays
    TLS. Missing or incomplete TLS configuration falls back to HTTP.
    """
    if (
        host not in ("127.0.0.1", "localhost", "::1")
        and not os.environ.get("DASHBOARD_PASSWORD", "").strip()
    ):
        logger.warning(
            "Dashboard bound to %s with no password set — notes, mic, speech, and "
            "Home Assistant control are exposed to your network. Set a password via "
            "the setup wizard, or bind dashboard_host to 127.0.0.1.",
            host,
        )

    ssl_kwargs = {}
    if ssl_certfile or ssl_keyfile:
        if not (ssl_certfile and ssl_keyfile):
            logger.warning(
                "Dashboard TLS needs BOTH dashboard_ssl_certfile and "
                "dashboard_ssl_keyfile — only one was set; serving over HTTP."
            )
        elif not Path(ssl_certfile).is_file() or not Path(ssl_keyfile).is_file():
            logger.warning(
                "Dashboard TLS cert/key not found (cert=%s, key=%s); serving over HTTP.",
                ssl_certfile,
                ssl_keyfile,
            )
        else:
            ssl_kwargs = {"ssl_certfile": ssl_certfile, "ssl_keyfile": ssl_keyfile}

    app = create_app(assistant, lifecycle=lifecycle, context=context)
    if ssl_kwargs:
        backend_port = _pick_free_local_port()
        uvicorn_config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=backend_port,
            log_level="warning",
            access_log=False,
            ws_ping_interval=None,
            **ssl_kwargs,
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
        uvicorn_thread = threading.Thread(target=uvicorn_server.run, daemon=True, name="dashboard-uvicorn")
        uvicorn_thread.start()
        try:
            start_tls_dispatcher(
                public_host=host,
                public_port=port,
                backend_host="127.0.0.1",
                backend_port=backend_port,
            )
        except OSError as e:
            logger.warning(
                "Could not start TLS dispatcher on %s:%s (%s). "
                "Users on http:// will see a TLS error; share the https:// URL directly.",
                host,
                port,
                e,
            )
        scheme = "https"
        logger.info(f"Dashboard listening on {scheme}://{host}:{port} (http→https redirect on the same port)")
        return uvicorn_thread

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        ws_ping_interval=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn")
    thread.start()
    logger.info(f"Dashboard listening on http://{host}:{port}")
    return thread
