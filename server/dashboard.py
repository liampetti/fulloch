"""FastAPI dashboard: text chat + live history view alongside the voice loop.

Wires into `core.assistant.Assistant` via `register_turn_listener` (for
SSE push) and `handle_text_turn` (for typed messages). The server runs
on a daemon thread under uvicorn; voice and text share `_chat_history`
and `_turn_lock` so the two inputs never race on the SLM.
"""

import asyncio
import json
import logging
import os
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SERVER_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _SERVER_DIR / "static"
_LOGO_PATH = _SERVER_DIR.parent / "fulloch.png"

HISTORY_LIMIT = 200
SUBSCRIBER_IDLE_KEEPALIVE_S = 15

# Optional bearer-token gate. When FULLOCH_DASHBOARD_TOKEN is set, every route
# except the unauthenticated shell (the HTML page + its logo) requires the token
# — supplied either as `Authorization: Bearer <token>` or, for EventSource which
# can't set headers, a `?token=<token>` query param. Unset = no auth (preserves
# the zero-config local-only experience); we warn loudly if that's paired with a
# non-loopback bind. See README "Exposing the dashboard".
_AUTH_EXEMPT_PATHS = frozenset({"/", "/logo.png", "/favicon.ico"})


class ChatRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


class MicRequest(BaseModel):
    enabled: bool


class FactRequest(BaseModel):
    text: str


class NoteRequest(BaseModel):
    content: str


class EntityDenyRequest(BaseModel):
    entity_id: str
    denied: bool


def create_app(assistant) -> FastAPI:
    app = FastAPI(title="Fulloch Dashboard")

    token = os.environ.get("FULLOCH_DASHBOARD_TOKEN", "").strip()
    if token:
        @app.middleware("http")
        async def _require_token(request: Request, call_next):
            if request.url.path not in _AUTH_EXEMPT_PATHS:
                header = request.headers.get("authorization", "")
                supplied = (
                    header[7:].strip()
                    if header.lower().startswith("bearer ")
                    else request.query_params.get("token", "").strip()
                )
                if not secrets.compare_digest(supplied, token):
                    return JSONResponse(
                        {"detail": "unauthorized"}, status_code=401
                    )
            return await call_next(request)
        logger.info("Dashboard bearer-token auth enabled")

    history_log: list = []
    history_lock = threading.Lock()
    subscribers: list[queue.Queue] = []
    subscribers_lock = threading.Lock()

    def on_turn(event: dict) -> None:
        with history_lock:
            history_log.append(event)
            if len(history_log) > HISTORY_LIMIT:
                del history_log[: len(history_log) - HISTORY_LIMIT]
        with subscribers_lock:
            dead = []
            for q in subscribers:
                try:
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
            for q in dead:
                subscribers.remove(q)

    assistant.register_turn_listener(on_turn)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/logo.png")
    def logo() -> FileResponse:
        return FileResponse(_LOGO_PATH, media_type="image/png")

    @app.get("/config")
    def get_config() -> JSONResponse:
        wakeword = getattr(assistant, "wakeword", "") or ""
        return JSONResponse({"wakeword": wakeword.title()})

    @app.get("/history")
    def get_history() -> JSONResponse:
        with history_lock:
            return JSONResponse(list(history_log))

    @app.post("/reset")
    def reset_chat() -> dict:
        with history_lock:
            history_log.clear()
        assistant._history.clear()
        on_turn({"role": "reset", "ts": time.time()})
        return {"ok": True}

    @app.get("/status")
    def get_status() -> JSONResponse:
        state = assistant.get_state()
        last_utterance = ""
        last_response = ""
        with history_lock:
            for event in reversed(history_log):
                role = event.get("role")
                if role == "user" and not last_utterance:
                    last_utterance = event.get("content", "")
                elif role == "assistant" and not last_response:
                    last_response = event.get("content", "")
                if last_utterance and last_response:
                    break
        return JSONResponse({
            "state": state,
            "mic_enabled": assistant.audio_capture.transcribing,
            "last_utterance": last_utterance,
            "last_response": last_response,
        })

    @app.post("/speak")
    def speak(req: SpeakRequest) -> dict:
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        threading.Thread(
            target=assistant.speak_proactive, args=(text,), daemon=True
        ).start()
        return {"ok": True}

    @app.post("/mic")
    def set_mic(req: MicRequest) -> dict:
        assistant.audio_capture.transcribing = req.enabled
        return {"ok": True, "mic_enabled": req.enabled}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        answer = assistant.handle_text_turn(req.text)
        return {"answer": answer}

    @app.get("/facts")
    def facts_list() -> JSONResponse:
        from tools.notes import list_facts
        return JSONResponse({"facts": list_facts()})

    @app.post("/facts")
    def facts_add(req: FactRequest) -> JSONResponse:
        from tools.notes import list_facts, remember_fact
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        remember_fact(text)
        return JSONResponse({"facts": list_facts()})

    @app.put("/facts/{idx}")
    def facts_update(idx: int, req: FactRequest) -> JSONResponse:
        from tools.notes import list_facts, update_fact
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        if not update_fact(idx, text):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.delete("/facts/{idx}")
    def facts_delete(idx: int) -> JSONResponse:
        from tools.notes import delete_fact, list_facts
        if not delete_fact(idx):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.get("/notes")
    def notes_list() -> JSONResponse:
        from tools.notes import list_note_files
        return JSONResponse({"notes": list_note_files()})

    @app.get("/notes/{name:path}")
    def notes_read(name: str) -> JSONResponse:
        from tools.notes import read_note_file
        content = read_note_file(name)
        if content is None:
            raise HTTPException(status_code=404, detail="note not found")
        return JSONResponse({"name": name, "content": content})

    @app.put("/notes/{name:path}")
    def notes_save(name: str, req: NoteRequest) -> JSONResponse:
        from tools.notes import read_note_file, save_note_file
        if not save_note_file(name, req.content):
            raise HTTPException(status_code=404, detail="note not found")
        return JSONResponse({"name": name, "content": read_note_file(name)})

    @app.get("/entities")
    def entities_list() -> JSONResponse:
        from tools._config import config
        if "home_assistant" not in config:
            return JSONResponse({"available": False, "entities": []})
        from tools import home_assistant as ha
        return JSONResponse({"available": True, "entities": ha.list_entities()})

    @app.post("/entities")
    def entities_set(req: EntityDenyRequest) -> JSONResponse:
        from tools._config import config
        if "home_assistant" not in config:
            raise HTTPException(status_code=404, detail="Home Assistant not configured")
        from tools import home_assistant as ha
        entity_id = (req.entity_id or "").strip()
        if not entity_id:
            raise HTTPException(status_code=400, detail="empty entity_id")
        ha.set_entity_denied(entity_id, req.denied)
        return JSONResponse({"available": True, "entities": ha.list_entities()})

    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        q: queue.Queue = queue.Queue()
        with subscribers_lock:
            subscribers.append(q)

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(
                            q.get, True, SUBSCRIBER_IDLE_KEEPALIVE_S
                        )
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def start_dashboard(
    assistant,
    host: str = "127.0.0.1",
    port: int = 8765,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
) -> threading.Thread:
    """Launch the dashboard on a daemon thread. Non-blocking.

    Pass both ``ssl_certfile`` and ``ssl_keyfile`` to serve over HTTPS (uvicorn
    terminates TLS). If only one is given, or a file is missing, TLS is skipped
    and the dashboard falls back to HTTP with a loud warning.
    """
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get(
        "FULLOCH_DASHBOARD_TOKEN", "").strip():
        logger.warning(
            "Dashboard bound to %s with NO auth token — notes, mic, speech, and "
            "Home Assistant control are exposed to your whole network. Set "
            "FULLOCH_DASHBOARD_TOKEN in .env or bind dashboard_host to 127.0.0.1.",
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
                "Dashboard TLS cert/key not found (cert=%s, key=%s); serving "
                "over HTTP.", ssl_certfile, ssl_keyfile,
            )
        else:
            ssl_kwargs = {"ssl_certfile": ssl_certfile, "ssl_keyfile": ssl_keyfile}

    app = create_app(assistant)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False,
        **ssl_kwargs,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn")
    thread.start()
    scheme = "https" if ssl_kwargs else "http"
    logger.info(f"Dashboard listening on {scheme}://{host}:{port}")
    return thread
