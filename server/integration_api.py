"""Token-authenticated HTTP API for the Home Assistant integration only."""

import asyncio
import json
import logging
import queue
import secrets
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .credentials_store import get_credential
from .lifecycle import AppContext

logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15


class _TextRequest(BaseModel):
    text: str


class _MicRequest(BaseModel):
    enabled: bool


def _tokens() -> list[str]:
    """Return only explicitly configured integration tokens, never satellite tokens."""
    value = get_credential("integration_tokens")
    if isinstance(value, list):
        return [str(token).strip() for token in value if str(token).strip()]
    return []


def create_integration_app(context: AppContext) -> FastAPI:
    """Build the deliberately small API consumed by the HACS integration."""
    app = FastAPI(title="Fulloch Integration API", docs_url=None, redoc_url=None, openapi_url=None)
    subscribers: list[queue.Queue] = []
    subscribers_lock = threading.Lock()
    last_utterance = ""
    last_response = ""
    history_lock = threading.Lock()

    @app.middleware("http")
    async def require_integration_token(request: Request, call_next):
        authorization = request.headers.get("authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        tokens = _tokens()
        if scheme.lower() != "bearer" or not supplied or not any(
            secrets.compare_digest(supplied, token) for token in tokens
        ):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    def on_turn(event: dict) -> None:
        nonlocal last_utterance, last_response
        if event.get("type") == "assistant.state":
            return
        role = event.get("role")
        content = event.get("content", "")
        with history_lock:
            if role == "user":
                last_utterance = content
            elif role == "assistant":
                last_response = content
        with subscribers_lock:
            dead = []
            for subscriber in subscribers:
                try:
                    subscriber.put_nowait(event)
                except Exception:
                    dead.append(subscriber)
            for subscriber in dead:
                subscribers.remove(subscriber)

    context.on_attach(lambda assistant: assistant.register_turn_listener(on_turn))

    def require_ready() -> None:
        if context.assistant is None or not context.lifecycle.is_ready():
            raise HTTPException(status_code=503, detail="assistant not ready (setup or model load in progress)")

    @app.get("/status")
    def status() -> JSONResponse:
        payload = context.lifecycle.snapshot()
        if context.assistant is None or not context.lifecycle.is_ready():
            payload.update({"state": "idle", "mic_enabled": False, "last_utterance": "", "last_response": ""})
        else:
            with history_lock:
                payload.update({
                    "state": context.assistant.get_state(),
                    "mic_enabled": context.assistant.audio_capture.mic_globally_enabled,
                    "last_utterance": last_utterance,
                    "last_response": last_response,
                })
        return JSONResponse(payload)

    @app.post("/speak")
    def speak(req: _TextRequest) -> dict:
        require_ready()
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        threading.Thread(target=context.assistant.speak_proactive, args=(text,), daemon=True).start()
        return {"ok": True}

    @app.post("/chat")
    def chat(req: _TextRequest) -> dict:
        require_ready()
        return {"answer": context.assistant.handle_text_turn(req.text)}

    @app.post("/mic")
    def mic(req: _MicRequest) -> dict:
        require_ready()
        context.assistant.audio_capture.mic_globally_enabled = req.enabled
        return {"ok": True, "mic_enabled": req.enabled}

    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        subscriber: queue.Queue = queue.Queue()
        with subscribers_lock:
            subscribers.append(subscriber)

        async def events():
            try:
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.to_thread(subscriber.get, True, _KEEPALIVE_SECONDS)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                    else:
                        yield f"data: {json.dumps(event)}\n\n"
            finally:
                with subscribers_lock:
                    if subscriber in subscribers:
                        subscribers.remove(subscriber)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def start_integration_api(context: AppContext, host: str, port: int) -> threading.Thread:
    """Launch the plain-HTTP integration API on a daemon thread."""
    server = uvicorn.Server(
        uvicorn.Config(
            create_integration_app(context), host=host, port=port, log_level="warning", access_log=False
        )
    )
    thread = threading.Thread(target=server.run, daemon=True, name="integration-api-uvicorn")
    thread.start()
    logger.info("Integration API listening on http://%s:%s", host, port)
    return thread
