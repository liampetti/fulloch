"""Token-authenticated HTTP API for the Home Assistant integration only."""

import asyncio
import json
import logging
import queue
import secrets
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .credentials_store import get_credential
from .lifecycle import AppContext

logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15
_SSE_SUBSCRIBER_QUEUE_SIZE = 100
_PROACTIVE_REQUEST_LIMIT = 2


class _TextRequest(BaseModel):
    text: str


class _MicRequest(BaseModel):
    enabled: bool


class _ThinkingRequest(BaseModel):
    task: str


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
    proactive_slots = threading.BoundedSemaphore(_PROACTIVE_REQUEST_LIMIT)
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
                except queue.Full:
                    dead.append(subscriber)
                except Exception:
                    dead.append(subscriber)
            for subscriber in dead:
                subscribers.remove(subscriber)

    def start_proactive(text: str) -> bool:
        if not proactive_slots.acquire(blocking=False):
            return False

        def run() -> None:
            try:
                context.assistant.speak_proactive(text)
            finally:
                proactive_slots.release()

        threading.Thread(target=run, daemon=True, name="integration-proactive").start()
        return True

    context.on_attach(lambda assistant: assistant.register_turn_listener(on_turn))

    def require_ready() -> None:
        if context.assistant is None or not context.lifecycle.is_ready():
            raise HTTPException(status_code=503, detail="assistant not ready (setup or model load in progress)")

    def thinking_assistant():
        require_ready()
        if not getattr(context.assistant, "thinking_enabled", False):
            raise HTTPException(status_code=409, detail="deliberate thinking is disabled")
        return context.assistant

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
                    "thinking_job": context.assistant.active_thinking_task(),
                })
        return JSONResponse(payload)

    @app.post("/speak")
    def speak(req: _TextRequest) -> dict:
        require_ready()
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        if not start_proactive(text):
            raise HTTPException(status_code=429, detail="too many pending speech requests")
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

    @app.post("/thinking/run")
    def run_thinking(req: _ThinkingRequest) -> dict:
        try:
            job = thinking_assistant().run_thinking_task(req.task)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job_id": job["id"], "status": job["status"]}

    @app.get("/thinking/{job_id}")
    def thinking_status(job_id: str) -> dict:
        job = thinking_assistant().thinking_task_status(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="thinking job not found")
        return job

    @app.post("/thinking/{job_id}/cancel")
    def cancel_thinking(job_id: str) -> dict:
        if not thinking_assistant().cancel_thinking_task(job_id):
            raise HTTPException(status_code=404, detail="thinking job cannot be cancelled")
        return {"ok": True}

    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        subscriber: queue.Queue = queue.Queue(maxsize=_SSE_SUBSCRIBER_QUEUE_SIZE)
        with subscribers_lock:
            subscribers.append(subscriber)

        async def events():
            try:
                next_keepalive = time.monotonic() + _KEEPALIVE_SECONDS
                while not await request.is_disconnected():
                    try:
                        event = subscriber.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if now >= next_keepalive:
                            yield ": keepalive\n\n"
                            next_keepalive = now + _KEEPALIVE_SECONDS
                        await asyncio.sleep(0.1)
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
