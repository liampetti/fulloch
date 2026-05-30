"""FastAPI dashboard: text chat + live history view alongside the voice loop.

Wires into `core.assistant.Assistant` via `register_turn_listener` (for
SSE push) and `handle_text_turn` (for typed messages). The server runs
on a daemon thread under uvicorn; voice and text share `_chat_history`
and `_turn_lock` so the two inputs never race on the SLM.
"""

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

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


class ChatRequest(BaseModel):
    text: str


class FactRequest(BaseModel):
    text: str


class NoteRequest(BaseModel):
    content: str


def create_app(assistant) -> FastAPI:
    app = FastAPI(title="Fulloch Dashboard")

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

    @app.get("/history")
    def get_history() -> JSONResponse:
        with history_lock:
            return JSONResponse(list(history_log))

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
    assistant, host: str = "127.0.0.1", port: int = 8765
) -> threading.Thread:
    """Launch the dashboard on a daemon thread. Non-blocking."""
    app = create_app(assistant)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn")
    thread.start()
    logger.info(f"Dashboard listening on http://{host}:{port}")
    return thread
