"""Dashboard chat/status, history/SSE and assistant-backed management routes."""

import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config_store import read_config
from .lifecycle import LOG_BUFFER, AppContext

_SERVER_DIR = Path(__file__).resolve().parent
_LOGO_PATH = _SERVER_DIR.parent / "fulloch.png"
_PARLOCH_PATH = _SERVER_DIR.parent / "parloch.png"
_SERVER_INSTANCE_ID = uuid.uuid4().hex
HISTORY_LIMIT = 200
SUBSCRIBER_IDLE_KEEPALIVE_S = 15
SSE_SUBSCRIBER_QUEUE_SIZE = 100
PROACTIVE_REQUEST_LIMIT = 2

# /status is polled frequently. Cache YAML parsing by path and mtime; edits
# invalidate automatically, including changes made by the settings routes.
_CONFIG_CACHE: dict = {"path": None, "mtime": None, "data": None}


def _read_config_cached(path: str) -> dict:
    """Return the cached config when its path and file mtime are unchanged."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return read_config(path)
    if (
        _CONFIG_CACHE["path"] == path
        and _CONFIG_CACHE["mtime"] == mtime
        and _CONFIG_CACHE["data"] is not None
    ):
        return _CONFIG_CACHE["data"]
    data = read_config(path)
    _CONFIG_CACHE["path"] = path
    _CONFIG_CACHE["mtime"] = mtime
    _CONFIG_CACHE["data"] = data
    return data


class SatelliteSettingRequest(BaseModel):
    enabled: bool


class ChatRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


class MicRequest(BaseModel):
    enabled: bool


class StopRequest(BaseModel):
    # Defaults to the active turn owner when omitted.
    satellite_id: Optional[str] = None


class FactRequest(BaseModel):
    text: str


class NoteRequest(BaseModel):
    content: str


class EntityDenyRequest(BaseModel):
    entity_id: str
    denied: bool


class ThinkingTaskRequest(BaseModel):
    task: str


def register_chat_routes(
    app: FastAPI,
    context: AppContext,
    *,
    require_ready: Callable[[], None],
    static_dir: Path,
) -> None:
    """Own per-app history, subscribers and proactive slots; attach via context.

    The context remains live so first-run setup can attach the assistant after
    registration. Locks and queues belong to this app, never to the module.
    """
    lifecycle = context.lifecycle
    history_log: list = []
    history_lock = threading.Lock()
    subscribers: list[queue.Queue] = []
    subscribers_lock = threading.Lock()
    proactive_slots = threading.BoundedSemaphore(PROACTIVE_REQUEST_LIMIT)
    startup_greeting_seeded = False

    def on_turn(event: dict) -> None:
        # Native lifecycle events aren't dashboard chat messages.
        if event.get("type") == "assistant.state":
            return
        with history_lock:
            history_log.append(event)
            if len(history_log) > HISTORY_LIMIT:
                del history_log[: len(history_log) - HISTORY_LIMIT]
        with subscribers_lock:
            dead = []
            for q in subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
                except Exception:
                    dead.append(q)
            for q in dead:
                subscribers.remove(q)

    def _start_proactive(*args, **kwargs) -> bool:
        """Bound pending speech work rather than growing request threads."""
        if not proactive_slots.acquire(blocking=False):
            return False

        def run() -> None:
            try:
                context.assistant.speak_proactive(*args, **kwargs)
            finally:
                proactive_slots.release()

        threading.Thread(target=run, daemon=True, name="dashboard-proactive").start()
        return True

    def _seed_startup_greeting(assistant) -> None:
        nonlocal startup_greeting_seeded
        greeting = getattr(assistant, "greeting_text", "").strip() if assistant else ""
        if not greeting:
            return
        with history_lock:
            if startup_greeting_seeded:
                return
            startup_greeting_seeded = True
        on_turn({"role": "assistant", "content": greeting, "ts": time.time(), "source": "startup"})

    def _attach_turn_listener(assistant) -> None:
        assistant.register_turn_listener(on_turn)
        _seed_startup_greeting(assistant)

    context.on_attach(_attach_turn_listener)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Keep users on setup/loading until the models and greeting are ready.
        ready = context.assistant is not None and lifecycle.is_ready()
        page = "index.html" if ready else "setup.html"
        html = (static_dir / page).read_text(encoding="utf-8")
        if ready:
            general = read_config(context.config_path).get("general") or {}
            prefs = {
                "theme": general.get("dashboard_theme", "auto"),
                "show_turn_details": general.get("dashboard_show_turn_details", False),
            }
            html = html.replace(
                "<script>\n  // Apply the configured appearance",
                f"<script>window.FULLOCH_DASHBOARD_PREFS = {json.dumps(prefs)};</script>\n<script>\n  // Apply the configured appearance",
            )
        return html

    def _is_remote_llm() -> bool:
        a = context.assistant
        return a is not None and getattr(a, "llm_backend", None) == "openai"

    @app.get("/logo.png")
    def logo(request: Request) -> FileResponse:
        # Wizard previews can force branding before the backend is running.
        q = request.query_params.get("remote")
        remote = (q == "1") if q is not None else _is_remote_llm()
        path = _PARLOCH_PATH if remote and _PARLOCH_PATH.is_file() else _LOGO_PATH
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/config")
    def get_config() -> JSONResponse:
        wakeword = getattr(context.assistant, "wakeword", "") or ""
        return JSONResponse({"wakeword": wakeword.title()})

    @app.get("/history")
    def get_history() -> JSONResponse:
        _seed_startup_greeting(context.assistant)
        with history_lock:
            return JSONResponse(list(history_log))

    @app.get("/reports/{note_id:path}")
    def get_report(note_id: str) -> Response:
        """Serve reports only from the dedicated reports note directory."""
        if not re.fullmatch(r"fulloch-reports/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", note_id):
            raise HTTPException(status_code=404, detail="report not found")
        from tools.notes_root import get_notes_root

        root = get_notes_root().resolve()
        path = (root / f"{note_id}.md").resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(status_code=404, detail="report not found")
        return Response(path.read_text(encoding="utf-8"), media_type="text/markdown")

    @app.post("/reset")
    def reset_chat() -> dict:
        nonlocal startup_greeting_seeded
        require_ready()
        with history_lock:
            history_log.clear()
            # An explicit reset suppresses later reinsertion of the greeting.
            startup_greeting_seeded = True
        context.assistant._history.clear()
        on_turn({"role": "reset", "ts": time.time()})
        return {"ok": True}

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Liveness probe: serving setup/loading is healthy even before READY."""
        return JSONResponse({"ready": True})

    @app.get("/status")
    def get_status(request: Request) -> JSONResponse:
        payload = lifecycle.snapshot()
        payload["server_instance_id"] = _SERVER_INSTANCE_ID
        payload["auth_enabled"] = bool(context.dashboard_password_hash)
        cfg = _read_config_cached(context.config_path)
        general = cfg.get("general") or {}
        cert = general.get("dashboard_ssl_certfile")
        key = general.get("dashboard_ssl_keyfile")
        if cert and key and Path(cert).is_file() and Path(key).is_file():
            payload["dashboard_url"] = f"{request.url.scheme}://{request.url.netloc}"
        if context.downloader is not None and context.downloader.active:
            payload["download"] = context.downloader.snapshot()
        if context.assistant is None or not lifecycle.is_ready():
            payload.update({
                "state": "idle", "mic_enabled": False, "last_utterance": "", "last_response": "",
                "log": LOG_BUFFER.tail(),
            })
            return JSONResponse(payload)
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
        owner_id = context.assistant._turn_arbiter.owner
        owner_sat = context.assistant.satellites.get(owner_id) if owner_id else None
        payload.update({
            "state": context.assistant.get_state(),
            "mic_enabled": context.assistant.audio_capture.mic_globally_enabled,
            "last_utterance": last_utterance,
            "last_response": last_response,
            "remote_llm": _is_remote_llm(),
            "llm_unreachable": _is_remote_llm()
            and bool(getattr(context.assistant, "remote_llm_unreachable", False)),
            "satellite_count": sum(1 for sid in context.assistant.satellites if sid != "dashboard-text"),
            "active_owner_id": owner_id,
            "active_owner_label": (owner_sat.label or owner_sat.ha_area_name) if owner_sat is not None else None,
            "last_turn_stats": context.assistant._last_turn_stats,
            "wakeword": {
                **getattr(context.assistant, "wakeword_metrics", {"status": "asr"}),
                **getattr(context.assistant.audio_capture, "wakeword_metrics", {}),
            },
            "asr_queue": (
                metrics
                if isinstance(metrics := getattr(context.assistant.audio_capture, "asr_queue_metrics", {}), dict)
                else {}
            ),
            "thinking_job": context.assistant.active_thinking_task(),
        })
        return JSONResponse(payload)

    @app.post("/speak")
    def speak(req: SpeakRequest) -> dict:
        require_ready()
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        if not _start_proactive(text):
            raise HTTPException(status_code=429, detail="too many pending speech requests")
        return {"ok": True}

    @app.post("/replay")
    def replay(req: SpeakRequest) -> dict:
        require_ready()
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        if not _start_proactive(text=text, emit_event=False):
            raise HTTPException(status_code=429, detail="too many pending speech requests")
        return {"ok": True}

    @app.post("/mic")
    def set_mic(req: MicRequest) -> dict:
        # HA's mic switch is global, unlike per-satellite half-duplex muting.
        require_ready()
        context.assistant.audio_capture.mic_globally_enabled = req.enabled
        return {"ok": True, "mic_enabled": req.enabled}

    @app.post("/thinking/run")
    def run_thinking_task(req: ThinkingTaskRequest) -> dict:
        require_ready()
        try:
            return context.assistant.run_thinking_task(req.task)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/thinking/{job_id}")
    def thinking_task_status(job_id: str) -> dict:
        require_ready()
        status = context.assistant.thinking_task_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="thinking job not found")
        return status

    @app.post("/thinking/{job_id}/cancel")
    def cancel_thinking_task(job_id: str) -> dict:
        require_ready()
        if not context.assistant.cancel_thinking_task(job_id):
            raise HTTPException(status_code=404, detail="thinking job not found or already finished")
        status = context.assistant.thinking_task_status(job_id)
        return {"ok": True, "job": status}

    @app.post("/stop")
    def stop_turn(req: StopRequest = StopRequest()) -> dict:  # noqa: B008
        # No body defaults to the current turn owner, for silent stand-down.
        require_ready()
        context.assistant.request_stop(req.satellite_id)
        return {"ok": True}

    @app.get("/satellites")
    def satellites() -> dict:
        require_ready()
        owner_id = context.assistant.conversation_owner_id
        items = []
        for satellite_id, session in context.assistant.satellites.items():
            if satellite_id == "dashboard-text":
                continue
            transport = "native" if session.device_id else "browser"
            state = "speaking" if session.tts_active.is_set() else "thinking" if session.active_session else "idle"
            items.append({
                "id": satellite_id,
                "label": session.label or session.ha_area_name or "This browser",
                "area": session.ha_area_name or "",
                "transport": transport,
                "device_id": session.device_id or "",
                "conversation_mode": session.conversation_mode,
                "conversation_owner": satellite_id == owner_id,
                "muted": session.user_muted,
                "state": state,
                "server_vad": session.server_vad,
            })
        items.sort(key=lambda item: (not item["conversation_owner"], item["label"].lower()))
        return {"satellites": items, "conversation_owner_id": owner_id}

    @app.post("/satellites/{satellite_id}/conversation-mode")
    def set_satellite_conversation_mode(satellite_id: str, req: SatelliteSettingRequest) -> dict:
        require_ready()
        ok, message = context.assistant.set_satellite_conversation_mode(satellite_id, req.enabled)
        session = context.assistant.satellites.get(satellite_id)
        active = bool(session and session.conversation_mode)
        context.assistant.send_satellite_control(satellite_id, {
            "type": "conversation_mode.changed",
            "enabled": active,
            "owner": active and context.assistant.conversation_owner_id == satellite_id,
            "message": message,
        })
        return {"ok": ok, "enabled": active, "message": message}

    @app.post("/satellites/{satellite_id}/mute")
    def set_satellite_mute(satellite_id: str, req: SatelliteSettingRequest) -> dict:
        require_ready()
        ok, message = context.assistant.set_satellite_user_muted(satellite_id, req.enabled)
        return {"ok": ok, "muted": req.enabled if ok else False, "message": message}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        require_ready()
        answer = context.assistant.handle_text_turn(req.text)
        return {"answer": answer}

    @app.get("/media-artwork/{entity_id}")
    def media_artwork(entity_id: str) -> Response:
        """Proxy a player's current artwork through the authenticated dashboard."""
        require_ready()
        if not re.fullmatch(r"media_player\.[A-Za-z0-9_]+", entity_id):
            raise HTTPException(status_code=404, detail="media player not found")
        import tools.ha_client as ha

        if not ha.HA_TOKEN or entity_id in ha._DENIED_ENTITIES:
            raise HTTPException(status_code=404, detail="media player not found")
        try:
            response = requests.get(
                f"{ha.HA_URL}/api/media_player_proxy/{entity_id}", headers=ha._get_headers(), timeout=ha.TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException:
            raise HTTPException(status_code=404, detail="artwork unavailable") from None
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if not content_type.startswith("image/") or len(response.content) > 5_000_000:
            raise HTTPException(status_code=404, detail="artwork unavailable")
        return Response(content=response.content, media_type=content_type, headers={"Cache-Control": "no-cache"})

    @app.get("/facts")
    def facts_list() -> JSONResponse:
        require_ready()
        from tools.notes import list_facts

        return JSONResponse({"facts": list_facts()})

    @app.post("/facts")
    def facts_add(req: FactRequest) -> JSONResponse:
        require_ready()
        from tools.notes import list_facts, remember_fact

        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        remember_fact(text)
        return JSONResponse({"facts": list_facts()})

    @app.put("/facts/{idx}")
    def facts_update(idx: int, req: FactRequest) -> JSONResponse:
        require_ready()
        from tools.notes import list_facts, update_fact

        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        if not update_fact(idx, text):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.delete("/facts/{idx}")
    def facts_delete(idx: int) -> JSONResponse:
        require_ready()
        from tools.notes import delete_fact, list_facts

        if not delete_fact(idx):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.get("/entities")
    def entities_list() -> JSONResponse:
        require_ready()
        from tools._config import config

        if "home_assistant" not in config:
            return JSONResponse({"available": False, "entities": []})
        from tools import home_assistant as ha

        return JSONResponse({"available": True, "entities": ha.list_entities()})

    @app.post("/entities")
    def entities_set(req: EntityDenyRequest) -> JSONResponse:
        require_ready()
        from tools._config import config

        if "home_assistant" not in config:
            raise HTTPException(status_code=404, detail="Home Assistant not configured")
        from tools import home_assistant as ha

        entity_id = (req.entity_id or "").strip()
        if not entity_id:
            raise HTTPException(status_code=400, detail="empty entity_id")
        ha.set_entity_denied(entity_id, req.denied)
        return JSONResponse({"available": True, "entities": ha.list_entities()})

    @app.get("/ha/areas")
    def ha_areas() -> JSONResponse:
        require_ready()
        from tools._config import config

        if "home_assistant" not in config:
            return JSONResponse({"available": False, "areas": []})
        from tools import home_assistant as ha

        return JSONResponse({"available": True, "areas": ha.list_areas()})

    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        q: queue.Queue = queue.Queue(maxsize=SSE_SUBSCRIBER_QUEUE_SIZE)
        with subscribers_lock:
            subscribers.append(q)

        async def gen():
            try:
                next_keepalive = time.monotonic() + SUBSCRIBER_IDLE_KEEPALIVE_S
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = q.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if now >= next_keepalive:
                            yield ": keepalive\n\n"
                            next_keepalive = now + SUBSCRIBER_IDLE_KEEPALIVE_S
                        await asyncio.sleep(0.1)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream")
