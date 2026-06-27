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
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from .lifecycle import (
    DOWNLOADING,
    ERROR,
    LOADING,
    LOG_BUFFER,
    READY,
    AppContext,
    Lifecycle,
)

logger = logging.getLogger(__name__)

_SERVER_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _SERVER_DIR / "static"
_LOGO_PATH = _SERVER_DIR.parent / "fulloch.png"
# Travelling-character image — shown instead of the default Fulloch logo when the
# language model runs off-device over a remote OpenAI-compatible endpoint.
_PARLOCH_PATH = _SERVER_DIR.parent / "parloch.png"
_VOICES_DIR = _SERVER_DIR.parent / "data" / "voices"

HISTORY_LIMIT = 200


def _schedule_restart(delay: float = 0.5) -> None:
    """Re-exec the process shortly, so a fresh start re-reads config.yml.

    Used by the dashboard Restart button and by the setup wizard when it
    reconfigures an *already-running* assistant (e.g. recovering from an ERROR
    where the first config failed to load) — main() is no longer waiting on the
    setup `proceed` event in that case, so a re-exec is the only way to apply the
    new backends. Falls back to a hard exit (compose `restart: unless-stopped`
    brings it back). Scheduled on a daemon thread so the caller can return its
    HTTP response first.
    """
    def _go():
        time.sleep(delay)
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception:  # noqa: BLE001 — last resort: exit, let the orchestrator restart
            os._exit(0)

    threading.Thread(target=_go, daemon=True, name="dashboard-restart").start()
SUBSCRIBER_IDLE_KEEPALIVE_S = 15

# Optional bearer-token gate. When FULLOCH_DASHBOARD_TOKEN is set, every route
# except the unauthenticated shell (the HTML page + its logo) requires the token
# — supplied either as `Authorization: Bearer <token>` or, for EventSource which
# can't set headers, a `?token=<token>` query param. Unset = no auth (preserves
# the zero-config local-only experience); we warn loudly if that's paired with a
# non-loopback bind. See README "Exposing the dashboard".
_AUTH_EXEMPT_PATHS = frozenset({"/", "/setup", "/logo.png", "/favicon.ico"})


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


class ConfigUpdateRequest(BaseModel):
    updates: dict


class ModelsRequest(BaseModel):
    tier: Optional[str] = None
    models: Optional[dict] = None


class VoiceRequest(BaseModel):
    instruct: str
    phrase: Optional[str] = None


class VoiceSaveRequest(BaseModel):
    name: str


class LlmTestRequest(BaseModel):
    base_url: str
    model: str
    api_key: Optional[str] = None


class LlmModelsRequest(BaseModel):
    base_url: str
    api_key: Optional[str] = None


class LlmSwitchRequest(BaseModel):
    model: str


def create_app(
    assistant=None,
    lifecycle: Optional[Lifecycle] = None,
    context: Optional[AppContext] = None,
) -> FastAPI:
    """Build the dashboard app.

    Pass a shared `context` (preferred) so the assistant can be attached after
    first-run setup without a second server. For backwards compatibility a bare
    `assistant` (+ optional `lifecycle`) is wrapped in a static context; when
    both are omitted the context is a setup-mode shell (no assistant). During
    setup, the app serves the setup page and assistant-backed routes return 503
    until the lifecycle reaches READY.
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

    def _require_ready() -> None:
        """Guard assistant-backed routes during setup / model load."""
        if context.assistant is None or not lifecycle.is_ready():
            raise HTTPException(
                status_code=503,
                detail="assistant not ready (setup or model load in progress)",
            )

    def _reset_marker_path() -> Path:
        """The setup-reset marker, alongside config.yml (see core/setup.py)."""
        return Path(context.config_path).parent / ".setup_pending"

    # Seed the live token from the env (preserves the env-configured behaviour);
    # the post-setup token step updates context.dashboard_token in place so the
    # console is gated immediately, no restart needed.
    if context.dashboard_token is None:
        context.dashboard_token = os.environ.get("FULLOCH_DASHBOARD_TOKEN", "").strip()
    if context.dashboard_token:
        logger.info("Dashboard bearer-token auth enabled")

    # Always installed; reads the live token each request and no-ops when unset
    # (so a token generated post-setup gates without re-creating the app).
    @app.middleware("http")
    async def _require_token(request: Request, call_next):
        token = context.dashboard_token
        if token and request.url.path not in _AUTH_EXEMPT_PATHS:
            header = request.headers.get("authorization", "")
            supplied = (
                header[7:].strip()
                if header.lower().startswith("bearer ")
                else request.query_params.get("token", "").strip()
            )
            if not secrets.compare_digest(supplied, token):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

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

    # Register the SSE turn listener as soon as the assistant exists — now if
    # already attached, or later via context.set_assistant after setup.
    context.on_attach(lambda a: a.register_turn_listener(on_turn))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # The dashboard (index.html) is served only once the assistant is fully
        # up — models loaded AND the opening greeting finished, i.e. lifecycle
        # READY. Until then (setup needed, downloading, loading, or a fatal
        # error) we serve setup.html, which routes itself to the wizard /
        # download progress / loading screen by phase. This keeps the user out
        # of the dashboard until it's actually usable; on READY the loading
        # screen redirects here and gets the dashboard.
        ready = context.assistant is not None and lifecycle.is_ready()
        page = "index.html" if ready else "setup.html"
        return (_STATIC_DIR / page).read_text(encoding="utf-8")

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page() -> str:
        return (_STATIC_DIR / "setup.html").read_text(encoding="utf-8")

    def _is_remote_llm() -> bool:
        """True when the language model runs off-device (a remote OpenAI endpoint)."""
        a = context.assistant
        return a is not None and getattr(a, "llm_backend", None) == "openai"

    @app.get("/logo.png")
    def logo(request: Request) -> FileResponse:
        # Swap to the travelling character when the LLM is remote. One endpoint backs
        # both the dashboard avatar and the favicon. no-cache so the browser
        # revalidates after a backend change + reload. `?remote=1`/`0` forces the
        # swap regardless of running state — the wizard/settings use it to preview
        # the character as the user selects an OpenAI endpoint, before any restart.
        q = request.query_params.get("remote")
        remote = (q == "1") if q is not None else _is_remote_llm()
        path = (_PARLOCH_PATH if remote and _PARLOCH_PATH.is_file()
                else _LOGO_PATH)
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/voice/sample")
    def voice_sample(name: str) -> FileResponse:
        """Preview clip for a selectable voice. Both Kokoro built-ins and Qwen
        clones have a data/voices/<name>.wav (bundled sample / clone reference)."""
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise HTTPException(status_code=400, detail="invalid voice name")
        path = _VOICES_DIR / f"{name}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no sample for that voice")
        return FileResponse(str(path), media_type="audio/wav")

    @app.get("/config")
    def get_config() -> JSONResponse:
        wakeword = getattr(context.assistant, "wakeword", "") or ""
        return JSONResponse({"wakeword": wakeword.title()})

    @app.get("/history")
    def get_history() -> JSONResponse:
        with history_lock:
            return JSONResponse(list(history_log))

    @app.post("/reset")
    def reset_chat() -> dict:
        _require_ready()
        with history_lock:
            history_log.clear()
        context.assistant._history.clear()
        on_turn({"role": "reset", "ts": time.time()})
        return {"ok": True}

    @app.post("/restart")
    def restart_app() -> JSONResponse:
        """Restart the Fulloch process so restart-flagged config changes apply.

        Re-execs `python app.py` in place — works natively and in the container
        (the new process re-reads config.yml); falls back to a hard exit, which
        the compose `restart: unless-stopped` policy brings back up. Scheduled
        just after the response so the client gets the 200 and can poll /status.
        """
        logger.info("Restart requested via dashboard")
        _schedule_restart()
        return JSONResponse({"restarting": True})

    @app.get("/status")
    def get_status() -> JSONResponse:
        # Lifecycle phase (NEEDS_SETUP/DOWNLOADING/LOADING/READY/ERROR) drives
        # wizard-vs-dashboard routing + the progress/loading screens. Always
        # available, even before the assistant exists.
        payload = lifecycle.snapshot()
        if context.downloader is not None and context.downloader.active:
            payload["download"] = context.downloader.snapshot()
        if context.assistant is None or not lifecycle.is_ready():
            payload.update({
                "state": "idle",
                "mic_enabled": False,
                "last_utterance": "",
                "last_response": "",
                # Tail of first-party logs so the loading screen can render a
                # live terminal of what's happening instead of a bare spinner.
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
        payload.update({
            "state": context.assistant.get_state(),
            "mic_enabled": context.assistant.audio_capture.transcribing,
            "last_utterance": last_utterance,
            "last_response": last_response,
            # Remote-LLM mode: the LLM is off-device; the UI swaps the character
            # + shows an "not fully local" note in the tagline.
            "remote_llm": _is_remote_llm(),
            # True when that off-device LLM is configured but unreachable — the UI
            # shows a red banner that we're degraded to regex/fast-path only.
            "llm_unreachable": _is_remote_llm() and bool(
                getattr(context.assistant, "remote_llm_unreachable", False)),
        })
        return JSONResponse(payload)

    @app.post("/speak")
    def speak(req: SpeakRequest) -> dict:
        _require_ready()
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        threading.Thread(
            target=context.assistant.speak_proactive, args=(text,), daemon=True
        ).start()
        return {"ok": True}

    @app.post("/mic")
    def set_mic(req: MicRequest) -> dict:
        _require_ready()
        context.assistant.audio_capture.transcribing = req.enabled
        return {"ok": True, "mic_enabled": req.enabled}

    @app.post("/stop")
    def stop_turn() -> dict:
        # Complete, silent stop: aborts the SLM/TTS of whatever turn is in
        # flight (voice or text) and stands down without a follow-up window.
        _require_ready()
        context.assistant.request_stop()
        return {"ok": True}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        _require_ready()
        answer = context.assistant.handle_text_turn(req.text)
        return {"answer": answer}

    @app.get("/facts")
    def facts_list() -> JSONResponse:
        _require_ready()
        from tools.notes import list_facts
        return JSONResponse({"facts": list_facts()})

    @app.post("/facts")
    def facts_add(req: FactRequest) -> JSONResponse:
        _require_ready()
        from tools.notes import list_facts, remember_fact
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        remember_fact(text)
        return JSONResponse({"facts": list_facts()})

    @app.put("/facts/{idx}")
    def facts_update(idx: int, req: FactRequest) -> JSONResponse:
        _require_ready()
        from tools.notes import list_facts, update_fact
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty fact")
        if not update_fact(idx, text):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.delete("/facts/{idx}")
    def facts_delete(idx: int) -> JSONResponse:
        _require_ready()
        from tools.notes import delete_fact, list_facts
        if not delete_fact(idx):
            raise HTTPException(status_code=404, detail="fact not found")
        return JSONResponse({"facts": list_facts()})

    @app.get("/notes")
    def notes_list() -> JSONResponse:
        _require_ready()
        from tools.notes import list_note_files
        return JSONResponse({"notes": list_note_files()})

    @app.get("/notes/{name:path}")
    def notes_read(name: str) -> JSONResponse:
        _require_ready()
        from tools.notes import read_note_file
        content = read_note_file(name)
        if content is None:
            raise HTTPException(status_code=404, detail="note not found")
        return JSONResponse({"name": name, "content": content})

    @app.put("/notes/{name:path}")
    def notes_save(name: str, req: NoteRequest) -> JSONResponse:
        _require_ready()
        from tools.notes import read_note_file, save_note_file
        if not save_note_file(name, req.content):
            raise HTTPException(status_code=404, detail="note not found")
        return JSONResponse({"name": name, "content": read_note_file(name)})

    @app.get("/entities")
    def entities_list() -> JSONResponse:
        _require_ready()
        from tools._config import config
        if "home_assistant" not in config:
            return JSONResponse({"available": False, "entities": []})
        from tools import home_assistant as ha
        return JSONResponse({"available": True, "entities": ha.list_entities()})

    @app.post("/entities")
    def entities_set(req: EntityDenyRequest) -> JSONResponse:
        _require_ready()
        from tools._config import config
        if "home_assistant" not in config:
            raise HTTPException(status_code=404, detail="Home Assistant not configured")
        from tools import home_assistant as ha
        entity_id = (req.entity_id or "").strip()
        if not entity_id:
            raise HTTPException(status_code=400, detail="empty entity_id")
        ha.set_entity_denied(entity_id, req.denied)
        return JSONResponse({"available": True, "entities": ha.list_entities()})

    # --- Setup / settings console (work with or without an assistant) ------
    # These are NOT _require_ready-gated: they drive first-run setup (no
    # assistant yet) and stay available afterwards as the settings console.

    @app.get("/setup/schema")
    def setup_schema() -> JSONResponse:
        from .config_store import settings_view
        return JSONResponse(settings_view(context.config_path))

    @app.get("/setup/preflight")
    def setup_preflight() -> JSONResponse:
        from .preflight import preflight
        return JSONResponse(preflight())

    @app.put("/config")
    def config_update(req: ConfigUpdateRequest) -> JSONResponse:
        from .config_store import ConfigValidationError, update_config
        try:
            applied = update_config(req.updates, context.config_path)
        except ConfigValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors) from e
        # Push to the running assistant so hot-applicable changes take effect
        # without a restart. A change needs a restart iff it wasn't applied live
        # — covers both restart-only fields and hot-capable fields the assistant
        # declined (e.g. a Qwen voice swap). In setup mode (no assistant yet) the
        # set is empty, so any change reports restart-required, which is correct:
        # nothing is live to apply it to.
        hot: set = set()
        if context.assistant is not None and lifecycle.is_ready():
            try:
                hot = context.assistant.apply_hot_config(applied)
            except Exception as e:  # noqa: BLE001 — never fail the save on hot-apply
                logger.warning("Hot-apply failed: %s", e)
        restart = any(a["path"] not in hot for a in applied)
        # Drop the coerced value from the response (may not be JSON-trivial; the
        # UI only needs path + whether a restart is still required).
        view = [{"path": a["path"], "apply": a["apply"],
                 "hot_applied": a["path"] in hot} for a in applied]
        return JSONResponse({"applied": view, "restart_required": restart})

    @app.post("/setup/models")
    def setup_models(req: ModelsRequest) -> JSONResponse:
        from .config_schema import TIER_PRESETS
        from .config_store import write_models
        models = req.models
        if req.tier:
            tier = next((t for t in TIER_PRESETS if t.id == req.tier), None)
            if tier is None:
                raise HTTPException(status_code=422, detail=f"unknown tier {req.tier!r}")
            models = tier.models
        if not models:
            raise HTTPException(status_code=422, detail="provide a tier or models block")
        write_models(models, context.config_path)
        return JSONResponse({"ok": True, "models": models})

    @app.post("/setup/plan")
    def setup_plan(req: ModelsRequest) -> JSONResponse:
        """Dry-run the download plan for a models selection: which assets are
        already on disk vs still need downloading. Drives the install view's
        pre-scan + per-model custom-path inputs. Writes nothing, downloads
        nothing — just resolves + inspects. Honours a custom `model` path per
        domain so the scan reflects a model the user already has elsewhere."""
        from core.backends import resolve_models

        from .config_schema import TIER_PRESETS
        from .downloader import plan_assets
        models = req.models
        if req.tier:
            tier = next((t for t in TIER_PRESETS if t.id == req.tier), None)
            if tier is None:
                raise HTTPException(status_code=422, detail=f"unknown tier {req.tier!r}")
            models = tier.models
        resolved = resolve_models(models)
        assets = plan_assets(resolved)
        return JSONResponse({"assets": [a.snapshot() for a in assets]})

    @app.post("/setup/install")
    def setup_install() -> JSONResponse:
        from core.backends import resolve_models

        from .config_store import read_config
        from .downloader import plan_assets
        cfg = read_config(context.config_path)
        resolved = resolve_models(cfg.get("models"))
        assets = plan_assets(resolved)

        def _done(ok: bool) -> None:
            if ok:
                # Setup finished — clear any reset marker so the next restart
                # stays in run mode instead of re-entering the wizard.
                try:
                    _reset_marker_path().unlink(missing_ok=True)
                except OSError:
                    pass
                context.lifecycle.set(LOADING, "starting assistant")
                if context.assistant is not None:
                    # Reconfigure of an already-running assistant (recovering
                    # from an ERROR where the first config failed to load).
                    # main() has long passed the setup `proceed` wait, so the
                    # only way to load the new backends is a fresh process.
                    _schedule_restart()
                else:
                    # First-run: release the setup block so main() builds and
                    # runs the assistant (which flips LOADING -> READY).
                    context.lifecycle.signal_proceed()
            else:
                snap = context.downloader.snapshot()
                context.lifecycle.set(ERROR, snap.get("error") or "download failed")

        context.lifecycle.set(DOWNLOADING, "downloading models")
        if not context.downloader.start(assets, on_complete=_done):
            raise HTTPException(status_code=409, detail="a download is already in progress")
        return JSONResponse({"started": True, "assets": [a.snapshot() for a in assets]})

    @app.post("/setup/reset")
    def setup_reset() -> JSONResponse:
        """Arm a re-run of the setup wizard on the next start.

        Backs up config.yml and drops the reset marker that detect_setup_state
        honours, so a restart re-enters setup mode with config + models still on
        disk (a reconfigure re-downloads nothing unless backends change). For a
        truly clean slate the user can delete the data volume themselves. Takes
        effect on restart — we don't tear down the live, already-loaded assistant.
        """
        cfg = Path(context.config_path)
        backup = None
        if cfg.is_file():
            backup = cfg.with_name(f"config.yml.bak-{int(time.time())}")
            shutil.copy2(cfg, backup)
        _reset_marker_path().write_text("setup reset requested\n")
        logger.info("Setup reset armed; wizard will run on next restart")
        return JSONResponse({
            "ok": True,
            "restart_required": True,
            "backup": backup.name if backup else None,
        })

    @app.get("/setup/progress")
    def setup_progress() -> JSONResponse:
        return JSONResponse(context.downloader.snapshot())

    @app.get("/setup/progress/stream")
    async def setup_progress_stream(request: Request) -> StreamingResponse:
        async def gen():
            while True:
                if await request.is_disconnected():
                    break
                snap = context.downloader.snapshot()
                yield f"data: {json.dumps(snap)}\n\n"
                if snap["state"] in ("done", "error", "idle"):
                    break
                await asyncio.sleep(1.0)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/setup/test-llm")
    def setup_test_llm(req: LlmTestRequest) -> JSONResponse:
        from core.llm_openai import test_connection
        return JSONResponse(test_connection(
            base_url=req.base_url, model=req.model, api_key=req.api_key or "",
        ))

    @app.post("/setup/list-llm-models")
    def setup_list_llm_models(req: LlmModelsRequest) -> JSONResponse:
        # Offer a model picker after a successful test connection (GET /v1/models)
        # instead of free-text entry. Available in setup mode (no assistant yet).
        from core.llm_openai import list_models
        return JSONResponse(list_models(
            base_url=req.base_url, api_key=req.api_key or "",
        ))

    @app.post("/llm/model")
    def switch_llm_model(req: LlmSwitchRequest) -> JSONResponse:
        """Hot-swap the remote LLM model on the live assistant, then persist it.

        Only meaningful when the running backend is OpenAI (see
        Assistant.set_llm_model — a local llama/none backend can't switch live);
        otherwise we report ok=False so the UI tells the user to restart. On a
        successful live swap we also write models.llm.model so it survives a
        restart; a persist failure is surfaced but doesn't undo the live swap.
        """
        _require_ready()
        result = context.assistant.set_llm_model(req.model)
        if result.get("ok"):
            from .config_store import set_llm_model_name
            try:
                set_llm_model_name(req.model, context.config_path)
            except Exception as e:  # noqa: BLE001
                result["persist_error"] = f"{type(e).__name__}: {e}"
        return JSONResponse(result)

    @app.get("/setup/token")
    def setup_token_status() -> JSONResponse:
        return JSONResponse({"enabled": bool(context.dashboard_token)})

    @app.post("/setup/token")
    def setup_token_generate() -> JSONResponse:
        # Generate, persist to .env, and apply live so the console is gated from
        # now on. Returned once for the user to copy.
        from .env_store import set_env_var
        tok = secrets.token_urlsafe(32)
        try:
            # Persist inside the single ./data volume so it survives image
            # updates (app.py loads ./data/.env at startup).
            set_env_var("FULLOCH_DASHBOARD_TOKEN", tok, path="./data/.env")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"could not write .env: {e}") from e
        context.dashboard_token = tok
        logger.info("Dashboard access token generated and applied")
        return JSONResponse({"token": tok})

    @app.get("/setup/voices")
    def setup_voices() -> JSONResponse:
        from core.voice_clone import list_voices
        return JSONResponse({"voices": list_voices()})

    @app.post("/setup/voice")
    def setup_voice_generate(req: VoiceRequest) -> Response:
        from core.voice_clone import audio_to_wav_bytes, generate
        try:
            audio, sr = generate(req.instruct, req.phrase)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return Response(content=audio_to_wav_bytes(audio, sr), media_type="audio/wav")

    @app.post("/setup/voice/save")
    def setup_voice_save(req: VoiceSaveRequest) -> JSONResponse:
        from core.voice_clone import list_voices, save_last
        name = (req.name or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="empty voice name")
        try:
            saved = save_last(name)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return JSONResponse({"saved": saved, "voices": list_voices()})

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
    assistant=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    lifecycle: Optional[Lifecycle] = None,
    context: Optional[AppContext] = None,
) -> threading.Thread:
    """Launch the dashboard on a daemon thread. Non-blocking.

    Pass both ``ssl_certfile`` and ``ssl_keyfile`` to serve over HTTPS (uvicorn
    terminates TLS). If only one is given, or a file is missing, TLS is skipped
    and the dashboard falls back to HTTP with a loud warning.

    Prefer a shared ``context`` (so the assistant can attach after setup); a
    bare ``assistant`` (may be None) + ``lifecycle`` still works.
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

    app = create_app(assistant, lifecycle=lifecycle, context=context)
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
