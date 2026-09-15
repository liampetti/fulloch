"""First-run setup and live settings routes, including model installation."""

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .backups import (
    _SAFE_BACKUP_NAME_RE,
    _create_backup,
    _list_backups,
    _restore_backup,
)
from .lifecycle import DOWNLOADING, ERROR, LOADING, AppContext

logger = logging.getLogger(__name__)
_VOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "voices"
# Passwords use the authentication module's dedicated endpoint.
_SETTABLE_CREDENTIALS = frozenset({"ha_token", "llm_api_key", "obsidian_token", "hf_token"})


class ConfigUpdateRequest(BaseModel):
    updates: dict


class TimezoneRequest(BaseModel):
    tz: str


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


class HaTestRequest(BaseModel):
    url: str
    token: Optional[str] = None


class PathTestRequest(BaseModel):
    path: str


class SetupCredentialRequest(BaseModel):
    key: str
    value: str


def _normalize_llm_url(url: str) -> str:
    """Prepend http:// and append /v1 if missing."""
    u = (url or "").strip()
    if not u:
        return u
    if not re.match(r"https?://", u, re.IGNORECASE):
        u = "http://" + u
    if not re.search(r"/v1/?$", u):
        u = u.rstrip("/") + "/v1"
    return u


def _schedule_restart(delay: float = 0.5, assistant=None) -> None:
    """Re-exec after the response; shut down the attached assistant first."""
    def _go():
        time.sleep(delay)
        try:
            if assistant is not None:
                assistant.shutdown()
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception:  # noqa: BLE001 — last resort: exit, let the orchestrator restart
            os._exit(0)

    threading.Thread(target=_go, daemon=True, name="dashboard-restart").start()


def _reset_marker_for(context: AppContext) -> Path:
    """The setup-reset marker, alongside config.yml (see core/setup.py)."""
    return Path(context.config_path).parent / ".setup_pending"


def _completion_marker_for(context: AppContext) -> Path:
    """Marker proving the wizard completed at least one successful install."""
    return Path(context.config_path).parent / ".setup_complete"


def start_auto_download(context: AppContext) -> None:
    """Download missing assets for an already-configured install.

    Called by app.py after create_app has supplied context.downloader. A
    successful automatic recovery continues startup without showing the wizard.
    """
    from core.backends import resolve_models

    from .config_store import read_config
    from .downloader import plan_assets

    cfg = read_config(context.config_path)
    resolved = resolve_models(cfg.get("models"))
    assets = plan_assets(resolved)

    def _done(ok: bool) -> None:
        if ok:
            _completion_marker_for(context).touch()
            try:
                _reset_marker_for(context).unlink(missing_ok=True)
            except OSError:
                pass
            context.lifecycle.set(LOADING, "starting assistant")
            context.lifecycle.signal_proceed()
        else:
            snap = context.downloader.snapshot()
            context.lifecycle.set(ERROR, snap.get("error") or "download failed")

    context.lifecycle.set(DOWNLOADING, "downloading models")
    context.downloader.start(assets, on_complete=_done)


def register_setup_routes(
    app: FastAPI,
    context: AppContext,
    *,
    require_ready: Callable[[], None],
    static_dir: Path,
) -> None:
    """Register settings against the shared, late-attachable application context."""
    lifecycle = context.lifecycle

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page() -> str:
        from .config_store import read_config

        general = read_config(context.config_path).get("general") or {}
        prefs = {"theme": general.get("dashboard_theme", "auto")}
        html = (static_dir / "setup.html").read_text(encoding="utf-8")
        return html.replace(
            "<script>\n  // Use the saved dashboard preference",
            f"<script>window.FULLOCH_DASHBOARD_PREFS = {json.dumps(prefs)};</script>\n<script>\n  // Use the saved dashboard preference",
        )

    @app.get("/voice/sample")
    def voice_sample(name: str) -> FileResponse:
        """Preview a bundled sample or a Qwen/Pocket clone reference WAV."""
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise HTTPException(status_code=400, detail="invalid voice name")
        path = _VOICES_DIR / f"{name}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no sample for that voice")
        return FileResponse(str(path), media_type="audio/wav")

    @app.post("/restart")
    def restart_app() -> JSONResponse:
        """Restart the process so restart-flagged configuration changes apply."""
        logger.info("Restart requested via dashboard")
        _schedule_restart(assistant=context.assistant)
        return JSONResponse({"restarting": True})

    # Setup routes remain available before the assistant is ready.
    @app.post("/setup/timezone")
    def setup_timezone(req: TimezoneRequest) -> JSONResponse:
        from utils import local_time

        if not local_time.is_valid_tz(req.tz):
            raise HTTPException(status_code=422, detail=f"Unknown timezone: {req.tz!r}")
        from .config_store import read_config, update_config

        configured_tz = ((read_config(context.config_path).get("general") or {}).get("timezone") or "").strip()
        if configured_tz:
            return JSONResponse({"ok": True, "configured": True})
        update_config({"general.timezone": req.tz}, context.config_path)
        local_time.set_tz(req.tz)
        return JSONResponse({"ok": True, "configured": False})

    @app.get("/setup/schema")
    def setup_schema() -> JSONResponse:
        from .config_store import settings_view
        from .credentials_store import load as load_creds

        schema = settings_view(context.config_path)
        creds = load_creds()
        schema["credentials"] = {k: bool(creds.get(k, "").strip()) for k in _SETTABLE_CREDENTIALS}
        return JSONResponse(schema)

    @app.get("/setup/preflight")
    def setup_preflight() -> JSONResponse:
        from .preflight import preflight

        return JSONResponse(preflight())

    @app.put("/config")
    def config_update(req: ConfigUpdateRequest) -> JSONResponse:
        from utils import local_time

        from .config_store import ConfigValidationError, update_config

        timezone = req.updates.get("general.timezone")
        if "general.timezone" in req.updates and not local_time.is_valid_tz(timezone):
            raise HTTPException(status_code=422, detail={"general.timezone": "unknown IANA timezone"})
        try:
            applied = update_config(req.updates, context.config_path)
        except ConfigValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors) from e
        # A change requires restart unless it was successfully applied live.
        hot: set = set()
        if context.assistant is not None and lifecycle.is_ready():
            try:
                hot = context.assistant.apply_hot_config(applied)
            except Exception as e:  # noqa: BLE001 — never fail the save on hot-apply
                logger.warning("Hot-apply failed: %s", e)
        if any(a["path"] == "notes.path" for a in applied):
            from tools.notes_root import refresh_notes_root

            refresh_notes_root()
        for change in applied:
            if change["path"] == "general.timezone":
                local_time.set_tz(change["value"])
                hot.add(change["path"])
        hot.update({
            a["path"] for a in applied
            if a["path"] in {"general.dashboard_theme", "general.dashboard_show_turn_details"}
        })
        restart = any(a["path"] not in hot for a in applied)
        view = [
            {"path": a["path"], "apply": a["apply"], "hot_applied": a["path"] in hot}
            for a in applied
        ]
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
        try:
            write_models(models, context.config_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return JSONResponse({"ok": True, "models": models})

    @app.post("/setup/plan")
    def setup_plan(req: ModelsRequest) -> JSONResponse:
        """Inspect the download plan, including custom paths, without writing."""
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

    @app.post("/setup/preflight-download")
    def setup_preflight_download(req: ModelsRequest = ModelsRequest()) -> JSONResponse:  # noqa: B008
        """Run disk, network and GPU checks before starting a model download."""
        from .config_schema import TIER_PRESETS
        from .config_store import read_config
        from .preflight import check_disk_for_models, check_gpu_for_models, check_network

        if req.tier:
            tier = next((t for t in TIER_PRESETS if t.id == req.tier), None)
            if tier is None:
                raise HTTPException(status_code=422, detail=f"unknown tier {req.tier!r}")
            models = tier.models
        elif req.models:
            models = req.models
        else:
            cfg = read_config(context.config_path)
            models = cfg.get("models") or {}
        errors: list[dict] = []
        ok, msg = check_disk_for_models(models)
        if not ok:
            errors.append({"check": "disk", "message": msg})
        ok, msg = check_network()
        if not ok:
            errors.append({"check": "network", "message": msg})
        ok, msg = check_gpu_for_models(models)
        if not ok:
            errors.append({"check": "gpu", "message": msg})
        return JSONResponse({"ok": not errors, "errors": errors})

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
                _completion_marker_for(context).touch()
                try:
                    _reset_marker_for(context).unlink(missing_ok=True)
                except OSError:
                    pass
                context.lifecycle.set(LOADING, "starting assistant")
                if context.assistant is not None:
                    _schedule_restart(assistant=context.assistant)
                else:
                    context.lifecycle.signal_proceed()
            else:
                snap = context.downloader.snapshot()
                context.lifecycle.set(ERROR, snap.get("error") or "download failed")

        context.lifecycle.set(DOWNLOADING, "downloading models")
        if not context.downloader.start(assets, on_complete=_done):
            raise HTTPException(status_code=409, detail="a download is already in progress")
        return JSONResponse({"started": True, "assets": [a.snapshot() for a in assets]})

    @app.post("/setup/retry-download")
    def setup_retry_download() -> JSONResponse:
        """Re-run the last download plan after a failed or interrupted download."""
        if context.downloader.active:
            raise HTTPException(status_code=409, detail="a download is already in progress")
        from core.backends import resolve_models

        from .config_store import read_config
        from .downloader import plan_assets

        cfg = read_config(context.config_path)
        resolved = resolve_models(cfg.get("models"))
        assets = plan_assets(resolved)

        def _done(ok: bool) -> None:
            if ok:
                _completion_marker_for(context).touch()
                try:
                    _reset_marker_for(context).unlink(missing_ok=True)
                except OSError:
                    pass
                context.lifecycle.set(LOADING, "starting assistant")
                if context.assistant is not None:
                    _schedule_restart(assistant=context.assistant)
                else:
                    context.lifecycle.signal_proceed()
            else:
                snap = context.downloader.snapshot()
                context.lifecycle.set(ERROR, snap.get("error") or "download failed")

        context.lifecycle.set(DOWNLOADING, "downloading models")
        context.downloader.start(assets, on_complete=_done)
        return JSONResponse({"started": True})

    @app.post("/setup/cancel-startup")
    def setup_cancel_startup() -> JSONResponse:
        """Re-exec to stop model transfer/load workers and return to setup."""
        if lifecycle.phase not in {DOWNLOADING, LOADING}:
            raise HTTPException(status_code=409, detail="startup is not in progress")
        _reset_marker_for(context).touch()
        lifecycle.set("NEEDS_SETUP", "startup cancelled; returning to setup")
        _schedule_restart(delay=0.1, assistant=context.assistant)
        return JSONResponse({"ok": True, "restarting": True})

    @app.post("/setup/reset")
    def setup_reset() -> JSONResponse:
        """Back up user state and request the setup wizard on next startup."""
        data_dir = Path(context.config_path).resolve().parent
        backup_dir = _create_backup(data_dir)
        _reset_marker_for(context).write_text("setup reset requested\n")
        logger.info("Setup reset armed; backup at %s", backup_dir)
        return JSONResponse({"ok": True, "restart_required": True, "backup": backup_dir.name})

    @app.get("/setup/backups")
    def setup_list_backups() -> dict:
        data_dir = Path(context.config_path).resolve().parent
        return {"backups": _list_backups(data_dir)}

    @app.post("/setup/backups/restore")
    def setup_restore_backup(req: dict) -> dict:
        """Restore a named backup, overwriting state without restarting."""
        name = (req or {}).get("name", "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="missing name")
        if not _SAFE_BACKUP_NAME_RE.match(name):
            raise HTTPException(status_code=400, detail="invalid backup name")
        data_dir = Path(context.config_path).resolve().parent
        backup_dir = data_dir / "backups" / name
        if not backup_dir.is_dir():
            raise HTTPException(status_code=404, detail="backup not found")
        restored = _restore_backup(backup_dir, data_dir)
        logger.info("Restored backup %s (%d files)", name, len(restored))
        return {"ok": True, "restored": restored}

    @app.post("/setup/regen-cert")
    def setup_regen_cert() -> JSONResponse:
        """Regenerate the self-signed dashboard certificate and require restart."""
        from core.tls_certs import regenerate_self_signed_cert

        from .config_store import read_config, update_config

        cfg = read_config(context.config_path)
        general = cfg.get("general") or {}
        cert_path = general.get("dashboard_ssl_certfile")
        key_path = general.get("dashboard_ssl_keyfile")
        certs_dir = str(Path(cert_path).parent) if cert_path else str(Path(context.config_path).parent / "certs")
        try:
            new_cert, new_key = regenerate_self_signed_cert(certs_dir)
            if not cert_path or not key_path:
                update_config(
                    {"general.dashboard_ssl_certfile": new_cert, "general.dashboard_ssl_keyfile": new_key},
                    context.config_path,
                )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"could not regenerate cert: {e}") from e
        logger.info("HTTPS certificate (re)generated via dashboard; restart required")
        return JSONResponse({"ok": True, "restart_required": True})

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
            base_url=_normalize_llm_url(req.base_url), model=req.model, api_key=req.api_key or "",
        ))

    @app.post("/setup/test-ha")
    def setup_test_ha(req: HaTestRequest) -> JSONResponse:
        import urllib.error
        import urllib.request

        from .credentials_store import get_credential

        url = req.url.rstrip("/") if req.url else ""
        if not url:
            return JSONResponse({"ok": False, "error": "No URL provided"})
        # A blank field retains the saved token.
        token = req.token or get_credential("ha_token")
        try:
            r = urllib.request.Request(
                f"{url}/api/", headers={"Authorization": f"Bearer {token}"} if token else {},
            )
            with urllib.request.urlopen(r, timeout=5) as resp:
                return JSONResponse({"ok": resp.status == 200, "status": resp.status})
        except urllib.error.HTTPError as e:
            return JSONResponse({"ok": False, "error": f"HTTP {e.code}"})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.post("/setup/test-path")
    def setup_test_path(req: PathTestRequest) -> JSONResponse:
        p = Path(req.path) if req.path else None
        return JSONResponse({"ok": bool(p and p.exists())})

    @app.post("/setup/list-llm-models")
    def setup_list_llm_models(req: LlmModelsRequest) -> JSONResponse:
        from core.llm_openai import list_models

        return JSONResponse(list_models(base_url=_normalize_llm_url(req.base_url), api_key=req.api_key or ""))

    @app.post("/llm/model")
    def switch_llm_model(req: LlmSwitchRequest) -> JSONResponse:
        """Hot-swap a remote model and persist it; report persist failures separately."""
        require_ready()
        result = context.assistant.set_llm_model(req.model)
        if result.get("ok"):
            from .config_store import set_llm_model_name

            try:
                set_llm_model_name(req.model, context.config_path)
            except Exception as e:  # noqa: BLE001
                result["persist_error"] = f"{type(e).__name__}: {e}"
        return JSONResponse(result)

    @app.get("/setup/credentials")
    def setup_get_credentials() -> JSONResponse:
        from .credentials_store import load as load_creds

        creds = load_creds()
        return JSONResponse({k: bool(creds.get(k, "").strip()) for k in _SETTABLE_CREDENTIALS})

    @app.post("/setup/credential")
    def setup_set_credential(req: SetupCredentialRequest) -> JSONResponse:
        from .credentials_store import set_credential

        if req.key not in _SETTABLE_CREDENTIALS:
            raise HTTPException(status_code=422, detail=f"unknown credential key: {req.key}")
        value = (req.value or "").strip()
        if not value:
            raise HTTPException(status_code=422, detail="value must not be empty")
        try:
            set_credential(req.key, value)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"could not write credentials: {e}") from e
        if req.key == "ha_token":
            try:
                import tools.ha_client as _ha

                _ha.HA_TOKEN = value
            except Exception:  # noqa: BLE001
                pass
        elif req.key == "llm_api_key" and context.assistant is not None:
            client = getattr(context.assistant, "slm_model", None)
            if hasattr(client, "set_api_key"):
                client.set_api_key(value)
        return JSONResponse({"ok": True})

    @app.get("/setup/voices")
    def setup_voices() -> JSONResponse:
        from core.voice_clone import list_voices

        return JSONResponse({"voices": list_voices()})

    @app.post("/setup/voice")
    def setup_voice_generate(req: VoiceRequest) -> Response:
        from core.voice_clone import audio_to_wav_bytes, generate, unload_model

        try:
            audio, sr = generate(req.instruct, req.phrase)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        finally:
            unload_model()
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
