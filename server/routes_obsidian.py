"""Obsidian plugin WebSocket bridge and vault configuration endpoints."""

import asyncio
import json
import logging
import os
import queue
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .lifecycle import AppContext

logger = logging.getLogger(__name__)
_OBSIDIAN_PLUGIN_ZIP = Path(__file__).resolve().parent.parent / "data" / "fulloch-obsidian-plugin.zip"


def register_obsidian_routes(app: FastAPI, context: AppContext) -> None:
    """Register the bridge using the shared editor state and current assistant."""
    lifecycle = context.lifecycle

    @app.websocket("/ws/obsidian")
    async def obsidian_ws(ws: WebSocket):
        """Exchange vault metadata, editor context, file changes and commands."""
        obsidian_token = (
            os.environ.get("OBSIDIAN_TOKEN", "").strip()
            or getattr(context, "obsidian_token", "")
        )
        if obsidian_token:
            query_token = ws.query_params.get("token", "")
            if not (query_token and secrets.compare_digest(query_token, obsidian_token)):
                await ws.close(code=1008)
                return
        if context.assistant is None or not lifecycle.is_ready():
            await ws.close(code=1013)
            return
        await ws.accept()
        logger.info("Obsidian plugin connected")
        cmd_q: queue.Queue = queue.Queue(maxsize=100)
        context.obsidian_cmd_q = cmd_q

        import tools.notes as _notes_module
        from tools.notes import set_obsidian_cmd_q

        set_obsidian_cmd_q(cmd_q)

        def _safe_reindex(full_path: Path) -> None:
            try:
                _notes_module._get_index().index_file(full_path)
            except Exception as e:
                logger.error("Reindex of %s failed: %s", full_path, e)

        async def _receive():
            try:
                while True:
                    msg = await ws.receive()
                    if "text" not in msg or not msg["text"]:
                        continue
                    try:
                        data = json.loads(msg["text"])
                    except (json.JSONDecodeError, Exception):
                        continue
                    kind = data.get("type")
                    if kind == "vault_metadata":
                        raw_path = data.get("vault_path")
                        if not raw_path:
                            continue
                        # Translate plugin host paths to the server's Docker mount.
                        from tools.notes_root import translate_vault_path

                        translated = translate_vault_path(raw_path)
                        resolved = translated.expanduser().resolve()
                        if not (resolved / ".obsidian").is_dir():
                            try:
                                await ws.send_json({"type": "vault_rejected", "reason": "not_a_vault"})
                            except Exception:
                                return
                            context.obsidian_vault_state["last_error"] = "not_a_vault"
                            logger.warning("Plugin reported non-vault path: %s", resolved)
                            continue
                        if not resolved.is_dir():
                            try:
                                await ws.send_json({"type": "vault_rejected", "reason": "unreadable"})
                            except Exception:
                                return
                            context.obsidian_vault_state["last_error"] = "unreadable"
                            continue
                        context.obsidian_vault_state["connected"] = True
                        context.obsidian_vault_state["vault_path"] = str(resolved)
                        context.obsidian_vault_state["vault_resolved_path"] = str(resolved)
                        context.obsidian_vault_state["plugin_vault_path"] = str(raw_path)
                        # open_file has no reverse translation back to the host.
                        context.obsidian_vault_state["path_navigation_mismatch"] = str(raw_path) != str(resolved)
                        context.obsidian_vault_state["last_connected_at"] = time.time()
                        context.obsidian_vault_state["last_error"] = None
                        logger.info("Obsidian vault adopted: %s (%d files)", resolved, len(data.get("files") or []))
                    elif kind == "context":
                        context.assistant.set_vault_context(current_file=data.get("file"))
                    elif kind == "file_changed":
                        rel = data.get("path")
                        if not rel:
                            continue
                        try:
                            # Metadata is editor state only; notes.path remains
                            # the storage root and fallback before metadata arrives.
                            vault = context.obsidian_vault_state.get("vault_resolved_path")
                            if not vault:
                                from tools.notes_root import get_notes_root

                                vault = get_notes_root()
                            full = Path(vault) / rel if vault else None
                            if full and full.is_file():
                                import threading as _th

                                _th.Thread(target=_safe_reindex, args=(full,), daemon=True).start()
                        except Exception as e:
                            logger.error("file_changed reindex failed: %s", e)
                    elif kind == "pong":
                        pass
            except (WebSocketDisconnect, RuntimeError):
                return

        async def _send():
            while True:
                try:
                    cmd = await asyncio.to_thread(lambda: cmd_q.get(timeout=1.0))
                except Exception:
                    try:
                        await ws.send_json({"type": "ping"})
                    except Exception:
                        return
                    continue
                try:
                    await ws.send_json(cmd)
                except Exception:
                    return

        recv_task = asyncio.create_task(_receive())
        send_task = asyncio.create_task(_send())
        try:
            _done, pending = await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            context.assistant.set_vault_context(None)
            context.obsidian_cmd_q = None
            set_obsidian_cmd_q(None)
            context.obsidian_vault_state["connected"] = False
            logger.info("Obsidian plugin disconnected")

    @app.get("/api/obsidian/status")
    def obsidian_status() -> dict:
        state = dict(context.obsidian_vault_state)
        from tools.notes_root import get_notes_root

        from .config_store import read_config

        state["notes_path"] = str(get_notes_root())
        state["allow_edit_delete"] = bool(
            (read_config(context.config_path).get("obsidian") or {}).get("allow_edit_delete", False)
        )
        return state

    @app.post("/api/obsidian/edit-capability")
    def obsidian_edit_capability(req: dict) -> dict:
        """Persist and live-apply the explicit active-note edit/delete opt-in."""
        enabled = (req or {}).get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="enabled must be a boolean")
        from tools._config import config as tools_config

        from .config_store import update_config

        update_config({"obsidian.allow_edit_delete": enabled}, context.config_path)
        tools_config.setdefault("obsidian", {})["allow_edit_delete"] = enabled
        return {"allow_edit_delete": enabled}

    @app.post("/api/obsidian/regenerate-token")
    def obsidian_regenerate_token() -> dict:
        from .credentials_store import set_credential

        new_token = secrets.token_hex(32)
        set_credential("obsidian_token", new_token)
        os.environ["OBSIDIAN_TOKEN"] = new_token
        return {"token": new_token}

    @app.post("/api/obsidian/show-token")
    def obsidian_show_token() -> dict:
        """Return the current token for the Connect modal, generating one if missing."""
        from .credentials_store import set_credential

        existing = os.environ.get("OBSIDIAN_TOKEN", "").strip()
        if not existing:
            existing = secrets.token_hex(32)
            set_credential("obsidian_token", existing)
            os.environ["OBSIDIAN_TOKEN"] = existing
        return {"token": existing}

    @app.post("/api/setup/detect-obsidian-vaults")
    def detect_obsidian_vaults() -> dict:
        """Best-effort scan for Obsidian vaults on the local filesystem."""
        from .config_schema import discover_obsidian_vaults

        return {"candidates": discover_obsidian_vaults()}

    @app.post("/api/setup/obsidian-vault")
    def setup_obsidian_vault(req: dict) -> dict:
        """Use a validated vault as the configured notes location."""
        from tools.notes_root import translate_vault_path

        from .credentials_store import set_credential

        raw = (req or {}).get("path", "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="missing path")
        translated = translate_vault_path(raw)
        resolved = translated.expanduser().resolve()
        if not (resolved / ".obsidian").is_dir():
            raise HTTPException(status_code=400, detail="not a vault (no .obsidian/ folder)")
        from tools.notes_root import refresh_notes_root

        from .config_store import update_config

        update_config({"notes.path": str(resolved)}, context.config_path)
        refresh_notes_root()
        existing = os.environ.get("OBSIDIAN_TOKEN", "").strip()
        if not existing:
            new_token = secrets.token_hex(32)
            set_credential("obsidian_token", new_token)
            os.environ["OBSIDIAN_TOKEN"] = new_token
        return {"vault_path": str(resolved)}

    @app.get("/api/obsidian/plugin.zip")
    def obsidian_plugin_zip() -> FileResponse:
        """Serve the transitional, separately built Obsidian plugin archive."""
        if not _OBSIDIAN_PLUGIN_ZIP.is_file():
            raise HTTPException(status_code=404, detail="plugin archive unavailable")
        return FileResponse(
            str(_OBSIDIAN_PLUGIN_ZIP), media_type="application/zip", filename="fulloch-obsidian-plugin.zip",
        )
