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
import re as _re
import secrets
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from .config_store import read_config
from .lifecycle import (
    DOWNLOADING,
    ERROR,
    LOADING,
    LOG_BUFFER,
    READY,
    AppContext,
    Lifecycle,
)
from .satellite_protocols import register_satellite_routes

logger = logging.getLogger(__name__)


# Config cache: the dashboard polls /status frequently and the YAML parse
# in read_config isn't free. Cache the (path, mtime, parsed-data) tuple
# so repeated reads within the same file mtime return the same dict object.
# A config edit invalidates the cache automatically — no need to wire cache
# invalidation through the wizard or the settings console. The
# /status handler is the only hot consumer; other config reads (wizard,
# settings) happen on user-initiated changes, not in a tight loop.
_CONFIG_CACHE: dict = {"path": None, "mtime": None, "data": None}


def _read_config_cached(path: str) -> dict:
    """Read config from `path`, cached by mtime.

    Returns the cached parsed config if `path`'s mtime is unchanged
    since the last read. Falls back to a fresh read on OSError (file
    missing) so the first-run / pre-bootstrap case still works.
    """
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


def _normalize_llm_url(url: str) -> str:
    """Prepend http:// and append /v1 if missing."""
    u = (url or "").strip()
    if not u:
        return u
    if not _re.match(r"https?://", u, _re.IGNORECASE):
        u = "http://" + u
    if not _re.search(r"/v1/?$", u):
        u = u.rstrip("/") + "/v1"
    return u


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


def _reset_marker_for(context: AppContext) -> Path:
    """The setup-reset marker, alongside config.yml (see core/setup.py)."""
    return Path(context.config_path).parent / ".setup_pending"


# Files backed up on /setup/reset. Relative to data_dir (the same dir that
# holds config.yml). Models and certs are excluded — they're regeneratable.
_BACKUP_FILES = (
    "config.yml",
    "credentials.json",
    ".env",
    "voice_denylist.json",
)
_BACKUP_DIRS = (
    "voices",  # clone audio + transcripts
)
# Restrict backup names to a strict shape so a user-supplied name can't escape
# data_dir via path traversal.
_SAFE_BACKUP_NAME_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}$")


def _create_backup(data_dir: Path) -> Path:
    """Snapshot the user's state files into data_dir/backups/<ts>/.

    Returns the backup directory path. Existing backups are kept; the new one
    is timestamped to the second (with a numeric suffix on collision).
    """
    backups_root = data_dir / "backups"
    backups_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
    target = backups_root / ts
    suffix = 0
    while target.exists():
        suffix += 1
        target = backups_root / f"{ts}-{suffix}"
    target.mkdir(parents=True, exist_ok=True)
    backed_up: list[str] = []
    for rel in _BACKUP_FILES:
        src = data_dir / rel
        if not src.is_file():
            continue
        shutil.copy2(src, target / rel)
        backed_up.append(rel)
    for rel in _BACKUP_DIRS:
        src = data_dir / rel
        if not src.is_dir():
            continue
        shutil.copytree(src, target / rel)
        backed_up.append(f"{rel}/")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": backed_up,
                "reason": "setup reset",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def _list_backups(data_dir: Path) -> list[dict]:
    """Return a list of backup summaries, newest first."""
    backups_root = data_dir / "backups"
    if not backups_root.is_dir():
        return []
    out: list[dict] = []
    for d in sorted(backups_root.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir() or not _SAFE_BACKUP_NAME_RE.match(d.name):
            continue
        meta = d / "meta.json"
        files: list[str] = []
        created_at = None
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                files = m.get("files", [])
                created_at = m.get("created_at")
            except Exception:
                pass
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        out.append(
            {
                "name": d.name,
                "created_at": created_at,
                "files": files,
                "size_bytes": size,
            }
        )
    return out


def _restore_backup(backup_dir: Path, data_dir: Path) -> list[str]:
    """Copy each file/dir in the backup back to data_dir, overwriting."""
    restored: list[str] = []
    for rel in _BACKUP_FILES:
        src = backup_dir / rel
        if not src.exists():
            continue
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
    for rel in _BACKUP_DIRS:
        src = backup_dir / rel
        if not src.is_dir():
            continue
        dst = data_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        restored.append(f"{rel}/")
    return restored


def start_auto_download(context: AppContext) -> None:
    """Kick off downloading missing model assets for an already-configured
    install, skipping the wizard UI entirely.

    Called by `app.py` at startup when `detect_setup_state` reports a valid,
    already-populated config that's merely missing assets (e.g. the user
    picked a new model in Settings and restarted) — there's nothing left to
    ask, so jump straight to the download/loading screens instead of
    re-showing the wizard. Requires `context.downloader` to already exist
    (set by `create_app`, which must run — i.e. the dashboard must already be
    started — before this is called).
    """
    from core.backends import resolve_models

    from .config_store import read_config
    from .downloader import plan_assets

    cfg = read_config(context.config_path)
    resolved = resolve_models(cfg.get("models"))
    assets = plan_assets(resolved)

    def _done(ok: bool) -> None:
        if ok:
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


SUBSCRIBER_IDLE_KEEPALIVE_S = 15

# Credential keys the UI can read/write via /setup/credentials and /setup/credential.
# dashboard_password has its own dedicated /setup/password endpoint.
_SETTABLE_CREDENTIALS = frozenset({"ha_token", "llm_api_key", "obsidian_token"})

# Session-cookie auth gate. When dashboard_password is set in credentials.json,
# every route except the login page, static assets, and the liveness probe
# requires a valid session cookie obtained via POST /auth/login. Unset = no auth (zero-config local-only).
# See README "Exposing the dashboard".
_AUTH_EXEMPT_PATHS = frozenset({
    "/login", "/auth/login", "/auth/logout",
    "/logo.png", "/parloch.png", "/favicon.ico",
    "/ready",
})

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fulloch — Log in</title>
  <link rel="icon" href="/logo.png" type="image/png">
  <style>
    :root{--bg:#f8faf9;--surface:#fff;--text:#1b2722;--text-muted:#64746e;
          --border:rgba(14,23,19,.1);--primary:#10b981;--primary-fg:#fff;--error:#c0392b}
    html.dark{--bg:#0e1713;--surface:#1b2722;--text:#f0f4f2;
              --text-muted:#92a19a;--border:rgba(110,231,183,.12)}
    *{box-sizing:border-box}
    body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
         display:flex;align-items:center;justify-content:center;padding:1.5rem}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
          padding:2rem;box-shadow:0 4px 24px rgba(0,0,0,.06);width:100%;max-width:22rem}
    .logo{display:flex;align-items:center;gap:.6rem;margin-bottom:1.5rem}
    .logo img{width:36px;height:36px}
    .logo span{font-size:1.15rem;font-weight:700}
    label{display:block;font-size:.85rem;font-weight:600;margin:.9rem 0 .3rem}
    input[type=password]{width:100%;font:inherit;font-size:1rem;padding:.5rem .7rem;
                         border-radius:8px;border:1px solid var(--border);
                         background:var(--bg);color:var(--text)}
    button{width:100%;margin-top:1.1rem;padding:.6rem;border:none;border-radius:10px;
           background:var(--primary);color:var(--primary-fg);font:inherit;font-size:1rem;
           font-weight:600;cursor:pointer}
    button:hover{filter:brightness(1.05)}
    .err{color:var(--error);font-size:.85rem;margin-top:.6rem;min-height:1.2em}
  </style>
  <script>
    (()=>{const s=localStorage.getItem('appearance');
    const d=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;
    if(s==='dark'||(s===null&&d))document.documentElement.classList.add('dark')})();
  </script>
</head>
<body>
  <div class="card">
    <div class="logo"><img src="/logo.png" alt=""><span>Fulloch</span></div>
    <label for="pw">Password</label>
    <input id="pw" type="password" autofocus autocomplete="current-password">
    <button id="btn">Log in</button>
    <div id="err" class="err"></div>
  </div>
  <script>
    async function login(){
      const pw=document.getElementById('pw').value;
      const err=document.getElementById('err');
      err.textContent='';
      const r=await fetch('/auth/login',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({password:pw})});
      if(r.ok){location.href='/';}
      else{err.textContent='Incorrect password.';document.getElementById('pw').select();}
    }
    document.getElementById('btn').addEventListener('click',login);
    document.getElementById('pw').addEventListener('keydown',e=>{if(e.key==='Enter')login();});
  </script>
</body>
</html>"""


class ChatRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


class MicRequest(BaseModel):
    enabled: bool


class StopRequest(BaseModel):
    # None (the default, and what the dashboard's Stop button sends today)
    # falls back to whichever satellite currently owns the TurnArbiter — the
    # dashboard has no per-connection satellite identity yet to supply one
    # explicitly (that lands with Phase 5's satellite-v2 protocol).
    satellite_id: Optional[str] = None


class FactRequest(BaseModel):
    text: str


class NoteRequest(BaseModel):
    content: str


class EntityDenyRequest(BaseModel):
    entity_id: str
    denied: bool


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
    # Split CSS / JS assets live under /static/. Mounted last so the explicit
    # /logo.png and /voice/sample routes above still take precedence on those
    # exact paths (StaticFiles only matches prefixes that have no explicit
    # route, so /logo.png is fine — but the mount order doesn't matter for
    # matching here, only for safety).
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), check_dir=False), name="static")

    register_satellite_routes(app, context, lifecycle)

    # Split CSS / JS assets live under /static/. Mounted last so the explicit
    # /logo.png and /voice/sample routes above still take precedence on those
    # exact paths (StaticFiles only matches prefixes that have no explicit
    # route, so /logo.png is fine — but the mount order doesn't matter for
    # matching here, only for safety).
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), check_dir=False), name="static")

    def _require_ready() -> None:
        """Guard assistant-backed routes during setup / model load."""
        if context.assistant is None or not lifecycle.is_ready():
            raise HTTPException(
                status_code=503,
                detail="assistant not ready (setup or model load in progress)",
            )

    @app.middleware("http")
    async def _no_cache_static(request, call_next):
        """Force revalidation on /static/* so satellite-protocol edits (e.g.
        the iOS loudspeaker / half-duplex mic mute) reach the browser without
        a hard-refresh. ETag-based revalidation is cheap; the cost is one
        round-trip per asset, which is irrelevant on a LAN device."""
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def _reset_marker_path() -> Path:
        return _reset_marker_for(context)

    # Seed the password hash from the environment (populated by inject_env()
    # at startup from credentials.json). A post-wizard /setup/password call
    # updates context.dashboard_password_hash in place — no restart needed.
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
            # Redirect HTML page requests to /login; API/WS calls get 401.
            upgrade = request.headers.get("upgrade", "").lower()
            accept = request.headers.get("accept", "")
            if upgrade != "websocket" and "text/html" in accept:
                from fastapi.responses import RedirectResponse
                return RedirectResponse("/login", status_code=303)
            return JSONResponse({"detail": "unauthorized"}, status_code=401)

        return await call_next(request)

    history_log: list = []
    history_lock = threading.Lock()
    subscribers: list[queue.Queue] = []
    subscribers_lock = threading.Lock()
    startup_greeting_seeded = False

    def on_turn(event: dict) -> None:
        # Native satellite lifecycle events share the assistant listener but
        # are delivered directly to `/ws/satellite-v2`; they are not chat
        # messages and have no dashboard timestamp/content fields.
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
                except Exception:
                    dead.append(q)
            for q in dead:
                subscribers.remove(q)

    def _seed_startup_greeting(assistant) -> None:
        """Add the warmed greeting once, even if dashboard attachment races boot."""
        nonlocal startup_greeting_seeded
        greeting = getattr(assistant, "greeting_text", "").strip() if assistant else ""
        if not greeting:
            return
        with history_lock:
            if startup_greeting_seeded:
                return
            startup_greeting_seeded = True
        on_turn({"role": "assistant", "content": greeting, "ts": time.time(), "source": "startup"})

    # Register the SSE turn listener as soon as the assistant exists — now if
    # already attached, or later via context.set_assistant after setup.
    def _attach_turn_listener(assistant) -> None:
        assistant.register_turn_listener(on_turn)
        _seed_startup_greeting(assistant)

    context.on_attach(_attach_turn_listener)

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
        html = (_STATIC_DIR / page).read_text(encoding="utf-8")
        if ready:
            from .config_store import read_config

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

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page() -> str:
        from .config_store import read_config

        general = read_config(context.config_path).get("general") or {}
        prefs = {"theme": general.get("dashboard_theme", "auto")}
        html = (_STATIC_DIR / "setup.html").read_text(encoding="utf-8")
        return html.replace(
            "<script>\n  // Use the saved dashboard preference",
            f"<script>window.FULLOCH_DASHBOARD_PREFS = {json.dumps(prefs)};</script>\n<script>\n  // Use the saved dashboard preference",
        )

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
        path = _PARLOCH_PATH if remote and _PARLOCH_PATH.is_file() else _LOGO_PATH
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/voice/sample")
    def voice_sample(name: str) -> FileResponse:
        """Preview a bundled sample or a Qwen/Pocket clone reference WAV."""
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
        _seed_startup_greeting(context.assistant)
        with history_lock:
            return JSONResponse(list(history_log))

    @app.post("/reset")
    def reset_chat() -> dict:
        nonlocal startup_greeting_seeded
        _require_ready()
        with history_lock:
            history_log.clear()
            # Clearing chat is explicit; don't reinsert the startup line on a
            # later history refresh.
            startup_greeting_seeded = True
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

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Liveness/readiness probe for `docker compose ps` and HEALTHCHECK.

        Returns 200 + `{"ready": true}` as soon as the dashboard server
        is serving HTTP — the wizard, download progress, or dashboard
        are all reachable from the browser, which is the right "ready"
        definition for a first-run UX (the assistant being fully
        *loaded* with the SLM/ASR/TTS warm can be minutes away on a
        first run; that's tracked separately by /status's `phase`).

        Semantics for Docker:
          - HEALTHCHECK passing means "the container is alive and the
            web UI is reachable". A user who hits the URL right now
            will see *something* useful (the wizard, a progress bar,
            or an actionable error), not a connection refused.
          - 503 would mean the dashboard server crashed. We don't
            currently return 503; the FastAPI app either responds
            (server alive) or doesn't (server dead → connection
            refused → Docker marks unhealthy via the connection
            failure, not via the HTTP code).

        Task 7a of docs/ease-of-use-tasks.md.
        """
        return JSONResponse({"ready": True})

    @app.get("/status")
    def get_status(request: Request) -> JSONResponse:
        # Lifecycle phase (NEEDS_SETUP/DOWNLOADING/LOADING/READY/ERROR) drives
        # wizard-vs-dashboard routing + the progress/loading screens. Always
        # available, even before the assistant exists.
        payload = lifecycle.snapshot()
        payload["auth_enabled"] = bool(context.dashboard_password_hash)
        # When the dashboard is serving over TLS, surface the URL the user
        # is currently accessing (built from the live request's scheme +
        # Host header — works for localhost, LAN IPs, and custom hostnames).
        # The setup wizard renders this as a banner with the cert-warning
        # copy + a copy button; HTTP installs omit the field entirely.
        cfg = _read_config_cached(context.config_path)
        general = cfg.get("general") or {}
        cert = general.get("dashboard_ssl_certfile")
        key = general.get("dashboard_ssl_keyfile")
        if (
            cert
            and key
            and Path(cert).is_file()
            and Path(key).is_file()
        ):
            payload["dashboard_url"] = f"{request.url.scheme}://{request.url.netloc}"
        if context.downloader is not None and context.downloader.active:
            payload["download"] = context.downloader.snapshot()
        if context.assistant is None or not lifecycle.is_ready():
            payload.update(
                {
                    "state": "idle",
                    "mic_enabled": False,
                    "last_utterance": "",
                    "last_response": "",
                    # Tail of first-party logs so the loading screen can render a
                    # live terminal of what's happening instead of a bare spinner.
                    "log": LOG_BUFFER.tail(),
                }
            )
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
        # Busy-status surface (#13): how many satellites are connected (the
        # "dashboard-text" pseudo-satellite doesn't count — it isn't a real
        # connection) and, if a turn is in flight, which one owns it. The UI
        # uses this to show "busy — talking to <label>" on every satellite
        # except the one currently active, instead of a silent stall.
        owner_id = context.assistant._turn_arbiter.owner
        owner_sat = context.assistant.satellites.get(owner_id) if owner_id else None
        payload.update(
            {
                "state": context.assistant.get_state(),
                "mic_enabled": context.assistant.audio_capture.mic_globally_enabled,
                "last_utterance": last_utterance,
                "last_response": last_response,
                # Remote-LLM mode: the LLM is off-device; the UI swaps the character
                # + shows an "not fully local" note in the tagline.
                "remote_llm": _is_remote_llm(),
                # True when that off-device LLM is configured but unreachable — the UI
                # shows a red banner that we're degraded to regex/fast-path only.
                "llm_unreachable": _is_remote_llm()
                and bool(getattr(context.assistant, "remote_llm_unreachable", False)),
                "satellite_count": sum(
                    1 for sid in context.assistant.satellites if sid != "dashboard-text"
                ),
                "active_owner_id": owner_id,
                "active_owner_label": (
                    (owner_sat.label or owner_sat.ha_area_name) if owner_sat is not None else None
                ),
                # A2: the last completed turn's latency breakdown (endpoint-wait,
                # ASR/LLM/TTS seconds, route, endpoint kind) — None until the
                # first turn finishes. Seeds the deferred turn-trace dashboard.
                "last_turn_stats": context.assistant._last_turn_stats,
            }
        )
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

    @app.post("/replay")
    def replay(req: SpeakRequest) -> dict:
        _require_ready()
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        threading.Thread(
            target=context.assistant.speak_proactive,
            kwargs={"text": text, "emit_event": False},
            daemon=True,
        ).start()
        return {"ok": True}

    @app.post("/mic")
    def set_mic(req: MicRequest) -> dict:
        # Genuinely global (HA switch entity, custom_components/fulloch/switch.py):
        # mutes/unmutes every connected satellite at once. Distinct from each
        # satellite's own SatelliteSession.transcribing, the internal
        # half-duplex self-mute during that satellite's own reply.
        _require_ready()
        context.assistant.audio_capture.mic_globally_enabled = req.enabled
        return {"ok": True, "mic_enabled": req.enabled}

    @app.post("/stop")
    def stop_turn(req: StopRequest = StopRequest()) -> dict:  # noqa: B008
        # Complete, silent stop: aborts the SLM/TTS of whatever turn is in
        # flight (voice or text) and stands down without a follow-up window.
        # req defaults to an empty body — the dashboard's Stop button sends
        # no JSON today, and request_stop()'s own fallback (the arbiter's
        # current owner) covers that case.
        _require_ready()
        context.assistant.request_stop(req.satellite_id)
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

    @app.get("/ha/areas")
    def ha_areas() -> JSONResponse:
        # Backs the browser satellite's area picker (6b) — buttons only render
        # when HA is actually configured; a thin/native satellite client
        # configures its area via YAML instead and never calls this.
        _require_ready()
        from tools._config import config

        if "home_assistant" not in config:
            return JSONResponse({"available": False, "areas": []})
        from tools import home_assistant as ha

        return JSONResponse({"available": True, "areas": ha.list_areas()})

    # --- Setup / settings console (work with or without an assistant) ------
    # These are NOT _require_ready-gated: they drive first-run setup (no
    # assistant yet) and stay available afterwards as the settings console.

    @app.post("/setup/timezone")
    def setup_timezone(req: TimezoneRequest) -> JSONResponse:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(req.tz)
        except (ZoneInfoNotFoundError, KeyError) as e:
            raise HTTPException(status_code=422, detail=f"Unknown timezone: {req.tz!r}") from e
        from utils.local_time import set_tz

        from .config_store import update_config

        update_config({"general.timezone": req.tz}, context.config_path)
        set_tz(req.tz)
        return JSONResponse({"ok": True})

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
        if any(a["path"] == "notes.path" for a in applied):
            from tools.notes_root import refresh_notes_root

            refresh_notes_root()
        # Dashboard preferences are read when the dashboard page is served;
        # they do not need an assistant reload to take effect.
        hot.update({
            a["path"] for a in applied
            if a["path"] in {"general.dashboard_theme", "general.dashboard_show_turn_details"}
        })
        restart = any(a["path"] not in hot for a in applied)
        # Drop the coerced value from the response (may not be JSON-trivial; the
        # UI only needs path + whether a restart is still required).
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

    @app.post("/setup/preflight-download")
    def setup_preflight_download() -> JSONResponse:
        """Run the blocking preflight (disk + network + GPU) for the wizard.

        Called on the "Start download" click so the user gets a clear
        pass/fail with one structured error per failed check before any
        model bytes flow. `ok` is True only when every check passes.
        Task 3 of docs/ease-of-use-tasks.md.
        """

        from .config_store import read_config
        from .preflight import (
            check_disk_for_models,
            check_gpu_for_models,
            check_network,
        )

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
                try:
                    _reset_marker_path().unlink(missing_ok=True)
                except OSError:
                    pass
                context.lifecycle.set(LOADING, "starting assistant")
                if context.assistant is not None:
                    _schedule_restart()
                else:
                    context.lifecycle.signal_proceed()
            else:
                snap = context.downloader.snapshot()
                context.lifecycle.set(ERROR, snap.get("error") or "download failed")

        context.lifecycle.set(DOWNLOADING, "downloading models")
        context.downloader.start(assets, on_complete=_done)
        return JSONResponse({"started": True})

    @app.post("/setup/reset")
    def setup_reset() -> JSONResponse:
        """Arm a re-run of the setup wizard on the next start.

        Backs up the user's state (config, credentials, obsidian override,
        voice denylist, voice clones) into a timestamped folder under
        data/backups/, then drops the reset marker that detect_setup_state
        honours. On restart, the wizard re-runs; credentials are reused from
        disk where the user already filled them in (they aren't wiped). Models
        and certs are NOT backed up — they're re-creatable on demand. The
        user can restore from any backup via /setup/backups/restore.
        """
        data_dir = Path(context.config_path).resolve().parent
        backup_dir = _create_backup(data_dir)
        _reset_marker_path().write_text("setup reset requested\n")
        logger.info("Setup reset armed; backup at %s", backup_dir)
        return JSONResponse(
            {
                "ok": True,
                "restart_required": True,
                "backup": backup_dir.name,
            }
        )

    @app.get("/setup/backups")
    def setup_list_backups() -> dict:
        """List available timestamped backups under data/backups/."""
        data_dir = Path(context.config_path).resolve().parent
        return {"backups": _list_backups(data_dir)}

    @app.post("/setup/backups/restore")
    def setup_restore_backup(req: dict) -> dict:
        """Restore a previous backup by name (e.g. "2026-07-02T120000").

        Copies each backed-up file back to its original location, overwriting.
        Does NOT restart — the user can review and restart manually.
        """
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
        """Enable or force-regenerate the dashboard's self-signed HTTPS cert.

        No cert configured yet: generates a fresh pair under data/certs and
        wires the paths into config.yml (same outcome as core.bootstrap's
        first-run seed, but via update_config since this is a live settings
        save rather than a fresh template — comments in config.yml are
        already stripped on any settings-console save, so no regression
        there). Cert already configured: overwrites the pair in place —
        unlike the idempotent startup path in core.bootstrap, so every
        device that already trusted the old cert will see the browser's
        "not private" warning again next visit. Useful after the LAN IP
        changes and the old cert's SANs go stale. Either way, a restart is
        needed to pick up the new files (uvicorn only reads them at
        startup).
        """
        from core.tls_certs import regenerate_self_signed_cert

        from .config_store import read_config, update_config

        cfg = read_config(context.config_path)
        general = cfg.get("general") or {}
        cert_path = general.get("dashboard_ssl_certfile")
        key_path = general.get("dashboard_ssl_keyfile")
        certs_dir = (
            str(Path(cert_path).parent)
            if cert_path
            else str(Path(context.config_path).parent / "certs")
        )

        try:
            new_cert, new_key = regenerate_self_signed_cert(certs_dir)
            if not cert_path or not key_path:
                update_config(
                    {
                        "general.dashboard_ssl_certfile": new_cert,
                        "general.dashboard_ssl_keyfile": new_key,
                    },
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

        return JSONResponse(
            test_connection(
                base_url=_normalize_llm_url(req.base_url),
                model=req.model,
                api_key=req.api_key or "",
            )
        )

    @app.post("/setup/test-ha")
    def setup_test_ha(req: HaTestRequest) -> JSONResponse:
        import urllib.error
        import urllib.request

        from .credentials_store import get_credential

        url = req.url.rstrip("/") if req.url else ""
        if not url:
            return JSONResponse({"ok": False, "error": "No URL provided"})
        # Blank token in the request means "keep the saved one" — the wizard
        # masks an already-set token behind a placeholder rather than
        # re-displaying the secret, so it submits "" unless the user retypes it.
        token = req.token or get_credential("ha_token")
        try:
            r = urllib.request.Request(
                f"{url}/api/",
                headers={"Authorization": f"Bearer {token}"} if token else {},
            )
            with urllib.request.urlopen(r, timeout=5) as resp:
                return JSONResponse({"ok": resp.status == 200, "status": resp.status})
        except urllib.error.HTTPError as e:
            return JSONResponse({"ok": False, "error": f"HTTP {e.code}"})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.post("/setup/test-path")
    def setup_test_path(req: PathTestRequest) -> JSONResponse:
        from pathlib import Path

        p = Path(req.path) if req.path else None
        return JSONResponse({"ok": bool(p and p.exists())})

    @app.post("/setup/list-llm-models")
    def setup_list_llm_models(req: LlmModelsRequest) -> JSONResponse:
        # Offer a model picker after a successful test connection (GET /v1/models)
        # instead of free-text entry. Available in setup mode (no assistant yet).
        from core.llm_openai import list_models

        return JSONResponse(
            list_models(
                base_url=_normalize_llm_url(req.base_url),
                api_key=req.api_key or "",
            )
        )

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

    @app.get("/login")
    def login_page() -> Response:
        if not context.dashboard_password_hash:
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/", status_code=303)
        return Response(content=_LOGIN_HTML, media_type="text/html")

    class LoginRequest(BaseModel):
        password: str

    @app.post("/auth/login")
    def auth_login(req: LoginRequest, response: Response) -> dict:
        import time

        from .auth import (
            COOKIE_MAX_AGE,
            SESSION_COOKIE,
            new_session_id,
            save_sessions,
            verify_password,
        )

        pw_hash = context.dashboard_password_hash
        if not pw_hash:
            return {"ok": True}
        if not verify_password(req.password, pw_hash):
            raise HTTPException(status_code=401, detail="incorrect password")
        sid = new_session_id()
        context.sessions[sid] = time.time()
        save_sessions(context.sessions)
        response.set_cookie(
            SESSION_COOKIE, sid,
            max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax", path="/",
        )
        return {"ok": True}

    @app.post("/auth/logout")
    def auth_logout(request: Request, response: Response) -> dict:
        from .auth import SESSION_COOKIE, save_sessions
        sid = request.cookies.get(SESSION_COOKIE, "")
        if sid:
            context.sessions.pop(sid, None)
            save_sessions(context.sessions)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    class SetupPasswordRequest(BaseModel):
        password: Optional[str] = None
        name: Optional[str] = None

    @app.post("/setup/password")
    def setup_set_password(req: SetupPasswordRequest) -> JSONResponse:
        from .auth import hash_password
        from .credentials_store import set_credential

        name = (req.name or "").strip()
        password = (req.password or "").strip()

        if name:
            try:
                from tools.notes import remember_fact
                remember_fact(f"The user's name is {name}")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not save user name fact: %s", e)

        if password:
            if len(password) < 8:
                raise HTTPException(status_code=422, detail="password must be at least 8 characters")
            pw_hash = hash_password(password)
            try:
                set_credential("dashboard_password", pw_hash)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"could not write credentials: {e}") from e
            context.dashboard_password_hash = pw_hash
            logger.info("Dashboard password set")

        return JSONResponse({"ok": True})

    class SetupCredentialRequest(BaseModel):
        key: str
        value: str

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
        # Live-reload without restart where possible.
        if req.key == "ha_token":
            try:
                import tools.home_assistant as _ha
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
                        event = await asyncio.to_thread(q.get, True, SUBSCRIBER_IDLE_KEEPALIVE_S)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws/obsidian")
    async def obsidian_ws(ws: WebSocket):
        """Obsidian plugin bridge.

        Plugin → server:
          {"type":"vault_metadata","vault_path":"...","files":[...],"daily_notes":{...}}
          {"type":"context","file":{"path":"...","name":"...","tags":[...],"links":[...],"backlinks":[...]}}
          {"type":"file_changed","path":"<vault-relative>"}
          {"type":"pong"}

        Server → plugin:
          {"type":"ping"}
          {"type":"open_file","path":"<absolute-path>"}
          {"type":"insert","text":"...","file":"..."}
          {"type":"vault_rejected","reason":"not_a_vault"|"unreadable"|"missing"}
        """
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
                        # The Obsidian plugin runs on the host, so it sends
                        # host paths. When Fulloch is in Docker, those paths
                        # need to be remapped to the in-container mount via
                        # `obsidian.path_translation` in config.yml.
                        from tools.notes_root import translate_vault_path
                        translated = translate_vault_path(raw_path)
                        resolved = translated.expanduser().resolve()
                        if not (resolved / ".obsidian").is_dir():
                            try:
                                await ws.send_json(
                                    {"type": "vault_rejected", "reason": "not_a_vault"}
                                )
                            except Exception:
                                return
                            context.obsidian_vault_state["last_error"] = "not_a_vault"
                            logger.warning("Plugin reported non-vault path: %s", resolved)
                            continue
                        if not resolved.is_dir():
                            try:
                                await ws.send_json(
                                    {"type": "vault_rejected", "reason": "unreadable"}
                                )
                            except Exception:
                                return
                            context.obsidian_vault_state["last_error"] = "unreadable"
                            continue
                        context.obsidian_vault_state["connected"] = True
                        context.obsidian_vault_state["vault_path"] = str(resolved)
                        context.obsidian_vault_state["vault_resolved_path"] = str(resolved)
                        context.obsidian_vault_state["last_connected_at"] = time.time()
                        context.obsidian_vault_state["last_error"] = None
                        logger.info(
                            "Obsidian vault adopted: %s (%d files)",
                            resolved,
                            len(data.get("files") or []),
                        )
                    elif kind == "context":
                        context.assistant.set_vault_context(current_file=data.get("file"))
                    elif kind == "file_changed":
                        rel = data.get("path")
                        if not rel:
                            continue
                        try:
                            # Notes storage stays configured through notes.path;
                            # metadata is editor state only. Fall back to that
                            # root when a plugin sends a change before metadata.
                            vault = context.obsidian_vault_state.get("vault_resolved_path")
                            if not vault:
                                from tools.notes_root import get_notes_root

                                vault = get_notes_root()
                            full = Path(vault) / rel if vault else None
                            if full and full.is_file():
                                import threading as _th
                                _th.Thread(
                                    target=_safe_reindex,
                                    args=(full,),
                                    daemon=True,
                                ).start()
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
            _done, pending = await asyncio.wait(
                [recv_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )
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
        """Return the current obsidian_token, generating one if missing.

        Used by the dashboard's "Connect Obsidian" modal to display the
        current token. Distinct from /regenerate-token (which rotates it).
        """
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
        # In Docker, the path the user pastes is the host path; remap to the
        # in-container path before validation so the wizard works regardless
        # of how the vault is mounted.
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
        """Serve the pre-built plugin zip from obsidian-plugin/fulloch.zip.

        The zip is committed in the repo so a fresh install has it available
        immediately, no release download required.
        """
        zip_path = _SERVER_DIR.parent / "obsidian-plugin" / "fulloch.zip"
        if not zip_path.is_file():
            raise HTTPException(status_code=404, detail="plugin zip not built yet")
        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename="fulloch.zip",
        )

    return app


def start_dashboard(
    assistant=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    lifecycle: Optional[Lifecycle] = None,
    context: Optional[AppContext] = None,
    http_redirect_port: Optional[int] = None,
) -> threading.Thread:
    """Launch the dashboard on a daemon thread. Non-blocking.

    Pass both ``ssl_certfile`` and ``ssl_keyfile`` to serve over HTTPS (uvicorn
    terminates TLS). If only one is given, or a file is missing, TLS is skipped
    and the dashboard falls back to HTTP with a loud warning.

    When HTTPS is enabled, a tiny TLS-sniffing dispatcher is started on the
    *public* port. It accepts raw TCP, peeks the first byte of every new
    connection, and either:
      - proxies through to a localhost-bound uvicorn (TLS backend), or
      - sends a single 308 redirect to ``https://{host}:{port}{path}``
    so a user typing ``http://`` lands on the TLS dashboard without a manual
    URL edit. The redirect, the TLS listener, and the FastAPI app all share
    the single configured port — no second port mapping required.

    The ``http_redirect_port`` argument is accepted for backward compatibility
    but ignored; the dispatcher is on the same port as the dashboard.

    Prefer a shared ``context`` (so the assistant can attach after setup); a
    bare ``assistant`` (may be None) + ``lifecycle`` still works.
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
        # Bind uvicorn (TLS) on 127.0.0.1, then front it with a dispatcher on
        # the public port. The OS picks an ephemeral internal port to avoid
        # colliding with anything the user might have on the chosen offset.
        backend_port = _pick_free_local_port()
        uvicorn_config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=backend_port,
            log_level="warning",
            access_log=False,
            # ESP32 WebSocket clients use the satellite protocol heartbeat for
            # liveness; Uvicorn's transport ping timeout disconnects them.
            ws_ping_interval=None,
            **ssl_kwargs,
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
        uvicorn_thread = threading.Thread(
            target=uvicorn_server.run, daemon=True, name="dashboard-uvicorn"
        )
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

    # No TLS — uvicorn binds the public port directly. No redirect needed.
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


def _pick_free_local_port() -> int:
    """Ask the OS for an ephemeral localhost port (port 0 → OS-assigned)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_TLS_HANDSHAKE_BYTE = 0x16  # first byte of a TLS ClientHello record


async def _dispatch_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    backend_host: str,
    backend_port: int,
    https_port: int,
) -> None:
    """Peek the first byte of a new connection; route to TLS proxy or 308.

    Module-level (rather than nested in start_tls_dispatcher) so tests can
    exercise the routing logic without binding a socket.

    Security note for future maintainers
    ------------------------------------
    This is an in-process port multiplexer — the same idea as sslh / hitch /
    HAProxy's `req_ssl_hello_type 1` check, but inside the Python process
    rather than as a separate reverse proxy. The protocol handling is
    standard (the 0x16 first byte is the IANA TLS record-type byte, peeked
    with MSG_PEEK so the kernel's socket buffer is untouched).

    For Fulloch's single-LAN-user use case that's fine. The classic
    cross-cutting risks for this pattern are resource-exhaustion rather than
    protocol ones — see sslh's security review by Matthias Gerstner
    (OpenSUSE, 2025) for the canonical writeup. Concretely, if this is ever
    exposed to anything beyond the home LAN, add:
      - a per-connection read timeout (a client that opens a TCP connection
        and never sends the first byte currently holds the dispatcher slot
        until the OS's TCP keepalive eventually reaps it — several minutes);
      - a cap on concurrent in-flight connections, with oldest dropped.
    A reverse-proxy deployment (Caddy / nginx / Traefik in front) sidesteps
    both concerns and is the right move for any multi-tenant exposure.
    """
    try:
        sock_info = writer.transport.get_extra_info("socket")
        if sock_info is None:
            return
        # asyncio (and uvloop) wraps the raw socket in a TransportSocket
        # that doesn't expose recv() directly. The peek has to go through
        # the underlying fd: dup it so closing our peek-socket doesn't
        # tear down the asyncio transport's socket, and do the MSG_PEEK
        # via the stdlib socket API. The kernel buffer is shared with the
        # asyncio reader, so a peek here doesn't consume the byte — the
        # TLS ClientHello or HTTP request line is still available when
        # we hand the reader off to the backend.
        raw_fd = sock_info.fileno()
        peek_sock = socket.socket(fileno=os.dup(raw_fd))
        try:
            peek_sock.setblocking(False)
            try:
                first = peek_sock.recv(1, socket.MSG_PEEK)
            except (BlockingIOError, InterruptedError, OSError):
                # No first byte yet — TCP connection is open but idle.
                # Pipe to the backend; if the client eventually sends
                # a TLS ClientHello the backend handles it, and if they
                # never send anything the kernel eventually reaps the
                # connection.
                await _pipe_to_backend(reader, writer, backend_host, backend_port)
                return
        finally:
            peek_sock.close()

        if first and first[0] == _TLS_HANDSHAKE_BYTE:
            await _pipe_to_backend(reader, writer, backend_host, backend_port)
        else:
            await _send_308(reader, writer, https_port)
    except Exception as e:  # noqa: BLE001 — connection-scoped; just close
        logger.debug("dispatcher connection error: %s", e)
    finally:
        try:
            writer.close()
        except Exception:
            pass


def start_tls_dispatcher(
    public_host: str, public_port: int, backend_host: str, backend_port: int
) -> threading.Thread:
    """Start a TLS-sniffing TCP dispatcher that fronts a TLS uvicorn backend.

    The dispatcher binds ``(public_host, public_port)`` and accepts raw TCP.
    For each connection it peeks the first byte (without consuming it):
      - ``0x16`` (TLS ClientHello) → open a localhost connection to the
        uvicorn backend, pipe bytes both ways until either side closes.
      - anything else → read the HTTP request, send a 308 to
        ``https://{host}:{public_port}{path}``, close.

    The peeked byte is not consumed, so the byte is still in the socket
    buffer when we hand the connection off — TLS Clients and HTTP request
    lines both stay intact.

    The dispatcher runs on its own asyncio event loop in a daemon thread so
    the rest of the process (FastAPI handlers, ASGI) is unaffected.
    """
    loop = asyncio.new_event_loop()

    async def _serve() -> None:
        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _dispatch_connection(
                reader, writer, backend_host, backend_port, public_port
            )

        server = await asyncio.start_server(_handle, public_host, public_port)
        async with server:
            await server.serve_forever()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve())
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True, name="dashboard-tls-dispatcher")
    thread.start()
    return thread


async def _pipe_to_backend(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    backend_host: str,
    backend_port: int,
) -> None:
    """Open a localhost connection to the TLS backend and bidirectionally pipe bytes."""
    try:
        backend_reader, backend_writer = await asyncio.open_connection(backend_host, backend_port)
    except Exception:
        return

    async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        finally:
            try:
                dst.close()
            except Exception:
                pass

    try:
        await asyncio.gather(
            _pump(client_reader, backend_writer),
            _pump(backend_reader, client_writer),
        )
    finally:
        for w in (client_writer, backend_writer):
            try:
                w.close()
            except Exception:
                pass


async def _send_308(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    https_port: int,
) -> None:
    """Read a plain-HTTP request and respond with a 308 to https://{host}:{port}{path}.

    Parses just enough of the request to build a sensible Location header —
    the request line for the path, and the ``Host:`` header for the hostname.
    Anything malformed gets a best-effort 308 to the bare origin.
    """
    try:
        data = await _read_until_double_crlf(reader)
        request_line, host = _parse_request_host(data)
        path = request_line or "/"
        location = f"https://{host}:{https_port}{path}"
    except Exception:
        location = f"https://{host}:{https_port}/"

    response = (
        b"HTTP/1.1 308 Permanent Redirect\r\n"
        b"Location: " + location.encode("ascii", errors="replace") + b"\r\n"
        b"Content-Length: 0\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    try:
        writer.write(response)
        await writer.drain()
    except Exception:
        pass


async def _read_until_double_crlf(reader: asyncio.StreamReader, max_bytes: int = 16384) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < max_bytes:
        chunk = await reader.read(1024)
        if not chunk:
            break
        buf += chunk
    return buf


def _parse_request_host(data: bytes) -> tuple[str, str]:
    """Best-effort: return (path, hostname) from a partial HTTP request.

    ``hostname`` is the inbound Host header with the port stripped — that's
    what we want in the redirect Location so a user typing
    ``http://192.168.1.10:8765/…`` lands on
    ``https://192.168.1.10:8765/…``.
    """
    if not data:
        return "/", "localhost"
    try:
        head = data.split(b"\r\n\r\n", 1)[0]
    except Exception:
        return "/", "localhost"
    lines = head.split(b"\r\n")
    path = "/"
    if lines:
        try:
            parts = lines[0].decode("latin-1", errors="replace").split(" ", 2)
            if len(parts) >= 2 and parts[1].startswith("/"):
                path = parts[1]
        except Exception:
            pass
    host = "localhost"
    for line in lines[1:]:
        if line.lower().startswith(b"host:"):
            try:
                host_header = line.split(b":", 1)[1].strip().decode("latin-1", errors="replace")
                host = host_header.split(":", 1)[0] or host
            except Exception:
                pass
            break
    return path, host
