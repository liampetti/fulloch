"""Dashboard login, logout and password setup; middleware lives in create_app."""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from .lifecycle import AppContext

logger = logging.getLogger(__name__)

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


class LoginRequest(BaseModel):
    password: str


class SetupPasswordRequest(BaseModel):
    password: Optional[str] = None
    name: Optional[str] = None


def register_auth_routes(app: FastAPI, context: AppContext) -> None:
    """Use the same password hash and sessions that the app middleware checks."""
    @app.get("/login")
    def login_page() -> Response:
        if not context.dashboard_password_hash:
            return RedirectResponse("/", status_code=303)
        return Response(content=_LOGIN_HTML, media_type="text/html")

    @app.post("/auth/login")
    def auth_login(req: LoginRequest, response: Response) -> dict:
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
