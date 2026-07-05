"""Dashboard password/session auth.

Verifies that a configured dashboard password gates the sensitive routes,
the login flow sets a session cookie, and that an unset password disables
auth entirely (the zero-config local-only path).
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.auth import SESSION_COOKIE, hash_password, new_session_id
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle


def _stub_assistant():
    assistant = MagicMock()
    assistant.register_turn_listener = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.transcribing = True
    assistant.wakeword = "hey atticus"
    assistant._history = []
    return assistant


def _ctx(pw_hash=None):
    ctx = AppContext(lifecycle=Lifecycle(phase=READY))
    ctx.dashboard_password_hash = pw_hash
    return ctx


def test_no_password_means_no_auth():
    client = TestClient(create_app(_stub_assistant(), context=_ctx()))
    assert client.get("/status").status_code == 200
    assert client.get("/history").status_code == 200


def test_password_blocks_unauthenticated_api_requests():
    client = TestClient(
        create_app(_stub_assistant(), context=_ctx(hash_password("testpass"))),
        follow_redirects=False,
    )
    assert client.get("/status").status_code == 401
    assert client.get("/history").status_code == 401
    assert client.get("/notes").status_code == 401


def test_login_page_is_exempt():
    client = TestClient(
        create_app(_stub_assistant(), context=_ctx(hash_password("testpass"))),
        follow_redirects=False,
    )
    assert client.get("/login").status_code == 200


def test_login_with_correct_password_sets_session_cookie():
    ctx = _ctx(hash_password("correctpass"))
    client = TestClient(create_app(_stub_assistant(), context=ctx))
    r = client.post("/auth/login", json={"password": "correctpass"})
    assert r.status_code == 200
    assert SESSION_COOKIE in r.cookies


def test_login_with_wrong_password_is_rejected():
    ctx = _ctx(hash_password("correctpass"))
    client = TestClient(create_app(_stub_assistant(), context=ctx))
    assert client.post("/auth/login", json={"password": "wrong"}).status_code == 401


def test_valid_session_grants_access():
    ctx = _ctx(hash_password("pass"))
    sid = new_session_id()
    ctx.sessions[sid] = True
    client = TestClient(create_app(_stub_assistant(), context=ctx))
    assert client.get("/status", cookies={SESSION_COOKIE: sid}).status_code == 200


def test_invalid_session_is_rejected():
    ctx = _ctx(hash_password("pass"))
    client = TestClient(create_app(_stub_assistant(), context=ctx))
    assert client.get("/status", cookies={SESSION_COOKIE: "bogus"}).status_code == 401


def test_logout_removes_session():
    ctx = _ctx(hash_password("pass"))
    sid = new_session_id()
    ctx.sessions[sid] = True
    client = TestClient(create_app(_stub_assistant(), context=ctx))
    r = client.post("/auth/logout", cookies={SESSION_COOKIE: sid})
    assert r.status_code == 200
    assert sid not in ctx.sessions


def test_login_page_redirects_to_root_when_no_password_set():
    # /login with no password configured → redirect to /
    client = TestClient(
        create_app(_stub_assistant(), context=_ctx(None)),
        follow_redirects=False,
    )
    r = client.get("/login")
    assert r.status_code == 303
    assert r.headers["location"] == "/"
