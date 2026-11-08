"""Authorization requirements for sensitive provisioning operations."""

import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from server.auth import SESSION_COOKIE
from server.satellite_authorization import authorize


@pytest.fixture
def protected():
    context = SimpleNamespace(
        dashboard_password_hash="password-hash", sessions={"session": time.time()}
    )
    app = FastAPI()

    @app.get("/protected")
    def route(request: Request):
        authorize(request, context)
        return {"ok": True}

    return TestClient(app), context


def test_password_session_and_header_are_required(protected):
    client, context = protected
    headers = {"x-fulloch-provisioning": "1"}
    cookies = {SESSION_COOKIE: "session"}
    assert client.get("/protected", headers=headers).status_code == 401
    assert client.get("/protected", cookies=cookies).status_code == 403
    assert client.get("/protected", headers=headers, cookies=cookies).status_code == 200
    context.dashboard_password_hash = None
    assert client.get("/protected", headers=headers, cookies=cookies).status_code == 403


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "null", "http://testserver:81", "http://testserver/path"]
)
def test_cross_origin_requests_are_rejected(protected, origin):
    client, _ = protected
    response = client.get(
        "/protected",
        headers={"x-fulloch-provisioning": "1", "origin": origin},
        cookies={SESSION_COOKIE: "session"},
    )
    assert response.status_code == 403


def test_same_origin_and_expired_sessions(protected):
    client, context = protected
    headers = {"x-fulloch-provisioning": "1", "origin": "http://testserver"}
    cookies = {SESSION_COOKIE: "session"}
    assert client.get("/protected", headers=headers, cookies=cookies).status_code == 200
    context.sessions["session"] = time.time() - 366 * 86400
    assert client.get("/protected", headers=headers, cookies=cookies).status_code == 401
