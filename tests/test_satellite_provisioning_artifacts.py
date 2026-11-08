"""Provisioning artifact endpoints enforce password sessions and origins."""

import time

import pytest
from fastapi.testclient import TestClient

from server import satellite_usb
from server.auth import SESSION_COOKIE
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    context = AppContext(Lifecycle(phase=READY), config_path=str(tmp_path / "config.yml"))
    context.sessions = {"test-session": time.time()}
    return TestClient(create_app(context=context)), context


def test_requires_password_session_header_and_origin(client):
    browser, context = client
    url = "/api/satellite-provisioning/waveshare/manifest"
    headers = {"x-fulloch-provisioning": "1"}
    cookies = {SESSION_COOKIE: "test-session"}
    assert browser.get(url, headers=headers, cookies=cookies).status_code == 403
    context.dashboard_password_hash = "enabled"
    assert browser.get(url, headers=headers).status_code == 401
    assert browser.get(url, cookies=cookies).status_code == 403
    assert (
        browser.get(
            url, headers={**headers, "origin": "https://evil.example"}, cookies=cookies
        ).status_code
        == 403
    )
    response = browser.get(url, headers={**headers, "origin": "http://testserver"}, cookies=cookies)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["board"] == "waveshare_touch_lcd_185c_box_rev2"


def test_only_valid_public_artifacts_are_served(client, monkeypatch):
    browser, context = client
    context.dashboard_password_hash = "enabled"
    headers = {"x-fulloch-provisioning": "1"}
    cookies = {SESSION_COOKIE: "test-session"}
    base = "/api/satellite-provisioning/waveshare/images/"
    response = browser.get(base + "bootloader.bin", headers=headers, cookies=cookies)
    assert response.status_code == 200
    assert response.content == satellite_usb.load_release()[1]["bootloader.bin"]
    assert response.headers["cache-control"] == "no-store"
    assert browser.get(base + "nvs.bin", headers=headers, cookies=cookies).status_code == 404

    def invalid():
        raise satellite_usb.ArtifactError("sensitive path not exposed")

    monkeypatch.setattr(satellite_usb, "load_release", invalid)
    response = browser.get(base + "bootloader.bin", headers=headers, cookies=cookies)
    assert response.status_code == 503
    assert "sensitive path" not in response.text
