"""End-to-end password-gated provisioning and satellite confirmation."""

import queue
import time
from unittest.mock import MagicMock

import pytest
from cryptography import x509
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server import config_store, satellite_waveshare
from server.auth import SESSION_COOKIE
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle
from tests.test_satellite_waveshare import _bundle, _certificate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    assistant = MagicMock()
    assistant.connect_satellite.return_value = queue.Queue()
    assistant.satellites = {}
    context = AppContext(
        Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml")
    )
    context.dashboard_password_hash = "enabled"
    context.sessions = {"session": time.time()}
    tokens = []

    def generate(config, credentials):
        satellite_waveshare.validate_bundle(config, credentials)
        tokens.append(credentials["fulloch"]["device_token"])
        return b"private-nvs"

    monkeypatch.setattr(satellite_waveshare, "generate_nvs", generate)
    cert = _certificate([x509.DNSName("device.example")])
    config, credentials = _bundle("device.example", cert)
    del credentials["fulloch"]
    with TestClient(create_app(context=context)) as client:
        client.cookies.set(SESSION_COOKIE, "session")
        client.headers["x-fulloch-provisioning"] = "1"
        yield client, context, tokens, {"config": config, "credentials": credentials}


def hello(device, token):
    return {
        "type": "satellite.hello",
        "token": token,
        "protocol": {"name": "satellite-v2", "major": 2, "minor": 4},
        "device": {"id": device},
        "capabilities": {},
    }


BASE = "/api/satellite-provisioning"


def test_automatic_setup_allocates_identity_and_browser_timezone(setup, monkeypatch):
    client, context, _, payload = setup
    from server import satellite_setup_discovery

    monkeypatch.setattr(satellite_setup_discovery, "posix_timezone", lambda name: "EST5EDT,M3.2.0,M11.1.0")
    captured = []
    monkeypatch.setattr(satellite_waveshare, "generate_nvs", lambda config, credentials: captured.append(config.copy()) or b"nvs")
    config_store.mutate_config(lambda doc: doc.update({
        "satellite_token_owners": {"kitchen-01": "existing"}, "satellite_tokens": ["existing"]
    }), context.config_path)
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    payload["auto_setup"] = True
    payload["config"]["device"] = {"id": "", "name": "Kitchen", "timezone": "America/New_York"}
    response = client.post(BASE + "/waveshare/transactions", json=payload)
    assert response.status_code == 200, response.text
    assert captured[0]["device"]["id"] == "kitchen-02"
    assert captured[0]["device"]["timezone"] == "EST5EDT,M3.2.0,M11.1.0"


def test_dashboard_ca_provisioned_automatically(setup, monkeypatch, tmp_path):
    from scripts.create_local_ca import issue_certificates

    client, _, _, payload = setup
    ca_path, leaf_path, _ = issue_certificates(tmp_path / "certs", ["device.example"], [], False)
    client.app.state.satellite_dashboard_certificate = leaf_path.read_text()
    client.app.state.satellite_dashboard_ca_path = ca_path
    payload["config"]["server"]["certificate_pem"] = ""
    captured = []

    def generate(config, credentials, *, identity_certificate=None):
        satellite_waveshare.validate_bundle(
            config, credentials, identity_certificate=identity_certificate
        )
        captured.append((config["server"]["certificate_pem"], identity_certificate))
        return b"nvs"

    monkeypatch.setattr(satellite_waveshare, "generate_nvs", generate)
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    result = client.post(BASE + "/waveshare/transactions", json=payload)
    assert result.status_code == 200, result.text
    assert captured == [(ca_path.read_text(), leaf_path.read_text())]


def test_dashboard_self_signed_provisioned_automatically(setup, monkeypatch, tmp_path):
    from core.tls_certs import ensure_self_signed_cert

    client, _, _, payload = setup
    path, _ = ensure_self_signed_cert(str(tmp_path / "certs"))
    from pathlib import Path

    cert = Path(path).read_text()
    client.app.state.satellite_dashboard_certificate = cert
    payload["config"]["server"].update(url="wss://127.0.0.1/ws/satellite-v2", certificate_pem="")
    captured = []
    monkeypatch.setattr(
        satellite_waveshare,
        "generate_nvs",
        lambda config, credentials: captured.append(config["server"]["certificate_pem"]) or b"nvs",
    )
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    result = client.post(BASE + "/waveshare/transactions", json=payload)
    assert result.status_code == 200, result.text
    assert captured == [cert]


def test_dashboard_ca_missing_fails_before_flashing(setup, tmp_path):
    from scripts.create_local_ca import issue_certificates

    client, _, tokens, payload = setup
    _, leaf_path, _ = issue_certificates(tmp_path / "certs", ["device.example"], [], False)
    client.app.state.satellite_dashboard_certificate = leaf_path.read_text()
    payload["config"]["server"]["certificate_pem"] = ""
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    result = client.post(BASE + "/waveshare/transactions", json=payload)
    assert result.status_code == 400
    assert "local CA" in result.text
    assert not tokens


def test_automatic_setup_rejects_unsupported_timezone_before_nvs(setup, monkeypatch):
    client, _, tokens, payload = setup
    from server import satellite_setup_discovery

    def reject(name):
        raise ValueError(satellite_setup_discovery.UNSUPPORTED_TIMEZONE)

    monkeypatch.setattr(satellite_setup_discovery, "posix_timezone", reject)
    payload["auto_setup"] = True
    payload["config"]["device"] = {"id": "", "name": "Kitchen", "timezone": "America/Nuuk"}
    response = client.post(BASE + "/waveshare/transactions", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == satellite_setup_discovery.UNSUPPORTED_TIMEZONE
    assert not tokens


def test_transaction_commits_only_on_expected_authenticated_session(setup):
    client, context, tokens, payload = setup
    assert client.post(BASE + "/waveshare/transactions", json=payload).status_code == 400
    assert client.post(BASE + "/migration", json={}).status_code == 400
    assert (
        client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": 1}).status_code
        == 400
    )
    assert (
        client.post(
            BASE + "/migration", json={"confirm_tokenless_reprovisioning": True}
        ).status_code
        == 200
    )
    response = client.post(BASE + "/waveshare/transactions", json=payload)
    assert response.status_code == 200, response.text
    assert tokens[0] not in response.text and "secret" not in response.text
    url = BASE + "/transactions/" + response.json()["transaction_id"]
    assert client.get(url + "/nvs").content == b"private-nvs"
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(hello("wrong", tokens[0]))
        assert ws.receive_json()["code"] == "auth"
    assert client.get(url).json()["state"] == "ready"
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(hello("test", tokens[0]))
        assert ws.receive_json()["type"] == "satellite.welcome"
        assert client.get(url).json()["state"] == "confirmed"
        assert client.get(url + "/nvs").status_code == 404
    config = config_store.read_config(context.config_path)
    assert config["satellite_token_owners"]["test"] == tokens[0]
    assert client.delete(BASE + "/devices/test/token").json()["revoked"]
    assert config_store.read_config(context.config_path)["satellite_token_required"] is True


def test_migration_disconnects_tokenless_but_preserves_existing_token(setup):
    client, context, _, _ = setup
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(hello("test", None))
        assert ws.receive_json()["type"] == "satellite.welcome"
        assert (
            client.post(
                BASE + "/migration", json={"confirm_tokenless_reprovisioning": True}
            ).status_code
            == 200
        )
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    config_store.mutate_config(
        lambda doc: doc.update({"satellite_tokens": ["legacy"]}), context.config_path
    )
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(hello("legacy-device", "legacy"))
        assert ws.receive_json()["type"] == "satellite.welcome"


def test_migration_preserves_an_already_connected_token_bearing_device(setup):
    client, context, _, _ = setup
    config_store.mutate_config(
        lambda doc: doc.update({"satellite_tokens": ["existing-token"]}), context.config_path
    )
    with client.websocket_connect("/ws/satellite-v2") as ws:
        ws.send_json(hello("existing", "existing-token"))
        assert ws.receive_json()["type"] == "satellite.welcome"
        assert (
            client.post(
                BASE + "/migration", json={"confirm_tokenless_reprovisioning": True}
            ).status_code
            == 200
        )
        # A protocol response proves the already-open socket survived migration.
        ws.send_json({"type": "unsupported-test-message"})
        assert ws.receive_json()["code"] == "protocol"
        # A second authenticated connection proves the legacy bearer remains valid.
        with client.websocket_connect("/ws/satellite-v2") as other:
            other.send_json(hello("existing", "existing-token"))
            assert other.receive_json()["type"] == "satellite.welcome"


def test_provisioning_responses_and_logs_do_not_expose_secrets(setup, caplog):
    client, _, tokens, payload = setup
    caplog.set_level("DEBUG")
    payload["credentials"]["wifi"]["password"] = "private-wifi-test-value"
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    response = client.post(BASE + "/waveshare/transactions", json=payload)
    assert response.status_code == 200
    url = BASE + "/transactions/" + response.json()["transaction_id"]
    status = client.get(url)
    cancelled = client.delete(url)
    assert tokens
    for secret in ("private-wifi-test-value", tokens[0], "private-nvs"):
        assert secret not in response.text + status.text + cancelled.text + caplog.text


def test_cancel_and_certificate_error_are_safe(setup):
    client, context, _, payload = setup
    client.post(BASE + "/migration", json={"confirm_tokenless_reprovisioning": True})
    payload["config"]["server"]["url"] = "wss://wrong.example/ws/satellite-v2"
    result = client.post(BASE + "/waveshare/transactions", json=payload)
    assert result.status_code == 400 and "does not match" in result.text
    assert "secret" not in result.text
    payload["config"]["server"]["url"] = "wss://device.example/ws/satellite-v2"
    result = client.post(BASE + "/waveshare/transactions", json=payload)
    url = BASE + "/transactions/" + result.json()["transaction_id"]
    assert client.delete(url).json()["state"] == "cancelled"
    assert client.get(url + "/nvs").status_code == 404
    assert not config_store.read_config(context.config_path)["satellite_token_candidates"]


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("post", BASE + "/migration"),
        ("post", BASE + "/waveshare/transactions"),
        ("get", BASE + "/transactions/id"),
        ("get", BASE + "/transactions/id/nvs"),
        ("delete", BASE + "/transactions/id"),
        ("delete", BASE + "/devices/test/token"),
    ],
)
def test_all_sensitive_routes_require_password_session_and_header(setup, method, url):
    client, context, _, _ = setup
    context.dashboard_password_hash = None
    assert client.request(method, url, json={}).status_code == 403
    context.dashboard_password_hash = "enabled"
    client.cookies.clear()
    assert client.request(method, url, json={}).status_code == 401
    client.cookies.set(SESSION_COOKIE, "session")
    del client.headers["x-fulloch-provisioning"]
    assert client.request(method, url, json={}).status_code == 403
