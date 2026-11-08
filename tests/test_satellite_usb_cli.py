"""The interactive fallback keeps secrets out of shell arguments and cleans up."""

import hashlib
import io
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from cryptography import x509
from fastapi.testclient import TestClient

from server import config_store, satellite_waveshare
from server.auth import hash_password
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle
from tests.test_satellite_waveshare import _certificate
from tools import satellite_usb_setup as cli


@pytest.mark.parametrize("fail", [False, True])
def test_cli_transaction_and_cleanup(monkeypatch, fail):
    answers = iter(
        [
            "https://device.example:8765",
            "",
            "REPROVISION",
            "kitchen",
            "Kitchen",
            "",
            "wifi",
            "",
            "",
            "",
            "",
            "",
            "/dev/ttyACM0",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    secrets = iter(["dashboard-password", "wifi-password"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: next(secrets))
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    calls = []
    image = b"image"

    class FakeClient:
        def __init__(self, url, ca):
            assert url == "https://device.example:8765"

        def json(self, path, **kwargs):
            calls.append((path, kwargs))
            if path == "/status":
                return {"auth_enabled": True}
            if path.endswith("/waveshare/transactions"):
                return {"transaction_id": "transaction"}
            if path.endswith("/waveshare/manifest"):
                return {
                    "images": [
                        {
                            "file": "application.bin",
                            "size": len(image),
                            "offset": 0x20000,
                            "sha256": hashlib.sha256(image).hexdigest(),
                        }
                    ],
                    "nvs": {"size": 3, "offset": 0x9000},
                }
            if path.endswith("/transactions/transaction"):
                return {"state": "confirmed"}
            return {"ok": True}

        def request(self, path, **kwargs):
            calls.append((path, kwargs))
            return b"nvs" if path.endswith("/nvs") else image

    monkeypatch.setattr(cli, "Client", FakeClient)
    paths = []

    def flash(command, check):
        assert "dashboard-password" not in command and "wifi-password" not in command
        paths.extend(Path(arg) for arg in command if arg.endswith(".bin"))
        assert all(path.exists() for path in paths)
        if fail:
            raise cli.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli.subprocess, "run", flash)
    assert cli.main([]) == (1 if fail else 0)
    assert paths and all(not path.exists() for path in paths)
    assert any(kwargs.get("method") == "DELETE" for _, kwargs in calls) is fail


def test_cli_rejects_untrusted_url_before_login():
    with pytest.raises(ValueError, match="HTTPS"):
        cli.Client("http://device.example", "")


@pytest.fixture
def real_dashboard(tmp_path, monkeypatch):
    """Exercise real auth/routes through urllib's request and response boundary."""
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr("server.auth.save_sessions", lambda *args: None)
    context = AppContext(Lifecycle(phase=READY), config_path=str(tmp_path / "config.yml"))
    context.dashboard_password_hash = hash_password("correct-password")
    context.sessions = {}
    with TestClient(create_app(context=context), base_url="https://testserver") as browser:

        class Opener:
            @contextmanager
            def open(self, request, timeout):
                response = browser.request(
                    request.get_method(),
                    urlsplit(request.full_url).path,
                    content=request.data,
                    headers=dict(request.header_items()),
                )
                if response.status_code >= 400:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        response.status_code,
                        "error",
                        {},
                        io.BytesIO(response.content),
                    )
                yield io.BytesIO(response.content)

        monkeypatch.setattr(cli.urllib.request, "build_opener", lambda *args: Opener())
        yield context


@pytest.mark.parametrize("fail", [False, True])
def test_real_password_session_provision_confirm_and_logout(
    real_dashboard, tmp_path, monkeypatch, fail
):
    cert = _certificate([x509.DNSName("device.example")])
    cert_path = tmp_path / "public-cert.pem"
    cert_path.write_text(cert)
    answers = iter(
        [
            "https://testserver",
            "",
            "REPROVISION",
            "kitchen",
            "Kitchen",
            "",
            "wifi",
            "wss://device.example/ws/satellite-v2",
            str(cert_path),
            "",
            "",
            "",
            "/dev/ttyACM0",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    passwords = iter(["correct-password", "wifi-password"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: next(passwords))
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(satellite_waveshare, "generate_nvs", lambda *args: b"x" * 0x10000)

    def flash(*args, **kwargs):
        config = config_store.read_config(real_dashboard.config_path)
        token = config["satellite_token_candidates"]["kitchen"]["token"]
        if fail:
            raise cli.subprocess.CalledProcessError(1, "esptool")
        assert real_dashboard.satellite_provisioning.confirm("kitchen", token)

    monkeypatch.setattr(cli.subprocess, "run", flash)
    assert cli.main([]) == (1 if fail else 0)
    assert not real_dashboard.sessions
    config = config_store.read_config(real_dashboard.config_path)
    assert not config["satellite_token_candidates"]
    assert ("kitchen" in config.get("satellite_token_owners", {})) is not fail


def test_real_session_revokes_exact_device_and_keeps_enforcement(real_dashboard, monkeypatch):
    config_store.mutate_config(
        lambda doc: doc.update(
            {
                "satellite_token_required": True,
                "satellite_tokens": ["owned", "other"],
                "satellite_token_owners": {"first/room": "owned", "second": "other"},
            }
        ),
        real_dashboard.config_path,
    )
    answers = iter(["https://testserver", "", "first/room", "REVOKE"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: "correct-password")
    assert cli.main(["--revoke"]) == 0
    config = config_store.read_config(real_dashboard.config_path)
    assert config["satellite_tokens"] == ["other"]
    assert config["satellite_token_required"] is True
    assert not real_dashboard.sessions


def test_wrong_password_cannot_mutate_configuration(real_dashboard, monkeypatch, capsys):
    answers = iter(["https://testserver", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: "incorrect")
    assert cli.main(["--revoke"]) == 1
    assert "Check the dashboard password" in capsys.readouterr().err
    assert not config_store.read_config(real_dashboard.config_path)
    assert not real_dashboard.sessions
