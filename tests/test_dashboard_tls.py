"""Dashboard HTTPS pass-through (optional uvicorn TLS).

start_dashboard forwards a cert/key pair to uvicorn.Config only when BOTH are
set and exist on disk; otherwise it falls back to plain HTTP. These tests patch
uvicorn so no socket is ever bound.
"""

from unittest.mock import MagicMock

import server.dashboard as dashboard


def _capture_config(monkeypatch):
    """Patch uvicorn so Config kwargs are captured and Server.run() is a no-op."""
    captured = {}

    def fake_config(app, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    fake_server = MagicMock()
    fake_server.return_value.run = MagicMock()

    monkeypatch.setattr(dashboard.uvicorn, "Config", fake_config)
    monkeypatch.setattr(dashboard.uvicorn, "Server", fake_server)
    return captured


def _stub_assistant():
    a = MagicMock()
    a.register_turn_listener = MagicMock()
    return a


def test_tls_enabled_when_both_files_exist(monkeypatch, tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x")
    key.write_text("x")
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    assert captured.get("ssl_certfile") == str(cert)
    assert captured.get("ssl_keyfile") == str(key)


def test_tls_skipped_when_only_one_key_set(monkeypatch, tmp_path):
    cert = tmp_path / "cert.pem"
    cert.write_text("x")
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=None,
    )

    assert "ssl_certfile" not in captured
    assert "ssl_keyfile" not in captured


def test_tls_skipped_when_files_missing(monkeypatch, tmp_path):
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(tmp_path / "nope.pem"),
        ssl_keyfile=str(tmp_path / "nokey.pem"),
    )

    assert "ssl_certfile" not in captured
    assert "ssl_keyfile" not in captured


def test_plain_http_by_default(monkeypatch):
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(_stub_assistant(), host="127.0.0.1", port=8765)

    assert "ssl_certfile" not in captured
    assert "ssl_keyfile" not in captured
