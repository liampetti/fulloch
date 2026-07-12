"""`GET /status` includes `dashboard_url` only when TLS is configured.

The setup wizard's first step reads this field to show a banner with
the dashboard URL + the cert-warning copy. HTTP installs omit the
field entirely (the banner is unnecessary when there's no cert to
warn about), so the wizard's `if status.dashboard_url` check doubles
as the TLS-gating condition.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server.dashboard import create_app
from server.lifecycle import NEEDS_SETUP, AppContext, Lifecycle


def _make_cert_files(tmp_path: Path) -> tuple[Path, Path]:
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    cert = cert_dir / "dashboard.crt"
    key = cert_dir / "dashboard.key"
    cert.write_text("placeholder cert")
    key.write_text("placeholder key")
    return cert, key


def _write_config(tmp_path: Path, *, tls: bool = False) -> Path:
    config_path = tmp_path / "config.yml"
    if tls:
        cert, key = _make_cert_files(tmp_path)
        body = textwrap.dedent(f"""\
            general:
              dashboard_ssl_certfile: "{cert}"
              dashboard_ssl_keyfile: "{key}"
            """)
    else:
        body = "general: {}\n"
    config_path.write_text(body)
    return config_path


def _client(tmp_path: Path, *, tls: bool = False) -> TestClient:
    config_path = _write_config(tmp_path, tls=tls)
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    lifecycle = Lifecycle(phase=NEEDS_SETUP)
    app = create_app(
        a,
        lifecycle=lifecycle,
        context=AppContext(lifecycle=lifecycle, config_path=str(config_path)),
    )
    return TestClient(app)


def test_dashboard_url_absent_when_tls_off(tmp_path):
    """HTTP install → no `dashboard_url` in /status → wizard doesn't show banner."""
    c = _client(tmp_path, tls=False)
    body = c.get("/status").json()
    assert "dashboard_url" not in body


def test_dashboard_url_present_when_tls_on(tmp_path):
    """HTTPS install → `dashboard_url` reflects the request's scheme + host."""
    c = _client(tmp_path, tls=True)
    body = c.get("/status").json()
    assert "dashboard_url" in body
    # TestClient uses `http://testserver` by default; the URL we build
    # should match. In production this is `https://<user's host>`.
    assert body["dashboard_url"].startswith(("http://", "https://"))
    assert "testserver" in body["dashboard_url"]


def test_dashboard_url_omitted_when_cert_files_missing(tmp_path):
    """TLS configured in config but the cert/key files don't exist on disk
    (e.g. a half-rewritten config, or a fresh checkout) → no URL → no
    banner. Mirrors the `start_dashboard` warning in server/dashboard.py
    that falls back to plain HTTP when the files can't be found."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        textwrap.dedent("""\
            general:
              dashboard_ssl_certfile: "/nope/missing.crt"
              dashboard_ssl_keyfile: "/nope/missing.key"
            """)
    )
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    lifecycle = Lifecycle(phase=NEEDS_SETUP)
    app = create_app(
        a,
        lifecycle=lifecycle,
        context=AppContext(lifecycle=lifecycle, config_path=str(config_path)),
    )
    body = TestClient(app).get("/status").json()
    assert "dashboard_url" not in body
