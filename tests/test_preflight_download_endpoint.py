"""Integration test for `POST /setup/preflight-download`.

The endpoint is a thin wrapper around the three check_* functions in
server/preflight.py. The unit tests in test_preflight_download.py
exercise each check in isolation; this one verifies the endpoint
shape (status code, JSON keys, error structure) and that all three
checks run on a single call.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.dashboard import create_app  # noqa: E402
from server.lifecycle import NEEDS_SETUP, AppContext, Lifecycle  # noqa: E402


def _make_client(tmp_path: Path) -> TestClient:
    config_path = tmp_path / "config.yml"
    config_path.write_text("general: {}\n")
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


def test_endpoint_returns_ok_when_all_checks_pass(tmp_path):
    c = _make_client(tmp_path)
    with patch("server.preflight.check_disk_for_models", return_value=(True, "")):
        with patch("server.preflight.check_network", return_value=(True, "")):
            with patch("server.preflight.check_gpu_for_models", return_value=(True, "")):
                body = c.post("/setup/preflight-download").json()
    assert body["ok"] is True
    assert body["errors"] == []


def test_endpoint_collects_all_errors_at_once(tmp_path):
    """All three checks fail → the user sees all three in one go, not one
    at a time. Saves round-trips for users with multiple setup issues."""
    c = _make_client(tmp_path)
    with patch("server.preflight.check_disk_for_models", return_value=(False, "disk full")):
        with patch("server.preflight.check_network", return_value=(False, "no internet")):
            with patch("server.preflight.check_gpu_for_models", return_value=(False, "no GPU")):
                body = c.post("/setup/preflight-download").json()
    assert body["ok"] is False
    assert len(body["errors"]) == 3
    by_check = {e["check"]: e["message"] for e in body["errors"]}
    assert by_check["disk"] == "disk full"
    assert by_check["network"] == "no internet"
    assert by_check["gpu"] == "no GPU"


def test_endpoint_reports_only_failed_checks(tmp_path):
    """A partial failure (one of three) reports exactly one error."""
    c = _make_client(tmp_path)
    with patch("server.preflight.check_disk_for_models", return_value=(True, "")):
        with patch("server.preflight.check_network", return_value=(False, "offline")):
            with patch("server.preflight.check_gpu_for_models", return_value=(True, "")):
                body = c.post("/setup/preflight-download").json()
    assert body["ok"] is False
    assert body["errors"] == [{"check": "network", "message": "offline"}]
