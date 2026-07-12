"""`GET /ready` is the liveness/readiness probe for `docker compose ps`
and the Docker HEALTHCHECK directive. The dashboard returns 200 +
`{"ready": true}` as soon as the web server is serving — the
"server is alive and the UI is reachable" signal, distinct from
the assistant being fully loaded (which can be minutes away on a
first run; that's tracked via /status's `phase`).

Task 7a of docs/ease-of-use-tasks.md.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server.dashboard import create_app
from server.lifecycle import AppContext, Lifecycle


def _client(phase):
    """Build a TestClient with the lifecycle forced into `phase`.

    /ready is supposed to be phase-agnostic (it reflects "server
    alive", not "assistant fully loaded"), so we vary `phase` across
    tests to make sure the endpoint doesn't accidentally return 503
    for any of them.
    """
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    lifecycle = Lifecycle(phase=phase)
    app = create_app(
        a,
        lifecycle=lifecycle,
        context=AppContext(
            lifecycle=lifecycle,
            config_path=str(Path("/tmp/fake-config.yml")),
        ),
    )
    return TestClient(app)


def test_ready_returns_200_in_needs_setup():
    """First-run state — wizard is renderable, /ready is 200."""
    r = _client(Lifecycle.phase if hasattr(Lifecycle, "phase") else "NEEDS_SETUP")
    # Use the actual phase constant value.
    from server.lifecycle import NEEDS_SETUP

    r = _client(NEEDS_SETUP)
    body = r.get("/ready")
    assert body.status_code == 200
    assert body.json() == {"ready": True}


def test_ready_returns_200_in_ready():
    """Assistant fully loaded — still 200 (the endpoint is the liveness probe,
    not the 'is the SLM warm' probe; that's /status's `phase` field)."""
    from server.lifecycle import READY

    r = _client(READY)
    body = r.get("/ready")
    assert body.status_code == 200
    assert body.json() == {"ready": True}


def test_ready_returns_200_in_downloading():
    """Models mid-download — the UI is showing a progress bar; from
    a `docker compose ps` perspective, the container is up and
    serving a useful page, so /ready is 200."""
    from server.lifecycle import DOWNLOADING

    r = _client(DOWNLOADING)
    body = r.get("/ready")
    assert body.status_code == 200


def test_ready_returns_200_in_error():
    """The dashboard is showing an actionable error. The container
    itself is alive; the user can see the error and fix it. HEALTHCHECK
    says 200 so Docker doesn't restart-loop a recoverable config
    error into a worse state."""
    from server.lifecycle import ERROR

    r = _client(ERROR)
    body = r.get("/ready")
    assert body.status_code == 200
