"""Thinking-task integration API contract."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.dashboard import create_app  # noqa: E402
from server.lifecycle import READY, AppContext, Lifecycle  # noqa: E402


def _client(tmp_path: Path, assistant: MagicMock) -> TestClient:
    config_path = tmp_path / "config.yml"
    config_path.write_text("general: {}\n")
    lifecycle = Lifecycle(phase=READY)
    return TestClient(
        create_app(
            assistant,
            lifecycle=lifecycle,
            context=AppContext(lifecycle=lifecycle, config_path=str(config_path), assistant=assistant),
        )
    )


def test_run_and_read_thinking_task(tmp_path):
    assistant = MagicMock()
    assistant.run_thinking_task.return_value = {"id": "job-1", "status": "QUEUED"}
    assistant.thinking_task_status.return_value = {"id": "job-1", "status": "RUNNING"}
    client = _client(tmp_path, assistant)

    response = client.post("/thinking/run", json={"task": "Research local weather"})
    assert response.status_code == 200
    assert response.json()["id"] == "job-1"
    assistant.run_thinking_task.assert_called_once_with("Research local weather")

    response = client.get("/thinking/job-1")
    assert response.status_code == 200
    assert response.json()["status"] == "RUNNING"


def test_thinking_task_disabled_and_cancelled_job_errors(tmp_path):
    assistant = MagicMock()
    assistant.run_thinking_task.side_effect = RuntimeError("thinking is disabled")
    assistant.cancel_thinking_task.return_value = False
    client = _client(tmp_path, assistant)

    assert client.post("/thinking/run", json={"task": "Research"}).status_code == 409
    assert client.post("/thinking/job-1/cancel").status_code == 404
