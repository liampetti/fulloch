"""Native llama-server MTP launch configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import slm  # noqa: E402


def test_mtp_server_command_enables_native_speculation(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "/opt/llama-cpp/llama-server")

    command = slm._local_server_command("model.gguf", 12288, 4, 512, mtp=True)

    assert command[:3] == ["/opt/llama-cpp/llama-server", "--model", "model.gguf"]
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-n-max") + 1] == str(slm.MTP_DRAFT_TOKENS)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--no-mmproj")] == "--no-mmproj"


def test_mtp_server_command_can_disable_flash_attention(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "/opt/llama-cpp/llama-server")

    command = slm._local_server_command("model.gguf", 12288, 4, 512, mtp=True, flash_attn=False)

    assert command[command.index("--flash-attn") + 1] == "off"


def test_server_command_disables_experimental_features_by_default(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "/opt/llama-cpp/llama-server")

    command = slm._local_server_command("model.gguf", 12288, 4, 512)

    assert "--spec-type" not in command
    assert command[command.index("--flash-attn") + 1] == "off"


def test_mtp_server_requires_native_binary(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "")

    with pytest.raises(RuntimeError, match="Local GGUF models require llama-server"):
        slm._local_server_command("model.gguf", 12288, 4, 512)


def test_standard_server_command_omits_mtp_flags(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "/opt/llama-cpp/llama-server")

    command = slm._local_server_command("gemma.gguf", 10240, 4, 512)

    assert "--spec-type" not in command
    assert "--spec-draft-n-max" not in command


def test_server_failure_detail_classifies_cuda_memory_fault(monkeypatch, tmp_path):
    log = tmp_path / "llama-server.log"
    log.write_text("CUDA error: an illegal memory access was encountered\n")
    monkeypatch.setattr(slm, "SERVER_LOG_PATH", log)

    detail = slm._server_failure_detail(1)

    assert "GPU memory fault" in detail
    assert str(log) in detail


def test_failure_capture_is_timestamped_and_appended(monkeypatch, tmp_path):
    events = tmp_path / "llama-server-events.log"
    server_log = tmp_path / "llama-server.log"
    server_log.write_text("stalled decode\n")
    monkeypatch.setattr(slm, "SERVER_EVENT_LOG_PATH", events)
    monkeypatch.setattr(slm, "SERVER_LOG_PATH", server_log)
    monkeypatch.setattr(slm.subprocess, "run", lambda *args, **kwargs: type("Run", (), {"stdout": "ok"})())

    class _Process:
        pid = 42

        @staticmethod
        def poll():
            return None

    slm._capture_local_server_failure("deadline exceeded", _Process())
    slm._capture_local_server_failure("deadline exceeded", _Process())

    text = events.read_text()
    assert text.count("local llama-server recovery") == 2
    assert "reason: deadline exceeded" in text
    assert "llama-server log tail:\nstalled decode" in text


def test_request_failure_is_persisted_before_recovery(monkeypatch, tmp_path):
    events = tmp_path / "llama-server-events.log"
    server_log = tmp_path / "llama-server.log"
    server_log.write_text("connection dropped\n")
    monkeypatch.setattr(slm, "SERVER_EVENT_LOG_PATH", events)
    monkeypatch.setattr(slm, "SERVER_LOG_PATH", server_log)

    class _Process:
        pid = 42

        @staticmethod
        def poll():
            return None

    slm._record_local_server_request_failure("APIConnectionError: reset", _Process())

    text = events.read_text()
    assert "local llama-server request failure" in text
    assert "reason: APIConnectionError: reset" in text
    assert "llama-server log tail:\nconnection dropped" in text
    snapshots = list(tmp_path.glob("llama-server-failure-*.log"))
    assert len(snapshots) == 1
    assert "2026" in snapshots[0].name
    assert snapshots[0].read_text().strip() == text.strip()


def test_diagnostic_log_rotation_keeps_five_archives(monkeypatch, tmp_path):
    log = tmp_path / "llama-server.log"
    monkeypatch.setattr(slm, "DIAGNOSTIC_LOG_MAX_BYTES", 1)
    for index in range(1, 6):
        (tmp_path / f"llama-server.log.{index}").write_text(f"old-{index}")
    log.write_text("2026-08-12T12:00:00Z ERROR current failure\n")

    slm._rotate_diagnostic_log(log)

    assert not log.exists()
    assert (tmp_path / "llama-server.log.1").read_text().endswith("ERROR current failure\n")
    assert (tmp_path / "llama-server.log.2").read_text() == "old-1"
    assert (tmp_path / "llama-server.log.5").read_text() == "old-4"
    assert not (tmp_path / "llama-server.log.6").exists()


def test_wait_for_local_server_retries_connection_reset(monkeypatch):
    class Process:
        returncode = None

        def poll(self):
            return None

    attempts = 0

    def open_health(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError("connection reset by peer")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        return Response()

    monkeypatch.setattr(slm, "urlopen", open_health)
    monkeypatch.setattr(slm.time, "sleep", lambda _seconds: None)

    slm._wait_for_local_server(Process(), 8081)

    assert attempts == 2
