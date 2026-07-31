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


def test_blackwell_disables_mtp_and_flash_attention(monkeypatch):
    monkeypatch.setattr(slm, "_blackwell_gpu", lambda: True)

    assert slm._mtp_supported() is False
    assert slm._flash_attn_supported() is False


def test_mtp_server_requires_native_binary(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "")

    with pytest.raises(RuntimeError, match="Local GGUF models require llama-server"):
        slm._local_server_command("model.gguf", 12288, 4, 512)


def test_standard_server_command_omits_mtp_flags(monkeypatch):
    monkeypatch.setattr(slm, "_local_server_binary", lambda: "/opt/llama-cpp/llama-server")

    command = slm._local_server_command("gemma.gguf", 10240, 4, 512)

    assert "--spec-type" not in command
    assert "--spec-draft-n-max" not in command
