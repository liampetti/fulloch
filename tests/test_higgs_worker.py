"""HiggsTTS.cpp process transport tests without a native runtime."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.higgs_worker import HiggsWorker  # noqa: E402


@pytest.fixture
def assets(tmp_path):
    paths = {}
    for name in ("higgs_server", "model.gguf", "tokenizer.json", "voice.wav"):
        path = tmp_path / name
        path.touch()
        paths[name] = path
    return paths


def test_command_includes_bounded_actions_and_voice_reference(assets):
    worker = HiggsWorker(
        server_path=assets["higgs_server"],
        model_path=assets["model.gguf"],
        tokenizer_path=assets["tokenizer.json"],
        reference_wav=assets["voice.wav"],
        reference_text="A voice reference.",
        max_actions=256,
    )

    command = worker._command()

    assert command[command.index("--max-actions") + 1] == "256"
    assert command[command.index("--ref-text") + 1] == "A voice reference."


@pytest.mark.parametrize("max_actions", (0, -1))
def test_actions_must_be_positive(assets, max_actions):
    with pytest.raises(ValueError, match="max_actions"):
        HiggsWorker(
            server_path=assets["higgs_server"],
            model_path=assets["model.gguf"],
            tokenizer_path=assets["tokenizer.json"],
            reference_wav=assets["voice.wav"],
            reference_text="A voice reference.",
            max_actions=max_actions,
        )


def test_synthesize_stream_decodes_native_pcm_frames(monkeypatch, assets):
    class FakeProcess:
        def poll(self):
            return None

    class FakeConnection:
        def __init__(self):
            pcm = struct.pack("!BI", 1, 8) + struct.pack("ff", 0.25, -0.5)
            self.data = bytearray(pcm + struct.pack("!BI", 2, 0))
            self.sent = b""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def sendall(self, data):
            self.sent += data

        def recv(self, length):
            result = self.data[:length]
            del self.data[:length]
            return bytes(result)

    connection = FakeConnection()
    monkeypatch.setattr("core.higgs_worker.socket.create_connection", lambda *_args, **_kwargs: connection)
    worker = HiggsWorker(
        server_path=assets["higgs_server"],
        model_path=assets["model.gguf"],
        tokenizer_path=assets["tokenizer.json"],
        reference_wav=assets["voice.wav"],
        reference_text="A voice reference.",
    )
    worker.process = FakeProcess()

    frames = list(worker.synthesize_stream("Hello"))

    assert len(frames) == 1
    assert np.allclose(frames[0], [0.25, -0.5])
    assert connection.sent.endswith(b"Hello")
