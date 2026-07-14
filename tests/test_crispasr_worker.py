"""CrispASR worker process isolation tests."""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.crispasr_worker as worker  # noqa: E402


class _Connection:
    def __init__(self):
        self.messages = [("ready", None)]

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.messages.append((True, message[0]))

    def close(self):
        pass


class _Listener:
    connection = _Connection()

    def __init__(self, address, family, authkey):
        self.address = address

    def accept(self):
        return self.connection

    def close(self):
        pass


class _Process:
    def __init__(self, args, cwd, env):
        self.args = args
        self.cwd = cwd
        self.env = env

    def poll(self):
        return 0


class _LiveProcess:
    def poll(self):
        return None


class _StreamConnection:
    def __init__(self):
        self.sent = []
        self.messages = [("stream", "one"), ("stream", "two"), ("done", None)]

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.messages.pop(0)


class _BrokenStreamConnection(_StreamConnection):
    def recv(self):
        raise EOFError


def test_worker_uses_module_subprocess_not_multiprocessing_spawn(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime" / "crispasr"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").touch()
    created = {}

    def popen(*args, **kwargs):
        created["process"] = _Process(*args, **kwargs)
        return created["process"]

    monkeypatch.setattr(worker, "Listener", _Listener)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    cuda_lib = tmp_path / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
    cuda_lib.mkdir(parents=True)
    monkeypatch.setattr(worker.site, "getsitepackages", lambda: [str(cuda_lib.parents[2])])

    instance = worker.CrispASRWorker(model_path="model.gguf", lib_dir=runtime.parent, backend="qwen3")

    assert created["process"].args[1:3] == ["-m", "core.crispasr_worker"]
    assert "--worker" in created["process"].args
    assert str(cuda_lib) in created["process"].env["LD_LIBRARY_PATH"]
    assert created["process"].env["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert "multiprocessing.get_context" not in Path(worker.__file__).read_text()
    instance.close()


def test_direct_tts_worker_requests_streaming_session(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime" / "crispasr"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").touch()
    created = {}

    def popen(*args, **kwargs):
        created["process"] = _Process(*args, **kwargs)
        return created["process"]

    _Listener.connection = _Connection()
    monkeypatch.setattr(worker, "Listener", _Listener)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)

    instance = worker.CrispASRWorker(
        model_path="model.gguf", lib_dir=runtime.parent, direct_tts=True
    )

    settings_arg = next(arg for arg in created["process"].args if arg.startswith("--settings="))
    settings = json.loads(base64.urlsafe_b64decode(settings_arg.split("=", 1)[1]).decode())
    assert settings["direct_tts"] is True
    instance.close()


def test_worker_stream_yields_each_pcm_chunk():
    instance = object.__new__(worker.CrispASRWorker)
    instance._process = _LiveProcess()
    instance._connection = _StreamConnection()

    assert list(instance.stream("synthesize_stream", text="hello")) == ["one", "two"]
    assert instance._connection.sent == [("synthesize_stream", {"text": "hello"})]
    instance._process = None  # Deliberately partial test double; don't run normal cleanup.


def test_worker_stream_marks_connection_broken_on_eof():
    instance = object.__new__(worker.CrispASRWorker)
    instance._process = _LiveProcess()
    instance._connection = _BrokenStreamConnection()
    instance._broken = False

    try:
        list(instance.stream("synthesize_stream", text="hello"))
    except RuntimeError as exc:
        assert "exited unexpectedly" in str(exc)
    else:
        raise AssertionError("EOF must fail the stream")
    assert instance.alive is False
    instance._process = None  # Deliberately partial test double; don't run normal cleanup.
