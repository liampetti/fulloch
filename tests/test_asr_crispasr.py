"""CrispASR GGUF ASR adapter contract tests (no native runtime/model required)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.asr_crispasr as asr  # noqa: E402
from core.asr_crispasr import CrispASRPipelineWrapper  # noqa: E402


class _Worker:
    def __init__(self):
        self.calls = []

    def call(self, command, **payload):
        self.calls.append((command, payload))
        return "hey atticus turn on the kitchen lights"


def test_pipeline_transcribes_with_context_and_tracks_duration():
    worker = _Worker()
    pipe = CrispASRPipelineWrapper(worker, language="English")
    pipe.context = "Technical terms: Atticus"

    out = pipe(np.zeros(16000, dtype=np.float32))

    assert out == [{"text": "hey atticus turn on the kitchen lights"}]
    command, payload = worker.calls[0]
    assert command == "transcribe"
    assert payload["language"] == "English"
    assert payload["context"] == "Technical terms: Atticus"
    assert payload["audio"].dtype == np.float32
    assert pipe.last_transcribe_seconds is not None


def test_pipeline_restarts_dead_worker_once(monkeypatch):
    class DeadWorker:
        alive = False

        def __init__(self):
            self.closed = False

        def call(self, *args, **kwargs):
            raise RuntimeError("worker exited")

        def close(self):
            self.closed = True

    class ReplacementWorker:
        alive = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []

        def call(self, command, **payload):
            self.calls.append((command, payload))
            return "recovered"

    dead = DeadWorker()
    created = []

    def make_worker(**kwargs):
        instance = ReplacementWorker(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(asr, "CrispASRWorker", make_worker)
    pipe = asr.CrispASRPipelineWrapper(
        dead,
        worker_config={"model_path": "model.gguf", "lib_dir": "runtime", "backend": "qwen3"},
    )

    assert pipe(np.zeros(8, dtype=np.float32)) == [{"text": "recovered"}]
    assert dead.closed is True
    assert created[0].calls[0][0] == "transcribe"
