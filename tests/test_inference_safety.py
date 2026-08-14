"""Focused tests for inference timeout containment and TTS job admission."""

import queue
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference_safety import (
    InferenceWatchdog,
    TtsQueueFullError,
    _record_timeout,
    submit_tts_job,
)


def test_watchdog_records_reason_then_exits(monkeypatch):
    recorded = []
    exited = threading.Event()

    monkeypatch.setattr("core.inference_safety._record_timeout", recorded.append)
    with InferenceWatchdog("test inference", timeout_seconds=0.01, exit_fn=lambda code: exited.set()):
        assert exited.wait(1)

    assert recorded == ["test inference"]


def test_watchdog_does_not_exit_after_completed_inference(monkeypatch):
    exited = threading.Event()
    monkeypatch.setattr("core.inference_safety._record_timeout", lambda reason: pytest.fail(reason))

    with InferenceWatchdog("completed inference", timeout_seconds=0.01, exit_fn=lambda code: exited.set()):
        pass

    assert not exited.wait(0.05)


def test_watchdog_diagnostic_file_requires_persistent_logging(monkeypatch, tmp_path):
    diagnostic = tmp_path / "inference_watchdog.log"
    monkeypatch.setattr("core.inference_safety._DIAGNOSTIC_PATH", diagnostic)
    monkeypatch.delenv("FULLOCH_PERSISTENT_LOGGING_ENABLED", raising=False)

    _record_timeout("test inference")

    assert not diagnostic.exists()


def test_tts_job_admission_is_nonblocking_when_full():
    jobs = queue.Queue(maxsize=1)
    jobs.put(object())

    with pytest.raises(TtsQueueFullError, match="queued speech limit"):
        submit_tts_job(jobs, object())
