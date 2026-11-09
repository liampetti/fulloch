import logging
import time

import pytest

from core.background_jobs import BackgroundJobManager, JobStatus
from core.thinking_source_store import load_document, store_document


def wait_terminal(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        # Terminal cleanup removes the checkpoint before the source files, all
        # under this lock. Do not mistake that intermediate state for completion.
        with manager._lock:
            job = manager._jobs[job_id]
            if job.status in {JobStatus.READY, JobStatus.FAILED} and not (
                manager._checkpoint_dir / f"{job_id}.json"
            ).exists():
                return job
        time.sleep(0.01)
    pytest.fail("job did not finish cleanup")


@pytest.mark.parametrize("fail", [False, True])
def test_terminal_job_deletes_working_files_and_logs_failure(tmp_path, monkeypatch, caplog, fail):
    sources = tmp_path / "sources"
    monkeypatch.setenv("THINKING_SOURCE_STORE_DIR", str(sources))
    checkpoints = tmp_path / "jobs"
    ledger = tmp_path / "ledger.json"

    def worker(job, _cancelled):
        job.record_outcome("lookup", "ok", "scope", {"sources": [{
            "url": "https://example.org", "text": "source body",
        }]}, (), None)
        job.stage = "Writing chapter"
        if fail:
            raise RuntimeError("backend unavailable")
        return "Finished report", ""

    manager = BackgroundJobManager(worker, checkpoint_dir=checkpoints, task_ledger_path=ledger)
    with caplog.at_level(logging.ERROR):
        job_id = manager.submit("Research", origin_satellite_id="thread", origin_source="conversation")
        job = wait_terminal(manager, job_id)
    assert job.status == (JobStatus.FAILED if fail else JobStatus.READY)
    assert not sources.exists()
    assert not checkpoints.exists()
    assert not ledger.exists()
    if fail:
        assert "Writing chapter" in caplog.text
        assert "backend unavailable" in caplog.text
    else:
        assert job.summary == "Finished report"


def test_terminal_cleanup_preserves_active_job_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("THINKING_SOURCE_STORE_DIR", str(tmp_path / "sources"))
    manager = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False,
                                   checkpoint_dir=tmp_path / "jobs")
    first = manager.submit("first")
    second = manager.submit("second")
    ref = store_document("https://example.org", "shared content")
    manager._jobs[second].artifacts["shared"] = {"source_ref": ref}
    assert manager.cancel(first)
    assert load_document(ref)["text"] == "shared content"
    assert manager.cancel(second)
    assert not (tmp_path / "sources").exists()
