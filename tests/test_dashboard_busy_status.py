"""Phase 4 (#13): `GET /status` surfaces `satellite_count` / `active_owner_id`
/ `active_owner_label` so the dashboard can show "busy — talking to <room>"
instead of a silent stall while another satellite (or the dashboard's own
text chat) owns the `TurnArbiter`.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.satellite import SatelliteSession
from server.dashboard import create_app


def _assistant():
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    return a


def _client(a):
    return TestClient(create_app(a))


def _connect(a, satellite_id, **overrides):
    overrides.setdefault("chunk_q", None)
    a.satellites[satellite_id] = SatelliteSession(id=satellite_id, **overrides)
    return a.satellites[satellite_id]


def test_zero_satellites_idle():
    a = _assistant()
    body = _client(a).get("/status").json()
    assert body["satellite_count"] == 0
    assert body["active_owner_id"] is None
    assert body["active_owner_label"] is None
    assert body["asr_queue"] == {}


def test_status_includes_asr_queue_metrics():
    a = _assistant()
    a.audio_capture.asr_queue_metrics = {"admitted": 2, "dropped": 1, "evicted": 1, "peak_depth": 16}

    assert _client(a).get("/status").json()["asr_queue"] == a.audio_capture.asr_queue_metrics


def test_last_turn_stats_none_before_any_turn():
    a = _assistant()
    body = _client(a).get("/status").json()
    assert body["last_turn_stats"] is None


def test_last_turn_stats_reflects_the_most_recent_turn():
    # A2: `Assistant._last_turn_stats` is set at the end of a turn (see
    # _run_half_duplex/_run_turn/handle_text_prompt) — /status exposes it
    # directly rather than the caller having to scrape SSE history.
    a = _assistant()
    a._last_turn_stats = {"total": 1.23, "route": "regex", "endpoint_kind": "soft"}
    body = _client(a).get("/status").json()
    assert body["last_turn_stats"] == {"total": 1.23, "route": "regex", "endpoint_kind": "soft"}


def test_one_satellite_idle():
    a = _assistant()
    _connect(a, "sat-a", label="kitchen")
    body = _client(a).get("/status").json()
    assert body["satellite_count"] == 1
    assert body["active_owner_id"] is None
    assert body["active_owner_label"] is None


def test_one_satellite_mid_turn_reports_owner_and_label():
    a = _assistant()
    _connect(a, "sat-a", label="kitchen")
    a._turn_arbiter.try_acquire("sat-a")
    body = _client(a).get("/status").json()
    assert body["satellite_count"] == 1
    assert body["active_owner_id"] == "sat-a"
    assert body["active_owner_label"] == "kitchen"


def test_two_satellites_mid_turn_reports_only_the_owner():
    a = _assistant()
    _connect(a, "sat-a", label="kitchen")
    _connect(a, "sat-b", label="office")
    a._turn_arbiter.try_acquire("sat-b")
    body = _client(a).get("/status").json()
    assert body["satellite_count"] == 2
    assert body["active_owner_id"] == "sat-b"
    assert body["active_owner_label"] == "office"


def test_owner_without_a_label_reports_null_label():
    a = _assistant()
    _connect(a, "sat-a")  # no label set
    a._turn_arbiter.try_acquire("sat-a")
    body = _client(a).get("/status").json()
    assert body["active_owner_id"] == "sat-a"
    assert body["active_owner_label"] is None


def test_owner_with_only_ha_area_name_reports_that_as_label():
    # Browser satellite with no configured `label` but a chosen HA room
    # (?area_name= on /ws/satellite) — the busy banner should still say
    # "talking to <room>" rather than falling back to nothing.
    a = _assistant()
    _connect(a, "sat-a", ha_area="living_room", ha_area_name="Living Room")
    a._turn_arbiter.try_acquire("sat-a")
    body = _client(a).get("/status").json()
    assert body["active_owner_id"] == "sat-a"
    assert body["active_owner_label"] == "Living Room"


def test_dashboard_text_pseudo_satellite_is_not_counted():
    # "dashboard-text" always exists in Assistant.satellites (see __init__)
    # but isn't a real connection, so it must not inflate satellite_count.
    a = _assistant()
    body = _client(a).get("/status").json()
    assert "dashboard-text" in a.satellites
    assert body["satellite_count"] == 0


def test_text_turn_holding_the_arbiter_reports_as_owner_with_no_label():
    a = _assistant()
    _connect(a, "sat-a", label="kitchen")
    a._turn_arbiter.try_acquire("dashboard-text")
    body = _client(a).get("/status").json()
    assert body["satellite_count"] == 1
    assert body["active_owner_id"] == "dashboard-text"
    assert body["active_owner_label"] is None


def test_status_polling_retains_live_research_progress_and_final_report_link(tmp_path, monkeypatch):
    from core.background_jobs import BackgroundJob, BackgroundJobManager, JobSnapshot, JobStatus
    from core.thinking_evidence import research_progress

    monkeypatch.setattr("core.background_jobs.time.time", lambda: 120.0)
    a = _assistant()
    manager = BackgroundJobManager(lambda *_: ("", ""), checkpoint_dir=tmp_path)
    a.thinking_jobs = manager
    job = BackgroundJob(
        "progress", JobSnapshot("Research F1 with additional constraints", task_record={
            "objective": "Research F1", "started_at": 110.0,
        }), status=JobStatus.RUNNING, created_at=100.0, stage="Planning research",
    )
    job.artifacts = {"web": {"data": {"sources": [{"url": "https://example.com/race", "eligible": True}]}}}
    job.fact_cards = [{"source_url": "https://example.com/race", "compatibility": "compatible"}]
    job.evidence_requirements = [{"identifier": "timing", "state": "open"}]
    manager._jobs[job.id] = job
    client = _client(a)

    first = client.get("/status").json()["thinking_job"]

    assert first["objective"] == "Research F1"
    assert first["stage"] == "Planning research"
    assert first["job_duration_seconds"] == 10.0 and first["elapsed_seconds"] == 20.0
    assert first["progress"] == research_progress(job)
    assert first["progress"]["processed_unique_documents"] == 1
    assert first["progress"]["unresolved_obligations"] == ["timing"]

    job.stage = "Writing chapters"
    job.chapter_plan = [{"id": "timing", "body": "Verified race timing.", "analysis_state": "complete"}]
    job.evidence_requirements[0]["state"] = "resolved"
    second = client.get("/status").json()["thinking_job"]

    assert second["stage"] == "Writing chapters"
    assert second["progress"]["saved_sections"] == 1
    assert second["progress"]["unresolved_obligations"] == []

    job.status = JobStatus.READY
    job.note_id = "fulloch-reports/f1"
    job.snapshot.task_record["finished_at"] = 120.0
    assert client.get("/status").json()["thinking_job"] is None
    final = client.get(f"/thinking/{job.id}").json()
    assert final["status"] == "READY"
    assert final["partial_artifact_link"] == "/reports/fulloch-reports/f1"
    assert final["progress"]["saved_sections"] == 1
