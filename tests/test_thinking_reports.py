"""Report storage, evidence, lifecycle notifications and grounded follow-ups."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core import thinking_reports as reports
from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus


@pytest.fixture
def report_root(tmp_path, monkeypatch):
    monkeypatch.setattr(reports.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(reports.notes, "_after_write", lambda _path: None)
    return tmp_path


def notifications(job, *, pending=None):
    state = SimpleNamespace(
        pending=pending or {},
        completed={},
        history=[],
        dispatch=MagicMock(),
        emit=MagicMock(),
        speak=MagicMock(),
    )
    reports.publish_status(
        job,
        pending_tasks=state.pending,
        completed_reports=state.completed,
        record_history=state.history.append,
        dispatch_event=state.dispatch,
        emit_turn_event=state.emit,
        speak_proactive=state.speak,
    )
    return state


def test_spoken_report_summary_is_short_and_sentence_complete():
    report = "One finding is useful. Two finding is useful. Three finding is useful. Four finding is useful. Five finding is useful."
    assert reports.spoken_summary(report) == "Four finding is useful. Five finding is useful."


def test_report_summary_keeps_only_the_first_three_summary_sentences():
    report = (
        "## Summary\n\nFirst key finding. Second key finding. Third key finding. "
        "Fourth finding should not be spoken.\n\n## Details\n\nMore detail."
    )
    assert (
        reports.report_summary(report)
        == "First key finding. Second key finding. Third key finding."
    )


def test_completed_report_is_available_after_a_satellite_reconnect(monkeypatch):
    completed = {
        "old-session": {
            "note_id": "fulloch-reports/2026-08-27-12345678",
            "task": "Compare options",
            "summary_delivered": False,
        }
    }
    monkeypatch.setattr(reports, "read_report", lambda _pending: "Finding. Conclusion.")
    entry = reports.completed_entry("new-session", completed, {"new-session": object()})
    response = reports.consume_report(entry)
    assert response == (
        "Here's the short version. Finding. Conclusion. "
        "The full report is saved in Fulloch Reports. Would you like me to read the full report?"
    )
    assert completed == {
        "new-session": {
            "note_id": "fulloch-reports/2026-08-27-12345678",
            "task": "Compare options",
            "summary_delivered": True,
        }
    }
    assert reports.consume_report(entry) == "Here is the full report. Finding. Conclusion."


def test_report_question_reads_saved_report_and_evidence_only(report_root):
    report_path = report_root / "fulloch-reports" / "2026-08-27-12345678.md"
    report_path.parent.mkdir()
    report_path.write_text(
        "# Deep Think Report\n\nThe installation cost is $4,000.", encoding="utf-8"
    )
    report_path.with_suffix(".evidence.json").write_text(
        '{"artifacts": {"artifact-001": {"data": {"price": 4000}}}}', encoding="utf-8"
    )
    generate = MagicMock(return_value="The report lists $4,000.")
    pending = {
        "note_id": "fulloch-reports/2026-08-27-12345678",
        "task": "Compare heating systems",
        "summary_delivered": True,
    }
    model, stats = object(), object()

    def cancel():
        return False

    answer = reports.answer_report(
        pending,
        "What did it say about installation costs?",
        model=model,
        generate=generate,
        max_new_tokens=256,
        cancel_check=cancel,
        stats=stats,
    )
    assert answer == "The report lists $4,000."
    captured = generate.call_args.kwargs
    assert generate.call_args.args == (model,)
    assert captured["user_prompt"] == "What did it say about installation costs?"
    assert "The installation cost is $4,000." in captured["system_prompt"]
    assert "artifact-001" in captured["system_prompt"]
    assert "Compare heating systems" not in captured["system_prompt"]
    assert "history" not in captured
    assert captured["stats"] is stats and captured["cancel_check"] is cancel


def test_completed_report_persists_typed_evidence_and_artifacts(report_root):
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Compare options"),
        summary="## Summary\n\nA scoped result.",
        evidence=[{"tool": "lookup", "status": "evidence", "artifact_id": "artifact-001"}],
        artifacts={"artifact-001": {"tool": "lookup", "data": {"price": 4000}}},
    )
    note_id = reports.save_report(job)
    evidence = json.loads((report_root / f"{note_id}.evidence.json").read_text(encoding="utf-8"))
    assert Path(note_id).parent == Path("fulloch-reports")
    assert "A scoped result." in (report_root / f"{note_id}.md").read_text(encoding="utf-8")
    assert evidence["task"] == "Compare options"
    assert evidence["evidence"][0]["artifact_id"] == "artifact-001"
    assert evidence["artifacts"]["artifact-001"]["data"]["price"] == 4000


def test_completed_travel_report_appends_retrieved_offers_and_source_link(report_root):
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Plan a trip"),
        summary="## Summary\n\nNo compatible itinerary was found.",
        artifacts={
            "artifact-001": {
                "data": {
                    "type": "travel_plan",
                    "representative": {"departure_date": "2026-09-23", "currency": "AUD"},
                    "leg_offers": [
                        [
                            {
                                "airlines": ["Emirates"],
                                "departure": {"id": "HND", "time": "2026-09-23 00:05"},
                                "arrival": {"id": "DXB", "time": "2026-09-23 05:40"},
                                "duration_minutes": 635,
                                "stops": 0,
                                "price": 1601,
                            }
                        ]
                    ],
                }
            }
        },
    )
    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")
    assert "## Retrieved Flight Offers" in report
    assert "Emirates: 2026-09-23 00:05 to 2026-09-23 05:40" in report
    assert "AUD 1601" in report
    assert "[Google Flights: HND to DXB on 2026-09-23]" in report


def test_completed_research_report_appends_artifact_references(report_root):
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Find papers"),
        summary="## Summary\n\nOne relevant paper was found.",
        artifacts={
            "artifact-001": {
                "data": {
                    "type": "paper_search",
                    "papers": [
                        {
                            "title": "Example Paper",
                            "source": "arXiv",
                            "url": "https://arxiv.org/abs/1234.5678",
                            "doi": "10.1000/example",
                        }
                    ],
                }
            }
        },
    )
    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")
    assert "## References" in report
    assert "[arXiv: Example Paper](https://arxiv.org/abs/1234.5678)" in report
    assert "[DOI: Example Paper](https://doi.org/10.1000/example)" in report


def test_completed_report_adds_a_scoped_summary_when_worker_omits_one(report_root):
    job = BackgroundJob(
        "12345678", JobSnapshot("Compare options"), summary="Option A has the lower price."
    )
    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")
    assert "## Summary" in report and "Option A has the lower price." in report
    assert "limited to the retrieved evidence" in report


def test_failed_thinking_job_does_not_offer_or_save_a_report(monkeypatch):
    save = MagicMock()
    monkeypatch.setattr(reports, "save_report", save)
    job = BackgroundJob(
        "12345678",
        JobSnapshot(
            "Today's finance summary",
            origin_source="conversation",
            origin_satellite_id="dashboard-text",
        ),
        status=JobStatus.FAILED,
        error="ReportSynthesisError: The final report generation returned no content.",
    )
    state = notifications(job, pending={"dashboard-text": "Today's finance summary"})
    assert job.note_id == "" and state.completed == {} and state.pending == {}
    save.assert_not_called()
    state.emit.assert_called_once_with(
        "assistant", "I couldn't complete that report.", "proactive", satellite_id="dashboard-text"
    )
    assert state.dispatch.call_args.args[0]["status"] == JobStatus.FAILED


def test_completed_report_history_keeps_summary_and_durable_filename():
    job = BackgroundJob(
        "12345678",
        JobSnapshot(
            "Compare heat pumps", origin_source="conversation", origin_satellite_id="dashboard-text"
        ),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        summary=(
            "## Summary\n\nHeat pumps are suitable for this insulated home, based on the retrieved quotes. "
            "Installation estimates vary by installer.\n\n## Analysis\n\n" + "detail " * 500
        ),
    )
    state = notifications(job)
    trace = state.history[-1]["content"]
    assert "Heat pumps are suitable" in trace and job.note_id in trace
    assert "detail detail" not in trace
    assert state.completed["dashboard-text"]["summary_delivered"] is False


@pytest.mark.parametrize(
    "source,task,summary,kind,title",
    [
        (
            {
                "type": "flight_search",
                "route": {"origin": "SYD", "destination": "NRT"},
                "offer": {},
            },
            "Find flights",
            "",
            "travel_report",
            "Travel Report",
        ),
        (
            None,
            "Compare heat pumps",
            "A heat pump is likely suitable with insulation improvements.",
            "generated_report",
            "Compare heat pumps",
        ),
        (
            {
                "type": "paper_search",
                "papers": [{"title": "A Paper", "year": 2026, "source": "arXiv"}],
            },
            "Find papers",
            "A relevant paper was found.",
            "research_report",
            "Research Report",
        ),
        (
            {"type": "finance_quote", "quote": {"name": "Tesla", "price": "250", "points": []}},
            "Analyse Tesla",
            "Tesla moved higher.",
            "finance_report",
            "Finance Report",
        ),
        (
            {"type": "finance_exchange_rate", "exchange_rate": {"rate": 1.5}},
            "Convert USD to AUD",
            "",
            "finance_report",
            "Finance Report",
        ),
    ],
)
def test_completed_job_emits_domain_report_card(source, task, summary, kind, title):
    job = BackgroundJob(
        "12345678",
        JobSnapshot(task, origin_source="integration"),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        artifact=source,
        summary=summary,
        created_at=1_788_000_000,
    )
    state = notifications(job)
    expected = {
        "type": kind,
        "title": title,
        "created_at": 1_788_000_000,
        "summary": summary,
        "report_url": "/reports/fulloch-reports/2026-08-27-12345678",
    }
    if source is not None:
        expected["data"] = source
    state.emit.assert_called_once_with(
        "assistant",
        "I've completed the report and saved the full version.",
        "proactive",
        artifact=expected,
    )


def test_finance_report_card_combines_watchlist_charts_with_market_context():
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Today's finance summary"),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        artifact={"type": "finance_market", "markets": [{"name": "S&P 500"}]},
        artifacts={
            "artifact-001": {"data": {"type": "finance_market", "markets": [{"name": "S&P 500"}]}},
            "artifact-002": {
                "data": {
                    "type": "finance_watchlist",
                    "quotes": [{"name": "Example Corp", "points": [{"value": 20.0}]}],
                }
            },
        },
    )
    artifact = reports.report_artifact(job)
    assert artifact["type"] == "finance_report"
    assert artifact["data"] == {
        "type": "finance_summary",
        "quotes": [{"name": "Example Corp", "points": [{"value": 20.0}]}],
        "markets": [{"name": "S&P 500"}],
    }


@pytest.mark.parametrize(
    "note_id", ["../secret", "fulloch-reports/invalid", "fulloch-reports/2026-08-27-12345678"]
)
def test_missing_or_invalid_reports_do_not_generate(report_root, note_id):
    generate = MagicMock()
    pending = {"note_id": note_id}
    assert reports.read_report(pending) is None and reports.read_evidence(pending) == {}
    assert (
        reports.answer_report(
            pending, "Read it", model=None, generate=generate, max_new_tokens=256, cancel_check=None
        )
        == "I can't retrieve that completed report right now."
    )
    generate.assert_not_called()


@pytest.mark.parametrize(
    "status,follow_up",
    [(JobStatus.NEEDS_INPUT, True), (JobStatus.READY, True), (JobStatus.FAILED, False)],
)
def test_conversation_notifications_start_targeted_proactive_speech(monkeypatch, status, follow_up):
    threads = []

    class ImmediateThread:
        def __init__(self, *, target, args, kwargs, **metadata):
            threads.append(metadata)
            self.call = lambda: target(*args, **kwargs)

        def start(self):
            self.call()

    monkeypatch.setattr(reports.threading, "Thread", ImmediateThread)
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Investigate", origin_source="conversation", origin_satellite_id="kitchen"),
        status=status,
        note_id="fulloch-reports/2026-08-27-12345678",
        summary="Reactive question: Which city?",
    )
    state = notifications(job)
    assert threads[0]["daemon"] is True
    state.speak.assert_called_once_with(
        state.emit.call_args.args[1], emit_event=False, satellite_id="kitchen", follow_up=follow_up
    )
    if status == JobStatus.NEEDS_INPUT:
        assert state.pending == {"kitchen": "Investigate"}
        assert state.history[0]["content"] == job.summary
