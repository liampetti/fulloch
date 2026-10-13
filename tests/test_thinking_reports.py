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
    assert reports.spoken_summary(report) == "One finding is useful. Two finding is useful."


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


def test_report_question_uses_saved_facts_but_not_debug_traces(report_root):
    report_path = report_root / "fulloch-reports" / "2026-08-27-12345678.md"
    report_path.parent.mkdir()
    report_path.write_text(
        "# Deep Think Report\n\nThe installation cost is $4,000.", encoding="utf-8"
    )
    report_path.with_suffix(".evidence.json").write_text(
        json.dumps({"artifacts": {"artifact-001": {"data": {"price": 4000}}},
                    "model_budget": {"messages": ["DEBUG PROMPT AND TOKEN STATS"]}}), encoding="utf-8"
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
    assert '"price":4000' in captured["system_prompt"]
    assert "DEBUG PROMPT AND TOKEN STATS" not in captured["system_prompt"]
    assert "Compare heating systems" not in captured["system_prompt"]
    assert "history" not in captured
    assert captured["stats"] is stats and captured["cancel_check"] is cancel


def test_completed_report_persists_facts_once_not_worker_diagnostics(report_root):
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
    assert evidence["fact_cards"][0]["record"]["price"] == 4000
    assert evidence["schema_version"] == 2
    for key in ("evidence", "artifacts", "model_budget", "report_audit", "report_versions", "synthesis_state"):
        assert key not in evidence


def test_evidence_only_report_never_announces_a_completed_analysis(report_root):
    job = BackgroundJob(
        "12345678", JobSnapshot("Research event", origin_source="integration"),
        status=JobStatus.READY,
        summary="## Evidence-only partial answer\n\nA few extracted facts are available.",
        synthesis_state={"limited_evidence_fallback": True, "partial_report": True},
    )

    state = notifications(job)
    report = (report_root / f"{job.note_id}.md").read_text()

    assert "**Completed:**" not in report
    assert "## Research Notes" in report
    # A generated report is announced as generated; the reader learns about
    # unresolved questions from the note at the end, not from a "partial"
    # headline. Evidence-only reports still avoid claiming a completed analysis.
    assert state.history[0]["content"].startswith("Retained research notes (not verified answers) for:")
    message = state.emit.call_args.args[1]
    assert "saved" in message and "completed analysis" not in message
    assert "partial" not in message.casefold()
    assert state.emit.call_args.kwargs["artifact"]["analysis_status"] == "available"


def test_blocked_ready_outcome_saves_notes_without_positive_report_or_verified_fact_language(report_root):
    job = BackgroundJob("blocked-notes", JobSnapshot("Upcoming event", origin_source="integration"),
                        status=JobStatus.READY, summary="## Research limitation\n\nExact upcoming identity is unresolved.",
                        synthesis_state={"delivery_assessment": {"status": "blocked"}})
    state = notifications(job)
    saved = (report_root / f"{job.note_id}.md").read_text()
    assert saved.startswith("# Deep Think Research Notes")
    assert "not verified answers" in state.history[0]["content"]
    assert state.emit.call_args.kwargs["artifact"]["analysis_status"] == "available"
    assert "saved the research report" in state.emit.call_args.args[1]
    assert "finished looking" not in state.emit.call_args.args[1]


def test_saving_fallback_has_one_bibliography_and_keeps_real_limitations(report_root):
    from core.thinking_fact_cards import build_fact_cards
    from core.thinking_fallback import evidence_fallback
    from core.thinking_worker import _fallback_report_bindings

    task = "Research event"
    registry = {}
    cards = build_fact_cards([{"status": "evidence", "evidence": {"sources": [{
        "title": "Event bulletin", "url": "https://example.com/event",
        "evidence": "The event starts at 11:30 on Sunday.", "eligible": True,
    }]}}], {}, task, source_registry=registry)
    cards[0]["limitations"] = (
        "Only a bounded extract of this page was retained; it is not exhaustive.; "
        "Times are provisional."
    )
    job = BackgroundJob("12345678", JobSnapshot(task), fact_cards=cards, source_registry=registry)
    job.summary = evidence_fallback(task, cards, state=job.synthesis_state)
    job.report_bindings = _fallback_report_bindings(job, cards)
    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text()
    assert report.count("## References") == 1
    assert report.count("](https://example.com/event)") == 1
    assert "bounded extract" not in report
    assert "Times are provisional." in report


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


def test_round_trip_appendix_retains_both_legs_all_offers_and_one_sources_heading(report_root):
    outbound = {
        "airlines": ["Example Air"],
        "departure": {"id": "HAM", "time": "2027-01-17 10:00"},
        "arrival": {"id": "SYD", "time": "2027-01-18 08:00"},
    }
    inbound = {
        "airlines": ["Example Air"],
        "departure": {"id": "SYD", "time": "2027-03-03 21:45"},
        "arrival": {"id": "HAM", "time": "2027-03-04 19:05"},
    }
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Find flights"),
        summary="## Summary\n\nTwo complete return options were retrieved.\n\n## Sources\n\nModel source.",
        artifacts={
            "artifact-001": {
                "data": {
                    "type": "flight_search",
                    "journey_type": "round_trip",
                    "route": {"origin": "HAM", "destination": "SYD"},
                    "departure_date": "2027-01-17",
                    "return_date": "2027-03-03",
                    "currency": "AUD",
                    "offers": [
                        {"price": 2443, "outbound": outbound, "inbound": inbound},
                        {"price": 2504, "outbound": outbound, "inbound": inbound},
                    ],
                }
            }
        },
    )

    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")

    assert "outbound 2027-01-17 10:00 to 2027-01-18 08:00" in report
    assert "return 2027-03-03 21:45 to 2027-03-04 19:05" in report
    assert "AUD 2443" in report and "AUD 2504" in report
    assert report.count("## Sources") == 1
    assert "returning 2027-03-03" in report


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


def test_completed_card_report_renders_only_bound_source_metadata(report_root):
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Find schedule"),
        summary="## Summary\n\nA schedule was retrieved.",
        fact_cards=[
            {
                "id": "schedule-1",
                "compatibility": "compatible",
                "label": "Fallback label",
                "source_url": "https://example.com/schedule",
            },
            {
                "id": "rejected-1",
                "compatibility": "out_of_scope",
                "label": "Rejected source",
                "source_url": "https://example.com/rejected",
            },
        ],
        report_bindings=[
            {
                "card_id": "schedule-1",
                "source_url": "https://example.com/schedule",
                "source_title": "Official event schedule",
                "source_date": "2026-09-28",
            }
        ],
    )

    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")

    assert "[Official event schedule (2026-09-28)](https://example.com/schedule)" in report
    assert "Rejected source" not in report


def test_completed_report_adds_a_scoped_summary_when_worker_omits_one(report_root):
    job = BackgroundJob(
        "12345678", JobSnapshot("Compare options"), summary="Option A has the lower price."
    )
    note_id = reports.save_report(job)
    report = (report_root / f"{note_id}.md").read_text(encoding="utf-8")
    assert "## Summary" in report and "Option A has the lower price." in report
    assert "limited to the retrieved evidence" not in report


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


@pytest.mark.parametrize("backend_lost", [False, True])
def test_search_outage_failure_is_explained_without_raw_diagnostics(monkeypatch, backend_lost):
    save = MagicMock()
    monkeypatch.setattr(reports, "save_report", save)
    error = "ReportSynthesisError: No verified evidence was retrieved; refusing to mark a source-less report ready."
    job = BackgroundJob(
        "search-outage",
        JobSnapshot("Analyse the next Formula 1 race", origin_source="conversation",
                    origin_satellite_id="dashboard-text"),
        status=JobStatus.FAILED,
        error=error,
        evidence=[{"tool": "external_information", "status": "unavailable",
                   "scope": "HTTPConnectionPool: No route to host at a private address.",
                   "evidence": {"sources": []}}],
        synthesis_state={"backend_recovery": {"state": "waiting_for_health_check"}} if backend_lost else {},
    )

    state = notifications(job, pending={"dashboard-text": "search-outage"})

    message = (
        "I ran into a model error, and web search was unavailable, so I couldn't complete that report."
        if backend_lost else
        "Web search was unavailable, so I couldn't gather the sources needed for that report."
    )
    state.emit.assert_called_once_with("assistant", message, "proactive", satellite_id="dashboard-text")
    event = state.dispatch.call_args.args[0]
    assert event["summary"] == message
    assert event["error"] == event["terminal_reason"] == error
    assert job.snapshot.task_record["failure_reason"] == error
    assert not state.pending and not state.completed
    save.assert_not_called()


@pytest.mark.parametrize("error,later_search", [
    ("ReportSynthesisError: The final report generation returned no content.", None),
    ("ReportSynthesisError: No verified evidence was retrieved; refusing to mark a source-less report ready.",
     "rejected"),
    ("ReportSynthesisError: No verified evidence was retrieved; refusing to mark a source-less report ready.",
     "evidence"),
])
def test_earlier_search_outage_does_not_explain_an_unrelated_final_failure(error, later_search):
    evidence = [{"tool": "external_information", "status": "unavailable"}]
    if later_search:
        evidence.append({"tool": "external_information", "status": later_search})
    job = BackgroundJob(
        "later-failure", JobSnapshot("Compare options", origin_source="conversation",
                                     origin_satellite_id="dashboard-text"),
        status=JobStatus.FAILED,
        error=error,
        evidence=evidence,
    )

    state = notifications(job)

    assert state.emit.call_args.args[1] == "I couldn't complete that report."


def test_worker_search_outage_reaches_the_conversational_failure_notification(monkeypatch):
    from core.thinking_worker import ReportSynthesisError
    from tests.test_thinking_worker import _web_worker
    from tools.tool_registry import ThinkingResult

    save = MagicMock()
    monkeypatch.setattr(reports, "save_report", save)
    invoke = MagicMock(return_value=ThinkingResult(
        "Search service could not be reached.", status="unavailable",
        scope="The web search did not return usable source material.",
        evidence={"query": "next Formula 1 race", "sources": []},
    ))
    service, _, _ = _web_worker([
        '{"tool":"external_information","args":["next Formula 1 race","research"]}',
    ], invoke)
    job = BackgroundJob(
        "worker-search-outage",
        JobSnapshot("Analyse the next Formula 1 race", origin_source="conversation",
                    origin_satellite_id="dashboard-text"),
    )

    with pytest.raises(ReportSynthesisError) as failure:
        service.run(job, lambda: False)
    job.status = JobStatus.FAILED
    job.error = f"{type(failure.value).__name__}: {failure.value}"
    state = notifications(job)

    invoke.assert_called_once()
    assert job.evidence[-1]["tool"] == "external_information"
    assert job.evidence[-1]["status"] == "unavailable"
    assert state.emit.call_args.args[1] == (
        "Web search was unavailable, so I couldn't gather the sources needed for that report."
    )
    assert state.dispatch.call_args.args[0]["error"] == job.error
    save.assert_not_called()


def test_failed_job_exports_verified_partial_report(report_root):
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Compare options", origin_source="conversation", origin_satellite_id="dashboard-text",
                    origin_turn_id="turn-123"),
        status=JobStatus.FAILED,
        summary="## Option comparison\n\nVerified finding. <!-- cards: C1 -->",
        error="RemoteUnreachable: audit connection dropped",
        report_bindings=[{"card_id": "C1"}],
        synthesis_state={"partial_report": True, "failed_stage": "Chapter 2", "process_generation": 7},
    )

    state = notifications(job)

    assert job.note_id
    saved = report_root / f"{job.note_id}.md"
    assert saved.is_file()
    assert "Verified finding." in saved.read_text(encoding="utf-8")
    event = state.dispatch.call_args.args[0]
    assert event["status"] == JobStatus.FAILED
    assert event["note_id"] == job.note_id
    assert event["partial_artifact_link"] == f"/reports/{job.note_id}"
    assert event["elapsed_seconds"] >= 0
    assert event["terminal_reason"] == "RemoteUnreachable: audit connection dropped"
    assert job.snapshot.task_record["failure_stage"] == "Chapter 2"
    assert job.snapshot.task_record["failure_reason"] == "RemoteUnreachable: audit connection dropped"
    assert job.snapshot.task_record["partial_artifact_link"] == f"/reports/{job.note_id}"
    assert event["turn_id"] == "turn-123"
    assert event["process_generation"] == 7
    state.emit.assert_called_once_with(
        "assistant", "I couldn't complete that report, but I've saved the supported partial work and research notes.",
        "proactive", satellite_id="dashboard-text",
    )


def test_queued_and_running_lifecycle_events_have_no_terminal_reason():
    for status in (JobStatus.QUEUED, JobStatus.RUNNING):
        job = BackgroundJob(
            f"lifecycle-{status.value.lower()}", JobSnapshot("Research a topic"),
            status=status, stage="Collecting evidence",
        )

        state = notifications(job)
        event = state.dispatch.call_args.args[0]

        assert event["terminal_reason"] == ""
        assert event["status"] == status


def test_running_progress_event_exposes_saved_sections_and_open_obligations():
    job = BackgroundJob(
        "progress-job", JobSnapshot("Compare Alpha and Beta"),
        status=JobStatus.RUNNING, stage="Writing section 2",
        chapter_plan=[{"id": "chapter-01", "body": "Accepted cited section.",
                       "analysis_state": "complete"}],
        evidence_requirements=[
            {"identifier": "identity", "state": "resolved"},
            {"identifier": "comparison", "state": "open"},
        ],
    )

    event = notifications(job).dispatch.call_args.args[0]

    assert event["progress"]["saved_sections"] == 1
    assert event["progress"]["unresolved_obligations"] == ["comparison"]
    assert event["terminal_reason"] == ""


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
        "summary": reports.report_summary(summary),
        "report_url": "/reports/fulloch-reports/2026-08-27-12345678",
        "analysis_status": "available",
        "progress": {
            "discovered_documents": 0, "fetched_documents": 0,
            "processed_unique_documents": 0, "eligible_documents": 0,
            "unprocessed_documents": 0, "rejected_documents": 0,
            "cited_documents": 0, "retrieval_artifacts": 0,
            "saved_sections": 0, "unresolved_obligations": [],
        },
    }
    if source is not None:
        expected["data"] = source
    state.emit.assert_called_once_with(
        "assistant",
        "I've saved the research report.",
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
        assert state.pending == {
            "kitchen": {"job_id": "12345678", "task": "Investigate", "question": "Which city?"}
        }
        assert state.history[0]["content"] == job.summary


def test_final_citation_guard_drops_paragraphs_missing_a_reference_entry():
    report = "Supported statement. [1]\n\nUnattached statement. [2]"
    references = "## References\n\n1. [Source](https://example.test/source)"

    assert reports._retain_referenced_paragraphs(report, references) == "Supported statement. [1]"
