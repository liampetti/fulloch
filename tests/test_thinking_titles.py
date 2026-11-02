"""Conversation-aware research titles and durable report presentation."""

from core import thinking_reports
from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus
from core.research_task import resolve_approved_research


def test_approval_uses_assignment_and_recent_user_context():
    history = [
        {"role": "user", "content": "How are my shares doing?"},
        {"role": "assistant", "content": "Would you like a report?"},
        {"role": "user", "content": "Should I buy, sell or hold?"},
        {"role": "user", "content": "yes"},
    ]
    task = resolve_approved_research("yes", {"research_task": "Analyze my portfolio"}, history)
    assert "Analyze my portfolio" in task
    assert "How are my shares doing?" in task
    assert "Should I buy, sell or hold?" in task
    assert resolve_approved_research("yes", None, history).startswith("Should I buy, sell or hold?")
    assert resolve_approved_research("Research solar", None, history) == "Research solar"
    assert resolve_approved_research("yes", None, []) == "yes"


def test_title_survives_checkpoint_and_matches_status_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(thinking_reports.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(thinking_reports.notes, "_after_write", lambda _: None)
    title = "Recent news and price movement for held shares"
    job = BackgroundJob("12345678abcdef", JobSnapshot("Analyze my shares"), title=title)
    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    assert restored.view()["title"] == title
    restored.status = JobStatus.READY
    restored.summary = "## Summary\n\nA finding."
    note_id = thinking_reports.save_report(restored)
    assert (tmp_path / (note_id + ".md")).read_text().startswith("# " + title + "\n")
    assert thinking_reports.read_report({"note_id": note_id}).startswith("# " + title + "\n")


def test_title_fallback_does_not_display_approval():
    assert BackgroundJob("id", JobSnapshot("yes")).display_title() == "Research investigation"
    assert BackgroundJob("id", JobSnapshot("Compare solar options")).display_title() == "Compare solar options"
