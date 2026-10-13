import threading

from core.idle_summaries import IdleSummaryQueue, validate_summary


def _summary(refs=None):
    return {
        "scope": "Question about route A",
        "findings": ["Verified finding"],
        "open_tasks": ["Check return timing"],
        "execution_outcomes": ["Search completed"],
        "source_references": refs or [],
    }


def test_summary_requires_all_fields_and_valid_source_references():
    assert validate_summary(_summary(["source-1"]), {"source-1"})
    assert validate_summary(_summary(["unknown"]), {"source-1"}) is None
    assert validate_summary({"scope": "incomplete"}, set()) is None


def test_stale_or_busy_summary_does_not_apply_and_only_one_is_queued():
    queue = IdleSummaryQueue()
    completed = threading.Event()
    version = ["v1"]
    applied = []

    def summarize():
        version[0] = "v2"
        completed.set()
        return _summary()

    def apply(value):
        applied.append(value)
        completed.set()

    assert queue.submit(
        "history", "v1", summarize=summarize, current_version=lambda: version[0],
        apply=apply, valid_source_refs=set(), is_idle=lambda: True,
    )
    assert not queue.submit(
        "history", "v1", summarize=_summary, current_version=lambda: version[0],
        apply=apply, valid_source_refs=set(), is_idle=lambda: True,
    )
    assert completed.wait(timeout=2)
    assert applied == []


def test_foreground_activity_prevents_idle_summary_call():
    queue = IdleSummaryQueue()
    calls = []
    assert not queue.submit(
        "job", "1", summarize=lambda: calls.append(True), current_version=lambda: "1",
        apply=lambda _value: None, valid_source_refs=set(), is_idle=lambda: False,
    )
    assert calls == []
