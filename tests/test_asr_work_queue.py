"""Dependency-light scheduling tests for bounded ASR work."""

import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.asr_work_queue import AsrWorkItem, AsrWorkQueue


def test_wake_candidate_evicts_oldest_ordinary_work_and_runs_first():
    work = AsrWorkQueue(maxsize=2)
    first = AsrWorkItem(("first",), "sat-a", "final")
    second = AsrWorkItem(("second",), "sat-b", "final")
    candidate = AsrWorkItem(("candidate",), "sat-c", "wake_candidate", candidate=True)

    assert work.offer(first) == (True, [])
    assert work.offer(second) == (True, [])
    admitted, evicted = work.offer(candidate)

    assert admitted is True
    assert evicted == [first]
    assert work.get_nowait()[:1] == ("candidate",)
    assert work.get_nowait() == ("second",)


def test_wake_candidate_queue_allows_one_pending_item_per_satellite():
    work = AsrWorkQueue(maxsize=2)
    first = AsrWorkItem(("first",), "sat-a", "wake_candidate", candidate=True)
    retry = AsrWorkItem(("retry",), "sat-a", "wake_candidate", candidate=True)

    assert work.offer(first) == (True, [])
    assert work.offer(retry) == (False, [])


def test_discarding_one_satellite_retains_other_work():
    work = AsrWorkQueue(maxsize=3)
    candidate = AsrWorkItem(("candidate",), "sat-a", "wake_candidate", candidate=True)
    ordinary = AsrWorkItem(("ordinary",), "sat-b", "final")

    work.offer(candidate)
    work.offer(ordinary)

    assert work.discard("sat-a") == [candidate]
    assert work.get_nowait() == ("ordinary",)


def test_capture_discard_preserves_newer_other_satellite_and_legacy_work():
    work = AsrWorkQueue(maxsize=5)

    def item(sid, capture_id, candidate=False):
        payload = (None,) * 12 + (capture_id,)
        return AsrWorkItem(payload, sid, "wake_candidate" if candidate else "final", candidate)

    committed = item("sat-a", 1, candidate=True)
    newer = item("sat-a", 2)
    other = item("sat-b", 1)
    legacy = AsrWorkItem(("legacy",), "sat-a", "final")
    for entry in (committed, newer, other, legacy):
        assert work.offer(entry)[0]

    assert work.discard("sat-a", capture_id=1) == [committed]
    assert work.get_nowait() == newer.payload
    assert work.get_nowait() == other.payload
    assert work.get_nowait() == legacy.payload
    # Removed candidate keys must not poison future admission.
    assert work.offer(item("sat-a", 2, candidate=True))[0]


def test_verification_precedes_final_and_full_candidates_are_never_evicted():
    work = AsrWorkQueue(maxsize=2)
    verification = AsrWorkItem(("verify",), "sat-a", "wake_verification", True)
    final = AsrWorkItem(("final",), "sat-a", "wake_candidate", True)
    assert work.offer(verification) == (True, [])
    assert work.offer(final) == (True, [])
    assert work.offer(AsrWorkItem(("other",), "sat-b", "wake_candidate", True)) == (False, [])
    assert work.get()[:1] == ("verify",)
    assert work.get()[:1] == ("final",)
    assert work.offer(verification) == (True, [])


def test_close_releases_consumer_and_rejects_new_work():
    work = AsrWorkQueue(maxsize=1)
    result = []
    consumer = threading.Thread(target=lambda: result.append(work.get()))
    consumer.start()
    work.close()
    consumer.join(timeout=1)
    assert not consumer.is_alive()
    assert result == [None]
    assert work.offer(AsrWorkItem(("late",), "sat-a", "final")) == (False, [])


def test_expired_candidate_is_dropped_before_asr_and_reported(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    expired = []
    work = AsrWorkQueue(maxsize=2, candidate_ttl_s=1, on_expired=expired.append)
    candidate = AsrWorkItem(("audio",) + (None,) * 13, "sat-a", "wake_candidate", True)
    assert work.offer(candidate) == (True, [])

    now[0] = 12.0
    with pytest.raises(queue.Empty):
        work.get_nowait()

    assert expired == [candidate]
    assert work.qsize() == 0


def test_live_candidate_payload_carries_expiry_for_result_admission():
    work = AsrWorkQueue(maxsize=1, candidate_ttl_s=30)
    candidate = AsrWorkItem(("audio",), "sat-a", "wake_candidate", True)
    work.offer(candidate)
    payload = work.get_nowait()

    assert payload[0] == "audio"
    assert isinstance(payload[1], float)
