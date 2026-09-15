"""Dependency-light scheduling tests for bounded ASR work."""

import sys
import threading
from pathlib import Path

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
    assert work.get_nowait() == ("candidate",)
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


def test_verification_precedes_final_and_full_candidates_are_never_evicted():
    work = AsrWorkQueue(maxsize=2)
    verification = AsrWorkItem(("verify",), "sat-a", "wake_verification", True)
    final = AsrWorkItem(("final",), "sat-a", "wake_candidate", True)
    assert work.offer(verification) == (True, [])
    assert work.offer(final) == (True, [])
    assert work.offer(AsrWorkItem(("other",), "sat-b", "wake_candidate", True)) == (False, [])
    assert work.get() == ("verify",)
    assert work.get() == ("final",)
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
