"""Dependency-light scheduling tests for bounded ASR work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio import AsrWorkItem, AsrWorkQueue


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
