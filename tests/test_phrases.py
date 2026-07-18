"""Hand-authored spoken phrases should not promise imminent completion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.phrases import REPLAN_STALL_PHRASES, THINKING_STALL_PHRASES  # noqa: E402


def test_progress_phrases_do_not_promise_completion_timing():
    assert "Almost there." not in REPLAN_STALL_PHRASES
    assert "Almost there." not in THINKING_STALL_PHRASES
