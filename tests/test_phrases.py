"""Hand-authored acknowledgements stay concise and conversational."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.phrases import ACK_CACHE_ATTRS, ACK_PHRASES  # noqa: E402


def test_acknowledgements_are_short_and_not_progress_narration():
    assert ACK_PHRASES == ["Okay.", "Yep.", "Aha."]
    assert "working" not in " ".join(ACK_PHRASES).lower()


def test_normal_stall_contexts_reuse_the_acknowledgements():
    assert "ack_cache" in ACK_CACHE_ATTRS
    assert "web_search_stall_cache" not in ACK_CACHE_ATTRS
    assert "thinking_stall_cache" in ACK_CACHE_ATTRS
