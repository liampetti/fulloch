"""Hand-authored acknowledgements stay concise and unambiguous."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.phrases import ACK_CACHE_ATTRS, ACK_PHRASES  # noqa: E402


def test_acknowledgements_are_short_and_not_progress_narration():
    answer_words = {
        "yes",
        "yep",
        "yeah",
        "yup",
        "affirmative",
        "correct",
        "right",
        "no",
        "nope",
        "nah",
        "never",
        "cannot",
        "cant",
        "dont",
        "wont",
        "maybe",
        "perhaps",
        "possibly",
        "probably",
        "likely",
        "unlikely",
        "depends",
        "unsure",
        "unknown",
    }
    progress_words = {
        "working",
        "searching",
        "checking",
        "looking",
        "finding",
        "fetching",
        "loading",
        "processing",
        "thinking",
        "calculating",
    }

    assert ACK_PHRASES
    assert len(ACK_PHRASES) == len(set(ACK_PHRASES))
    for phrase in ACK_PHRASES:
        words = phrase.removesuffix(".").lower().split()
        assert words
        assert len(words) <= 3
        assert phrase.endswith(".")
        assert "?" not in phrase
        assert "!" not in phrase
        assert not answer_words.intersection(words)
        assert not progress_words.intersection(words)


def test_normal_stall_contexts_reuse_the_acknowledgements():
    assert "ack_cache" in ACK_CACHE_ATTRS
    assert "web_search_stall_cache" not in ACK_CACHE_ATTRS
    assert "thinking_stall_cache" in ACK_CACHE_ATTRS
