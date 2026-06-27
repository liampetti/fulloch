"""The free-text spoken summaries (web-search result + cancelled thinking) run
without the agent GBNF grammar, so on a remote LLM nothing bounds their length.
Both must pass a finite max_new_tokens cap, or a turn balloons to thousands of
tokens (~45s) instead of a few sentences. See SUMMARY_MAX_NEW_TOKENS."""

import types
from unittest.mock import patch

from core.assistant import SUMMARY_MAX_NEW_TOKENS, Assistant


def _capture_generate():
    captured = {}

    def fake_generate(model, **kw):
        captured.update(kw)
        return "a short answer"

    return captured, fake_generate


def test_web_summariser_caps_output_tokens():
    captured, fake = _capture_generate()
    self_ = types.SimpleNamespace(slm_model=object(), web_summary_prompt="sys")
    with patch("core.assistant.generate_slm", fake):
        out = Assistant._summarise_search_result(self_, "raw snippets", cancel_check=None)
    assert out == "a short answer"
    assert captured["max_new_tokens"] == SUMMARY_MAX_NEW_TOKENS


def test_partial_thinking_summary_caps_output_tokens():
    import core.assistant as a

    captured, fake = _capture_generate()
    self_ = types.SimpleNamespace(
        slm_model=object(),
        greeting_prompt="sys",
        _last_thinking_partial="some reasoning",
        _last_thinking_question="q",
        _last_thinking_cancelled_at=0.0,
    )
    # Within the TTL so the summary actually runs (monotonic ~ small at import).
    with (
        patch.object(a, "THINKING_PARTIAL_TTL_S", 1e12),
        patch("core.assistant.get_partial_thinking_summary_prompt", lambda q, p: "u"),
        patch("core.assistant.generate_slm", fake),
    ):
        out = Assistant._summarise_partial_thinking(self_, cancel_check=None)
    assert out == "a short answer"
    assert captured["max_new_tokens"] == SUMMARY_MAX_NEW_TOKENS
