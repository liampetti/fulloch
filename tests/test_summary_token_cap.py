"""Web-search summaries run without the agent GBNF grammar, so output is capped."""

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
