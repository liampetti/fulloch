"""Tests for thinking-mode wiring: deep_think tool + generate_slm `/think`."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import slm  # noqa: E402
from tools import thinking  # noqa: E402


class TestDeepThinkTool:
    def test_returns_thinking_prefix(self):
        result = thinking.deep_think("whether to switch banks")
        assert result.startswith(thinking.THINKING_PREFIX)
        assert "whether to switch banks" in result

    def test_prefix_constant_matches_returned_text(self):
        # The assistant uses THINKING_PREFIX as a marker — keeping them in
        # sync via the constant means renaming only happens in one place.
        result = thinking.deep_think("anything")
        assert thinking.THINKING_PREFIX in result


class TestSummarizeThinkingTool:
    def test_returns_summary_prefix(self):
        result = thinking.summarize_thinking()
        assert result == thinking.SUMMARY_PREFIX

    def test_summary_prefix_distinct_from_thinking_prefix(self):
        # The two sentinels must differ so _handle_wakeword can route them
        # separately. Catches accidental reuse if anyone renames either.
        assert thinking.SUMMARY_PREFIX != thinking.THINKING_PREFIX


class TestGenerateSlmThinkingMode:
    """Verify the /think directive is appended only when requested."""

    @pytest.fixture
    def fake_model(self):
        """A stand-in for Llama that records the messages it was given."""
        model = MagicMock()
        model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "ok"}}]}
        ])
        return model

    def test_appends_think_directive_when_enabled(self, fake_model):
        slm.generate_slm(
            fake_model,
            user_prompt="what should I do",
            thinking_mode=True,
        )
        messages = fake_model.create_chat_completion.call_args.kwargs["messages"]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"].endswith("/think")
        assert "what should I do" in user_msg["content"]

    def test_does_not_append_think_directive_by_default(self, fake_model):
        slm.generate_slm(
            fake_model,
            user_prompt="what should I do",
        )
        messages = fake_model.create_chat_completion.call_args.kwargs["messages"]
        user_msg = messages[-1]
        assert user_msg["content"] == "what should I do"

    def test_thinking_mode_with_history(self, fake_model):
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
        ]
        slm.generate_slm(
            fake_model,
            user_prompt="now",
            history=history,
            thinking_mode=True,
        )
        messages = fake_model.create_chat_completion.call_args.kwargs["messages"]
        # System + history (2) + new user (1) = 4
        assert len(messages) == 4
        # Only the new user message gets the directive, not the history
        assert messages[1]["content"] == "earlier"
        assert messages[2]["content"] == "reply"
        assert messages[3]["content"].endswith("/think")


class TestThinkingPromptStripping:
    """The <think>...</think> block should not reach the TTS path."""

    def test_clean_for_tts_strips_think_blocks(self):
        from core.text_utils import clean_for_tts
        raw = (
            "<think>OK so they're asking about cars. "
            "Let me weigh the options...</think>"
            "Electric cars are probably worth it if you have home charging."
        )
        cleaned = clean_for_tts(raw)
        assert "<think>" not in cleaned
        assert "weigh the options" not in cleaned
        assert "Electric cars" in cleaned


def _import_assistant_module():
    """Import core.assistant with the heavy audio/ASR/SLM/TTS deps stubbed."""
    import types
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice", "warmup_model", "synthesize", "play_chunks",
            "speak_stream", "set_output_device", "set_tts_active_event", "model",
        ],
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        sys.modules[name] = mod
    import core.assistant as assistant  # noqa: E402
    return assistant


class TestThinkingLoopFix:
    """deep_think must run a single out-of-loop free-text call and answer —
    not re-enter the grammar-constrained agent loop (which re-emitted
    deep_think until MAX_AGENT_CALLS and never produced an answer).
    """

    def test_thinking_uses_free_text_prompt(self):
        import inspect
        a = _import_assistant_module()
        src = inspect.getsource(a.Assistant._handle_wakeword)
        # The thinking branch runs the dedicated free-text prompt...
        assert "get_thinking_system_prompt" in src
        # ...and does NOT pass the agent grammar on that call (free text so
        # Qwen3 can emit a <think> block the grammar would otherwise forbid).
        snippet = src[src.index("if saw_thinking"):]
        assert "grammar=" not in snippet, "thinking call must not be grammar-constrained"

    def test_no_thinking_replan_loop(self):
        import inspect
        a = _import_assistant_module()
        src = inspect.getsource(a.Assistant._handle_wakeword)
        # The old looping flag is gone — thinking terminates the turn.
        assert "thinking_for_next" not in src

    def test_thinking_branch_returns(self):
        import inspect
        a = _import_assistant_module()
        src = inspect.getsource(a.Assistant._handle_wakeword)
        # The saw_thinking block ends by returning the cleaned answer rather
        # than `continue`-ing back into the loop.
        block = src[src.index("if saw_thinking"):]
        block = block[: block.index("if replan")] if "if replan" in block else block
        assert "return cleaned" in block
        assert "continue" not in block
