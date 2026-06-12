"""Context-overflow handling.

When a turn's assembled prompt no longer fits the SLM context window the old
behaviour was a silent failure: llama.cpp raised an opaque ValueError that the
caller's generic `except` only logged, so the user heard nothing. These tests
pin the replacement:

  1. `generate_slm` raises a typed `ContextExhaustedError` — proactively from a
     token estimate, and as a backstop around llama.cpp's own overflow error.
  2. The agent loop catches it at both history-carrying SLM call sites, clears
     `_history`, and returns a spoken apology so the user knows what happened.
"""

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slm import ContextExhaustedError, generate_slm  # noqa: E402


class _FakeModel:
    """Minimal stand-in for the llama.cpp model surface generate_slm touches."""

    def __init__(self, n_ctx, prompt_tokens, raise_overflow=False):
        self._n_ctx = n_ctx
        self._prompt_tokens = prompt_tokens
        self._raise_overflow = raise_overflow

    def n_ctx(self):
        return self._n_ctx

    def tokenize(self, _b):
        return [0] * self._prompt_tokens

    def create_chat_completion(self, **_kw):
        # llama.cpp raises the overflow error lazily, when the stream is first
        # iterated — mirror that so the backstop's placement is exercised.
        if self._raise_overflow:
            raise ValueError(
                "Requested tokens (99) exceed context window of 16384"
            )
        yield {"choices": [{"delta": {"content": "ok"}}]}


def test_preflight_raises_when_prompt_exceeds_budget():
    # prompt_tokens == n_ctx leaves zero headroom for the reply → overflow.
    model = _FakeModel(n_ctx=1000, prompt_tokens=1000)
    with pytest.raises(ContextExhaustedError):
        generate_slm(model, user_prompt="hi", system_prompt="sys")


def test_preflight_passes_within_budget():
    model = _FakeModel(n_ctx=16384, prompt_tokens=100)
    assert generate_slm(model, user_prompt="hi", system_prompt="sys") == "ok"


def test_backstop_retypes_llama_overflow():
    # Preflight undershoots (tokenize reports a tiny count) but llama.cpp still
    # overflows on eval — the ValueError must be re-typed, not leaked raw.
    model = _FakeModel(n_ctx=16384, prompt_tokens=10, raise_overflow=True)
    with pytest.raises(ContextExhaustedError):
        generate_slm(model, user_prompt="hi", system_prompt="sys")


def test_unrelated_valueerror_not_swallowed():
    class _Boom(_FakeModel):
        def create_chat_completion(self, **_kw):
            raise ValueError("something else entirely")
            yield  # pragma: no cover - keeps this a generator

    with pytest.raises(ValueError) as exc:
        generate_slm(_Boom(16384, 10), user_prompt="hi", system_prompt="sys")
    assert not isinstance(exc.value, ContextExhaustedError)


def _import_assistant_module():
    """Import core.assistant with audio/asr/tts stubbed (core.slm stays real)."""
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice", "warmup_model", "synthesize", "play_chunks",
            "speak_stream", "set_output_device", "set_tts_active_event", "model",
        ],
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


def test_context_exhausted_reply_clears_history():
    a = _import_assistant_module()
    fake = SimpleNamespace(_history=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ])
    result = a.Assistant._context_exhausted_reply(fake)
    assert result == a.CONTEXT_EXHAUSTED_REPLY
    assert fake._history == []


def test_both_slm_calls_guard_context_exhaustion():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop.run)
    # Agent call + thinking call must each be wrapped.
    assert src.count("except ContextExhaustedError") == 2
    assert "_context_exhausted_reply()" in src
