"""Context-overflow handling.

When a turn's assembled prompt no longer fits the SLM context window, the
server returns a typed context error that the agent loop recovers from. These
tests pin the recovery behaviour:

   1. The agent loop catches `ContextExhaustedError` at both history-carrying
      SLM call sites. The
     recovery wrapper sheds the oldest history and retries (so a long
     conversation degrades gracefully); only when even the recent floor won't
     fit does it clear `_history` and return a spoken apology.
"""

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

def _import_assistant_module():
    """Import core.assistant with audio/asr/tts stubbed (core.slm stays real)."""
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice",
            "warmup_model",
            "synthesize",
            "play_chunks",
            "speak_stream",
            "set_output_device",
            "set_tts_active_event",
            "model",
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
    fake = SimpleNamespace(
        _history=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
    )
    result = a.Assistant._context_exhausted_reply(fake)
    assert result == a.CONTEXT_EXHAUSTED_REPLY
    assert result == "I've lost some conversation context. Could you ask that again?"
    assert fake._history == []


def test_custom_llm_uses_its_filename_in_loading_status():
    a = _import_assistant_module()
    spec = SimpleNamespace(
        default_model="./data/models/qwen3.5-9b-mtp/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        display_name="Qwen3.5 9B MTP (local)",
    )
    cfg = {
        "backend": "llama",
        "model": "./data/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf",
        "spec": spec,
    }
    assert a.Assistant._loading_display_name(cfg) == "Custom local model (Qwen3.6-35B-A3B-UD-IQ4_NL.gguf)"


def test_both_slm_calls_guard_context_exhaustion():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # Agent call + thinking call must each be wrapped.
    assert src.count("except ContextExhaustedError") == 2
    assert "_context_exhausted_reply()" in src


def test_shed_oldest_history_keeps_recent_and_turn_boundary():
    """Shedding drops the oldest entries, keeps the recent tail, and leaves the
    trimmed history starting on a `user` turn boundary."""
    a = _import_assistant_module()
    fake = SimpleNamespace(
        _history=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
        ]
    )
    assert a.Assistant._shed_oldest_history(fake) is True
    assert len(fake._history) < 8
    assert fake._history[0]["role"] == "user"  # no orphaned tool/assistant head


def test_shed_oldest_history_at_floor_returns_false():
    """At/below the recent floor there's nothing safe to shed → caller clears."""
    a = _import_assistant_module()
    fake = SimpleNamespace(_history=[{"role": "user", "content": "x"}] * 3)
    assert a.Assistant._shed_oldest_history(fake) is False
    assert len(fake._history) == 3  # untouched


def test_recovery_sheds_then_retries_instead_of_clearing(monkeypatch):
    """On overflow the wrapper sheds oldest history and retries — and succeeds
    with the conversation tail intact, rather than wiping everything."""
    a = _import_assistant_module()
    fake = SimpleNamespace(
        slm_model=object(), _history=[{"role": "user", "content": f"m{i}"} for i in range(8)]
    )
    fake._shed_oldest_history = lambda: a.Assistant._shed_oldest_history(fake)

    calls = {"n": 0}

    def fake_gen(_model, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise a.ContextExhaustedError("too big")
        return "ok"

    monkeypatch.setattr(a, "generate_slm", fake_gen)
    result = a.Assistant._generate_with_context_recovery(fake, history=fake._history)
    assert result == "ok"
    assert calls["n"] == 2  # failed once, retried once
    assert 0 < len(fake._history) < 8  # tail preserved, not cleared


def test_recovery_clears_history_and_retries_current_turn(monkeypatch):
    """If the recent floor overflows, discard it and retry the current turn."""
    a = _import_assistant_module()
    fake = SimpleNamespace(
        slm_model=object(),
        _history=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ],
    )
    fake._shed_oldest_history = lambda: a.Assistant._shed_oldest_history(fake)

    calls = {"n": 0}

    def overflow_then_succeed(_model, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise a.ContextExhaustedError("nope")
        return "ok"

    monkeypatch.setattr(a, "generate_slm", overflow_then_succeed)
    assert a.Assistant._generate_with_context_recovery(fake, history=fake._history) == "ok"
    assert fake._history == [{"role": "user", "content": "x"}]
    assert calls["n"] == 2


def test_recovery_reraises_when_empty_context_overflows(monkeypatch):
    """Only a request too large for an empty context reaches the apology path."""
    a = _import_assistant_module()
    fake = SimpleNamespace(slm_model=object(), _history=[])
    fake._shed_oldest_history = lambda: a.Assistant._shed_oldest_history(fake)

    def always_overflow(_model, **_kw):
        raise a.ContextExhaustedError("nope")

    monkeypatch.setattr(a, "generate_slm", always_overflow)
    with pytest.raises(a.ContextExhaustedError):
        a.Assistant._generate_with_context_recovery(fake, history=fake._history)
