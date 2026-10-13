"""Failure-diagnosis + ERROR-state escalation.

When a backend can't load (or the running model crashes — typically out of GPU
memory), the assistant should flip the app to the ERROR lifecycle phase with a
plain-language explanation, so the web UI bounces the user to the setup screen's
red alert instead of hanging on a spinner / failing silently turn after turn.

The full Assistant is too heavy to instantiate, so we exercise the pure helpers
(`_diagnose_failure`, `_enter_error_state`, `_note_runtime_error`) on a bare
instance built with `__new__`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _bare_assistant():
    # Importing the module is cheap (model loads are deferred); only *constructing*
    # an Assistant pulls in audio/ASR/TTS, which we avoid via __new__. Skip cleanly
    # if the heavy import deps aren't present (e.g. the CPU-only test image).
    try:
        import core.assistant as assistant
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core.assistant import unavailable: {exc}")
    return assistant.Assistant.__new__(assistant.Assistant)


class _Spec:
    def __init__(self, display_name, vram_gb=0):
        self.display_name = display_name
        self.vram_gb = vram_gb


def test_oom_is_fatal_with_friendly_message():
    inst = _bare_assistant()
    # The opaque llama.cpp OOM-at-context-creation error.
    fatal, msg = inst._diagnose_failure(
        ValueError("Failed to create llama_context"),
        spec=_Spec("Gemma 4 12B QAT (local)", vram_gb=8),
    )
    assert fatal is True
    assert "memory" in msg.lower()
    # Actionable: tells the user to pick something lighter.
    assert "lighter" in msg.lower() or "smaller" in msg.lower()


def test_cuda_error_is_fatal():
    inst = _bare_assistant()
    fatal, msg = inst._diagnose_failure(RuntimeError("CUDA error: an illegal memory access"))
    assert fatal is True


def test_ordinary_error_is_not_fatal():
    inst = _bare_assistant()
    # A plain value error (e.g. a bad config field) shouldn't nuke the app.
    fatal, _ = inst._diagnose_failure(ValueError("missing required field"))
    assert fatal is False


def test_enter_error_state_sets_lifecycle_error():
    inst = _bare_assistant()

    class _LC:
        def __init__(self):
            self.calls = []

        def set(self, phase, detail="", **extra):
            self.calls.append((phase, detail))

    inst.lifecycle = _LC()
    inst._enter_error_state("too big for your card")
    assert inst.lifecycle.calls == [("ERROR", "too big for your card")]


def test_enter_error_state_noop_without_lifecycle():
    inst = _bare_assistant()
    inst.lifecycle = None
    inst._enter_error_state("whatever")  # must not raise


def test_note_runtime_error_escalates_only_fatal():
    inst = _bare_assistant()

    class _LC:
        def __init__(self):
            self.phase = None

        def set(self, phase, detail="", **extra):
            self.phase = phase

    inst.lifecycle = _LC()
    inst._note_runtime_error(ValueError("just a hiccup"))
    assert inst.lifecycle.phase is None  # transient -> no escalation

    inst._note_runtime_error(RuntimeError("CUDA error: out of memory"))
    assert inst.lifecycle.phase == "ERROR"  # fatal -> bounce to setup


def test_recovery_status_resolves_once_and_enters_current_prompt(monkeypatch):
    import core.assistant as assistant

    inst = _bare_assistant()
    inst._history = []
    inst.thinking_jobs = None
    inst.slm_model = SimpleNamespace(_fulloch_backend_state="READY", _fulloch_process_generation=2)
    inst._backend_status_events = {
        "dashboard-text": {"state": "OUTAGE", "generation": 1, "occurred_at": 1.0}
    }
    events = []
    prompts = []
    inst._emit_turn_event = lambda *args, **kwargs: events.append((args, kwargs))
    monkeypatch.setattr(
        assistant.turn_history,
        "generate_with_recovery",
        lambda _model, **kwargs: prompts.append(kwargs["system_prompt"]) or "ok",
    )

    assert inst._generate_with_context_recovery(system_prompt="agent prompt") == "ok"
    assert "currently available" in prompts[0]
    assert inst._backend_status_events["dashboard-text"]["state"] == "RESOLVED"
    assert len(events) == 1
    assert events[0][0][1] == "The AI server is back online"

    inst._generate_with_context_recovery(system_prompt="next prompt")
    assert len(events) == 1


def test_old_backend_generation_cannot_resolve_newer_outage_callback(monkeypatch):
    import core.assistant as assistant

    inst = _bare_assistant()
    inst._history = []
    inst.thinking_jobs = None
    inst.slm_model = SimpleNamespace(_fulloch_backend_state="READY", _fulloch_process_generation=3)
    event = {"state": "OUTAGE", "generation": 4, "occurred_at": 1.0}
    inst._backend_status_events = {"dashboard-text": event}
    announced = []
    prompts = []
    inst._emit_turn_event = lambda *args, **kwargs: announced.append(args)
    monkeypatch.setattr(
        assistant.turn_history, "generate_with_recovery",
        lambda _model, **kwargs: prompts.append(kwargs["system_prompt"]) or "greeting",
    )

    assert inst._generate_with_context_recovery(system_prompt="Greeting") == "greeting"
    assert event["state"] == "OUTAGE"
    assert announced == []

    inst.slm_model._fulloch_process_generation = 4
    assert inst._generate_with_context_recovery(system_prompt="Greeting") == "greeting"
    assert event["state"] == "RESOLVED"
    assert len(announced) == 1
    assert "currently available" in prompts[-1]

    inst._generate_with_context_recovery(system_prompt="Next greeting")
    assert len(announced) == 1
