"""A2: AgentLoop sets `TurnStats.route` ("regex" / "agent" / "no_llm") so
`GET /status` and the structured per-turn log line can show which path
resolved a turn. Host stubs mirror the pattern in test_llm_openai.py's
`test_agent_loop_degrades_to_regex_on_remote_unreachable` and test_intents.py's
`TestNoLlmReactive` — the minimal set of host attributes `AgentLoop._run`
touches before returning, not a full Assistant.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slm import RemoteUnreachable  # noqa: E402
from core.turn_stats import TurnStats  # noqa: E402


def _host(**overrides):
    base = {
        "llm_enabled": True,
        "_history": [],
        "_history_for": lambda session: [],
        "_trim_history": lambda: None,
        "_emit_agent_event": lambda *a, **k: None,
        "_compact_completed_turns": lambda: None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_route_is_regex_when_catchall_reply_resolves_without_slm(monkeypatch):
    import core.agent_loop as al

    # catchAll's own {"reply": ...} shape (e.g. the note-delete refusal) is a
    # complete turn resolved on iteration 0 — the SLM is never called.
    monkeypatch.setattr(al, "catchAll", lambda prompt: {"reply": "no can do"})
    stats = TurnStats()
    loop = al.AgentLoop(_host(), session=None, source="text", stats=stats)
    out = loop.run("delete my note")
    assert out == "no can do"
    assert stats.route == "regex"


def test_route_is_no_llm_when_llm_disabled(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    spoken = {}
    host = _host(
        llm_enabled=False,
        _record_spoken=lambda s: spoken.__setitem__("said", s),
        _speak_no_ai_fallback=lambda session, source, satellite_id=None: (
            spoken.__setitem__("said", "NO_AI") or "NO_AI"
        ),
    )
    stats = TurnStats()
    loop = al.AgentLoop(host, session=None, source="text", stats=stats)
    out = loop.run("tell me a joke")
    assert out == "NO_AI"
    assert stats.route == "no_llm"


def test_route_is_agent_once_slm_call_starts(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)

    def _raise(**k):
        raise RemoteUnreachable("down")

    host = _host(
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],  # empty -> progress watchdog is a no-op
        play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda ok, error="": None,
        _speak_llm_error_fallback=lambda session, source, satellite_id=None: "LLM SERVER UNREACHABLE",
        _generate_with_context_recovery=_raise,
    )
    stats = TurnStats()
    loop = al.AgentLoop(host, session=None, source="text", stats=stats)
    out = loop.run("tell me a story about the sea")
    # Route is set the moment the SLM call is attempted, before its outcome
    # is known — even though this particular call fails over to a fallback.
    assert out == "LLM SERVER UNREACHABLE"
    assert stats.route == "agent"


def test_route_stays_none_without_a_stats_object():
    import core.agent_loop as al

    # stats=None (e.g. a caller that doesn't track turn stats) must not raise.
    host = _host(llm_enabled=False, _speak_no_ai_fallback=lambda *a, **k: "NO_AI")
    loop = al.AgentLoop(host, session=None, source="text", stats=None)
    assert loop.run("tell me a joke") == "NO_AI"
