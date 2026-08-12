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

from core.agent_loop import _llm_unavailable_label  # noqa: E402
from core.slm import RemoteUnreachable  # noqa: E402
from core.turn_stats import TurnStats  # noqa: E402


def _host(**overrides):
    base = {
        "llm_enabled": True,
        "_history": [],
        "_history_for": lambda session: [],
        "_turn_local": types.SimpleNamespace(sink=None, tts_active_event=None),
        "_trim_history": lambda: None,
        "_emit_agent_event": lambda *a, **k: None,
        "_compact_completed_turns": lambda: None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_local_llm_failure_label_names_llama_server():
    assert _llm_unavailable_label(_host(llm_backend="local")) == "Local llama-server unavailable"


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


def test_topicless_web_search_reuses_prior_question(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: prompt)
    captured = {}
    monkeypatch.setattr(
        al.AgentLoop,
        "_run_without_llm",
        lambda self, prompt, emission: captured.setdefault("emission", emission) or "searched",
    )
    history = [{"role": "user", "content": "did we ever find out what that supermarket staple is"}]
    host = _host(
        llm_enabled=False,
        _history_for=lambda session: history,
    )
    loop = al.AgentLoop(host, session=None, source="text")
    loop.run("can you search the internet")

    assert captured["emission"] == {
        "actions": [
            {
                "intent": "external_information",
                "args": ["did we ever find out what that supermarket staple is"],
            }
        ]
    }


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


def test_remote_cooldown_skips_generation_and_uses_outage_fallback(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    calls = {"generate": 0, "fallback": 0}
    host = _host(
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: (
            calls.__setitem__("fallback", calls["fallback"] + 1) or "REMOTE UNAVAILABLE"
        ),
        _generate_with_context_recovery=lambda **kwargs: (
            calls.__setitem__("generate", calls["generate"] + 1)
            or (_ for _ in ()).throw(AssertionError("cooldown must skip generation"))
        ),
    )

    out = al.AgentLoop(host, session=None, source="voice").run("tell me a story")

    assert out == "REMOTE UNAVAILABLE"
    assert calls == {"generate": 0, "fallback": 1}


def test_remote_cooldown_keeps_regex_reply_available(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: {"reply": "The time is noon."})
    host = _host(
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: "REMOTE UNAVAILABLE",
    )

    out = al.AgentLoop(host, session=None, source="voice").run("what time is it")

    assert out == "The time is noon."


def test_non_balanced_satellite_message_uses_agent_but_keeps_regex_fallback(monkeypatch):
    import core.agent_loop as al

    emission = {"actions": [{"intent": "send_satellite_message", "args": ["kitchen", "Dinner is ready"]}]}
    monkeypatch.setattr(al, "catchAll", lambda prompt: emission)
    captured = {}
    host = _host(
        personality="wry",
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: "REMOTE UNAVAILABLE",
    )

    def fallback(self, prompt, fallback_emission, unavailable_fallback=None):
        captured["emission"] = fallback_emission
        return "queued"

    monkeypatch.setattr(
        al.AgentLoop,
        "_run_without_llm",
        fallback,
    )

    out = al.AgentLoop(host, session=None, source="text").run("tell kitchen dinner is ready")

    assert out == "queued"
    assert captured["emission"] == emission


def test_music_search_ack_plays_before_regex_spotify_dispatch(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(
        al,
        "catchAll",
        lambda prompt: {"actions": [{"intent": "play_song", "args": ["Take Five"]}]},
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: name == "play_song")
    monkeypatch.setattr(al.intents, "handle_action", lambda action: "Playing Take Five.")
    calls = []
    host = _host(
        music_search_stall_cache=[("pcm", 16000)],
        _play_random_ack=lambda session, cache: calls.append((session, cache)),
        _record_spoken=lambda spoken: None,
    )

    out = al.AgentLoop(host, session=None, source="voice").run("play Take Five")

    assert out == "Playing Take Five."
    assert calls == [(None, host.music_search_stall_cache)]


def test_wry_announcement_fallback_changes_an_echoed_llm_message():
    import core.agent_loop as al

    raw = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "dinner is ready"]}]}
    emission = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "dinner is ready"]}]}

    al._apply_announcement_fallback(_host(personality="wry"), "tell kitchen dinner is ready", raw, emission)

    assert emission["actions"][0]["args"][1] == "dinner is ready. The kitchen's patience has been noted."


def test_announcement_fallback_keeps_verbatim_and_safety_messages_literal():
    import core.agent_loop as al

    raw = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "take your medication"]}]}
    emission = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "take your medication"]}]}

    al._apply_announcement_fallback(_host(personality="wry"), "tell kitchen verbatim: take your medication", raw, emission)

    assert emission == raw


def test_delivery_is_not_spoken_for_lock_actions():
    import core.agent_loop as al

    assert not al._can_speak_delivery(_host(personality="wry"), [{"intent": "ha_lock", "args": ["front door"]}])
    assert al._can_speak_delivery(_host(personality="wry"), [{"intent": "turn_on", "args": ["lamp"]}])


def test_delivery_replaces_raw_tool_results_after_success(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(
        al,
        "catchAll",
        lambda prompt: {
            "actions": [
                {"intent": "turn_on", "args": ["lights"]},
                {"intent": "play_song", "args": ["music"]},
            ],
            "delivery": "The lights are on and music is playing.",
        },
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: name in {"turn_on", "play_song"})
    monkeypatch.setattr(
        al.intents,
        "handle_action",
        lambda action: "Raw result for " + action["intent"],
    )
    spoken = []
    host = _host(personality="wry", _record_spoken=spoken.append)

    out = al.AgentLoop(host, session=None, source="text").run("turn on lights and play music")

    assert out == "The lights are on and music is playing."
    assert spoken == [out]
