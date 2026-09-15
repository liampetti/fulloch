"""Agent-loop behavior with deterministic model, search, and playback services."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from core import agent_loop as al
from core.agent_search import normalise_search_query
from core.tts_session import TtsSession
from core.turn_stats import TurnStats
from tools.tool_registry import ToolRegistry


@pytest.fixture
def turn(monkeypatch):
    history = []
    events = []
    active_watchdogs = []
    watchdogs = []
    session = TtsSession()
    registry = ToolRegistry()
    search = Mock(return_value="User question: raw search snippets")
    save = Mock(return_value="Saved the note.")
    registry.register_tool(lambda query: search(query), name="external_information")
    registry.register_tool(lambda text: save(text), name="append_to_today")
    monkeypatch.setattr(al, "tool_registry", registry)
    monkeypatch.setattr(al.intents, "tool_registry", registry)
    catch = Mock(return_value=None)
    monkeypatch.setattr(al, "catchAll", catch)

    real_watchdog = al.ThinkingWatchdog

    @contextmanager
    def watchdog(clips, *args, **kwargs):
        entry = (clips, kwargs["max_stalls"])
        watchdogs.append(entry)
        active_watchdogs.append(entry)
        try:
            with real_watchdog(clips, *args, **kwargs):
                yield
        finally:
            active_watchdogs.pop()

    monkeypatch.setattr(al, "ThinkingWatchdog", watchdog)
    host = SimpleNamespace(
        llm_enabled=True,
        grammar=None,
        wakeword_name="Fulloch",
        tts_session=session,
        replan_stall_cache=[],
        web_search_stall_cache=[(["search clip"], 24000)],
        note_write_stall_cache=[],
        _turn_local=SimpleNamespace(sink=None, tts_active_event=None),
        _history_for=lambda satellite: history,
        _trim_history=Mock(),
        _compact_completed_turns=Mock(),
        _emit_agent_event=Mock(),
        _record_spoken=Mock(),
        _play_random_ack=Mock(),
        _note_llm_remote_status=Mock(),
        _generate_with_context_recovery=Mock(),
        _summarise_search_result=Mock(return_value="Grounded findings."),
        _speak_tool_unavailable_fallback=Mock(return_value="Tool unavailable."),
        play_chunks=Mock(side_effect=lambda *args, **kwargs: events.append("play")),
    )
    stats = TurnStats()
    return SimpleNamespace(
        host=host, history=history, events=events, search=search, save=save,
        catch=catch, session=session, stats=stats, watchdogs=watchdogs,
        active_watchdogs=active_watchdogs,
        loop=al.AgentLoop(host, session=session, source="text", stats=stats),
    )


def _search(query="latest news"):
    return {"intent": "external_information", "args": [query]}


def _emissions(turn, *emissions):
    turn.host._generate_with_context_recovery.side_effect = [
        json.dumps(emission) if isinstance(emission, dict) else emission
        for emission in emissions
    ]


def test_search_stall_precedes_dispatch_and_summary_is_in_history_before_replan(turn):
    turn.catch.return_value = {"actions": [_search()]}

    def search(query):
        turn.events.append("search")
        return "User question: raw search snippets"

    def summarise(raw, cancel_check, *, stats):
        assert raw == "User question: raw search snippets"
        assert turn.active_watchdogs == [(turn.host.web_search_stall_cache, 2)]
        turn.events.append("summary")
        return "Grounded findings."

    def generate(**kwargs):
        assert turn.active_watchdogs == [(turn.host.replan_stall_cache, 0)]
        assert {"role": "tool", "name": "external_information", "content": "Grounded findings."} in kwargs["history"]
        assert "raw search snippets" not in json.dumps(kwargs["history"])
        turn.events.append("replan")
        return '{"reply": "Ungrounded replacement."}'

    turn.search.side_effect = search
    turn.host._summarise_search_result.side_effect = summarise
    turn.host._generate_with_context_recovery.side_effect = generate

    assert turn.loop.run("Find the latest news") == "Grounded findings."
    assert turn.events == ["play", "search", "summary", "replan"]
    assert turn.history[-1] == {"role": "assistant", "content": '{"reply": "Grounded findings."}'}
    turn.catch.assert_called_once_with("Find the latest news")
    turn.search.assert_called_once_with("latest news")
    assert turn.stats.tool_dispatches == 1


def test_search_summary_survives_a_follow_up_note_write(turn):
    turn.catch.return_value = {"actions": [_search()]}
    _emissions(turn, {"actions": [{"intent": "append_to_today", "args": ["Grounded findings."]}]})

    assert turn.loop.run("Find news and save it") == "Grounded findings. Saved the note."
    turn.save.assert_called_once_with("Grounded findings.")
    turn.host._record_spoken.assert_called_once_with("Grounded findings. Saved the note.")


def test_search_discards_bundled_actions_and_replans_from_findings(turn):
    turn.catch.return_value = {"actions": [
        _search(), {"intent": "append_to_today", "args": ["Unresearched guess"]},
    ]}
    _emissions(turn, {"reply": "Done"})

    assert turn.loop.run("Find the news") == "Grounded findings."
    turn.save.assert_not_called()
    turn.host._generate_with_context_recovery.assert_called_once()


@pytest.mark.parametrize("query,dispatches", [(" Latest  NEWS! ", 1), ("weather tomorrow", 2)])
def test_search_cache_reuses_only_equivalent_queries(turn, query, dispatches):
    turn.catch.return_value = {"actions": [_search()]}
    turn.loop.on_slm_start = Mock()
    _emissions(turn, {"actions": [_search(query)]}, {"reply": "Done"})

    assert turn.loop.run("Find the news") == "Grounded findings."
    assert turn.search.call_args_list == [call("latest news")] + (
        [call(query)] if dispatches == 2 else []
    )
    assert turn.host._summarise_search_result.call_count == dispatches
    assert turn.host.play_chunks.call_count == dispatches
    assert turn.stats.tool_dispatches == dispatches
    assert turn.host._generate_with_context_recovery.call_count == 2
    turn.loop.on_slm_start.assert_called_once_with()


def test_search_cache_is_scoped_to_one_turn(turn):
    turn.catch.return_value = {"actions": [_search()]}
    _emissions(turn, {"reply": "Done"}, {"reply": "Done"})

    assert turn.loop.run("Find the news") == "Grounded findings."
    assert turn.loop.run("Find the news again") == "Grounded findings."
    assert turn.search.call_args_list == [call("latest news"), call("latest news")]
    assert turn.host._summarise_search_result.call_count == 2


def test_cancellation_during_search_stall_prevents_dispatch(turn):
    turn.catch.return_value = {"actions": [_search()]}
    turn.host.play_chunks.side_effect = lambda *args, **kwargs: turn.session.stop()

    assert turn.loop.run("Find the news") == ""
    turn.search.assert_not_called()
    turn.host._summarise_search_result.assert_not_called()
    turn.host._generate_with_context_recovery.assert_not_called()


def test_cancellation_during_summary_does_not_publish_findings_or_replan(turn):
    turn.catch.return_value = {"actions": [_search()]}

    def summarise(*args, **kwargs):
        turn.session.stop()
        return "Cancelled findings."

    turn.host._summarise_search_result.side_effect = summarise
    assert turn.loop.run("Find the news") == ""
    assert not any(message["role"] == "tool" for message in turn.history)
    turn.host._generate_with_context_recovery.assert_not_called()
    assert turn.active_watchdogs == []


def test_search_call_cap_returns_latest_distinct_grounded_summary(turn):
    turn.catch.return_value = {"actions": [_search("first query")]}
    turn.host._summarise_search_result.side_effect = ["First findings.", "Latest findings."]
    _emissions(turn, *[
        {"actions": [_search("second query")]}
        for _ in range(al.MAX_AGENT_CALLS_PER_TURN - 1)
    ])

    assert turn.loop.run("Research the news") == "Latest findings."
    assert turn.search.call_count == 2
    assert turn.host._summarise_search_result.call_count == 2
    turn.host._record_spoken.assert_called_once_with("Latest findings.")


def test_agent_generation_runs_inside_progress_watchdog(turn):
    turn.loop.on_slm_start = Mock()

    def generate(**kwargs):
        turn.loop.on_slm_start.assert_called_once_with()
        assert turn.active_watchdogs == [(turn.host.replan_stall_cache, 1)]
        return '{"reply": "An answer."}'

    turn.host._generate_with_context_recovery.side_effect = generate
    assert turn.loop.run("Explain clouds") == "An answer."
    assert turn.active_watchdogs == []
    assert turn.watchdogs == [(turn.host.replan_stall_cache, 1)]


def test_prose_emission_is_returned_as_reply(turn):
    _emissions(turn, "  A plain prose answer.  ")
    assert turn.loop.run("Explain clouds") == "A plain prose answer."
    assert turn.history[-1]["content"] == '{"reply": "A plain prose answer."}'


@pytest.mark.parametrize("emission", ["", '{"reply":', '[{"intent":'])
def test_empty_or_fragmented_emission_requests_clarification(turn, emission):
    _emissions(turn, emission)
    assert turn.loop.run("Explain clouds") in {"Sorry, can you repeat that", "I don't understand"}
    turn.search.assert_not_called()
    turn.save.assert_not_called()


def test_bundled_reply_is_spoken_after_real_tool_dispatch(turn):
    _emissions(turn, {"actions": [
        {"intent": "append_to_today", "args": ["A fact"]},
        {"intent": "reply", "args": ["I've saved this to your notes."]},
    ]})

    assert turn.loop.run("Remember a fact") == "I've saved this to your notes."
    turn.save.assert_called_once_with("A fact")
    turn.host._speak_tool_unavailable_fallback.assert_not_called()


@pytest.mark.parametrize("unknown_first", [False, True])
def test_unknown_tool_blocks_entire_batch_before_side_effects(turn, unknown_first):
    actions = [
        {"intent": "append_to_today", "args": ["A fact"]},
        {"intent": "invented_tool", "args": []},
    ]
    _emissions(turn, {"actions": list(reversed(actions)) if unknown_first else actions})

    assert turn.loop.run("Do these tasks") == "Tool unavailable."
    turn.save.assert_not_called()
    turn.search.assert_not_called()
    turn.host._speak_tool_unavailable_fallback.assert_called_once_with(
        turn.session, "text", satellite_id=None,
    )
    assert turn.stats.tool_dispatches == 0


@pytest.mark.parametrize("args,expected", [
    (["  Today's News!  "], "today's news"),
    (["the   latest  news"], "the latest news"),
    ([], "__default__"),
    ([123], None),
    ({"query": "Latest News!"}, "latest news"),
    ({}, "__default__"),
    ("Today's News", "today's news"),
])
def test_normalise_search_query(args, expected):
    assert normalise_search_query(args) == expected


def test_search_and_unavailable_phrase_caches_have_dedicated_pools():
    from core.assistant import ACK_CACHE_ATTRS, STARTUP_CACHE_SPECS
    from utils.phrases import TOOL_UNAVAILABLE_PHRASES, WEB_SEARCH_PHRASES

    specs = {attr: pool for attr, pool, _ in STARTUP_CACHE_SPECS}
    assert "web_search_stall_cache" not in ACK_CACHE_ATTRS
    assert specs["web_search_stall_cache"] == WEB_SEARCH_PHRASES
    assert specs["tool_unavailable_cache"] == TOOL_UNAVAILABLE_PHRASES


@pytest.mark.parametrize("source,cached", [("voice", True), ("voice", False), ("text", True)])
def test_tool_unavailable_fallback_records_phrase_and_avoids_double_playback(source, cached):
    from core.assistant import Assistant
    from utils.phrases import TOOL_UNAVAILABLE_PHRASES

    session = TtsSession()
    host = SimpleNamespace(
        tool_unavailable_cache=[(["clip"], 24000)] if cached else [],
        _record_spoken=Mock(), _emit_turn_event=Mock(), play_chunks=Mock(),
    )
    result = Assistant._speak_tool_unavailable_fallback(host, session, source, "kitchen")
    phrase = TOOL_UNAVAILABLE_PHRASES[0]
    host._record_spoken.assert_called_once_with(phrase)
    if source == "voice" and cached:
        assert result == ""
        host.play_chunks.assert_called_once_with(["clip"], 24000, session=session)
        host._emit_turn_event.assert_called_once_with(
            "assistant", phrase, "voice", satellite_id="kitchen",
        )
    else:
        assert result == phrase
        host.play_chunks.assert_not_called()
        host._emit_turn_event.assert_not_called()
