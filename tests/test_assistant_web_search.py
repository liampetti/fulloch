"""Surface checks for the web-search stall timing + summary-surfacing fixes.

The full Assistant can't be instantiated in tests (it loads audio/ASR/SLM/TTS),
so — like test_assistant_ack.py — these assert the wiring via source inspection.

Two behaviours are pinned:
  1. The "searching the web" stall plays BEFORE dispatching the search tool
     (so it covers the slow SearXNG round-trip), not only before the fast
     summarise step that follows.
  2. A web-search summary produced earlier in the turn is surfaced in the
     final spoken output even when a later side-effect action (e.g. write_note)
     is what terminates the turn.
"""

import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """`_import_assistant_module` installs fake `core.*` modules into sys.modules
    so the assistant imports without the heavy pipeline. Those must not leak into
    later test files — a fake `core.slm` (no GRAMMAR_FILE) was shadowing the real
    module and breaking test_intents' grammar check. Snapshot before each test;
    on teardown drop anything imported during it and reinstate the originals.
    """
    saved = dict(sys.modules)
    yield
    for name in list(sys.modules):
        if name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)


def _import_assistant_module():
    fake = {
        "core.audio": ["AudioCapture", "resolve_device"],
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
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        if name == "core.slm":
            # Real exception class — assistant.py does `except
            # ContextExhaustedError`, which a lambda stub can't satisfy.
            # Both are imported by name in core.agent_loop, so the stub must
            # provide them for this file to import in isolation.
            mod.ContextExhaustedError = type("ContextExhaustedError", (RuntimeError,), {})
            mod.RemoteUnreachable = type("RemoteUnreachable", (RuntimeError,), {})
        sys.modules[name] = mod

    import core.assistant as assistant  # noqa: E402

    return assistant


def test_search_stall_plays_before_dispatch():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # The stall must be gated on is_web_search and sit before handle_action.
    assert "is_web_search" in src, "search stall not gated on is_web_search"
    stall_pos = src.index("is_web_search")
    # Key on handle_action(action) — the main-loop dispatch — not
    # handle_action(deferred) in the reply branch which appears earlier.
    dispatch_pos = src.index("handle_action(action)")
    assert stall_pos < dispatch_pos, "stall must play before the search dispatch"


def test_search_uses_dedicated_progress_clips_during_summary():
    a = _import_assistant_module()
    from utils.phrases import WEB_SEARCH_PHRASES

    assert "web_search_stall_cache" not in a.ACK_CACHE_ATTRS
    specs = {attr: pool for attr, pool, _ in a.STARTUP_CACHE_SPECS}
    assert specs["web_search_stall_cache"] == WEB_SEARCH_PHRASES

    src = inspect.getsource(a.AgentLoop._run)
    summary_pos = src.index("Summarising web search payload")
    watchdog_pos = src.index("ThinkingWatchdog(", summary_pos)
    generate_pos = src.index("_summarise_search_result(", summary_pos)
    assert watchdog_pos < generate_pos
    assert "max_stalls=2" in src[watchdog_pos:generate_pos]


def test_summary_surfaced_in_spoken_output():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # The web summary is tracked across replan iterations and folded into the
    # terminal spoken output so a trailing write_note doesn't bury it.
    assert "web_summary_text" in src, "web summary not tracked for spoken output"


def test_agent_generation_wrapped_in_progress_watchdog():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # A slow (esp. remote) agent generation must get periodic progress stalls,
    # not just the one-shot replan stall — so it isn't silent for many seconds.
    wd_pos = src.index("ThinkingWatchdog(")
    gen_pos = src.index("_generate_with_context_recovery(")
    assert wd_pos < gen_pos, "watchdog must wrap the agent generation call"
    assert "replan_stall_cache" in src[wd_pos:gen_pos], "watchdog should use replan stalls"
    assert "max_stalls=1 if iteration == 0 else 0" in src[wd_pos:gen_pos]


def test_is_web_search_importable():
    import utils.intents as intents

    assert callable(intents.is_web_search)
    assert intents.WEB_SEARCH_TOOL == "external_information"


def test_prose_emission_recovered_as_reply():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # A grammar-less remote model replying in prose (not the JSON envelope) must
    # not drop the turn — the prose is treated as a reply and falls through to
    # the reply branch, rather than returning "I don't understand".
    assert 'emission = {"reply": prose}' in src, "prose emissions should recover as a reply"
    # ...but a malformed JSON *fragment* (starts with { or [) still gives up,
    # so we don't read brace-junk aloud.
    assert 'prose[:1] in ("{", "[")' in src, "JSON fragments must not be spoken as prose"


def test_bundled_reply_pseudo_action_split_out():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # A `reply` bundled inside `actions` must be split out (so real tools still
    # dispatch and the guard doesn't block the turn), and spoken at the terminal.
    assert "REPLY_PSEUDO_INTENTS" in src, "pseudo-reply actions must be recognised"
    assert "bundled_reply" in src, "bundled reply must be captured and spoken"
    # The split must happen before the hallucinated-tool guard, so a bundled
    # `reply` doesn't trip it.
    split_pos = src.index("REPLY_PSEUDO_INTENTS")
    guard_pos = src.index('is_registered_tool(a.get("intent"))')
    assert split_pos < guard_pos, "pseudo-reply split must run before the tool guard"


def test_reply_branch_prefers_grounded_web_summary():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # After a web search, a replan `reply` is the model re-answering from a
    # compressed summary — a fabrication opening. The grounded summary (built
    # from the actual snippets) must be preferred over the model's reply.
    reply_branch = src.index("# Reply branch")
    grounded_use = src.index("web_summary_text", reply_branch)
    model_reply_return = src.index("strip_unfounded_save_claim(reply", reply_branch)
    assert grounded_use < model_reply_return, (
        "reply branch must prefer the grounded web summary over the model's reply"
    )


def test_hallucinated_tool_blocked_before_dispatch():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # The guard checks the loaded registry (direct match) and speaks the canned
    # fallback, and it must run BEFORE the main-loop dispatch so a hallucinated
    # tool can't replan into a fabricated answer or cause partial side effects.
    assert "is_registered_tool" in src, "hallucinated-tool guard not wired"
    guard_pos = src.index("is_registered_tool")
    fallback_pos = src.index("_speak_tool_unavailable_fallback")
    dispatch_pos = src.index("handle_action(action)")
    assert guard_pos < dispatch_pos, "guard must run before the dispatch loop"
    assert fallback_pos < dispatch_pos, "canned fallback must replace dispatch"


def test_tool_unavailable_cache_wired():
    a = _import_assistant_module()
    # The canned clips are pre-rendered and re-rendered on a voice change.
    specs = {attr: pool for attr, pool, _ in a.STARTUP_CACHE_SPECS}
    assert "tool_unavailable_cache" in specs
    src = inspect.getsource(a.Assistant._speak_tool_unavailable_fallback)
    assert "tool_unavailable_cache" in src and "_record_spoken" in src


def test_search_cached_before_dispatch():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # Repeated identical queries within a turn must reuse the cached summary
    # rather than re-dispatching the search tool.
    assert "search_cache" in src, "per-turn search cache missing"
    cache_pos = src.index("search_cache.get(")
    dispatch_pos = src.index("handle_action(action)")
    assert cache_pos < dispatch_pos, "cache lookup must precede the dispatch"


def test_web_search_always_replans():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop._run)
    # A summarised web search must hand control back to the agent every time
    # (not only when bundled with other actions), so the agent decides the
    # next move from the findings. The old `len(actions) > 1` gate is gone.
    assert "if web_summarised or step.should_replan or lookup_replan:" in src, (
        "web search should unconditionally replan"
    )
    assert "pending_side_effects" not in src, "deferred side-effect machinery should be removed"


def _normalise():
    # _normalise_search_query moved to core.agent_loop with the loop extraction.
    _import_assistant_module()  # installs the stubs core.agent_loop loads under
    import core.agent_loop as al

    return al._normalise_search_query


class TestNormaliseSearchQuery:
    def test_lowercases_and_strips(self):
        assert _normalise()(["  Today's News!  "]) == "today's news"

    def test_collapses_whitespace(self):
        assert _normalise()(["the   latest  news"]) == "the latest news"

    def test_variants_of_same_query_share_key(self):
        normalise = _normalise()
        assert normalise(["today's news"]) == normalise(["Today's news."])

    def test_no_args_maps_to_default_sentinel(self):
        assert _normalise()([]) == "__default__"

    def test_non_string_arg_returns_none(self):
        assert _normalise()([123]) is None

    def test_dict_args_use_first_value(self):
        # A grammar-less remote model can emit kwargs-style args; indexing
        # args[0] on a dict raised KeyError(0) (the "Turn error: 0" crash).
        normalise = _normalise()
        assert normalise({"query": "Latest News!"}) == "latest news"
        assert normalise({}) == "__default__"

    def test_bare_string_arg(self):
        assert _normalise()("Today's News") == "today's news"
