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

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_assistant_module():
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice", "warmup_model", "synthesize", "play_chunks",
            "speak_stream", "set_output_device", "set_tts_active_event",
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
            mod.ContextExhaustedError = type(
                "ContextExhaustedError", (RuntimeError,), {}
            )
        sys.modules[name] = mod

    import core.assistant as assistant  # noqa: E402
    return assistant


def test_search_stall_plays_before_dispatch():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop.run)
    # The stall must be gated on is_web_search and sit before handle_action.
    assert "is_web_search" in src, "search stall not gated on is_web_search"
    stall_pos = src.index("is_web_search")
    # Key on handle_action(action) — the main-loop dispatch — not
    # handle_action(deferred) in the reply branch which appears earlier.
    dispatch_pos = src.index("handle_action(action)")
    assert stall_pos < dispatch_pos, "stall must play before the search dispatch"


def test_summary_surfaced_in_spoken_output():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop.run)
    # The web summary is tracked across replan iterations and folded into the
    # terminal spoken output so a trailing write_note doesn't bury it.
    assert "web_summary_text" in src, "web summary not tracked for spoken output"


def test_is_web_search_importable():
    import utils.intents as intents
    assert callable(intents.is_web_search)
    assert intents.WEB_SEARCH_TOOL == "external_information"


def test_search_cached_before_dispatch():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop.run)
    # Repeated identical queries within a turn must reuse the cached summary
    # rather than re-dispatching the search tool.
    assert "search_cache" in src, "per-turn search cache missing"
    cache_pos = src.index("search_cache.get(")
    dispatch_pos = src.index("handle_action(action)")
    assert cache_pos < dispatch_pos, "cache lookup must precede the dispatch"


def test_web_search_always_replans():
    a = _import_assistant_module()
    src = inspect.getsource(a.AgentLoop.run)
    # A summarised web search must hand control back to the agent every time
    # (not only when bundled with other actions), so the agent decides the
    # next move from the findings. The old `len(actions) > 1` gate is gone.
    assert "if web_summarised or step.should_replan:" in src, \
        "web search should unconditionally replan"
    assert "pending_side_effects" not in src, \
        "deferred side-effect machinery should be removed"


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
