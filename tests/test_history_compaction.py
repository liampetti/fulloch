"""History compaction.

Bare planning emissions (`{"actions": ...}` scaffolding) are in-turn-only:
the agent needs to see them only while composing that turn's reply, which is
itself recorded as a `{"reply": ...}` entry, so `_compact_completed_turns`
(run at the start of each turn) drops them entirely.

Tool results are different: they're truncated to a short trace rather than
dropped outright. Dropping them entirely used to leave history with nothing
but `{"reply": ...}` entries — indistinguishable from a fact the model merely
asserted out loud — so a relative follow-up ("brighten them now") had no
signal that a tool genuinely ran last turn versus the model just having said
a number. Truncating (instead of keeping the raw payload, which can be
multi-KB for a note read or search result) keeps that signal without
re-bloating history.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


def _compact(history):
    a = _import_assistant_module()
    fake = SimpleNamespace(_history=list(history))
    a.Assistant._compact_completed_turns(fake)
    return fake._history, a


def test_conversation_pairs_survive_and_short_tool_trace_kept():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"actions": [{"intent": "x", "args": []}]}'},
        {"role": "tool", "name": "x", "content": "short result"},
        {"role": "assistant", "content": '{"reply": "did it"}'},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": '{"reply": "answer two"}'},
    ]
    out, _ = _compact(history)
    assert out == [
        {"role": "user", "content": "u1"},
        {"role": "tool", "name": "x", "content": "short result"},
        {"role": "assistant", "content": '{"reply": "did it"}'},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": '{"reply": "answer two"}'},
    ]


def test_long_tool_message_is_truncated_not_dropped():
    history = [
        {"role": "user", "content": "u"},
        {"role": "tool", "name": "read_note", "content": "note body " * 500},
        {"role": "assistant", "content": '{"reply": "summary"}'},
    ]
    out, a = _compact(history)
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "read_note"
    assert len(tool_msgs[0]["content"]) <= a.COMPACTED_TOOL_TRACE_MAX_CHARS + 1  # +1 for the ellipsis char
    assert tool_msgs[0]["content"].endswith("…")
    assert {"role": "user", "content": "u"} in out


def test_keeps_unparseable_assistant_content():
    # A plain-text assistant entry (not JSON) is conservatively kept.
    history = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "plain reply not json"},
    ]
    out, _ = _compact(history)
    assert out == history


def test_noop_when_already_clean():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"reply": "r1"}'},
    ]
    out, _ = _compact(history)
    assert out == history


def test_proactive_fit_compacts_tool_payloads_before_shedding_turns():
    a = _import_assistant_module()
    history = [
        {"role": "user", "content": "remember decision"},
        {"role": "tool", "name": "search", "content": "evidence " * 2000},
        {"role": "assistant", "content": '{"reply":"decision: choose option A"}'},
        {"role": "user", "content": "current question"},
    ]

    size = a.turn_history.fit_prompt_history(
        history, system_prompt="system", user_prompt=None, context_tokens=4096,
        output_reserve=512, margin=256,
    )

    assert size <= 4096 - 512 - 256
    assert history[0]["content"] == "remember decision"
    assert len(next(item["content"] for item in history if item["role"] == "tool")) <= 161
    assert history[-1]["content"] == "current question"


def test_oversized_fixed_context_fails_explicitly():
    a = _import_assistant_module()
    try:
        a.turn_history.fit_prompt_history(
            [], system_prompt="fixed" * 1000, user_prompt="question",
            context_tokens=1024, output_reserve=512, margin=256,
        )
    except a.turn_history.ContextExhaustedError as exc:
        assert "Fixed prompt requires" in str(exc)
    else:
        raise AssertionError("oversized fixed input should not trigger an unbounded retry")


def test_heuristic_overflow_does_not_reject_fresh_foreground_request_or_lose_outcome():
    a = _import_assistant_module()
    history = [
        {"role": "user", "content": "Research my Home Assistant history and Spotify activity."},
        {"role": "tool", "name": "plan_progress", "content": "Execution outcome: entity history fetched."},
        {"role": "user", "content": "Current constraints: use the latest week; compare with Spotify."},
    ]

    size = a.turn_history.fit_prompt_history(
        history, system_prompt="fixed request material " * 1200,
        user_prompt="Answer using the latest-week constraint and fetched outcome.",
        context_tokens=1024, output_reserve=256, margin=128,
        allow_estimated_overflow=True,
    )

    contents = [item["content"] for item in history]
    assert size > 1024 - 256 - 128
    assert "Current constraints: use the latest week; compare with Spotify." in contents
    assert "Execution outcome: entity history fetched." in contents


def test_pre_call_shedding_preserves_decisions_open_tasks_and_execution_outcomes():
    a = _import_assistant_module()
    history = [
        {"role": "user", "content": "old request " * 100},
        {"role": "assistant", "content": "unrelated completed detail " * 100},
        {"role": "assistant", "content": "Decision: choose option A"},
        {"role": "tool", "name": "plan_progress", "content": "Open task: verify timing"},
        {"role": "tool", "name": "lookup", "content": "Execution outcome: source lookup completed"},
        {"role": "user", "content": "current question"},
    ]

    a.turn_history.fit_prompt_history(
        history, system_prompt="system", user_prompt=None, context_tokens=1024,
        output_reserve=256, margin=128,
    )

    contents = [message["content"] for message in history]
    assert "Decision: choose option A" in contents
    assert "Open task: verify timing" in contents
    assert "Execution outcome: source lookup completed" in contents
    assert "current question" in contents
    assert "unrelated completed detail " * 100 not in contents
