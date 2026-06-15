"""History compaction.

Tool results and bare planning emissions are in-turn scaffolding: the agent
needs a tool result only while composing that turn's reply, which is itself
recorded as a `{"reply": ...}` entry. `_compact_completed_turns` (run at the
start of each turn) strips that scaffolding so history stays just the user's
turns and Fulloch's replies — keeping multi-KB tool dumps from accumulating
and evicting the conversation / blowing the context window.
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


def _compact(history):
    a = _import_assistant_module()
    fake = SimpleNamespace(_history=list(history))
    a.Assistant._compact_completed_turns(fake)
    return fake._history


def test_conversation_pairs_survive():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"actions": [{"intent": "x", "args": []}]}'},
        {"role": "tool", "name": "x", "content": "big dump"},
        {"role": "assistant", "content": '{"reply": "did it"}'},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": '{"reply": "answer two"}'},
    ]
    out = _compact(history)
    assert out == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"reply": "did it"}'},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": '{"reply": "answer two"}'},
    ]


def test_drops_all_tool_messages():
    history = [
        {"role": "user", "content": "u"},
        {"role": "tool", "name": "read_note", "content": "note body " * 500},
        {"role": "assistant", "content": '{"reply": "summary"}'},
    ]
    out = _compact(history)
    assert all(m["role"] != "tool" for m in out)
    assert {"role": "user", "content": "u"} in out


def test_keeps_unparseable_assistant_content():
    # A plain-text assistant entry (not JSON) is conservatively kept.
    history = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "plain reply not json"},
    ]
    out = _compact(history)
    assert out == history


def test_noop_when_already_clean():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"reply": "r1"}'},
    ]
    out = _compact(history)
    assert out == history
