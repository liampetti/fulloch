"""Native tool capability policy adapters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.capabilities import native_capabilities  # noqa: E402
from tools.tool_registry import tool_registry  # noqa: E402


def test_native_capability_preserves_registry_dispatch(monkeypatch):
    monkeypatch.setattr(tool_registry, "is_available", lambda _name: True)
    monkeypatch.setattr(tool_registry, "execute_tool", lambda name, args, kwargs: f"{name}:{args}:{kwargs}")

    capability = native_capabilities(["calculate"])["calculate"]

    assert capability.source == "native"
    assert capability.access_class == "read"
    assert capability.invoke(["2 + 2"], {}) == "calculate:['2 + 2']:{}"
    assert capability.format_result("result") == "result"


def test_unclassified_native_tools_default_to_execute(monkeypatch):
    monkeypatch.setattr(tool_registry, "is_available", lambda _name: True)

    capability = native_capabilities(["write_note"])["write_note"]

    assert capability.access_class == "execute"


def test_unavailable_or_unknown_native_tools_are_not_adapted(monkeypatch):
    monkeypatch.setattr(tool_registry, "is_available", lambda _name: False)

    assert native_capabilities(["calculate", "not_a_tool"]) == {}
