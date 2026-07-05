"""The reminder-poll thread must respect the same config gate as the tool loader.

Importing tools.home_assistant has side effects — it registers the HA tools into
the global registry and connects to HA (default URL + credentials.json token) — so a
commented-out `home_assistant:` block must NOT trigger that import via the
reminder poll. Otherwise HA is silently re-enabled despite being disabled.
"""

import sys
import types

import core.assistant as a
import tools._config as cfg


def _stub_self():
    """Minimal stand-in: _start_reminder_poll only touches _reminder_poll_loop."""
    return types.SimpleNamespace(_reminder_poll_loop=lambda: None)


def _patch_thread(monkeypatch, counter):
    class _FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            counter["n"] += 1

    monkeypatch.setattr(a.threading, "Thread", _FakeThread)


def test_reminder_poll_skipped_when_ha_not_configured(monkeypatch):
    # Tripwire: any attribute access on tools.home_assistant means it was
    # imported despite HA being disabled. The gate must return before the import.
    class _Tripwire(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"tools.home_assistant.{name} accessed despite HA disabled")

    monkeypatch.setitem(sys.modules, "tools.home_assistant", _Tripwire("tools.home_assistant"))
    started = {"n": 0}
    _patch_thread(monkeypatch, started)
    monkeypatch.setattr(cfg, "config", {"general": {}})

    a.Assistant._start_reminder_poll(_stub_self())
    assert started["n"] == 0  # no thread, and no tripwire AssertionError


def test_reminder_poll_starts_when_ha_configured(monkeypatch):
    fake_ha = types.ModuleType("tools.home_assistant")
    fake_ha.HA_CONFIG = {"reminder_poll": True}
    fake_ha._reminder_calendar_entity = lambda: "calendar.fulloch"
    monkeypatch.setitem(sys.modules, "tools.home_assistant", fake_ha)
    started = {"n": 0}
    _patch_thread(monkeypatch, started)
    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})

    a.Assistant._start_reminder_poll(_stub_self())
    assert started["n"] == 1
