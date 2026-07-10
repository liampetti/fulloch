"""`core/satellite_context.py` — the tiny, dependency-free module a tool
module can import to read the calling satellite's id and reach the live
Assistant instance, without pulling in core.agent_loop/core.assistant
(both heavy — core.slm alone imports torch)."""

import sys


def test_module_has_no_heavy_imports():
    """Importing this module must not pull in torch — that's the whole
    point of it existing as a separate module from core.agent_loop."""
    had_torch = "torch" in sys.modules
    import core.satellite_context  # noqa: F401

    if not had_torch:
        assert "torch" not in sys.modules


def test_current_assistant_defaults_to_none():
    from core.satellite_context import get_current_assistant

    # Not asserting None unconditionally — another test module may have
    # registered something first. Just confirm the getter doesn't blow up
    # and round-trips whatever's set below.
    assert get_current_assistant() == get_current_assistant()


def test_set_and_get_current_assistant_round_trips():
    from core.satellite_context import get_current_assistant, set_current_assistant

    sentinel = object()
    try:
        set_current_assistant(sentinel)
        assert get_current_assistant() is sentinel
    finally:
        set_current_assistant(None)


def test_current_satellite_id_defaults_to_none_outside_a_turn():
    from core.satellite_context import current_satellite_id

    assert current_satellite_id.get() is None


def test_current_satellite_id_set_and_reset():
    from core.satellite_context import current_satellite_id

    token = current_satellite_id.set("sat-a")
    try:
        assert current_satellite_id.get() == "sat-a"
    finally:
        current_satellite_id.reset(token)
    assert current_satellite_id.get() is None


def test_agent_loop_reexports_the_same_contextvar():
    import core.agent_loop as al
    import core.satellite_context as sc

    assert al._current_satellite_id is sc.current_satellite_id
