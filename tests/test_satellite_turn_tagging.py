"""Phase 6a (#14): chat-bubble events (`_emit_turn_event`, consumed via
`/stream`) are tagged with `satellite_id`/`satellite_label` so the dashboard
can show a "from <label>" pill on a turn. Unlike `_emit_agent_event` (which
reuses the `_current_satellite_id` contextvar), `_emit_turn_event` takes the
id explicitly — half its call sites fire outside the ctxvar's scope (before
`AgentLoop.run()` starts, or after it's already returned).
"""

from unittest.mock import MagicMock, patch

from core.satellite import SatelliteSession
from core.tts_session import TtsSession


def _make_assistant(**kwargs):
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus", **kwargs)
    return a


class TestEmitTurnEventTagging:
    def test_labelled_satellite_is_tagged(self):
        a = _make_assistant()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", chunk_q=None, label="kitchen")
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "the light is on", "voice", satellite_id="sat-a")

        assert events[-1]["satellite_id"] == "sat-a"
        assert events[-1]["satellite_label"] == "kitchen"

    def test_unlabelled_satellite_reports_null_label(self):
        a = _make_assistant()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", chunk_q=None)  # no label set
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "ok", "voice", satellite_id="sat-a")

        assert events[-1]["satellite_id"] == "sat-a"
        assert events[-1]["satellite_label"] is None

    def test_unknown_satellite_id_reports_null_label(self):
        a = _make_assistant()
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "ok", "voice", satellite_id="never-connected")

        assert events[-1]["satellite_id"] == "never-connected"
        assert events[-1]["satellite_label"] is None

    def test_no_satellite_id_reports_null_for_both(self):
        a = _make_assistant()
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "ok", "voice")

        assert events[-1]["satellite_id"] is None
        assert events[-1]["satellite_label"] is None

    def test_ha_area_name_is_used_when_no_label_set(self):
        """A browser satellite with no configured `label` but a chosen HA room
        (SatelliteSession.ha_area_name, set from /ws/satellite's ?area_name=)
        still gets a location pill — the room choice is the fallback."""
        a = _make_assistant()
        a.satellites["sat-a"] = SatelliteSession(
            id="sat-a", chunk_q=None, ha_area="living_room", ha_area_name="Living Room"
        )
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "the light is on", "voice", satellite_id="sat-a")

        assert events[-1]["satellite_label"] == "Living Room"

    def test_label_takes_precedence_over_ha_area_name(self):
        a = _make_assistant()
        a.satellites["sat-a"] = SatelliteSession(
            id="sat-a", chunk_q=None, label="kitchen", ha_area_name="Living Room"
        )
        events = []
        a.register_turn_listener(events.append)

        a._emit_turn_event("assistant", "ok", "voice", satellite_id="sat-a")

        assert events[-1]["satellite_label"] == "kitchen"


class TestHandleTextTurnTagging:
    def test_user_and_assistant_events_tagged_dashboard_text(self):
        a = _make_assistant()
        a.models_ready.set()
        a.llm_enabled = False  # regex-only bypass, no SLM needed
        events = []
        a.register_turn_listener(events.append)

        a.handle_text_turn("what time is it")

        user_events = [e for e in events if e.get("role") == "user"]
        assistant_events = [e for e in events if e.get("role") == "assistant"]
        assert user_events and all(e["satellite_id"] == "dashboard-text" for e in user_events)
        assert assistant_events and all(
            e["satellite_id"] == "dashboard-text" for e in assistant_events
        )

    def test_busy_bounce_is_tagged_dashboard_text(self):
        a = _make_assistant()
        a.models_ready.set()
        a._turn_arbiter.try_acquire("sat-a")  # something else already owns the arbiter
        events = []
        a.register_turn_listener(events.append)

        a.handle_text_turn("what time is it")

        assistant_events = [e for e in events if e.get("role") == "assistant"]
        assert assistant_events
        assert assistant_events[-1]["satellite_id"] == "dashboard-text"


class TestFallbackHelpersTagSatellite:
    def test_speak_no_ai_fallback_tags_the_given_satellite(self):
        from utils.phrases import NO_AI_PHRASES

        a = _make_assistant()
        a.satellites["sat-a"] = SatelliteSession(id="sat-a", chunk_q=None, label="office")
        a.no_ai_cache = [(["chunk"], 24000)] * len(NO_AI_PHRASES)
        a._tts_module = MagicMock()
        events = []
        a.register_turn_listener(events.append)

        a._speak_no_ai_fallback(TtsSession(), "voice", satellite_id="sat-a")

        assistant_events = [e for e in events if e.get("role") == "assistant"]
        assert assistant_events[-1]["satellite_id"] == "sat-a"
        assert assistant_events[-1]["satellite_label"] == "office"
