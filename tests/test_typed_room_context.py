"""Typed browser room context through HTTP, real dispatch, and HA boundaries."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from core.satellite import SatelliteSession
from core.satellite_context import current_satellite_ha_area
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle
from tools import ha_client, ha_devices
from tools.tool_registry import tool_registry


@pytest.fixture
def room_chat(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr("core.satellite_context._current_assistant", None)
    monkeypatch.setattr("server.auth.load_sessions", lambda: {})
    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "test-token")
    monkeypatch.setattr(ha_client, "_AREA_MAP", {
        "downstairs_office": "Downstairs Office", "kitchen": "Kitchen",
    })
    aliases = {
        "entry light": "light.entry_light",
        "downstairs office": "light.office",
        "office pendant": "light.office_pendant",
        "kitchen light": "light.kitchen",
    }
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", aliases)
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {k: [v] for k, v in aliases.items()})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    monkeypatch.setattr(ha_client, "_render_template", Mock(side_effect=lambda template: json.dumps(
        ["light.office", "light.office_pendant"] if "downstairs_office" in template
        else ["light.kitchen"]
    )))
    post = Mock(return_value=Mock())
    monkeypatch.setattr(ha_client, "_post", post)
    # Ensure both normal dispatch and the local tool schema are exercised,
    # regardless of the developer machine's integration configuration.
    for attr in ("_tools", "_schemas", "_aliases", "_availability"):
        monkeypatch.setattr(tool_registry, attr, {})
    for name in ("turn_on", "turn_off", "toggle"):
        tool_registry.register_tool(getattr(ha_devices, name), name=name)
    tool_registry.register_tool(ha_devices.set_ha_brightness, name="ha_set_brightness")

    with patch("core.assistant.AudioCapture", return_value=MagicMock()):
        from core.assistant import Assistant

        assistant = Assistant(wakeword="hey atticus", barge_in="off")
    assistant.models_ready.set()
    assistant.llm_enabled = False
    assistant.laya_enabled = False
    assistant._generate_with_context_recovery = Mock(side_effect=AssertionError("No LLM needed"))
    config = tmp_path / "config.yml"
    config.write_text("general: {}\n")
    context = AppContext(lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(config))
    with TestClient(create_app(context=context)) as http:
        yield http, assistant, post


@pytest.mark.parametrize("text,service,data", [
    ("turn on lights", "turn_on", {}),
    ("turn off lights", "turn_off", {}),
    ("toggle lights", "toggle", {}),
    ("set lights to 50 percent", "turn_on", {"brightness": 127}),
])
def test_typed_fast_path_targets_pinned_room_without_voice_connection(room_chat, text, service, data):
    http, assistant, post = room_chat
    response = http.post("/chat", json={"text": text, "ha_area": "downstairs_office"})

    assert response.status_code == 200
    assert "Downstairs Office" in response.json()["answer"]
    post.assert_called_once_with(f"/api/services/light/{service}", json={
        "entity_id": ["light.office", "light.office_pendant"], **data,
    })
    assert assistant._last_turn_stats["route"] == "regex"
    assistant._generate_with_context_recovery.assert_not_called()
    assert current_satellite_ha_area.get() is None
    assert assistant.satellites["dashboard-text"].ha_area is None


def test_original_three_turn_conversation_stays_on_fast_path(room_chat):
    http, assistant, post = room_chat
    for text in (
        "turn off lights", "turn off downstairs office lights", "turn them back on",
    ):
        response = http.post("/chat", json={"text": text, "ha_area": "downstairs_office"})
        assert response.status_code == 200
        assert assistant._last_turn_stats["route"] == "regex"
    assert [call.args[0] for call in post.call_args_list] == [
        "/api/services/light/turn_off", "/api/services/light/turn_off", "/api/services/light/turn_on",
    ]
    assert [call.kwargs["json"]["entity_id"] for call in post.call_args_list] == [
        ["light.office", "light.office_pendant"], "light.office", "light.office",
    ]
    assistant._generate_with_context_recovery.assert_not_called()


def test_bare_light_follow_up_reuses_the_pinned_room(room_chat):
    http, assistant, post = room_chat
    for text in ("turn off lights", "turn them back on"):
        response = http.post("/chat", json={"text": text, "ha_area": "downstairs_office"})
        assert response.status_code == 200
        assert assistant._last_turn_stats["route"] == "regex"
    assert post.call_count == 2
    assert all(call.kwargs["json"]["entity_id"] == ["light.office", "light.office_pendant"]
               for call in post.call_args_list)


def test_voice_fast_path_still_uses_satellite_room(room_chat):
    _http, assistant, post = room_chat
    assistant.satellites["office-voice"] = SatelliteSession(
        id="office-voice", chunk_q=None, ha_area="downstairs_office",
    )
    assert "Downstairs Office" in assistant._handle_wakeword(
        "turn off lights", source="voice", satellite_id="office-voice",
    )
    post.assert_called_once_with("/api/services/light/turn_off", json={
        "entity_id": ["light.office", "light.office_pendant"],
    })
    assistant._generate_with_context_recovery.assert_not_called()
    assert current_satellite_ha_area.get() is None


def test_room_default_keeps_denylist_enforcement(room_chat, monkeypatch):
    http, _assistant, post = room_chat
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.office_pendant"}))
    response = http.post("/chat", json={"text": "turn off lights", "ha_area": "downstairs_office"})
    assert response.status_code == 200
    post.assert_called_once_with("/api/services/light/turn_off", json={"entity_id": ["light.office"]})


def test_room_is_request_scoped_and_does_not_leak_to_roomless_callers(room_chat):
    http, _assistant, post = room_chat
    for area, target in (("downstairs_office", ["light.office", "light.office_pendant"]),
                         ("kitchen", ["light.kitchen"]), (None, "light.entry_light")):
        response = http.post("/chat", json={"text": "turn off lights", "ha_area": area})
        assert response.status_code == 200
        assert post.call_args.kwargs["json"]["entity_id"] == target


def test_explicit_target_still_overrides_pinned_room(room_chat):
    http, _assistant, post = room_chat
    http.post("/chat", json={"text": "turn off kitchen light", "ha_area": "downstairs_office"})
    assert post.call_args.kwargs["json"]["entity_id"] == "light.kitchen"


def test_invalid_room_is_rejected_without_dispatch(room_chat):
    http, assistant, post = room_chat
    response = http.post("/chat", json={"text": "turn off lights", "ha_area": "unknown"})
    assert response.status_code == 400
    assert "no command was sent" in response.json()["detail"]
    post.assert_not_called()
    assert assistant._history == []


@pytest.mark.parametrize("raw", [None, "[]", "not json", "{}", '[null]'])
def test_failed_room_lookup_never_fuzzy_matches_entry_light(room_chat, monkeypatch, raw):
    http, assistant, post = room_chat
    monkeypatch.setattr(ha_client, "_render_template", Mock(return_value=raw))
    response = http.post("/chat", json={"text": "turn off lights", "ha_area": "downstairs_office"})
    assert "No command was sent" in response.json()["answer"]
    post.assert_not_called()
    assistant._generate_with_context_recovery.assert_not_called()


def test_typed_llm_prompt_receives_the_same_room_as_fast_path(room_chat, monkeypatch):
    http, assistant, post = room_chat
    assistant.llm_enabled = True
    assistant.replan_stall_cache = []
    assistant.grammar = None
    assistant._generate_with_context_recovery = Mock(return_value='{"reply":"Which device?"}')
    prompt_builder = Mock(return_value="system prompt")
    monkeypatch.setattr("core.agent_loop.get_agent_system_prompt", prompt_builder)

    response = http.post("/chat", json={"text": "what is in this room?", "ha_area": "downstairs_office"})
    assert response.status_code == 200
    assert prompt_builder.call_args.kwargs["satellite_area"] == "downstairs_office"
    assert prompt_builder.call_args.kwargs["vault_context"] == assistant._vault_current_file
    post.assert_not_called()
    assert current_satellite_ha_area.get() is None
