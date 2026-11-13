"""HassIL provisional routing through live-ASR, endpoint, and HA dispatch seams.

The transcriber and agent router are real. ASR events and HA HTTP are mocked,
so tests need neither a model nor a network and do not depend on wall-clock sleeps.
"""

import shutil
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from core.asr import AsrInput, stream_generator
from core.asr_parakeet import LiveAsrEvent, LiveAsrResult
from core.audio import AudioCapture
from core.satellite import SatelliteSession
from tools import ha_client, ha_devices
from tools.tool_registry import tool_registry
from utils.hassil_fastpath import HassilFastPath

ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_schema_matches_runtime_fallbacks():
    from server.config_schema import field_for
    from server.config_store import read_config

    audio = AudioCapture(use_vad=False)
    example = read_config(str(ROOT / "data/config.example.yml"))["general"]
    assert audio._vad_soft_endpoint_silence_ms == 100
    assert audio._vad_endpoint_silence_ms == 1500
    assert field_for("general.vad_soft_endpoint_silence_ms").default == 100
    assert field_for("general.vad_endpoint_silence_ms").default == 1500
    assert example["vad_soft_endpoint_silence_ms"] == audio._vad_soft_endpoint_silence_ms
    assert example["vad_endpoint_silence_ms"] == audio._vad_endpoint_silence_ms


def test_endpoint_overrides_remain_independent():
    audio = AudioCapture(use_vad=False, vad_soft_endpoint_silence_ms=250, vad_endpoint_silence_ms=900)
    assert audio._vad_soft_endpoint_silence_ms == 250
    assert audio._vad_endpoint_silence_ms == 900


@pytest.fixture
def voice_route(tmp_path, monkeypatch):
    for language, source in (
        ("en", ROOT / "third_party/home-assistant-intents/en/2025.6.23.yaml"),
        ("fr", ROOT / "tests/fixtures/hassil/fr/2025.6.23.yaml"),
        ("de", ROOT / "tests/fixtures/hassil/de/2025.6.23.yaml"),
    ):
        target = tmp_path / language / "2025.6.23.yaml"
        target.parent.mkdir()
        shutil.copyfile(source, target)

    aliases = {
        "kitchen lights": ["light.kitchen_lights"],
        "lumière de la cuisine": ["light.kitchen_lights"],
        "küchenlicht": ["light.kitchen_lights"],
        "front door": ["lock.front"],
        "bedroom blinds": ["cover.blinds"],
        "thermostat": ["climate.thermostat"],
        "movie night": ["scene.movie_night"],
        "morning routine": ["script.morning_routine"],
    }
    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "test-token")
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", aliases)
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {name: ids[0] for name, ids in aliases.items()})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "kitchen"})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    monkeypatch.setattr(ha_client, "_area_entities", lambda *_: ["light.kitchen_lights"])
    http = Mock(return_value=Mock())
    monkeypatch.setattr(ha_client, "_post", http)

    for attr in ("_tools", "_schemas", "_aliases", "_availability"):
        monkeypatch.setattr(tool_registry, attr, {})
    for name, fn in (
        ("turn_on", ha_devices.turn_on), ("turn_off", ha_devices.turn_off),
        ("ha_set_brightness", ha_devices.set_ha_brightness),
        ("ha_lock", ha_devices.lock), ("ha_unlock", ha_devices.unlock),
        ("ha_close_cover", ha_devices.close_cover), ("ha_set_climate", ha_devices.set_climate),
        ("ha_hassil_area", ha_devices.hassil_area),
    ):
        tool_registry.register_tool(fn, name=name)

    with patch("core.assistant.AudioCapture", return_value=MagicMock()):
        from core.assistant import Assistant

        assistant = Assistant(wakeword="atticus", barge_in="wakeword")
    assistant.audio_capture = AudioCapture(use_vad=False)
    assistant.satellites["sat-a"] = SatelliteSession(id="sat-a")
    assistant._hassil_fastpath = HassilFastPath(["en", "fr", "de"], str(tmp_path))
    assistant.llm_enabled = False
    assistant.laya_enabled = False
    assistant.asr_stream_generator = stream_generator
    assistant._asr_verification_context = None
    assistant._start_turn = Mock(side_effect=lambda prompt, **kw: assistant._handle_wakeword(
        prompt, satellite_id=kw["satellite_id"], source="voice",
    ))
    return assistant, http


def _item(text, *, provisional, capture_id=1):
    """An already-decoded test transcript with recorder metadata."""
    return (text, 10.0, -20.0, provisional, "sat-a", None, False, False,
            None, None, False, None, capture_id)


def _put_live_partial(assistant, text, capture_id=1):
    """Pass a no-endpoint live event through the production event router."""
    pcm = np.zeros(1600, dtype=np.float32)
    item = _item(pcm, provisional=True, capture_id=capture_id)
    event = LiveAsrEvent(LiveAsrResult(text, pcm, capture_id=capture_id), item, False)

    class Worker:
        _closed = Event()
        _closed.set()

        def __init__(self):
            self.events = iter([event, None])

        def get_event(self):
            return next(self.events)

        def is_current_result(self, satellite_id, result):
            return True

    assistant._route_live_asr_events(Worker())


def _drive(assistant, first, final, *, live=False, before_final=None):
    if live:
        _put_live_partial(assistant, first)
    else:
        assistant.audio_capture._put_utterance(_item(first, provisional=True), satellite_id="sat-a", kind="soft")

    def pipe(stream, **kwargs):
        audio = next(stream)
        yield {"text": audio.text if isinstance(audio, LiveAsrResult) else audio}
        if before_final is not None:
            before_final()
        # Queue after the provisional has been processed, so this really
        # exercises late-final deduplication, not just queue pruning.
        assistant.audio_capture._put_utterance(_item(final, provisional=False), satellite_id="sat-a", kind="final")
        assistant.audio_capture.audio_queue.close()
        for audio in stream:
            yield {"text": audio}

    assistant.asr_pipe = pipe
    assistant._run_transcriber_loop()


@pytest.mark.parametrize("live", [True, False], ids=["live-partial", "soft-endpoint"])
@pytest.mark.parametrize("command", [
    "turn on the kitchen lights",
    "allume la lumière de la cuisine",
    "schalte das küchenlicht ein",
])
def test_safe_multilingual_partial_dispatches_once_before_final(voice_route, live, command):
    assistant, http = voice_route
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        target = ["light.kitchen_lights"] if command == "turn on the kitchen lights" else "light.kitchen_lights"
        http.assert_called_once_with("/api/services/light/turn_on", json={"entity_id": target})

    _drive(assistant, text, text, live=live, before_final=check_early)
    assistant._start_turn.assert_called_once()
    http.assert_called_once()


@pytest.mark.parametrize("command,service", [
    ("turn on the kitchen lights", "/api/services/light/turn_on"),
    ("allume la lumière de la cuisine", "/api/services/light/turn_on"),
    ("schalte das küchenlicht ein", "/api/services/light/turn_on"),
    ("set the kitchen lights to 50 percent", "/api/services/light/turn_on"),
    ("règle la luminosité de la lumière de la cuisine à 25 pour cent", "/api/services/light/turn_on"),
    ("dimme küchenlicht auf 25 prozent", "/api/services/light/turn_on"),
])
def test_conversation_mode_dispatches_resolved_light_partial_immediately(voice_route, command, service):
    assistant, http = voice_route
    assistant.satellites["sat-a"].conversation_mode = True
    assistant._queue_conversation_turn = Mock()
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assistant._queue_conversation_turn.assert_not_called()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        assert http.call_count == 1
        assert http.call_args.args[0] == service

    _drive(assistant, text, text, live=True, before_final=check_early)
    assistant._start_turn.assert_called_once()
    http.assert_called_once()


def test_conversation_mode_light_area_dispatches_before_hard_endpoint(voice_route):
    assistant, http = voice_route
    assistant.satellites["sat-a"].conversation_mode = True
    assistant._queue_conversation_turn = Mock()
    text = "atticus turn on the lights in the kitchen"

    def check_early():
        assistant._start_turn.assert_called_once()
        assistant._queue_conversation_turn.assert_not_called()
        http.assert_called_once()

    _drive(assistant, text, text, live=True, before_final=check_early)
    assistant._start_turn.assert_called_once()
    assistant._queue_conversation_turn.assert_not_called()
    assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
    http.assert_called_once()


@pytest.mark.parametrize("command,brightness", [
    ("dim the lights", 76), ("brighten the lights", 255),
    ("tamise la lumière", 76), ("augmente la lumière", 255),
    ("dimme das licht", 76), ("mach das licht heller", 255),
])
def test_default_brightness_dispatches_immediately_once(voice_route, monkeypatch, command, brightness):
    assistant, http = voice_route
    assistant.satellites["sat-a"].conversation_mode = True
    assistant._queue_conversation_turn = Mock()
    monkeypatch.setattr(ha_devices, "_bare_light_area_entities", lambda _: (["light.kitchen_lights"], "kitchen"))
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assistant._queue_conversation_turn.assert_not_called()
        http.assert_called_once_with("/api/services/light/turn_on", json={
            "entity_id": ["light.kitchen_lights"], "brightness": brightness,
        })

    _drive(assistant, text, text, live=True, before_final=check_early)
    http.assert_called_once()


@pytest.mark.parametrize("still_paused", [True, False])
def test_guarded_pause_commits_timer_only_if_speaker_has_not_resumed(voice_route, still_paused):
    assistant, _ = voice_route
    timer = Mock(return_value="Timer started")
    tool_registry.register_tool(timer, name="start_countdown")
    assistant.satellites["sat-a"].vad_endpointer = SimpleNamespace(
        soft_endpointed=still_paused, soft_pause_generation=3,
    )
    text = "atticus set a timer for five minutes"
    item = (*_item(text, provisional=True), None, 3)
    assistant.audio_capture._put_utterance(item, satellite_id="sat-a", kind="provisional")
    assistant.audio_capture.audio_queue.close()
    assistant.asr_pipe = lambda stream, **kw: ({"text": audio} for audio in stream)
    assistant._run_transcriber_loop()
    assert timer.call_count == int(still_paused)
    assert assistant._start_turn.call_count == int(still_paused)


def test_verified_wakeword_soft_snapshot_dispatches_once(voice_route):
    assistant, http = voice_route
    sat = assistant.satellites["sat-a"]
    sat.kws_candidate = True
    sat.kws_verified = True
    sat.kws_capture_id = 7
    sat.audio_capture_id = 1
    text = "atticus turn on the kitchen lights"
    item = (text, 10.0, -20.0, True, "sat-a", None, False, True, None,
            sat.protocol_state_generation, False, 7, 1, None, False)
    assistant.audio_capture._put_utterance(item, satellite_id="sat-a", kind="provisional", candidate=True)
    assistant.audio_capture.audio_queue.close()
    assistant.asr_pipe = lambda stream, **kw: (
        {"text": audio.pcm if isinstance(audio, AsrInput) else audio} for audio in stream
    )
    assistant._run_transcriber_loop()
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
    http.assert_called_once()


@pytest.mark.parametrize("conversation", [False, True], ids=["wakeword", "conversation"])
@pytest.mark.parametrize("command", [
    "turn on lights", "turn off the lights", "licht einschalten", "schalte das licht aus",
    "allume la lumière", "éteins la lumière",
])
def test_satellite_room_light_commands_dispatch_on_partial(voice_route, monkeypatch, conversation, command):
    assistant, http = voice_route
    sat = assistant.satellites["sat-a"]
    sat.ha_area = "kitchen"
    sat.conversation_mode = conversation
    assistant._queue_conversation_turn = Mock()
    monkeypatch.setattr(ha_devices, "_bare_light_area_entities", lambda _: (["light.kitchen_lights"], "kitchen"))
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assistant._queue_conversation_turn.assert_not_called()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        assert http.call_count == 1

    _drive(assistant, text, text, live=True, before_final=check_early)
    http.assert_called_once()


@pytest.mark.parametrize("command,service,brightness", [
    ("turn off light", "turn_off", None),
    ("turn off lights", "turn_off", None),
    ("licht ausschalten", "turn_off", None),
    ("éteins la lumière", "turn_off", None),
    ("dim light", "turn_on", 76),
    ("dim lights", "turn_on", 76),
    ("dimme das licht", "turn_on", 76),
    ("tamise la lumière", "turn_on", 76),
])
def test_bare_light_commands_control_all_satellite_room_lights_early(voice_route, monkeypatch, command, service, brightness):
    assistant, http = voice_route
    assistant.satellites["sat-a"].ha_area = "kitchen"
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "pendant", "light.pendant")
    monkeypatch.setattr(ha_client, "_render_template", lambda _: '["light.kitchen_lights", "light.pendant", "switch.other"]')
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        data = {"entity_id": ["light.kitchen_lights", "light.pendant"]}
        if brightness is not None:
            data["brightness"] = brightness
        http.assert_called_once_with(f"/api/services/light/{service}", json=data)

    _drive(assistant, text, text, live=True, before_final=check_early)
    http.assert_called_once()


@pytest.mark.parametrize("command,operation,value", [
    ("turn on downstairs office lights", "on", None),
    ("turn off downstairs office lights", "off", None),
    ("dim downstairs office lights", "brightness", 76),
])
def test_named_area_lights_dispatch_early_to_all_lights(voice_route, monkeypatch, command, operation, value):
    assistant, http = voice_route
    assistant.satellites["sat-a"].ha_area = "kitchen"
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "kitchen", "downstairs_office": "downstairs office"})
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "office fixture", "light.office")
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "pendant", "light.pendant")
    monkeypatch.setattr(ha_client, "_area_entities", lambda area, domain: ["light.office", "light.pendant"])
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        data = {"entity_id": ["light.office", "light.pendant"]}
        if value is not None:
            data["brightness"] = value
        service = "turn_off" if operation == "off" else "turn_on"
        http.assert_called_once_with(f"/api/services/light/{service}", json=data)

    _drive(assistant, text, text, live=True, before_final=check_early)
    http.assert_called_once()


@pytest.mark.parametrize("llm_enabled", [False, True])
def test_repaired_area_command_and_contextual_followup_use_live_fastpath(voice_route, monkeypatch, llm_enabled):
    assistant, http = voice_route
    assistant.llm_enabled = llm_enabled
    assistant._generate_with_context_recovery = Mock(side_effect=AssertionError("must not call the LLM"))
    satellite = assistant.satellites["sat-a"]
    satellite.ha_area = "kitchen"
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "kitchen", "downstairs_office": "downstairs office"})
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "office fixture", "light.office")
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "pendant", "light.pendant")
    monkeypatch.setattr(ha_client, "_area_entities", lambda area, domain: ["light.office", "light.pendant"])
    first = "atticus turn on downstairs office slides"

    def check_first():
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        http.assert_called_once_with("/api/services/light/turn_on", json={"entity_id": ["light.office", "light.pendant"]})

    _drive(assistant, first, first, live=True, before_final=check_first)
    assert satellite.fastpath_light_context["target"] == "downstairs_office"
    assistant.audio_capture = AudioCapture(use_vad=False)
    assistant._start_turn.reset_mock()
    http.reset_mock()
    satellite.committed_capture_id = 0
    second = "atticus brighton then"

    def check_second():
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        http.assert_called_once_with("/api/services/light/turn_on", json={
            "entity_id": ["light.office", "light.pendant"], "brightness_step_pct": 20,
        })

    _drive(assistant, second, second, live=True, before_final=check_second)
    http.assert_called_once()
    assistant._generate_with_context_recovery.assert_not_called()


@pytest.mark.parametrize("command", ["licht ausschalten", "éteins la lumière"])
def test_language_yaml_command_targets_satellite_room_early(voice_route, monkeypatch, command):
    assistant, http = voice_route
    assistant.satellites["sat-a"].ha_area = "kitchen"
    monkeypatch.setattr(ha_devices, "_bare_light_area_entities", lambda _: (["light.kitchen_lights"], "kitchen"))
    text = f"atticus {command}"

    def check_early():
        assistant._start_turn.assert_called_once()
        assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "soft"
        http.assert_called_once_with("/api/services/light/turn_off", json={"entity_id": ["light.kitchen_lights"]})

    _drive(assistant, text, text, live=True, before_final=check_early)
    http.assert_called_once()


def test_area_entity_prefix_waits_for_complete_plural_target(voice_route, monkeypatch):
    assistant, http = voice_route
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"downstairs_office": "downstairs office"})
    monkeypatch.setitem(ha_client._ENTITY_ALIASES, "downstairs office", "light.office")
    monkeypatch.setitem(ha_client._ENTITY_ALIASES_MULTI, "downstairs office", ["light.office"])
    monkeypatch.setattr(ha_client, "_area_entities", lambda *_: ["light.office"])

    def held():
        assistant._start_turn.assert_not_called()
        http.assert_not_called()

    _drive(assistant, "atticus turn on downstairs office", "atticus turn on downstairs office lights", live=True, before_final=held)
    assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "hard"
    http.assert_called_once()


@pytest.mark.parametrize("command,service", [
    ("lock the front door", "/api/services/lock/lock"),
    ("unlock the front door", "/api/services/lock/unlock"),
    ("close the bedroom blinds", "/api/services/cover/close_cover"),
    ("set thermostat to 21 degrees", "/api/services/climate/set_temperature"),
    ("turn on movie night", "/api/services/scene/turn_on"),
    ("run morning routine", "/api/services/script/turn_on"),
])
def test_unsafe_partial_waits_for_hard_endpoint(voice_route, command, service):
    assistant, http = voice_route
    text = f"atticus {command}"

    def check_held():
        assistant._start_turn.assert_not_called()
        http.assert_not_called()

    _drive(assistant, text, text, live=True, before_final=check_held)
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "hard"
    assert http.call_count == 1
    assert http.call_args.args[0] == service


@pytest.mark.parametrize("partial", ["turn on the", "tell me something unrelated"])
def test_incomplete_or_unmatched_partial_waits_then_final_executes(voice_route, partial):
    assistant, http = voice_route

    def check_held():
        assistant._start_turn.assert_not_called()
        http.assert_not_called()

    _drive(assistant, f"atticus {partial}", "atticus turn on the kitchen lights", before_final=check_held)
    assistant._start_turn.assert_called_once()
    assert assistant._start_turn.call_args.kwargs["endpoint_kind"] == "hard"
    http.assert_called_once()


@pytest.mark.parametrize("blocked", [True, False], ids=["denied", "ambiguous"])
def test_blocked_or_ambiguous_partial_never_dispatches(voice_route, monkeypatch, blocked):
    assistant, http = voice_route
    if blocked:
        monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.kitchen_lights"}))
    else:
        aliases = dict(ha_client._ENTITY_ALIASES_MULTI)
        aliases["kitchen lights"] = ["light.kitchen_lights", "switch.kitchen"]
        monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", aliases)

    def check_held():
        assistant._start_turn.assert_not_called()
        http.assert_not_called()

    text = "atticus turn on the kitchen lights"
    _drive(assistant, text, text, live=True, before_final=check_held)
    http.assert_not_called()


@pytest.mark.parametrize("text", ["atticus um wie viel uhr ist es", "atticus um wie viel uhr ist das"])
def test_german_time_live_partial_dispatches_without_ha_discovery(voice_route, monkeypatch, text):
    assistant, http = voice_route
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    clock = Mock(return_value="The time")
    tool_registry.register_tool(clock, name="get_current_time")
    def check_early():
        assistant._start_turn.assert_called_once()
        clock.assert_called_once_with()
        http.assert_not_called()

    _drive(assistant, text, text, live=True, before_final=check_early)
    clock.assert_called_once_with()
