"""HassIL provisional routing through live-ASR, endpoint, and HA dispatch seams.

The transcriber and agent router are real. ASR events and HA HTTP are mocked,
so tests need neither a model nor a network and do not depend on wall-clock sleeps.
"""

import shutil
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from core.asr import stream_generator
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
        http.assert_called_once_with("/api/services/light/turn_on", json={"entity_id": "light.kitchen_lights"})

    _drive(assistant, text, text, live=live, before_final=check_early)
    assistant._start_turn.assert_called_once()
    http.assert_called_once()


@pytest.mark.parametrize("command,service", [
    ("lock the front door", "/api/services/lock/lock"),
    ("unlock the front door", "/api/services/lock/unlock"),
    ("close the bedroom blinds", "/api/services/cover/close_cover"),
    ("set thermostat to 21 degrees", "/api/services/climate/set_temperature"),
    ("turn on movie night", "/api/services/scene/turn_on"),
    ("run morning routine", "/api/services/script/turn_on"),
    ("turn on the lights in the kitchen", "/api/services/light/turn_on"),
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
