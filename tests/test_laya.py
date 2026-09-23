"""Laya's bounded semantic-command adapter without model downloads."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.laya import LayaRouter, _timer_duration_candidates


class _Agent:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.questions = []
        self.commands = []

    def predict(self, state, questions):
        self.commands.append(state["command"])
        self.questions.append(questions)
        return {"answers": {"route": next(self.answers)}}


def test_routes_a_high_confidence_media_command(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name in {"pause", "resume"})
    router = LayaRouter(
        _Agent(
            [
                {"choice": "pause", "confidence": 0.97},
            ]
        )
    )

    assert router.route("please stop whatever is playing") == {
        "actions": [{"intent": "pause", "args": []}]
    }


def test_rejects_low_confidence_before_selecting_an_action(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_current_time")
    agent = _Agent([{"choice": "get_current_time", "confidence": 0.89}])

    assert LayaRouter(agent).route("can you tell me something") is None
    assert len(agent.questions) == 1


def test_rejects_an_unsupported_action_choice(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_current_time")
    router = LayaRouter(
        _Agent(
            [
                {"choice": "unsupported", "confidence": 0.98},
            ]
        )
    )

    assert router.route("tell me the latest weather on mars") is None


def test_entity_actions_require_a_selected_live_alias(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "turn_on")
    monkeypatch.setattr("core.laya._entity_candidates", lambda _command, _domain=None: {"kitchen lamp": "light.kitchen"})
    router = LayaRouter(
        _Agent(
            [
                {"choice": "turn_on", "confidence": 0.99},
                {"choice": "kitchen lamp", "confidence": 0.99},
            ]
        )
    )

    assert router.route("turn on the kitchen lamp") == {
        "actions": [{"intent": "turn_on", "args": ["light.kitchen"]}]
    }


def test_entity_actions_reject_a_missing_candidate(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "turn_off")
    monkeypatch.setattr("core.laya._entity_candidates", lambda _command, _domain=None: {})
    router = LayaRouter(_Agent([{"choice": "turn_off", "confidence": 0.99}]))

    assert router.route("turn off the mystery light") is None


def test_rejects_non_finite_model_confidence(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_current_time")
    router = LayaRouter(_Agent([{"choice": "get_current_time", "confidence": float("nan")}]))

    assert router.route("what time is it") is None


def test_action_schema_stays_within_the_calibrated_choice_budget(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda _name: True)
    agent = _Agent([{"choice": "get_current_time", "confidence": 0.99}])

    assert LayaRouter(agent).route("what time is it") == {
        "actions": [{"intent": "get_current_time", "args": []}]
    }
    assert len(agent.questions[0]["route"]["criteria"]) == 10


def test_home_action_is_prioritised_within_the_choice_budget(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda _name: True)
    monkeypatch.setattr("core.laya._entity_candidates", lambda _command: {"office light": "light.office"})
    agent = _Agent([
        {"choice": "turn_off", "confidence": 0.99},
        {"choice": "office light", "confidence": 0.99},
    ])

    assert LayaRouter(agent).route("shut down the office light") == {
        "actions": [{"intent": "turn_off", "args": ["light.office"]}]
    }
    assert "turn_off" in agent.questions[0]["route"]["criteria"]


def test_unresolved_explicit_media_target_does_not_use_the_default_player(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "pause")
    monkeypatch.setattr("core.laya._entity_candidates", lambda _command, _domain=None: {})

    assert LayaRouter(_Agent([{"choice": "pause", "confidence": 0.99}])).route(
        "pause the music in the mystery room"
    ) is None


def test_weather_with_a_date_is_not_routed_as_default_weather(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_weather_forecast")

    assert LayaRouter(_Agent([{"choice": "get_weather_forecast", "confidence": 0.99}])).route(
        "what is the weather tomorrow"
    ) is None


def test_routes_temperature_to_a_climate_or_sensor(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_temperature")
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, domain=None, preferred=None: {"upstairs": "climate.upstairs"}
        if domain == "climate" else {},
    )

    assert LayaRouter(_Agent([
        {"choice": "get_temperature", "confidence": 0.99},
        {"choice": "upstairs", "confidence": 0.99},
    ])).route("what's the temperature upstairs") == {
        "actions": [{"intent": "get_temperature", "args": ["climate.upstairs"]}]
    }


def test_timer_candidates_normalise_simple_and_compound_durations():
    assert _timer_duration_candidates("start a twenty five minute countdown") == {
        "twenty five minute": "25 minute"
    }
    assert _timer_duration_candidates("start a one hour and thirty minute countdown") == {
        "one hour thirty minute": "1 hour 30 minute"
    }
    assert _timer_duration_candidates("set a timer to one minute and thirty-five seconds") == {
        "one minute thirty-five seconds": "1 minute 35 seconds"
    }


def test_routes_timer_cancellation_without_an_id(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "cancel_timer")

    assert LayaRouter(_Agent([{"choice": "cancel_timer", "confidence": 0.99}])).route(
        "please stop the countdown"
    ) == {"actions": [{"intent": "cancel_timer", "args": []}]}


def test_brightness_follow_up_uses_prior_target_and_raw_percentage(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "ha_set_brightness")
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, _domain=None, preferred=None: {preferred: preferred} if preferred else {},
    )
    agent = _Agent([
        {"choice": "ha_set_brightness", "confidence": 0.99},
        {"choice": "downstairs office lights", "confidence": 0.99},
        {"choice": "100 percent", "confidence": 0.99},
    ])

    assert LayaRouter(agent).route(
        "can you brighten them again",
        context={
            "request": "can you dim the downstairs office lights",
            "intent": "ha_set_brightness",
            "target": "downstairs office lights",
            "result": "Lights in downstairs office at 30 percent",
        },
    ) == {"actions": [{"intent": "ha_set_brightness", "args": ["downstairs office lights", "100 percent"]}]}
    assert "Previous request: can you dim the downstairs office lights." in agent.commands[0]


def test_prior_media_target_is_not_offered_for_a_light_action(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "ha_set_brightness")
    monkeypatch.setattr("core.laya._entity_candidates", lambda *_args, **_kwargs: {})

    assert LayaRouter(_Agent([{"choice": "ha_set_brightness", "confidence": 0.99}])).route(
        "brighten them",
        context={"request": "turn down the TV", "intent": "ha_volume_down", "target": "TV", "result": "TV quieter"},
    ) is None


def test_brightness_follow_up_reuses_a_prior_turn_on_target(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "ha_set_brightness")
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, _domain=None, preferred=None: {preferred: preferred} if preferred else {},
    )

    assert LayaRouter(_Agent([
        {"choice": "ha_set_brightness", "confidence": 0.99},
        {"choice": "downstairs lights", "confidence": 0.99},
        {"choice": "30 percent", "confidence": 0.99},
    ])).route("now dim them", context={
        "request": "turn on downstairs lights",
        "intent": "turn_on",
        "target": "downstairs lights",
        "result": "downstairs lights on",
    }) == {"actions": [{"intent": "ha_set_brightness", "args": ["downstairs lights", 30]}]}


def test_turn_on_follow_up_reuses_a_prior_turn_off_target(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "turn_on")
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, _domain=None, preferred=None: {preferred: preferred} if preferred else {},
    )

    assert LayaRouter(_Agent([
        {"choice": "turn_on", "confidence": 0.99},
        {"choice": "downstairs lights", "confidence": 0.99},
    ])).route("now turn them back on", context={
        "request": "turn off downstairs lights",
        "intent": "turn_off",
        "target": "downstairs lights",
        "result": "downstairs lights off",
    }) == {"actions": [{"intent": "turn_on", "args": ["downstairs lights"]}]}


@pytest.mark.parametrize(
    "command,context,expected",
    [
        ("turn them back on", {"intent": "turn_off", "target": "office lights"}, {"intent": "turn_on", "args": ["office lights"]}),
        ("now brighten them", {"intent": "turn_on", "target": "office lights"}, {"intent": "ha_set_brightness", "args": ["office lights", 100]}),
        ("turn it down", {"intent": "ha_volume_set", "target": "TV"}, {"intent": "ha_volume_down", "args": ["TV"]}),
        ("pause it", {"intent": "ha_volume_set", "target": "TV"}, {"intent": "pause", "args": ["TV"]}),
        ("stop it", {"intent": "ha_open_cover", "target": "kitchen blind"}, {"intent": "ha_stop_cover", "args": ["kitchen blind"]}),
        ("lock it", {"intent": "ha_unlock", "target": "front door"}, {"intent": "ha_lock", "args": ["front door"]}),
    ],
)
def test_contextual_follow_ups_bypass_model(monkeypatch, command, context, expected):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == expected["intent"])
    agent = _Agent([])
    assert LayaRouter(agent).route(command, context=context) == {"actions": [expected]}
    assert agent.commands == []


def test_routes_safe_home_overview_without_arguments(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "get_home_overview")
    assert LayaRouter(_Agent([{"choice": "get_home_overview", "confidence": 0.99}])).route(
        "give me a home status"
    ) == {"actions": [{"intent": "get_home_overview", "args": []}]}


def test_routes_fan_speed_with_a_normalised_percentage(monkeypatch):
    monkeypatch.setattr("core.laya.tool_registry.is_available", lambda name: name == "ha_set_fan_speed")
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, _domain=None, preferred=None: {"bedroom fan": "fan.bedroom"},
    )
    assert LayaRouter(_Agent([
        {"choice": "ha_set_fan_speed", "confidence": 0.99},
        {"choice": "bedroom fan", "confidence": 0.99},
        {"choice": "seventy percent", "confidence": 0.99},
    ])).route("set the bedroom fan speed to seventy percent") == {
        "actions": [{"intent": "ha_set_fan_speed", "args": ["fan.bedroom", 70]}]
    }
