"""End-to-end Laya fallback scenarios without a model download or HA server."""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_loop import AgentLoop
from core.laya import LayaRouter


@dataclass(frozen=True)
class Scenario:
    name: str
    query: str
    choices: tuple[str, ...]
    expected: dict | None
    context: dict | None = None


class _ScriptedLaya:
    def __init__(self, choices):
        self._choices = iter(choices)
        self.commands = []

    def predict(self, state, _questions):
        self.commands.append(state["command"])
        return {"answers": {"route": {"choice": next(self._choices), "confidence": 0.99}}}


_OFFICE_DIM = {
    "request": "dim the office lights",
    "intent": "ha_set_brightness",
    "target": "office lights",
    "value": "30 percent",
    "result": "Office lights at 30 percent",
}
_OFFICE_BRIGHT = {**_OFFICE_DIM, "request": "brighten the office lights", "value": "100 percent"}
_TV_VOLUME = {
    "request": "set the TV volume to thirty percent",
    "intent": "ha_volume_set",
    "target": "TV",
    "value": "thirty percent",
    "result": "TV volume 30",
}


SCENARIOS = (
    Scenario("paraphrase darker", "Could you make the office a bit darker?", ("ha_set_brightness", "office lights", "30 percent"), {"intent": "ha_set_brightness", "args": ["office lights", "30 percent"]}),
    Scenario("paraphrase brighter", "Could you make the office as bright as possible?", ("ha_set_brightness", "office lights", "100 percent"), {"intent": "ha_set_brightness", "args": ["office lights", "100 percent"]}),
    Scenario("spoken brightness", "Set the office lights to seventy percent", ("ha_set_brightness", "office lights", "seventy percent"), {"intent": "ha_set_brightness", "args": ["office lights", "seventy percent"]}),
    Scenario("spoken volume", "Set the TV sound to thirty", ("ha_volume_set", "TV", "thirty"), {"intent": "ha_volume_set", "args": ["TV", "thirty"]}),
    Scenario("asr room typo", "Turn off the kichen light", ("turn_off", "kitchen light"), {"intent": "turn_off", "args": ["kitchen light"]}),
    Scenario("asr colour typo", "Make the livng room lamps blue", ("ha_set_color", "living room lamps", "blue"), {"intent": "ha_set_color", "args": ["living room lamps", "blue"]}),
    Scenario("asr lock typo", "Lock the frnt door", ("ha_lock", "front door"), {"intent": "ha_lock", "args": ["front door"]}),
    Scenario("asr cover typo", "Open the garag door", ("ha_open_cover", "garage door"), {"intent": "ha_open_cover", "args": ["garage door"]}),
    Scenario("follow up brighten", "Can you brighten them again?", ("ha_set_brightness", "office lights", "100 percent"), {"intent": "ha_set_brightness", "args": ["office lights", "100 percent"]}, _OFFICE_DIM),
    Scenario("follow up dim", "Can you dim them again?", ("ha_set_brightness", "office lights", "30 percent"), {"intent": "ha_set_brightness", "args": ["office lights", "30 percent"]}, _OFFICE_BRIGHT),
    Scenario("changed target", "Now do the kitchen instead", ("ha_set_brightness", "kitchen lights", "30 percent"), {"intent": "ha_set_brightness", "args": ["kitchen lights", "30 percent"]}, _OFFICE_DIM),
    Scenario("media follow up", "Pause it", ("pause", "TV"), {"intent": "pause", "args": ["TV"]}, _TV_VOLUME),
    Scenario("context override brightness", "Brighten the kitchen lights", ("ha_set_brightness", "kitchen lights", "100 percent"), {"intent": "ha_set_brightness", "args": ["kitchen lights", "100 percent"]}, _OFFICE_DIM),
    Scenario("context override colour", "Make the kitchen lights red", ("ha_set_color", "kitchen lights", "red"), {"intent": "ha_set_color", "args": ["kitchen lights", "red"]}, _OFFICE_DIM),
    Scenario("time", "Could you tell me the time please", ("get_current_time",), {"intent": "get_current_time", "args": []}),
    Scenario("timer", "Start a twenty five minute countdown", ("start_countdown", "twenty five minute"), {"intent": "start_countdown", "args": ["25 minute"]}),
    Scenario("timer status", "How much time is left on my timers", ("get_timer_status",), {"intent": "get_timer_status", "args": []}),
    Scenario("unsupported joke", "Tell me a joke about a penguin", ("unsupported",), None),
    Scenario("unsupported music search", "Play the Beatles", ("unsupported",), None),
    Scenario("unsupported general question", "What is the capital of France?", ("unsupported",), None),
)


def _host(router, context, spoken):
    history = []
    return types.SimpleNamespace(
        llm_enabled=False,
        laya_enabled=True,
        laya_router=router,
        _laya_context=context,
        _history_for=lambda _satellite: history,
        _trim_history=lambda: None,
        _compact_completed_turns=lambda: None,
        _emit_agent_event=lambda *_args, **_kwargs: None,
        _speak_no_ai_fallback=lambda *_args, **_kwargs: spoken.append("NO_AI") or "NO_AI",
        _record_spoken=lambda text: spoken.append(text),
        _record_laya_action=lambda *_args: None,
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_laya_pipeline_dispatches_expected_tool_call(monkeypatch, scenario):
    import core.agent_loop as agent_loop
    import utils.intents as intents

    monkeypatch.setattr(agent_loop, "catchAll", lambda prompt: prompt)
    monkeypatch.setattr(
        "core.laya.tool_registry.is_available",
        lambda name: scenario.expected is None or name == scenario.expected["intent"],
    )
    expected_args = scenario.expected["args"] if scenario.expected is not None else []
    target = expected_args[0] if expected_args and isinstance(expected_args[0], str) else None
    monkeypatch.setattr(
        "core.laya._entity_candidates",
        lambda _command, _domain=None, preferred=None: {target: target} if target else {},
    )
    calls = []
    monkeypatch.setattr(intents, "handle_action", lambda action: calls.append(action) or "tool completed")

    model = _ScriptedLaya(scenario.choices)
    spoken = []
    result = AgentLoop(_host(LayaRouter(model), scenario.context, spoken), source="text").run(scenario.query)

    if scenario.expected is None:
        assert calls == []
        assert result == "NO_AI"
    else:
        assert calls == [scenario.expected]
        assert result == "tool completed"
    # Some strictly unambiguous pronoun follow-ups are now resolved from the
    # successful prior target without calling the semantic classifier.
    if scenario.context is not None and model.commands:
        assert "Previous request:" in model.commands[0]
