"""Timer artifact tests."""

import datetime

import pytest


class _Timer:
    def __init__(self, interval, function, args=()):
        self.interval = interval
        self.function = function
        self.args = args
        self.daemon = False
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def test_start_countdown_attaches_timer_artifact(monkeypatch):
    import tools.time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    monkeypatch.setattr(timers.time, "time", lambda: 1000)

    result = timers.start_countdown("2 minutes", "Check the pasta")

    assert result == "Timer started for 2 minutes"
    assert result.artifact == {
        "type": "timers",
        "timers": [
            {
                "id": next(iter(timers.active_timers)),
                "label": "Check the pasta",
                "ends_at": 1120,
                "duration": 120,
                "remaining": 120,
            }
        ],
    }


def test_compound_countdown_preserves_every_duration_component(monkeypatch):
    import tools.time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    monkeypatch.setattr(timers.time, "time", lambda: 1000)

    result = timers.start_countdown("one minute and thirty-five seconds")

    assert result == "Timer started for 1 minute 35 seconds"
    assert result.artifact["timers"][0]["duration"] == 95
    assert result.artifact["timers"][0]["remaining"] == 95


def test_bare_numeric_countdown_remains_seconds(monkeypatch):
    import tools.time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    monkeypatch.setattr(timers.time, "time", lambda: 1000)

    result = timers.start_countdown("60")

    assert result == "Timer started for 1 minute"


def test_cancel_timer_without_id_cancels_the_only_active_timer(monkeypatch):
    import tools.time_tools as timers

    timer = _Timer(60, lambda: None)
    timer.start_time = 1000
    timer.reminder = None
    monkeypatch.setattr(timers, "active_timers", {"timer_1": timer})

    result = timers.cancel_timer()

    assert result == "Timer timer_1 cancelled"
    assert timer.cancelled is True
    assert result.artifact == {"type": "timers", "timers": []}


def test_extend_timer_preserves_id_and_attaches_updated_artifact(monkeypatch):
    import tools.time_tools as timers

    old = _Timer(60, lambda: None)
    old.start_time = 1000
    old.reminder = "Tea"
    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {"timer_1": old})
    monkeypatch.setattr(timers.time, "time", lambda: 1010)

    result = timers.extend_timer("timer_1", "1 minute")

    assert result == "Timer timer_1 extended by 1 minute"
    assert old.cancelled is True
    assert result.artifact["timers"][0]["id"] == "timer_1"
    assert result.artifact["timers"][0]["remaining"] == 110


def test_time_outcome_preserves_timezone_and_dispatch(monkeypatch):
    from tools import time_tools
    from tools.tool_registry import tool_registry
    from utils.intents import StepKind, classify_step

    now = datetime.datetime(2026, 11, 5, 23, 12, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(time_tools._local_time, "now", lambda: now)
    result = tool_registry.execute_tool("get_current_time")
    step = classify_step(result)
    assert step.kind is StepKind.NORMAL
    assert step.operation_outcome.success
    assert step.operation_outcome.values["datetime"] == now
    assert "twenty twenty-six" in step.text


def test_time_tool_uses_response_language(monkeypatch):
    from core.satellite_context import current_response_language
    from tools import time_tools

    now = datetime.datetime(2026, 11, 5, 23, 12, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(time_tools._local_time, "now", lambda: now)
    token = current_response_language.set("fr")
    try:
        result = time_tools.get_current_time()
    finally:
        current_response_language.reset(token)
    assert "novembre" in result
    assert "cinq" in result
    assert "twenty" not in result
    assert result.operation_outcome.values["datetime"] == now


def test_timer_structured_success_and_artifacts(monkeypatch):
    from tools import time_tools as timers
    from utils.intents import classify_step

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    monkeypatch.setattr(timers.time, "time", lambda: 1000)
    started = timers.start_countdown("90 seconds")
    timer_id = started.operation_outcome.values["timer_id"]
    assert started.operation_outcome.values["duration_seconds"] == 90
    assert classify_step(started).artifact == started.artifact
    status = timers.get_timer_status(timer_id)
    assert status.operation_outcome.values["timers"] == [{"timer_id": timer_id, "remaining_seconds": 90}]
    cancelled = timers.cancel_timer(timer_id)
    assert cancelled.operation_outcome.success
    assert cancelled.operation_outcome.code == "timer_cancelled"
    assert cancelled.artifact == {"type": "timers", "timers": []}


@pytest.mark.parametrize("call, code", [
    (lambda t: t.start_countdown("not a duration"), "invalid_duration"),
    (lambda t: t.start_countdown("193 hours"), "duration_too_long"),
    (lambda t: t.cancel_timer(), "no_active_timers"),
    (lambda t: t.cancel_timer("timer_404"), "timer_not_found"),
    (lambda t: t.extend_timer("timer_404", "1 minute"), "timer_not_found"),
])
def test_timer_failures_are_not_successful_even_when_normal(monkeypatch, call, code):
    from tools import time_tools as timers
    from utils.intents import StepKind, classify_step

    monkeypatch.setattr(timers, "active_timers", {})
    result = call(timers)
    step = classify_step(result)
    assert step.kind is StepKind.NORMAL
    assert not step.operation_outcome.success
    assert step.operation_outcome.code == code


def test_empty_status_is_a_successful_query(monkeypatch):
    from tools import time_tools as timers

    monkeypatch.setattr(timers, "active_timers", {})
    result = timers.get_timer_status()
    assert result == "No active timers"
    assert result.operation_outcome.success
    assert result.operation_outcome.values == {"timers": []}


def test_timer_limit_and_ambiguous_cancel_are_failures(monkeypatch):
    from tools import time_tools as timers

    active = {f"timer_{i}": _Timer(60, lambda: None) for i in range(timers.MAX_ACTIVE_TIMERS)}
    monkeypatch.setattr(timers, "active_timers", active)
    limit = timers.start_countdown("1 minute")
    ambiguous = timers.cancel_timer()
    assert not limit.operation_outcome.success
    assert limit.operation_outcome.code == "timer_limit"
    assert not ambiguous.operation_outcome.success
    assert ambiguous.operation_outcome.code == "ambiguous_timer"
    assert len(active) == timers.MAX_ACTIVE_TIMERS


def test_timer_retains_language_across_extension_and_localizes_completion(monkeypatch):
    from core.satellite_context import current_timer_response_language
    from tools import time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    monkeypatch.setattr(timers.time, "time", lambda: 1000)
    announcements = []
    monkeypatch.setattr(timers, "_speak_proactive", lambda text, **kwargs: announcements.append((text, kwargs)))
    token = current_timer_response_language.set("fr")
    try:
        result = timers.start_countdown("60 seconds")
    finally:
        current_timer_response_language.reset(token)
    timer_id = result.operation_outcome.values["timer_id"]
    assert result.operation_outcome.values["response_language"] == "fr"
    timers.extend_timer(timer_id, "10 seconds")
    timer = timers.active_timers[timer_id]
    assert timer.response_language == "fr"
    timer.function(*timer.args)
    assert announcements == [("Votre minuteur est terminé.", {"alarm": True})]
    assert timer_id not in timers.active_timers
    assert current_timer_response_language.get() == "en"


def test_timer_reminder_text_is_preserved_regardless_of_language(monkeypatch):
    from core.satellite_context import current_timer_response_language
    from tools import time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    spoken = []
    monkeypatch.setattr(timers, "_speak_proactive", lambda text, **_: spoken.append(text))
    token = current_timer_response_language.set("de")
    try:
        result = timers.start_countdown("60", "Vérifie le thé!")
    finally:
        current_timer_response_language.reset(token)
    timer = timers.active_timers[result.operation_outcome.values["timer_id"]]
    timer.function(*timer.args)
    assert spoken == ["Vérifie le thé!"]


def test_timer_callback_binds_and_restores_spoken_language(monkeypatch):
    from core.satellite_context import current_response_language, current_timer_response_language
    from tools import time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    seen = []
    monkeypatch.setattr(timers, "_speak_proactive", lambda text, **_: seen.append(current_response_language.get()))
    token = current_timer_response_language.set("fr")
    try:
        result = timers.start_countdown("60", "Vérifie dans 20 minutes")
    finally:
        current_timer_response_language.reset(token)
    timer = timers.active_timers[result.operation_outcome.values["timer_id"]]
    timer.function(*timer.args)
    assert seen == ["fr"]
    assert current_response_language.get() == "en"


def test_timer_completion_falls_back_if_response_file_disappears(monkeypatch, tmp_path):
    from core.satellite_context import (
        current_timer_response_intents_dir,
        current_timer_response_language,
    )
    from tools import time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    spoken = []
    monkeypatch.setattr(timers, "_speak_proactive", lambda text, **_: spoken.append(text))
    language = current_timer_response_language.set("fr")
    directory = current_timer_response_intents_dir.set(str(tmp_path))
    try:
        result = timers.start_countdown("60")
    finally:
        current_timer_response_intents_dir.reset(directory)
        current_timer_response_language.reset(language)
    timer = timers.active_timers[result.operation_outcome.values["timer_id"]]
    timer.function(*timer.args)
    assert spoken == ["Your timer is up."]


def test_timer_languages_are_isolated_between_concurrent_turns(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from core.satellite_context import current_timer_response_language
    from tools import time_tools as timers

    monkeypatch.setattr(timers.threading, "Timer", _Timer)
    monkeypatch.setattr(timers, "active_timers", {})
    barrier = Barrier(2)

    def create(language):
        token = current_timer_response_language.set(language)
        try:
            barrier.wait(timeout=5)
            result = timers.start_countdown("60")
            return result.operation_outcome.values["timer_id"]
        finally:
            current_timer_response_language.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {language: pool.submit(create, language) for language in ("fr", "de")}
        ids = {language: future.result(timeout=5) for language, future in futures.items()}
    for language, timer_id in ids.items():
        assert timers.active_timers[timer_id].response_language == language
    assert current_timer_response_language.get() == "en"
