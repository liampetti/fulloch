"""Timer artifact tests."""


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
