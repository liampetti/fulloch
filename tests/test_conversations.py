"""Durable conversation and dashboard management regression tests."""

import threading
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from server.conversations import ConversationStore
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle


def test_store_persists_snapshots_search_and_restores_context(tmp_path, monkeypatch):
    path = tmp_path / "conversations.sqlite3"
    store = ConversationStore(path)
    monkeypatch.setattr("server.conversations.embed", lambda texts, query=False: np.array(
        [[1.0, 0.0] if "garden" in text.lower() else [0.0, 1.0] for text in texts], dtype=np.float32
    ))
    store.append({"role": "user", "content": "Garden project", "source": "text", "ts": 1})
    store.append({"role": "assistant", "content": "Plant herbs", "source": "text", "ts": 2},
                 [{"role": "user", "content": "Garden project"}])
    first = store.save([{"role": "user", "content": "Garden project"}])
    store.clear()
    store.append({"role": "user", "content": "Weather forecast", "source": "text", "ts": 3})
    second = store.save([])

    assert first["id"] != second["id"]
    assert len(store.list(limit=1)["conversations"]) == 1
    assert store.list(limit=1)["next_offset"] == 1
    assert [item["id"] for item in store.list(query="garden")["conversations"]] == [first["id"]]
    reopened = ConversationStore(path)
    events, context, saved = reopened.load(first["id"])
    assert saved is None
    assert events[0]["content"] == "Garden project"
    assert context == [{"role": "user", "content": "Garden project"}]
    assert reopened.active() == (events, context)
    assert reopened.delete(first["id"])
    assert not reopened.delete(first["id"])
    assert reopened.load(first["id"]) is None


def _assistant():
    listeners = []
    assistant = SimpleNamespace(
        greeting_text="", register_turn_listener=listeners.append, _history=[],
        _turn_lock=threading.Lock(), _turn_arbiter=SimpleNamespace(owner=None),
        _laya_context=None,
    )
    return assistant, listeners


def test_dashboard_restart_save_load_clear_and_busy_protection(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("general: {}\n")
    first, listeners = _assistant()
    context = AppContext(lifecycle=Lifecycle(phase=READY), assistant=first, config_path=str(config))
    client = TestClient(create_app(context=context))
    listeners[0]({"role": "user", "source": "text", "satellite_id": "dashboard-text", "content": "Hello", "ts": 1})
    first._history[:] = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    listeners[0]({"role": "assistant", "source": "text", "satellite_id": "dashboard-text", "content": "Hi", "ts": 2})
    saved = client.post("/conversations").json()
    assert saved["id"]
    assert client.get("/conversations").json()["conversations"][0]["id"] == saved["id"]

    second, second_listeners = _assistant()
    restart = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=second, config_path=str(config),
    )))
    assert [event["content"] for event in restart.get("/history").json()] == ["Hello", "Hi"]
    assert second._history == first._history
    second._turn_lock.acquire()
    try:
        assert restart.post("/reset").status_code == 409
        assert restart.post(f"/conversations/{saved['id']}/load", json={"current": "clear"}).status_code == 409
    finally:
        second._turn_lock.release()
    assert restart.post("/reset").status_code == 200
    assert second._history == []
    assert restart.post(f"/conversations/{saved['id']}/load", json={"current": "clear"}).status_code == 200
    assert [event["content"] for event in restart.get("/history").json()] == ["Hello", "Hi"]
    assert second._history == first._history
    assert restart.delete(f"/conversations/{saved['id']}").status_code == 200
    assert restart.post(f"/conversations/{saved['id']}/load", json={"current": "clear"}).status_code == 404
    assert second_listeners


def test_transcript_retains_more_than_old_limit_while_context_is_bounded(tmp_path):
    assistant, listeners = _assistant()
    context = AppContext(lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml"))
    client = TestClient(create_app(context=context))
    for index in range(205):
        listeners[0]({"role": "user", "source": "text", "content": f"Message {index}", "ts": index})
    assistant._history[:] = [{"role": "user", "content": f"Message {index}"} for index in range(75)]
    listeners[0]({"role": "assistant", "source": "text", "content": "Finished", "ts": 205})
    assert len(client.get("/history").json()) == 206
    snapshot = client.post("/conversations").json()

    other, _ = _assistant()
    restarted = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=other, config_path=context.config_path,
    )))
    assert len(restarted.get("/history").json()) == 206
    assert len(other._history) == 50
    assert other._history[0]["content"] == "Message 25"
    assert restarted.post(f"/conversations/{snapshot['id']}/load", json={"current": "clear"}).status_code == 200
    assert len(restarted.get("/history").json()) == 206
    assert len(other._history) == 50


def test_load_with_save_preserves_current_before_replacing(tmp_path):
    assistant, listeners = _assistant()
    client = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml"),
    )))
    listeners[0]({"role": "user", "source": "text", "content": "First", "ts": 1})
    first = client.post("/conversations").json()["id"]
    assert client.post("/reset").status_code == 200
    listeners[0]({"role": "user", "source": "text", "content": "Second", "ts": 2})
    assert client.post(f"/conversations/{first}/load", json={"current": "save"}).status_code == 200
    snapshots = client.get("/conversations").json()["conversations"]
    assert {item["title"] for item in snapshots} == {"First", "Second"}
    assert client.get("/history").json()[0]["content"] == "First"


def test_failed_transcript_write_is_visible_and_blocks_incomplete_save(tmp_path, monkeypatch):
    assistant, listeners = _assistant()
    client = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml"),
    )))
    monkeypatch.setattr(ConversationStore, "append", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    listeners[0]({"role": "user", "source": "text", "content": "Not persisted", "ts": 1})
    assert client.get("/history").json()[0]["content"] == "Not persisted"
    assert client.get("/conversations/status").json() == {"persisted": False}
    assert client.post("/conversations").status_code == 503
    assert client.post("/reset").status_code == 200
    assert client.get("/conversations/status").json() == {"persisted": True}


def test_dashboard_has_no_idle_timeout_but_voice_still_does(monkeypatch):
    from core import assistant as assistant_module

    histories = {"dashboard-text": [{"role": "user", "content": "hello"}],
                 "kitchen": [{"role": "user", "content": "hello"}]}
    sessions = {name: SimpleNamespace(last_turn_end=1.0, skip_followup_self_echo=True) for name in histories}
    assistant = SimpleNamespace(_history_for_id=histories.__getitem__, satellites=sessions, _laya_context={})
    monkeypatch.setattr(assistant_module.time, "monotonic", lambda: 5000.0)
    assistant_module.Assistant._maybe_reset_session(assistant, "dashboard-text")
    assistant_module.Assistant._maybe_reset_session(assistant, "kitchen")
    assert histories["dashboard-text"]
    assert not histories["kitchen"]
    assert sessions["kitchen"].skip_followup_self_echo is False


def test_snapshot_restores_satellite_context_with_voice_timeout(tmp_path, monkeypatch):
    from core import assistant as assistant_module

    assistant, listeners = _assistant()
    assistant._satellite_histories = {"kitchen": [{"role": "user", "content": "Turn on lamp"}]}
    assistant.satellites = {"kitchen": SimpleNamespace(last_turn_end=0.0, skip_followup_self_echo=False)}
    assistant._history_for_id = lambda satellite_id: assistant._satellite_histories[satellite_id]
    client = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml"),
    )))
    listeners[0]({"role": "user", "source": "voice", "satellite_id": "kitchen", "content": "Turn on lamp", "ts": 1})
    listeners[0]({"role": "assistant", "source": "voice", "satellite_id": "kitchen", "content": "Lamp on", "ts": 2})
    identifier = client.post("/conversations").json()["id"]
    assistant._satellite_histories["kitchen"] = []
    monkeypatch.setattr(assistant_module.time, "monotonic", lambda: 1000.0)
    assert client.post(f"/conversations/{identifier}/load", json={"current": "clear"}).status_code == 200
    assert assistant._satellite_histories["kitchen"][0]["content"] == "Turn on lamp"
    monkeypatch.setattr(assistant_module.time, "monotonic", lambda: 2801.0)
    assistant_module.Assistant._maybe_reset_session(assistant, "kitchen")
    assert assistant._satellite_histories["kitchen"] == []


def test_clearing_chat_discards_visible_voice_context_only(tmp_path):
    assistant, listeners = _assistant()
    assistant._satellite_histories = {
        "kitchen": [{"role": "user", "content": "Current voice conversation"}],
        "office": [{"role": "user", "content": "Separate conversation"}],
    }
    client = TestClient(create_app(context=AppContext(
        lifecycle=Lifecycle(phase=READY), assistant=assistant, config_path=str(tmp_path / "config.yml"),
    )))
    listeners[0]({"role": "user", "source": "voice", "satellite_id": "kitchen", "content": "Hello", "ts": 1})
    assert client.post("/reset").status_code == 200
    assert "kitchen" not in assistant._satellite_histories
    assert assistant._satellite_histories["office"][0]["content"] == "Separate conversation"
