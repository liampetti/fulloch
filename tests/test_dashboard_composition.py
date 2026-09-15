"""Route families share a live context while each app owns its chat state."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from server.dashboard import create_app
from server.lifecycle import LOADING, NEEDS_SETUP, READY, AppContext, Lifecycle


def test_late_attachment_and_chat_state_are_isolated_per_app(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr("server.auth.load_sessions", lambda: {})
    monkeypatch.setattr("core.satellite_context.set_current_assistant", lambda _assistant: None)
    config = tmp_path / "config.yml"
    config.write_text("general: {}\n")
    contexts = [
        AppContext(lifecycle=Lifecycle(phase=NEEDS_SETUP), config_path=str(config))
        for _ in range(2)
    ]
    apps = [create_app(context=context) for context in contexts]
    clients = [TestClient(app) for app in apps]
    listeners = [[], []]

    for client in clients:
        assert client.post("/chat", json={"text": "hello"}).status_code == 503
        assert client.post("/llm/model", json={"model": "new"}).status_code == 503
        assert client.get("/history").json() == []

    for index, context in enumerate(contexts):
        context.set_assistant(SimpleNamespace(
            greeting_text=f"greeting {index}",
            register_turn_listener=listeners[index].append,
            handle_text_turn=lambda text: f"reply: {text}",
            set_llm_model=lambda model: {"ok": False, "model": model},
            _history=[],
        ))
        context.lifecycle.set(LOADING)
        assert clients[index].post("/chat", json={"text": "hello"}).status_code == 503
        assert clients[index].post("/llm/model", json={"model": "new"}).status_code == 503
        context.lifecycle.set(READY)
        assert clients[index].post("/chat", json={"text": "hello"}).json() == {"answer": "reply: hello"}
        assert clients[index].post("/llm/model", json={"model": "new"}).json() == {"ok": False, "model": "new"}
        assert len(listeners[index]) == 1

    async def exercise_streams():
        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        streams = [
            await next(route.endpoint for route in app.routes if route.path == "/stream")(ConnectedRequest())
            for app in apps
        ]
        try:
            for index in range(2):
                event = {"role": "user", "content": f"turn {index}", "ts": index}
                listeners[index][0](event)
                frame = await asyncio.wait_for(anext(streams[index].body_iterator), timeout=1)
                assert json.loads(frame.removeprefix("data: ")) == event
        finally:
            for stream in streams:
                await stream.body_iterator.aclose()

    asyncio.run(exercise_streams())
    for index, client in enumerate(clients):
        assert [event["content"] for event in client.get("/history").json()] == [f"greeting {index}", f"turn {index}"]
    assert clients[0].post("/reset").json() == {"ok": True}
    assert [event["role"] for event in clients[0].get("/history").json()] == ["reset"]
    assert len(clients[1].get("/history").json()) == 2


def test_ha_credentials_and_artwork_share_live_client_state(tmp_path, monkeypatch):
    from server import credentials_store, routes_chat
    from tools import ha_client

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("HA_TOKEN", "old-token")
    monkeypatch.setattr(ha_client, "HA_TOKEN", "old-token")
    monkeypatch.setattr(ha_client, "HA_URL", "http://ha.test")
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    config = tmp_path / "config.yml"
    config.write_text("general: {}\n")
    context = AppContext(lifecycle=Lifecycle(phase=READY), config_path=str(config))
    context.set_assistant(SimpleNamespace(greeting_text="", register_turn_listener=Mock()))
    response = Mock(content=b"image", headers={"content-type": "image/png"})
    fetch = Mock(return_value=response)
    monkeypatch.setattr(routes_chat.requests, "get", fetch)

    with TestClient(create_app(context=context)) as client:
        result = client.post("/setup/credential", json={"key": "ha_token", "value": "new-token"})
        assert result.status_code == 200
        assert credentials_store.get_credential("ha_token") == "new-token"
        assert ha_client.HA_TOKEN == "new-token"

        artwork = client.get("/media-artwork/media_player.kitchen")
        assert artwork.status_code == 200
        assert artwork.content == b"image"
        assert artwork.headers["content-type"] == "image/png"
        fetch.assert_called_once_with(
            "http://ha.test/api/media_player_proxy/media_player.kitchen",
            headers=ha_client._get_headers(),
            timeout=ha_client.TIMEOUT,
        )
        assert fetch.call_args.kwargs["headers"]["Authorization"] == "Bearer new-token"

        monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"media_player.kitchen"}))
        assert client.get("/media-artwork/media_player.kitchen").status_code == 404
        assert fetch.call_count == 1
