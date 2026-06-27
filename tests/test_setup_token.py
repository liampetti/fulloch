"""Post-setup access token: generation, live gating, .env persistence (Step 5)."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import env_store  # noqa: E402
from server import lifecycle as lc  # noqa: E402
from server.dashboard import create_app  # noqa: E402
from server.lifecycle import AppContext, Lifecycle  # noqa: E402


def _client(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("general:\n  wakeword: hey atticus\n")
    ctx = AppContext(lifecycle=Lifecycle(phase=lc.NEEDS_SETUP), config_path=str(cfg))
    return TestClient(create_app(context=ctx)), ctx


def test_token_status_and_generation_gates_live(tmp_path, monkeypatch):
    written = {}
    monkeypatch.setattr(env_store, "set_env_var",
                        lambda k, v, **kw: written.update({k: v}))
    client, ctx = _client(tmp_path)

    # No token yet — open, status reports disabled.
    assert client.get("/setup/token").json() == {"enabled": False}
    assert client.get("/status").status_code == 200

    # Generate — returns once, applied live, persisted to .env.
    r = client.post("/setup/token")
    assert r.status_code == 200
    token = r.json()["token"]
    assert token and ctx.dashboard_token == token
    assert written.get("FULLOCH_DASHBOARD_TOKEN") == token

    # Now the console is gated: no creds -> 401, correct token -> 200.
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # The unauthenticated shell + setup page stay reachable so the SPA can load.
    assert client.get("/").status_code == 200
    assert client.get("/setup").status_code == 200


def test_schema_exposes_backend_options(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/setup/schema").json()
    assert set(body["backends"]) == {"asr", "tts", "llm"}
    llm = {b["backend"]: b for b in body["backends"]["llm"]}
    assert llm["none"]["implemented"] is True
    assert llm["openai"]["implemented"] is True  # remote backend (Step 6)
    assert llm["llama"]["cpu_ok"] is False


# --- env_store --------------------------------------------------------------

def test_env_store_replaces_existing_key(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OTHER=keep\nFULLOCH_DASHBOARD_TOKEN=old\n# a comment\n")
    env_store.set_env_var("FULLOCH_DASHBOARD_TOKEN", "new", path=str(p))
    text = p.read_text()
    assert "FULLOCH_DASHBOARD_TOKEN=new" in text
    assert "FULLOCH_DASHBOARD_TOKEN=old" not in text
    assert "OTHER=keep" in text and "# a comment" in text


def test_env_store_appends_when_absent(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OTHER=keep\n")
    env_store.set_env_var("NEWKEY", "v", path=str(p))
    assert "NEWKEY=v" in p.read_text()
    assert "OTHER=keep" in p.read_text()


def test_env_store_seeds_from_example(tmp_path):
    (tmp_path / ".env.example").write_text("# template\nSEARXNG_SECRET=\n")
    p = tmp_path / ".env"
    env_store.set_env_var("FULLOCH_DASHBOARD_TOKEN", "abc", path=str(p))
    text = p.read_text()
    assert "# template" in text
    assert "FULLOCH_DASHBOARD_TOKEN=abc" in text
