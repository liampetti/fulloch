"""Credentials store — atomic JSON read/write."""

import json
import os
from pathlib import Path

from server import credentials_store


def test_load_returns_empty_dict_when_file_absent(tmp_path):
    assert credentials_store.load(str(tmp_path / "credentials.json")) == {}


def test_set_credential_creates_file(tmp_path):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("ha_token", "mytoken", path=path)
    assert json.loads(Path(path).read_text())["ha_token"] == "mytoken"


def test_set_credential_updates_existing_key(tmp_path):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("ha_token", "old", path=path)
    credentials_store.set_credential("ha_token", "new", path=path)
    assert json.loads(Path(path).read_text())["ha_token"] == "new"


def test_set_credential_preserves_other_keys(tmp_path):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("ha_token", "tok", path=path)
    credentials_store.set_credential("llm_api_key", "key", path=path)
    data = json.loads(Path(path).read_text())
    assert data["ha_token"] == "tok"
    assert data["llm_api_key"] == "key"


def test_inject_env_sets_env_var(tmp_path, monkeypatch):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("ha_token", "ha-test-token", path=path)
    monkeypatch.delenv("HA_TOKEN", raising=False)
    credentials_store.inject_env(path=path)
    assert os.environ.get("HA_TOKEN") == "ha-test-token"


def test_inject_env_does_not_override_existing(tmp_path, monkeypatch):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("ha_token", "from-file", path=path)
    monkeypatch.setenv("HA_TOKEN", "from-env")
    credentials_store.inject_env(path=path)
    assert os.environ.get("HA_TOKEN") == "from-env"


def test_hf_token_is_injected_into_environment(tmp_path, monkeypatch):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("hf_token", "hf-test-token", path=path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    credentials_store.inject_env(path=path)
    assert os.environ.get("HF_TOKEN") == "hf-test-token"


def test_get_credential_returns_empty_when_absent(tmp_path):
    assert credentials_store.get_credential("ha_token", str(tmp_path / "credentials.json")) == ""


def test_get_credential_returns_stored_value(tmp_path):
    path = str(tmp_path / "credentials.json")
    credentials_store.set_credential("obsidian_token", "obs-tok", path=path)
    assert credentials_store.get_credential("obsidian_token", path=path) == "obs-tok"


def test_schema_exposes_backend_options(tmp_path):
    from fastapi.testclient import TestClient

    from server import lifecycle as lc
    from server.dashboard import create_app
    from server.lifecycle import AppContext, Lifecycle

    cfg = tmp_path / "config.yml"
    cfg.write_text("general:\n  wakeword: hey atticus\n")
    ctx = AppContext(lifecycle=Lifecycle(phase=lc.NEEDS_SETUP), config_path=str(cfg))
    client = TestClient(create_app(context=ctx))
    body = client.get("/setup/schema").json()
    assert set(body["backends"]) == {"asr", "tts", "llm"}
    llm = {b["backend"]: b for b in body["backends"]["llm"]}
    assert llm["none"]["implemented"] is True
    assert llm["openai"]["implemented"] is True
    assert llm["llama"]["cpu_ok"] is False
    assert body["credentials"]["hf_token"] is False
