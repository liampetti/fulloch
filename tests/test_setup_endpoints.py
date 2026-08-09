"""Setup / settings console endpoints (v2.2 Step 4).

Drives the config/install/voice/preflight routes against a setup-mode app
(no assistant) with a fake download manager — no network, no models.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import lifecycle as lc  # noqa: E402
from server.dashboard import create_app  # noqa: E402
from server.downloader import DownloadManager  # noqa: E402
from server.lifecycle import AppContext, Lifecycle  # noqa: E402


def _client(tmp_path, downloader=None):
    cfg = tmp_path / "config.yml"
    cfg.write_text("general:\n  wakeword: hey atticus\n")
    life = Lifecycle(phase=lc.NEEDS_SETUP, detail="first run")
    ctx = AppContext(lifecycle=life, config_path=str(cfg), downloader=downloader)
    return TestClient(create_app(context=ctx)), ctx, str(cfg)


def test_schema_endpoint_lists_fields(tmp_path):
    client, _, _ = _client(tmp_path)
    body = client.get("/setup/schema").json()
    paths = {f["path"] for f in body["fields"]}
    assert "general.wakeword" in paths
    assert body["wakeword_presets"] and body["tier_presets"]


def test_config_put_validates_and_writes(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.put("/config", json={"updates": {"general.wakeword": "computer"}})
    assert r.status_code == 200
    assert r.json()["restart_required"] is True
    assert "computer" in Path(path).read_text()


def test_config_put_rejects_unknown_key(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.put("/config", json={"updates": {"general.bogus": "x"}})
    assert r.status_code == 422


def test_models_endpoint_writes_tier(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post("/setup/models", json={"tier": "cpu_local"})
    assert r.status_code == 200
    assert "qwen-onnx" in Path(path).read_text()
    assert "pocket-tts-onnx" in Path(path).read_text()


def test_models_endpoint_rejects_unknown_tier(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.post("/setup/models", json={"tier": "mega"}).status_code == 422


def test_models_endpoint_accepts_regex_only_llm(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post(
        "/setup/models",
        json={
            "models": {
                "asr": {"backend": "qwen-onnx"},
                "tts": {"backend": "pocket-tts-onnx"},
                "llm": {"backend": "none"},
            }
        },
    )
    assert r.status_code == 200
    assert "backend: none" in Path(path).read_text()


def test_models_endpoint_persists_experimental_llm_options(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post(
        "/setup/models",
        json={
            "models": {
                "asr": {"backend": "qwen-onnx"},
                "tts": {"backend": "pocket-tts-onnx"},
                "llm": {"backend": "local", "local_model": "qwen", "mtp": True, "flash_attn": True},
            }
        },
    )
    assert r.status_code == 200
    text = Path(path).read_text()
    assert "mtp: true" in text
    assert "flash_attn: true" in text


def test_cancel_startup_arms_setup_marker(tmp_path, monkeypatch):
    client, ctx, path = _client(tmp_path)
    ctx.lifecycle.set(lc.DOWNLOADING, "downloading")
    restart_delay = []

    def schedule_restart(*, delay):
        restart_delay.append(delay)

    monkeypatch.setattr("server.dashboard._schedule_restart", schedule_restart)

    r = client.post("/setup/cancel-startup")

    assert r.status_code == 200
    assert r.json()["restarting"] is True
    assert (Path(path).parent / ".setup_pending").is_file()
    assert ctx.lifecycle.phase == lc.NEEDS_SETUP
    assert restart_delay == [0.1]


def test_plan_endpoint_reports_presence_and_domain(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/setup/plan", json={"tier": "cpu_local"})
    assert r.status_code == 200
    assets = r.json()["assets"]
    asr = next(a for a in assets if a["key"].startswith("asr:"))
    # Shape: each model asset carries its domain + a boolean presence flag
    # (actual presence depends on what's on disk, so don't assert the value).
    assert asr["domain"] == "asr" and isinstance(asr["present"], bool)
    assert "dest" in asr
    assert any(a["key"] == "bge" for a in assets)


def test_plan_endpoint_honours_custom_present_path(tmp_path):
    client, _, _ = _client(tmp_path)
    gguf = tmp_path / "have" / "model.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("x")
    r = client.post(
        "/setup/plan",
        json={
            "models": {
                "asr": {"backend": "moonshine"},
                "tts": {"backend": "kokoro-onnx"},
                "llm": {"backend": "llama", "model": str(gguf)},
            }
        },
    )
    llm = next(a for a in r.json()["assets"] if a["key"] == "llm:llama")
    assert llm["present"] is True  # found at the custom path -> no download


def test_install_runs_download_and_signals_proceed(tmp_path):
    # Fake manager that succeeds instantly (incl. dir-snapshot assets so the
    # CPU stack's ONNX models never reach the network).
    fake = DownloadManager(
        snapshot_fn=lambda *a: None,
        file_fn=lambda *a: None,
        url_fn=lambda *a: None,
        dir_snapshot_fn=lambda *a: None,
    )
    client, ctx, path = _client(tmp_path, downloader=fake)
    # A leftover reset marker must be cleared once setup completes.
    marker = Path(path).parent / ".setup_pending"
    marker.write_text("x")
    # Pick a CPU stack (no GGUF/grammar gymnastics), then install.
    client.post("/setup/models", json={"tier": "cpu_local"})
    r = client.post("/setup/install")
    assert r.status_code == 200 and r.json()["started"] is True

    # On success the lifecycle is released so main() proceeds to Phase B. Wait
    # on `proceed` itself (the completion signal) rather than `active`, which
    # flips a beat earlier in _run.
    assert ctx.lifecycle.proceed.wait(timeout=5), "install did not complete"
    prog = client.get("/setup/progress").json()
    assert prog["state"] == "done"
    assert not marker.exists()  # reset marker cleared on completion
    assert (Path(path).parent / ".setup_complete").is_file()


def test_reset_endpoint_arms_wizard_and_backs_up(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post("/setup/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["restart_required"]
    # Marker dropped next to config so the next restart re-enters setup.
    assert (Path(path).parent / ".setup_pending").is_file()
    # Current config preserved as a timestamped backup directory under data/backups/.
    assert body["backup"] and (Path(path).parent / "backups" / body["backup"]).is_dir()


def test_reset_backup_includes_credentials(tmp_path):
    """A reset must snapshot credentials.json too, not just config.yml."""
    client, _, path = _client(tmp_path)
    data_dir = Path(path).parent
    (data_dir / "credentials.json").write_text('{"ha_token": "secret"}', encoding="utf-8")
    r = client.post("/setup/reset")
    assert r.status_code == 200
    backup_dir = data_dir / "backups" / r.json()["backup"]
    assert (backup_dir / "config.yml").is_file()
    assert (backup_dir / "credentials.json").is_file()
    assert (backup_dir / "meta.json").is_file()
    assert "credentials.json" in (backup_dir / "meta.json").read_text()


def test_list_backups_returns_empty_then_entries(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.get("/setup/backups")
    assert r.status_code == 200
    assert r.json() == {"backups": []}
    client.post("/setup/reset")
    r = client.get("/setup/backups")
    assert r.status_code == 200
    backups = r.json()["backups"]
    assert len(backups) == 1
    assert "config.yml" in backups[0]["files"]


def test_restore_backup_replaces_files(tmp_path):
    client, _, path = _client(tmp_path)
    data_dir = Path(path).parent
    (data_dir / "credentials.json").write_text('{"ha_token": "first"}', encoding="utf-8")
    client.post("/setup/reset")
    # Mutate after the backup
    (data_dir / "config.yml").write_text("general:\n  wakeword: changed\n", encoding="utf-8")
    (data_dir / "credentials.json").write_text('{"ha_token": "mutated"}', encoding="utf-8")
    # Find the backup
    backups = client.get("/setup/backups").json()["backups"]
    name = backups[0]["name"]
    r = client.post("/setup/backups/restore", json={"name": name})
    assert r.status_code == 200
    assert "config.yml" in r.json()["restored"]
    assert (data_dir / "credentials.json").read_text() == '{"ha_token": "first"}'
    assert "wakeword: hey atticus" in (data_dir / "config.yml").read_text()


def test_restore_backup_rejects_unsafe_name(tmp_path):
    client, _, _ = _client(tmp_path)
    for bad in ("", "../etc", "2026-07-02", "name with space"):
        r = client.post("/setup/backups/restore", json={"name": bad})
        assert r.status_code == 400, bad
    r = client.post("/setup/backups/restore", json={"name": "2099-01-01T000000"})
    assert r.status_code == 404


def test_regen_cert_endpoint_enables_https_from_scratch(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post("/setup/regen-cert")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["restart_required"]
    text = Path(path).read_text()
    assert "dashboard_ssl_certfile" in text and "dashboard_ssl_keyfile" in text
    certs_dir = Path(path).parent / "certs"
    assert (certs_dir / "dashboard.crt").is_file()
    assert (certs_dir / "dashboard.key").is_file()


def test_regen_cert_endpoint_overwrites_existing_pair(tmp_path):
    from core.tls_certs import ensure_self_signed_cert

    certs_dir = tmp_path / "certs"
    cert_path, key_path = ensure_self_signed_cert(str(certs_dir))
    original = Path(cert_path).read_bytes()

    cfg = tmp_path / "config.yml"
    cfg.write_text(
        f"general:\n  wakeword: hey atticus\n"
        f"  dashboard_ssl_certfile: {cert_path}\n"
        f"  dashboard_ssl_keyfile: {key_path}\n"
    )
    life = Lifecycle(phase=lc.NEEDS_SETUP, detail="first run")
    ctx = AppContext(lifecycle=life, config_path=str(cfg), downloader=None)
    client = TestClient(create_app(context=ctx))

    r = client.post("/setup/regen-cert")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["restart_required"]
    assert Path(cert_path).read_bytes() != original


def test_test_llm_endpoint(tmp_path, monkeypatch):
    import core.llm_openai as llmo

    monkeypatch.setattr(llmo, "test_connection", lambda **kw: {"ok": True, "error": None})
    client, _, _ = _client(tmp_path)
    r = client.post(
        "/setup/test-llm", json={"base_url": "http://x/v1", "model": "m", "api_key": ""}
    )
    assert r.status_code == 200 and r.json()["ok"] is True


def test_models_endpoint_writes_openai_block(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post(
        "/setup/models",
        json={
            "models": {
                "asr": {"backend": "qwen"},
                "tts": {"backend": "qwen"},
                "llm": {"backend": "openai", "model": "gpt-x", "base_url": "http://x/v1"},
            }
        },
    )
    assert r.status_code == 200
    text = Path(path).read_text()
    assert "openai" in text and "gpt-x" in text and "http://x/v1" in text


def test_preflight_endpoint(tmp_path):
    client, _, _ = _client(tmp_path)
    body = client.get("/setup/preflight").json()
    assert "disk_free_gb" in body and "gpu" in body and "ram" in body and "tier_fit" in body
    assert "ram_gb" in body["tier_fit"][0]


def test_voices_endpoint_lists(tmp_path, monkeypatch):
    import core.voice_clone as vc

    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "atticus.wav").write_text("x")
    (voices / "atticus.txt").write_text("hi")
    monkeypatch.setattr(vc, "VOICES_DIR", str(voices))
    client, _, _ = _client(tmp_path)
    assert client.get("/setup/voices").json()["voices"] == ["atticus"]


def test_voice_save_without_generate_409(tmp_path):
    import core.voice_clone as vc

    vc._last = None
    client, _, _ = _client(tmp_path)
    assert client.post("/setup/voice/save", json={"name": "x"}).status_code == 409


def test_restart_endpoint_reexecs(tmp_path, monkeypatch):
    import os
    import sys
    import time

    # CRITICAL: stub the actual restart so the test process isn't replaced/killed.
    calls = []
    monkeypatch.setattr(os, "execv", lambda *a: calls.append(a))
    monkeypatch.setattr(os, "_exit", lambda *a: calls.append(("_exit",)))
    client, _, _ = _client(tmp_path)
    r = client.post("/restart")
    assert r.status_code == 200 and r.json()["restarting"] is True
    time.sleep(1.0)  # let the scheduled restart thread fire (0.5s delay) under the stub
    assert calls and calls[0][0] == sys.executable  # re-exec'd the python interpreter


def test_voice_sample_serves_wav(tmp_path, monkeypatch):
    import server.dashboard as dash

    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "af_heart.wav").write_bytes(b"RIFFfakewav")
    monkeypatch.setattr(dash, "_VOICES_DIR", voices)
    client, _, _ = _client(tmp_path)
    r = client.get("/voice/sample?name=af_heart")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.get("/voice/sample?name=missing").status_code == 404
    # Path-traversal / empty names are rejected before any file access.
    assert client.get("/voice/sample?name=../config").status_code == 400
    assert client.get("/voice/sample?name=").status_code == 400


def test_list_voices_default_txt_fallback(tmp_path):
    from core.voice_clone import list_voices

    (tmp_path / "af_heart.wav").write_bytes(b"x")  # no per-voice .txt
    (tmp_path / "atticus.wav").write_bytes(b"x")
    (tmp_path / "atticus.txt").write_text("hi")
    # No default.txt → only the .wav+.txt pair is usable.
    assert list_voices(str(tmp_path)) == ["atticus"]
    # default.txt present → every .wav lists; 'default' itself is not a voice.
    (tmp_path / "default.txt").write_text("shared transcript")
    assert list_voices(str(tmp_path)) == ["af_heart", "atticus"]
