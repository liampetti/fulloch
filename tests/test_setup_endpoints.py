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


def test_models_endpoint_rejects_unknown_tier(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.post("/setup/models", json={"tier": "mega"}).status_code == 422


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
    r = client.post("/setup/plan", json={"models": {
        "asr": {"backend": "moonshine"}, "tts": {"backend": "kokoro-onnx"},
        "llm": {"backend": "llama", "model": str(gguf)},
    }})
    llm = next(a for a in r.json()["assets"] if a["key"] == "llm:llama")
    assert llm["present"] is True  # found at the custom path -> no download


def test_install_runs_download_and_signals_proceed(tmp_path):
    # Fake manager that succeeds instantly (incl. dir-snapshot assets so the
    # CPU stack's ONNX models never reach the network).
    fake = DownloadManager(
        snapshot_fn=lambda *a: None, file_fn=lambda *a: None, url_fn=lambda *a: None,
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


def test_reset_endpoint_arms_wizard_and_backs_up(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post("/setup/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["restart_required"]
    # Marker dropped next to config so the next restart re-enters setup.
    assert (Path(path).parent / ".setup_pending").is_file()
    # Current config preserved as a timestamped backup.
    assert body["backup"] and (Path(path).parent / body["backup"]).is_file()


def test_test_llm_endpoint(tmp_path, monkeypatch):
    import core.llm_openai as llmo
    monkeypatch.setattr(llmo, "test_connection",
                        lambda **kw: {"ok": True, "error": None})
    client, _, _ = _client(tmp_path)
    r = client.post("/setup/test-llm",
                    json={"base_url": "http://x/v1", "model": "m", "api_key": ""})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_models_endpoint_writes_openai_block(tmp_path):
    client, _, path = _client(tmp_path)
    r = client.post("/setup/models", json={"models": {
        "asr": {"backend": "qwen"}, "tts": {"backend": "qwen"},
        "llm": {"backend": "openai", "model": "gpt-x", "base_url": "http://x/v1"},
    }})
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
    (tmp_path / "af_heart.wav").write_bytes(b"x")   # no per-voice .txt
    (tmp_path / "atticus.wav").write_bytes(b"x")
    (tmp_path / "atticus.txt").write_text("hi")
    # No default.txt → only the .wav+.txt pair is usable.
    assert list_voices(str(tmp_path)) == ["atticus"]
    # default.txt present → every .wav lists; 'default' itself is not a voice.
    (tmp_path / "default.txt").write_text("shared transcript")
    assert list_voices(str(tmp_path)) == ["af_heart", "atticus"]
