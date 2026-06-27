"""First-run setup detection, lifecycle, and dashboard setup mode (v2.2 Step 3).

An empty `data/models` (or missing config) must boot into setup mode: the
dashboard serves a setup page, reports NEEDS_SETUP on `/status`, and every
assistant-backed route returns 503 — without crashing on absent models.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.setup import detect_setup_state  # noqa: E402
from server import lifecycle as lc  # noqa: E402
from server.dashboard import create_app  # noqa: E402
from server.lifecycle import Lifecycle  # noqa: E402

# --- detect_setup_state -----------------------------------------------------


def _mk_hub(models_dir: Path, repo_id: str) -> None:
    (models_dir / "hub" / f"models--{repo_id.replace('/', '--')}").mkdir(parents=True)


def test_no_config_is_first_run():
    d = detect_setup_state(None)
    assert d.needs_setup and not d.config_present
    assert d.config_error is None


def test_config_without_general_is_first_run():
    d = detect_setup_state({"search": {}})
    assert d.needs_setup and not d.config_present


def test_missing_required_key_is_config_error():
    d = detect_setup_state({"general": {"voice_clone": "atticus"}})
    assert d.needs_setup and d.config_present
    assert d.config_error and "general.wakeword" in d.config_error


def test_existing_install_with_assets_present(tmp_path):
    models = tmp_path / "models"
    _mk_hub(models, "Qwen/Qwen3-ASR-1.7B")
    _mk_hub(models, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    config = {"general": {"wakeword": "hey atticus"}, "models": {"llm": {"backend": "none"}}}
    d = detect_setup_state(config, models_dir=str(models))
    assert not d.needs_setup
    assert d.missing_assets == []


def test_cpu_variant_forces_setup_for_gpu_only_backends(tmp_path, monkeypatch):
    """On the CPU image, a config resolving to gpu_only backends (e.g. the qwen
    defaults, even with the GPU assets on disk from a shared ./data) must route
    to the wizard, not try to load qwen_asr and crash."""
    import core.setup as setup

    models = tmp_path / "models"
    _mk_hub(models, "Qwen/Qwen3-ASR-1.7B")
    _mk_hub(models, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    # No models: block -> resolves to the qwen GPU defaults.
    config = {"general": {"wakeword": "hey atticus"}}

    monkeypatch.setattr(setup, "variant", lambda: "cpu")
    d = detect_setup_state(config, models_dir=str(models))
    assert d.needs_setup and d.config_present
    assert "GPU-only" in d.reason

    # Same config on the GPU image runs (assets present, no grammar needed for
    # the default llama... it does need the grammar, so assert the gpu path is
    # at least not blocked by the variant guard).
    monkeypatch.setattr(setup, "variant", lambda: "gpu")
    d_gpu = detect_setup_state(config, models_dir=str(models))
    assert "GPU-only" not in (d_gpu.reason or "")


def test_cpu_variant_runs_with_cpu_backends(tmp_path, monkeypatch):
    """The CPU image runs normally when the config picks CPU-capable backends."""
    import core.setup as setup

    # Explicit paths under tmp_path so the asset check is hermetic (path-style
    # models resolve to the literal path, not models_dir).
    asr_dir = tmp_path / "asr-onnx"
    asr_dir.mkdir()
    (asr_dir / "model.onnx").write_text("x")
    tts_dir = tmp_path / "tts-onnx"
    tts_dir.mkdir()
    (tts_dir / "model.onnx").write_text("x")
    config = {
        "general": {"wakeword": "hey atticus"},
        "models": {
            "asr": {"backend": "qwen-onnx", "model": str(asr_dir)},
            "tts": {"backend": "kokoro-onnx", "model": str(tts_dir)},
            "llm": {"backend": "none"},
        },
    }
    monkeypatch.setattr(setup, "variant", lambda: "cpu")
    d = detect_setup_state(config, models_dir=str(tmp_path / "models"))
    assert not d.needs_setup


def test_reset_marker_forces_setup(tmp_path):
    # An armed reset re-runs the wizard even though config + assets are present.
    models = tmp_path / "models"
    _mk_hub(models, "Qwen/Qwen3-ASR-1.7B")
    _mk_hub(models, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    config = {"general": {"wakeword": "hey atticus"}, "models": {"llm": {"backend": "none"}}}
    marker = tmp_path / ".setup_pending"

    # No marker -> ready.
    d = detect_setup_state(config, models_dir=str(models), reset_marker=str(marker))
    assert not d.needs_setup

    # Marker present -> wizard, config still seen as present.
    marker.write_text("reset requested\n")
    d = detect_setup_state(config, models_dir=str(models), reset_marker=str(marker))
    assert d.needs_setup and d.config_present
    assert "reset" in d.reason


def test_remote_llm_needs_no_local_asset(tmp_path):
    # A remote OpenAI-compatible LLM has a model *name*, not a local asset —
    # it must not keep the install stuck reporting setup-needed.
    models = tmp_path / "models"
    _mk_hub(models, "Qwen/Qwen3-ASR-1.7B")
    _mk_hub(models, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    config = {
        "general": {"wakeword": "hey atticus"},
        "models": {"llm": {"backend": "openai", "model": "gpt-x", "base_url": "http://x/v1"}},
    }
    d = detect_setup_state(config, models_dir=str(models))
    assert not d.needs_setup
    assert d.missing_assets == []


def test_missing_model_assets_needs_setup(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    config = {"general": {"wakeword": "hey atticus"}, "models": {"llm": {"backend": "none"}}}
    d = detect_setup_state(config, models_dir=str(models))
    assert d.needs_setup
    # Both Qwen backends' hub dirs are absent.
    joined = " ".join(d.missing_assets)
    assert "asr:qwen" in joined and "tts:qwen" in joined


def test_llama_requires_gguf_and_grammar(tmp_path):
    models = tmp_path / "models"
    _mk_hub(models, "Qwen/Qwen3-ASR-1.7B")
    _mk_hub(models, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    gguf = models / "model.gguf"
    config = {
        "general": {"wakeword": "hey atticus"},
        "models": {"llm": {"backend": "llama", "model": str(gguf)}},
    }

    # gguf + grammar both missing
    d = detect_setup_state(config, models_dir=str(models))
    assert d.needs_setup
    joined = " ".join(d.missing_assets)
    assert "llm:llama" in joined and "grammar" in joined

    # provide both -> ready
    gguf.write_text("x")
    (models / "grammars").mkdir()
    (models / "grammars" / "agent.gbnf").write_text('root ::= "x"')
    d2 = detect_setup_state(config, models_dir=str(models))
    assert not d2.needs_setup


# --- Lifecycle --------------------------------------------------------------


def test_lifecycle_transitions_and_snapshot():
    life = Lifecycle(phase=lc.NEEDS_SETUP, detail="first run", missing_assets=["asr"])
    snap = life.snapshot()
    assert snap["phase"] == lc.NEEDS_SETUP
    assert snap["detail"] == "first run"
    assert snap["missing_assets"] == ["asr"]
    assert not life.is_ready()

    life.set(lc.READY)
    assert life.is_ready()
    assert life.snapshot()["phase"] == lc.READY

    assert not life.proceed.is_set()
    life.signal_proceed()
    assert life.proceed.is_set()


# --- Dashboard setup mode (no assistant) ------------------------------------


def _setup_client():
    life = Lifecycle(
        phase=lc.NEEDS_SETUP, detail="first run", missing_assets=["asr:qwen (Qwen/Qwen3-ASR-1.7B)"]
    )
    return TestClient(create_app(None, lifecycle=life)), life


def test_setup_mode_status_reports_phase():
    client, _ = _setup_client()
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["phase"] == lc.NEEDS_SETUP
    assert body["state"] == "idle"
    assert body["mic_enabled"] is False
    assert body["missing_assets"]


def test_setup_mode_serves_setup_page():
    client, _ = _setup_client()
    for path in ("/", "/setup"):
        r = client.get(path)
        assert r.status_code == 200
        assert "First-run setup" in r.text


def test_setup_mode_assistant_routes_return_503():
    client, _ = _setup_client()
    assert client.post("/chat", json={"text": "hi"}).status_code == 503
    assert client.post("/mic", json={"enabled": True}).status_code == 503
    assert client.post("/stop").status_code == 503
    assert client.get("/facts").status_code == 503


def test_ready_lifecycle_with_no_assistant_still_503s():
    # Defensive: even if a stray READY phase is reported, a None assistant
    # must not let routes dereference it.
    client = TestClient(create_app(None, lifecycle=Lifecycle(phase=lc.READY)))
    assert client.post("/chat", json={"text": "hi"}).status_code == 503


def test_loading_phase_serves_setup_loading_view_not_dashboard():
    # Assistant is attached (post set_assistant) but still loading models /
    # warming up — the dashboard must NOT be served yet; '/' returns the setup
    # page (which routes itself to the loading screen by phase).
    from unittest.mock import MagicMock

    life = Lifecycle(phase=lc.LOADING, detail="loading models")
    client = TestClient(create_app(MagicMock(), lifecycle=life))
    body = client.get("/").text
    assert "Fulloch — Setup" in body and "Fulloch</title>" not in body


def test_ready_phase_serves_dashboard():
    from unittest.mock import MagicMock

    life = Lifecycle(phase=lc.READY)
    client = TestClient(create_app(MagicMock(), lifecycle=life))
    body = client.get("/").text
    assert "<title>Fulloch</title>" in body  # the dashboard, not the setup page
