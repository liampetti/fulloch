"""Download manager + pre-flight (v2.2 Step 4)."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backends import resolve_models  # noqa: E402
from server import downloader as dl  # noqa: E402
from server import preflight as pf  # noqa: E402


def _wait_done(mgr, timeout=5.0):
    deadline = time.time() + timeout
    while mgr.active and time.time() < deadline:
        time.sleep(0.01)


def test_plan_assets_covers_backends_and_always_required():
    resolved = resolve_models({"asr": {"backend": "qwen"},
                               "tts": {"backend": "qwen"},
                               "llm": {"backend": "llama"}})
    assets = dl.plan_assets(resolved, models_dir="/tmp/nope")
    keys = {a.key for a in assets}
    assert "asr:qwen" in keys and "tts:qwen" in keys and "llm:llama" in keys
    assert "bge" in keys and "grammar" in keys
    # GGUF SLM is a single-file download; the rest are snapshots/url.
    llm = next(a for a in assets if a.key == "llm:llama")
    assert llm.kind == "file" and llm.filename.endswith(".gguf")


def test_plan_skips_none_llm():
    resolved = resolve_models({"asr": {"backend": "moonshine"},
                               "tts": {"backend": "kokoro-onnx"},
                               "llm": {"backend": "none"}})
    keys = {a.key for a in dl.plan_assets(resolved)}
    assert not any(k.startswith("llm:") for k in keys)
    assert "asr:moonshine" in keys and "tts:kokoro-onnx" in keys
    # Kokoro-ONNX is a directory model fetched as a partial dir_snapshot.
    kok = next(a for a in dl.plan_assets(resolved) if a.key == "tts:kokoro-onnx")
    assert kok.kind == "dir_snapshot" and kok.allow


def test_download_runs_all_and_reports_done(tmp_path):
    calls = []
    mgr = dl.DownloadManager(
        snapshot_fn=lambda repo, dest: calls.append(("snap", repo)),
        file_fn=lambda repo, fn, dest: calls.append(("file", repo, fn)),
        url_fn=lambda url, dest: calls.append(("url", url)),
    )
    resolved = resolve_models({"llm": {"backend": "llama"}})
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    done = {}
    mgr.start(assets, on_complete=lambda ok: done.setdefault("ok", ok))
    _wait_done(mgr)
    snap = mgr.snapshot()
    assert snap["state"] == "done"
    assert snap["completed"] == snap["total"] == len(assets)
    assert done["ok"] is True
    assert calls  # the fakes ran


def test_download_marks_error_and_stops(tmp_path):
    def boom(*a):
        raise RuntimeError("network down")

    mgr = dl.DownloadManager(snapshot_fn=boom, file_fn=boom, url_fn=boom)
    resolved = resolve_models({"asr": {"backend": "qwen"}, "llm": {"backend": "none"}})
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    err = {}
    mgr.start(assets, on_complete=lambda ok: err.setdefault("ok", ok))
    _wait_done(mgr)
    snap = mgr.snapshot()
    assert snap["state"] == "error"
    assert err["ok"] is False
    assert any(a["status"] == "error" for a in snap["assets"])


def test_already_present_assets_skip(tmp_path):
    # Pre-create the grammar so it's marked done without downloading.
    (tmp_path / "grammars").mkdir()
    (tmp_path / "grammars" / "json.gbnf").write_text("x")
    ran = []
    mgr = dl.DownloadManager(
        snapshot_fn=lambda *a: ran.append("snap"),
        url_fn=lambda *a: ran.append("url"),
        dir_snapshot_fn=lambda *a: ran.append("dir"),
    )
    resolved = resolve_models({"asr": {"backend": "moonshine"}, "tts": {"backend": "kokoro-onnx"},
                               "llm": {"backend": "none"}})
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    mgr.start(assets)
    _wait_done(mgr)
    grammar = next(a for a in mgr.snapshot()["assets"] if a["key"] == "grammar")
    assert grammar["status"] == "done"
    assert "url" not in ran  # grammar was skipped


def test_snapshot_exposes_domain_and_present(tmp_path):
    resolved = resolve_models({"asr": {"backend": "moonshine"},
                               "tts": {"backend": "kokoro-onnx"}, "llm": {"backend": "none"}})
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    asr = next(a for a in assets if a.key == "asr:moonshine").snapshot()
    assert asr["domain"] == "asr" and asr["present"] is False and "dest" in asr
    # The always-required extras carry no domain.
    bge = next(a for a in assets if a.key == "bge").snapshot()
    assert bge["domain"] is None


def test_custom_gguf_path_is_planned_and_present_in_place(tmp_path):
    # A user-supplied .gguf path should be present-checked (and loaded) in place,
    # not re-downloaded to the default location.
    gguf = tmp_path / "mine" / "custom-model.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("weights")
    resolved = resolve_models({"asr": {"backend": "moonshine"},
                               "tts": {"backend": "kokoro-onnx"},
                               "llm": {"backend": "llama", "model": str(gguf)}})
    llm = next(a for a in dl.plan_assets(resolved, models_dir=str(tmp_path)) if a.key == "llm:llama")
    assert llm.kind == "file" and llm.filename == "custom-model.gguf"
    assert Path(llm.dest) == gguf.parent
    assert llm.snapshot()["present"] is True  # already on disk -> no download


def test_custom_dir_path_honoured_for_snapshot_backend(tmp_path):
    # A custom local folder for a normally-HF-snapshot backend (qwen TTS) must be
    # present-checked at that folder, not fall back to the default repo's hub.
    tts_dir = tmp_path / "my-tts"
    tts_dir.mkdir()
    (tts_dir / "model.bin").write_text("x")  # non-empty -> present
    resolved = resolve_models({"asr": {"backend": "moonshine"},
                               "tts": {"backend": "qwen", "model": str(tts_dir)},
                               "llm": {"backend": "none"}})
    tts = next(a for a in dl.plan_assets(resolved, models_dir=str(tmp_path)) if a.key == "tts:qwen")
    assert tts.kind == "dir_snapshot" and Path(tts.dest) == tts_dir
    assert tts.snapshot()["present"] is True


# --- preflight --------------------------------------------------------------

def test_disk_free_is_positive():
    assert pf.disk_free_gb(".") > 0


def test_tier_fit_badges_without_gpu():
    gpu = {"available": False, "name": None, "vram_gb": None}
    fits = pf.tier_fit(gpu, disk_gb=999)
    by_id = {f["id"]: f for f in fits}
    # GPU-needing tiers warn on a CPU-only box; CPU stacks (no VRAM need) are ok.
    assert by_id["cpu_local"]["badge"] == "ok"
    assert by_id["cpu_server"]["badge"] == "ok"
    assert by_id["full"]["badge"] == "warn"


def test_tier_fit_warns_on_low_disk():
    gpu = {"available": True, "name": "X", "vram_gb": 24}
    fits = pf.tier_fit(gpu, disk_gb=0.1)
    assert all(f["badge"] == "warn" for f in fits if f["download_gb"] > 0.1)


def test_tier_fit_warns_on_low_ram():
    gpu = {"available": False, "name": None, "vram_gb": None}
    # CPU tiers need ~4.9 GB RAM (4.5 ASR + 0.4 TTS); 2 GB available should warn.
    ram = {"total_gb": 4.0, "available_gb": 2.0}
    fits = pf.tier_fit(gpu, disk_gb=999, ram=ram)
    by_id = {f["id"]: f for f in fits}
    assert by_id["cpu_local"]["badge"] == "warn"
    assert "RAM" in by_id["cpu_local"]["reason"]
    assert by_id["cpu_server"]["badge"] == "warn"


def test_tier_fit_ok_with_sufficient_ram():
    gpu = {"available": False, "name": None, "vram_gb": None}
    ram = {"total_gb": 16.0, "available_gb": 8.0}
    fits = pf.tier_fit(gpu, disk_gb=999, ram=ram)
    by_id = {f["id"]: f for f in fits}
    assert by_id["cpu_local"]["badge"] == "ok"
    assert by_id["cpu_server"]["badge"] == "ok"


def test_tier_fit_skips_ram_check_when_unknown():
    gpu = {"available": False, "name": None, "vram_gb": None}
    # ram=None (unreadable) and available_gb=None should not trigger a warn.
    fits_no_ram = pf.tier_fit(gpu, disk_gb=999, ram=None)
    fits_null = pf.tier_fit(gpu, disk_gb=999, ram={"total_gb": None, "available_gb": None})
    for fits in (fits_no_ram, fits_null):
        by_id = {f["id"]: f for f in fits}
        assert by_id["cpu_local"]["badge"] == "ok"


def test_tier_fit_exposes_ram_gb():
    gpu = {"available": False, "name": None, "vram_gb": None}
    fits = pf.tier_fit(gpu, disk_gb=999)
    by_id = {f["id"]: f for f in fits}
    # cpu_local: qwen-onnx (4.5) + kokoro-onnx (0.4) = 4.9
    assert by_id["cpu_local"]["ram_gb"] == pytest.approx(4.9)
    # full: only qwen-onnx ASR runs on CPU (4.5); GPU TTS/LLM have no ram_gb
    assert by_id["full"]["ram_gb"] == pytest.approx(4.5)


def test_ram_info_returns_dict():
    info = pf.ram_info()
    assert "total_gb" in info and "available_gb" in info
    # On Linux (CI / dev box) both should be readable; skip assertion on other OS.
    import os
    if os.path.exists("/proc/meminfo"):
        assert info["total_gb"] is not None and info["total_gb"] > 0
        assert info["available_gb"] is not None


def test_preflight_shape():
    out = pf.preflight(".")
    assert "disk_free_gb" in out and "gpu" in out and "ram" in out and "tier_fit" in out
    assert "ram_gb" in out["tier_fit"][0]
