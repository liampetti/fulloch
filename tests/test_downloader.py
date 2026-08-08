"""Download manager + pre-flight (v2.2 Step 4)."""

import sys
import threading
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


def test_asset_tqdm_stub_survives_ensure_lock_cycle():
    """Reproduces huggingface_hub's tqdm.contrib.concurrent.ensure_lock sequence
    (used during parallel multi-file snapshot downloads), which does
    `getattr(cls, "_lock", None)`, `set_lock(...)`, then `del cls._lock` when no
    prior lock existed. A stub that only tracks the lock in a closure — instead
    of a real `_lock` class attribute — breaks on that final `del`.
    """
    asset = dl.Asset(key="asr:qwen-onnx", label="x", kind="dir_snapshot", dest="/tmp/x")
    tqdm_class = dl._make_asset_tqdm(asset, threading.Lock())

    old_lock = getattr(tqdm_class, "_lock", None)
    assert old_lock is None
    lock = old_lock or tqdm_class.get_lock()
    tqdm_class.set_lock(lock)
    del tqdm_class._lock  # must not raise AttributeError


def test_asset_tqdm_stub_propagates_iterable_errors():
    """The outer bar wraps a `concurrent.futures.Executor.map()` result
    iterator; a failed per-file download surfaces as an exception when that
    iterator is consumed. A stub whose `__iter__` doesn't actually iterate the
    wrapped iterable (e.g. `return iter([])`) would silently swallow it.
    """
    asset = dl.Asset(key="asr:qwen-onnx", label="x", kind="dir_snapshot", dest="/tmp/x")
    tqdm_class = dl._make_asset_tqdm(asset, threading.Lock())

    def gen():
        yield 1
        raise ValueError("boom")

    seen = []
    with pytest.raises(ValueError):
        for item in tqdm_class(gen(), total=2):
            seen.append(item)
    assert seen == [1]


def test_byte_progress_patches_hf_tqdm_and_restores():
    """`_byte_progress` must patch huggingface_hub's internal per-file tqdm
    class (the only place real byte totals are known — `snapshot_download`'s
    own `tqdm_class=` kwarg only sees file *counts*), accumulate concurrent
    bars' totals/updates onto the asset, and restore the original class after.
    """
    hf_utils_tqdm = pytest.importorskip("huggingface_hub.utils.tqdm")
    original = hf_utils_tqdm.tqdm

    asset = dl.Asset(key="asr:qwen-onnx", label="x", kind="dir_snapshot", dest="/tmp/x")
    lock = threading.Lock()

    with dl._byte_progress(asset, lock):
        assert hf_utils_tqdm.tqdm is not original
        bar1 = hf_utils_tqdm.tqdm(total=1000)
        bar2 = hf_utils_tqdm.tqdm(total=2000)
        bar1.update(500)
        bar2.update(1000)
        assert asset.bytes_total == 3000
        assert asset.bytes_done == 1500

    assert hf_utils_tqdm.tqdm is original


def test_plan_assets_covers_backends_and_always_required():
    resolved = resolve_models(
        {"asr": {"backend": "qwen"}, "tts": {"backend": "qwen"}, "llm": {"backend": "llama"}}
    )
    assets = dl.plan_assets(resolved, models_dir="/tmp/nope")
    keys = {a.key for a in assets}
    assert "asr:qwen" in keys and "tts:qwen" in keys and "llm:llama" in keys
    assert "bge" in keys and "grammar" in keys
    # GGUF SLM is a single-file download; the rest are snapshots/url.
    llm = next(a for a in assets if a.key == "llm:llama")
    assert llm.kind == "file" and llm.filename.endswith(".gguf")


def test_plan_skips_none_llm():
    resolved = resolve_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "none"},
        }
    )
    keys = {a.key for a in dl.plan_assets(resolved)}
    assert not any(k.startswith("llm:") for k in keys)
    assert "asr:moonshine" in keys and "tts:kokoro-onnx" in keys
    # Kokoro-ONNX is a directory model fetched as a partial dir_snapshot.
    kok = next(a for a in dl.plan_assets(resolved) if a.key == "tts:kokoro-onnx")
    assert kok.kind == "dir_snapshot" and kok.allow


def test_snapshot_without_completion_marker_is_not_present(tmp_path):
    asset = dl.Asset(key="asr:qwen", label="x", kind="snapshot", dest=str(tmp_path), repo="Qwen/model")
    (tmp_path / "models--Qwen--model").mkdir()

    assert dl._already_present(asset) is False

    (tmp_path / "models--Qwen--model" / dl.COMPLETE_SENTINEL).touch()
    assert dl._already_present(asset) is True


def test_legacy_complete_hf_snapshot_is_present_without_marker(tmp_path):
    root = tmp_path / "models--Qwen--model"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("abc")
    (root / "snapshots" / "abc").mkdir(parents=True)
    (root / "snapshots" / "abc" / "config.json").write_text("{}")

    asset = dl.Asset(key="asr:qwen", label="x", kind="snapshot", dest=str(tmp_path), repo="Qwen/model")
    assert dl._already_present(asset) is True


def test_plan_downloads_compound_crispasr_tts_model_to_one_directory(tmp_path):
    resolved = resolve_models(
        {
            "asr": {"backend": "qwen-gguf"},
            "tts": {"backend": "qwen-gguf"},
            "llm": {"backend": "none"},
        }
    )
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    asr = next(asset for asset in assets if asset.key == "asr:qwen-gguf")
    tts = [asset for asset in assets if asset.key.startswith("tts:qwen-gguf")]
    assert asr.kind == "file" and asr.filename.endswith("q4_k.gguf")
    assert len(tts) == 2
    assert {asset.filename for asset in tts} == {
        "qwen3-tts-12hz-1.7b-base-f16.gguf",
        "qwen3-tts-tokenizer-12hz.gguf",
    }
    assert len({asset.dest for asset in tts}) == 1


def test_plan_downloads_pocket_tts_english_bundle_only():
    resolved = resolve_models(
        {"asr": {"backend": "moonshine"}, "tts": {"backend": "pocket-tts-onnx"}, "llm": {"backend": "none"}}
    )
    pocket = next(a for a in dl.plan_assets(resolved) if a.key == "tts:pocket-tts-onnx")
    assert pocket.kind == "dir_snapshot"
    assert pocket.allow == ["pocket_tts_onnx.py", "onnx/english_2026-04/*"]


def test_plan_downloads_pocket_tts_gguf_file():
    resolved = resolve_models(
        {"asr": {"backend": "moonshine"}, "tts": {"backend": "pocket-tts-gguf"}, "llm": {"backend": "none"}}
    )
    pocket = next(a for a in dl.plan_assets(resolved) if a.key == "tts:pocket-tts-gguf")
    assert pocket.kind == "file"
    assert pocket.filename == "pocket-tts-english-q8_0.gguf"


def test_plan_downloads_compound_small_crispasr_tts_model(tmp_path):
    resolved = resolve_models(
        {
            "asr": {"backend": "qwen-gguf-small"},
            "tts": {"backend": "qwen-gguf-small"},
            "llm": {"backend": "none"},
        }
    )
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    asr = next(asset for asset in assets if asset.key == "asr:qwen-gguf-small")
    tts = [asset for asset in assets if asset.key.startswith("tts:qwen-gguf-small")]
    assert asr.filename == "qwen3-asr-0.6b-q4_k.gguf"
    assert {asset.filename for asset in tts} == {
        "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
        "qwen3-tts-tokenizer-12hz.gguf",
    }


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
    resolved = resolve_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "none"},
        }
    )
    assets = dl.plan_assets(resolved, models_dir=str(tmp_path))
    mgr.start(assets)
    _wait_done(mgr)
    grammar = next(a for a in mgr.snapshot()["assets"] if a["key"] == "grammar")
    assert grammar["status"] == "done"
    assert "url" not in ran  # grammar was skipped


def test_sentinel_only_dir_is_not_present(tmp_path):
    # A dir_snapshot dest holding only the completion sentinel (no actual model
    # files) simulates a run killed between mkdir+touch and the real transfer —
    # must be treated as missing so it re-downloads, not silently loaded as done.
    asset = dl.Asset(
        key="asr:qwen-onnx-small",
        label="Qwen3-ASR 0.6B",
        kind="dir_snapshot",
        dest=str(tmp_path / "qwen3-asr-0.6b-onnx"),
        repo="Daumee/Qwen3-ASR-0.6B-ONNX-CPU",
    )
    d = Path(asset.dest)
    d.mkdir(parents=True)
    (d / dl.COMPLETE_SENTINEL).touch()
    assert dl._already_present(asset) is False

    (d / "onnx_models").mkdir()
    (d / "onnx_models" / "encoder_conv.onnx").write_text("x")
    assert dl._already_present(asset) is True


def test_snapshot_exposes_domain_and_present(tmp_path):
    resolved = resolve_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "none"},
        }
    )
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
    resolved = resolve_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "llama", "model": str(gguf)},
        }
    )
    llm = next(
        a for a in dl.plan_assets(resolved, models_dir=str(tmp_path)) if a.key == "llm:llama"
    )
    assert llm.kind == "file" and llm.filename == "custom-model.gguf"
    assert Path(llm.dest) == gguf.parent
    assert llm.snapshot()["present"] is True  # already on disk -> no download


def test_custom_dir_path_honoured_for_snapshot_backend(tmp_path):
    # A custom local folder for a normally-HF-snapshot backend (qwen TTS) must be
    # present-checked at that folder, not fall back to the default repo's hub.
    tts_dir = tmp_path / "my-tts"
    tts_dir.mkdir()
    (tts_dir / "model.bin").write_text("x")  # non-empty -> present
    resolved = resolve_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "qwen", "model": str(tts_dir)},
            "llm": {"backend": "none"},
        }
    )
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
    # CPU tiers need ~5.2 GB RAM (4.5 qwen-onnx ASR + 0.7 Pocket TTS); 1 GB available should warn.
    ram = {"total_gb": 4.0, "available_gb": 1.0}
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
    # cpu_local: qwen-onnx (4.5) + pocket-tts-onnx (0.7) = 5.2
    assert by_id["cpu_local"]["ram_gb"] == pytest.approx(5.2)
    # full: ASR/TTS/LLM are all GPU-resident now, none set ram_gb
    assert by_id["full"]["ram_gb"] == pytest.approx(0.0)


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
