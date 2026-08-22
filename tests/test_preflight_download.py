"""Blocking preflight checks that run on the wizard's "Start download" click.

These complement the *advisory* preflight snapshot in `preflight()`
(tier_fit badges, RAM/VRAM, used at boot) — these checks are the
hard-fail gate just before bytes start flowing. The user clicked
"Start download"; we owe them a clear "X is broken" before the
download itself produces a less-specific error.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import preflight as pf  # noqa: E402

# --- check_disk_for_models -------------------------------------------------


def test_check_disk_passes_when_free_exceeds_needed():
    """Plausibly-sized tier on a normal disk → ok, no message."""
    models = {"asr": {"backend": "qwen-onnx"}, "tts": {"backend": "kokoro-onnx"}}
    with patch.object(pf, "disk_free_gb", return_value=9999.0):
        ok, msg = pf.check_disk_for_models(models)
    assert ok is True
    assert msg == ""


def test_check_disk_fails_with_useful_message_when_full():
    """Tiny free disk vs. a real download size → fail with both numbers."""
    models = {"asr": {"backend": "qwen-onnx"}, "tts": {"backend": "kokoro-onnx"}}
    with patch.object(pf, "disk_free_gb", return_value=0.1):
        ok, msg = pf.check_disk_for_models(models)
    assert ok is False
    # Message should mention both free and needed so the user can act on it.
    assert "free" in msg.lower()
    assert "need" in msg.lower()


def test_check_disk_passes_for_zero_download_tier():
    """A tier with no download (regex-only, remote LLM) → ok regardless of disk."""
    # cpu_local uses regex-only LLM with no model download. We construct a
    # synthetic models dict the resolver treats as download-free by
    # bypassing resolve_models' own download_size_gb.
    with patch.object(pf, "disk_free_gb", return_value=0.0):
        with patch.object(pf, "_models_download_gb", return_value=0.0):
            ok, msg = pf.check_disk_for_models({})
    assert ok is True
    assert msg == ""


# --- check_network ---------------------------------------------------------


def test_check_network_passes_against_real_huggingface():
    """Real HEAD against huggingface.co. Skip if no network in CI."""
    pytest = __import__("pytest")
    try:
        ok, msg = pf.check_network(timeout=5.0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"network check skipped: {e}")
    if not ok:
        # The CI sandbox might block outbound. Treat connection failure
        # as a skip so this test doesn't fail when there's no network.
        if "can't reach" in msg or "timeout" in msg:
            pytest.skip(f"network check skipped: {msg}")
    assert ok is True
    assert msg == ""


def test_check_network_fails_on_unreachable_host():
    """A 0.0.0.0 address with a port nothing is listening on → fail fast."""
    # Use a TCP port that should be unbound (assuming standard test env).
    # Port 1 is reserved (tcpmux) and almost never listening.
    ok, msg = pf.check_network(url="http://127.0.0.1:1/", timeout=1.0)
    assert ok is False
    assert msg  # non-empty failure message
    assert "127.0.0.1" in msg or "reach" in msg.lower() or "timeout" in msg.lower()


# --- check_gpu_for_models --------------------------------------------------


def test_check_gpu_passes_for_cpu_tier_without_gpu():
    """A CPU-friendly tier never trips the GPU check, even on a GPU-less box."""
    # A models dict where every backend has cpu_ok=True (mock at the spec
    # level to keep the test independent of TIER_PRESETS changing).
    class _Spec:
        cpu_ok = True

    with patch.object(pf, "_needs_gpu", return_value=False):
        with patch.object(pf, "gpu_info", return_value={"available": False, "name": None, "vram_gb": None}):
            ok, msg = pf.check_gpu_for_models({})
    assert ok is True
    assert msg == ""


def test_check_gpu_fails_for_gpu_tier_without_gpu():
    """A GPU-only tier with no GPU visible → fail with a clear message."""
    with patch.object(pf, "_needs_gpu", return_value=True):
        with patch.object(pf, "gpu_info", return_value={"available": False, "name": None, "vram_gb": None}):
            ok, msg = pf.check_gpu_for_models({})
    assert ok is False
    assert "GPU" in msg or "gpu" in msg.lower()


def test_check_gpu_passes_for_gpu_tier_with_gpu():
    """A GPU-only tier with a visible GPU → ok."""
    with patch.object(pf, "_needs_gpu", return_value=True):
        with patch.object(pf, "_models_vram_gb", return_value=16.0):
            with patch.object(pf, "gpu_info", return_value={"available": True, "name": "RTX 5060 Ti", "vram_gb": 16.0}):
                ok, msg = pf.check_gpu_for_models({})
    assert ok is True
    assert msg == ""


def test_check_gpu_fails_when_visible_gpu_lacks_required_vram():
    with patch.object(pf, "_needs_gpu", return_value=True):
        with patch.object(pf, "_models_vram_gb", return_value=16.0):
            with patch.object(pf, "gpu_info", return_value={"available": True, "name": "RTX", "vram_gb": 8.0}):
                ok, msg = pf.check_gpu_for_models({})
    assert ok is False
    assert "16.0GB" in msg and "8.0GB" in msg
