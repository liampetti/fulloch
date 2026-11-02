"""Preset hardware classes, bundled wakeword assets and verification policy."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.voice_reference import voice_reference
from core.wakeword_policy import verify_asr_wakeword
from server.config_schema import TIER_PRESETS, wakeword_presets_as_dicts
from server.preflight import tier_fit


@pytest.mark.parametrize("vram,expected", [
    (0, {"lite"}), (6, {"lite"}), (8, {"economy", "lite"}),
    (10, {"economy", "lite"}), (12, {"premium", "economy", "lite"}),
    (16, {"pro", "premium", "economy", "lite"}),
    (24, {"pro", "premium", "economy", "lite"}),
    (32, {"max", "pro", "premium", "economy", "lite"}),
])
def test_fit_uses_hardware_classes(vram, expected):
    fits = tier_fit({"available": bool(vram), "vram_gb": vram}, 999, {"available_gb": 64})
    assert {f["id"] for f in fits if f["badge"] == "ok"} == expected
    assert next(f for f in fits if f["id"] == "lite")["vram_gb"] == 0


def test_presets_select_expected_stacks():
    tiers = {t.id: t for t in TIER_PRESETS}
    assert {t.id: t.vram_class_gb for t in TIER_PRESETS} == {
        "max": 32, "pro": 16, "premium": 12, "economy": 8, "lite": 0,
    }
    assert tiers["max"].models["llm"]["local_model"] == "qwen27b"
    for tier in ("pro", "premium", "economy"):
        assert tiers[tier].models["llm"]["local_model"] == "ornith"
    assert tiers["premium"].models["asr"]["backend"] == "orukeet"
    assert tiers["premium"].models["tts"]["backend"] == "pocket-tts-pytorch"
    for tier in ("economy", "lite"):
        assert tiers[tier].models["asr"]["backend"] == "moonshine"
        assert tiers[tier].models["tts"]["backend"] == "kokoro-onnx"


def test_gpu_probe_uses_retail_capacity_despite_driver_reservations(monkeypatch):
    import sys

    from server.preflight import gpu_info

    torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _: SimpleNamespace(name="16 GB card", total_memory=int(15.92 * 1024**3)),
    ))
    monkeypatch.setitem(sys.modules, "torch", torch)
    gpu = gpu_info()
    assert gpu["vram_gb"] == 16
    assert gpu["vram_reported_gb"] == pytest.approx(15.92)
    fits = {f["id"]: f for f in tier_fit(gpu, 999)}
    assert fits["pro"]["badge"] == "ok"


@pytest.mark.parametrize("backend", ["moonshine", "moonshine-tiny", "parakeet-onnx"])
def test_unreliable_asr_trusts_acoustic_gate_but_not_arbitrary_speech(backend):
    assert verify_asr_wakeword(backend, {"backend": "openwakeword"}) is False
    assert verify_asr_wakeword(backend, {}) is True
    assert verify_asr_wakeword(backend, {"backend": "asr", "verify_asr": False}) is True
    assert verify_asr_wakeword(backend, {"backend": "openwakeword", "verify_asr": True}) is True


@pytest.mark.parametrize("backend", ["qwen", "qwen-onnx", "orukeet", "parakeet"])
def test_reliable_asr_verifies_by_default_and_allows_config_override(backend):
    assert verify_asr_wakeword(backend, {"backend": "openwakeword"}) is True
    assert verify_asr_wakeword(backend, {"backend": "openwakeword", "verify_asr": False}) is False


def test_upstream_classifiers_are_bundled_with_attribution_and_hashes():
    root = Path(__file__).resolve().parents[1]
    notice = (root / "third_party/openwakeword-models/NOTICE.md").read_text()
    presets = {p["id"]: p for p in wakeword_presets_as_dicts()}
    for name, expected in {
        "alexa": "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e",
        "hey_jarvis": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    }.items():
        p = presets[name]
        assert "CC BY-NC-SA" in p["license"]
        assert hashlib.sha256((root / p["model"]).read_bytes()).hexdigest() == expected
        assert expected in notice


def test_custom_reference_requires_own_transcript(tmp_path):
    audio = tmp_path / "reference.wav"
    audio.touch()
    (tmp_path / "default.txt").write_text("not this recording")
    with pytest.raises(FileNotFoundError, match="transcript"):
        voice_reference(str(audio), tmp_path)
    transcript = audio.with_suffix(".txt")
    transcript.write_text("this recording")
    assert voice_reference(str(audio), tmp_path) == (audio, transcript)


def test_named_reference_retains_bundled_transcript_fallback(tmp_path):
    audio = tmp_path / "atticus.wav"
    audio.touch()
    transcript = tmp_path / "default.txt"
    transcript.write_text("bundled sample")
    assert voice_reference("atticus", tmp_path) == (audio, transcript)
