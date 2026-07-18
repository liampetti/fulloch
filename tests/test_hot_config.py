"""Live (no-restart) config hot-apply.

Covers the settings-console path that pushes changed config to the running
assistant: `AudioCapture`'s live setters and `Assistant.apply_hot_config`,
including the TTS-backend gating (voice swap live on Kokoro/Pocket,
restart-only on Qwen).
"""

import queue
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio import AudioCapture, dbfs_to_rms  # noqa: E402

# --- AudioCapture live setters ---------------------------------------------


def test_set_use_vad_cannot_enable_without_model():
    # use_vad=False keeps Silero out, so _vad_available is False and VAD can't
    # be turned on live — the setter reports False (caller flags restart).
    ac = AudioCapture(use_vad=False)
    assert ac.set_use_vad(True) is False
    assert ac._use_vad_enabled is False
    # Disabling is always fine and idempotent.
    assert ac.set_use_vad(False) is True
    assert ac._use_vad_enabled is False


def test_set_barge_in_threshold_recomputes_rms():
    ac = AudioCapture(use_vad=False)
    ac.set_barge_in_threshold_dbfs(-40.0)
    assert ac.barge_in_threshold_dbfs == -40.0
    assert ac._barge_in_rms == dbfs_to_rms(-40.0)


def test_set_vad_min_speech_ms_converts_to_samples():
    ac = AudioCapture(use_vad=False)
    ac.set_vad_min_speech_ms(500)
    assert ac.vad_min_speech_samples == int(ac.sample_rate * 500 / 1000)


def test_set_vad_params_noop_without_endpointer():
    # No VAD model loaded -> the call must be a harmless no-op, not a crash.
    ac = AudioCapture(use_vad=False)
    ac.set_vad_params(threshold=0.7, endpoint_silence_ms=2000)  # no raise


# --- Assistant.apply_hot_config --------------------------------------------


def _import_assistant_module():
    """Import core.assistant with its heavy submodules stubbed (see
    test_assistant_ack for the same technique)."""
    fake = {
        "core.audio": ["AudioCapture", "dbfs_to_rms"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice",
            "warmup_model",
            "synthesize",
            "play_chunks",
            "speak_stream",
            "set_output_device",
            "set_tts_active_event",
            "model",
        ],
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        if name == "core.slm":
            mod.ContextExhaustedError = type("ContextExhaustedError", (RuntimeError,), {})
            mod.RemoteUnreachable = type("RemoteUnreachable", (RuntimeError,), {})
        sys.modules[name] = mod
    import core.assistant as assistant  # noqa: E402

    return assistant


def _make_assistant(backend="kokoro-onnx"):
    """A bare Assistant (no __init__) wired with just the attrs apply_hot_config
    touches — class attrs (_HOT_CONFIG_PATHS etc.) resolve via the real class."""
    assistant = _import_assistant_module()
    A = assistant.Assistant
    obj = A.__new__(A)
    obj._tts_backend = backend
    obj._tts_module = MagicMock()
    obj.audio_capture = MagicMock()
    obj.audio_capture.set_use_vad.return_value = True
    obj.barge_in = "wakeword"
    obj.conversation_mode_default = False
    obj.follow_up_seconds = 5.0
    obj.tts_speed = 1.0
    obj.whisper_gain = 0.30
    obj.voice_clone = "af_heart"
    obj.voice_clone_prompt = "old-prompt"
    obj.llm_enabled = True
    obj.synthesize = MagicMock(return_value=([], 24000))
    # Stub the (threaded) cache re-render so the test doesn't synthesise.
    obj._rerender_phrase_caches = MagicMock()
    return obj


def test_hot_apply_endpointing_knobs():
    obj = _make_assistant()
    changes = [
        {"path": "general.vad_threshold", "value": 0.7},
        {"path": "general.vad_endpoint_silence_ms", "value": 2000},
        {"path": "general.vad_min_speech_ms", "value": 400},
        {"path": "general.barge_in_threshold_dbfs", "value": -40.0},
        {"path": "general.barge_in", "value": "off"},
        {"path": "general.conversation_mode_default", "value": True},
        {"path": "general.follow_up_time", "value": "0s"},
    ]
    applied = obj.apply_hot_config(changes)
    assert applied == {c["path"] for c in changes}
    obj.audio_capture.set_vad_params.assert_any_call(threshold=0.7)
    obj.audio_capture.set_vad_params.assert_any_call(endpoint_silence_ms=2000)
    obj.audio_capture.set_vad_min_speech_ms.assert_called_once_with(400)
    obj.audio_capture.set_barge_in_threshold_dbfs.assert_called_once_with(-40.0)
    assert obj.barge_in == "off"
    assert obj.conversation_mode_default is True
    assert obj.follow_up_seconds == 0  # parse_barge_time("0s")


def test_hot_apply_use_vad_declined_needs_restart():
    obj = _make_assistant()
    obj.audio_capture.set_use_vad.return_value = False  # model not loaded
    applied = obj.apply_hot_config([{"path": "general.use_vad", "value": True}])
    assert applied == set()  # not applied -> caller flags restart


def test_hot_apply_voice_and_speed_live_on_kokoro():
    obj = _make_assistant(backend="kokoro-onnx")
    obj._tts_module.set_voice.return_value = "new-prompt"
    applied = obj.apply_hot_config(
        [
            {"path": "general.voice_clone", "value": "am_onyx"},
            {"path": "general.tts_speed", "value": 1.25},
        ]
    )
    assert applied == {"general.voice_clone", "general.tts_speed"}
    obj._tts_module.set_voice.assert_called_once_with("am_onyx")
    assert obj.voice_clone == "am_onyx"
    assert obj.voice_clone_prompt == "new-prompt"
    obj._tts_module.set_speed.assert_called_once_with(1.25)
    assert obj.tts_speed == 1.25
    obj._rerender_phrase_caches.assert_called_once()


def test_hot_apply_voice_restart_only_on_qwen():
    obj = _make_assistant(backend="qwen")
    applied = obj.apply_hot_config(
        [
            {"path": "general.voice_clone", "value": "atticus"},
            {"path": "general.tts_speed", "value": 1.25},
        ]
    )
    # Qwen voice swap needs a restart (clone warmup + cache re-render).
    assert "general.voice_clone" not in applied
    obj._tts_module.set_voice.assert_not_called()
    # Qwen has no speed knob — a restart wouldn't help, so it's reported applied
    # (no-op) rather than nagging for one; set_speed must NOT be called.
    assert "general.tts_speed" in applied
    obj._tts_module.set_speed.assert_not_called()


def test_hot_apply_voice_live_on_pocket_tts():
    obj = _make_assistant(backend="pocket-tts-onnx")
    obj._tts_module.set_voice.return_value = "new-prompt"
    applied = obj.apply_hot_config([{"path": "general.voice_clone", "value": "atticus"}])
    assert applied == {"general.voice_clone"}
    obj._tts_module.set_voice.assert_called_once_with("atticus")
    obj._rerender_phrase_caches.assert_called_once()


def test_hot_apply_whisper_gain_clamps_to_valid_pcm_range():
    obj = _make_assistant()
    applied = obj.apply_hot_config([{"path": "general.whisper_gain", "value": 1.5}])
    assert applied == {"general.whisper_gain"}
    assert obj.whisper_gain == 1.0


def test_whisper_gain_scales_all_pcm_sent_to_the_sink():
    assistant = _import_assistant_module()
    sink = queue.Queue()
    assistant._GainSink(sink, 0.30).put((np.array([1.0, -0.5], dtype=np.float32), None))
    chunk, _ = sink.get_nowait()
    assert np.allclose(chunk, [0.30, -0.15])


def test_whisper_request_uses_configured_gain_for_every_backend():
    assistant = _import_assistant_module()
    obj = assistant.Assistant.__new__(assistant.Assistant)
    obj._tts_backend = "pocket-tts-onnx"
    obj.whisper_gain = 0.30
    satellite = types.SimpleNamespace(higgs_delivery="", tts_gain=1.0)

    assert obj._prepare_delivery_request("Can you whisper that to me?", satellite) == "Can you whisper that to me?"
    assert satellite.tts_gain == 0.30


def test_hot_apply_ignores_restart_only_paths():
    obj = _make_assistant()
    applied = obj.apply_hot_config([{"path": "general.wakeword", "value": "computer"}])
    assert applied == set()
