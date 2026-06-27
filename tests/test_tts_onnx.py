"""Kokoro 82M ONNX TTS backend (registry + a guarded functional test)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import backends as b  # noqa: E402

_MODEL_DIR = Path("data/models/kokoro-82m-onnx")
_HAVE_MODEL = any((_MODEL_DIR / "onnx").glob("model*.onnx")) if (_MODEL_DIR / "onnx").is_dir() else False


def test_kokoro_onnx_registered_cpu_offerable():
    spec = b.get_spec("tts", "kokoro-onnx")
    assert spec.implemented and spec.cpu_ok and not spec.gpu_only
    assert b.is_offerable(spec, "cpu") and b.is_offerable(spec, "gpu")
    assert b.get_loader("tts", "kokoro-onnx").__name__ == "load_tts"
    # Partial-download patterns so the big repo isn't fully fetched (the model
    # + the voice packs only).
    assert spec.hf_allow and any(".onnx" in p for p in spec.hf_allow)
    assert any("voices" in p for p in spec.hf_allow)


def test_torch_kokoro_backend_removed():
    with pytest.raises(ValueError):
        b.get_spec("tts", "kokoro")


def test_cpu_stacks_use_kokoro_onnx():
    from server.config_schema import TIER_PRESETS
    for tid in ("cpu_local", "cpu_server"):
        tier = next(t for t in TIER_PRESETS if t.id == tid)
        assert tier.models["tts"]["backend"] == "kokoro-onnx"


def test_fragmenter_splits_on_clause_punctuation():
    from core import tts_onnx
    text = ("Sure, the weather today is mostly sunny with a high of twenty-two "
            "degrees, and a light breeze. Tomorrow looks similar though.")
    frags = list(tts_onnx._iter_fragments(text))
    # One fragment per clause/sentence boundary (commas + the period).
    assert frags == [
        "Sure,",
        "the weather today is mostly sunny with a high of twenty-two degrees,",
        "and a light breeze.",
        "Tomorrow looks similar though.",
    ]
    # Reconstruct losslessly (no dropped/added words).
    assert " ".join(frags).split() == text.split()
    # Non-trivial input never yields an empty fragment.
    assert all(f.strip() for f in frags)


def test_fragmenter_handles_short_and_empty():
    from core import tts_onnx
    assert list(tts_onnx._iter_fragments("Hi there.")) == ["Hi there."]
    assert list(tts_onnx._iter_fragments("   ")) == []


def test_desegment_splits_runons_but_leaves_real_words():
    pytest.importorskip("wordninja")
    import wordninja

    from core import tts_onnx
    tts_onnx._wordninja = wordninja
    # The OOV run-on that misaki drops is recovered into its real words.
    assert tts_onnx._desegment("partlycloudy") == "partly cloudy"
    # Surrounding punctuation is preserved around the split.
    assert tts_onnx._desegment("It is partlycloudy today.") == "It is partly cloudy today."
    # Ordinary long words, multi-word text, and alphanumerics are untouched.
    for s in ("temperature", "information", "thunderstorms", "hello world", "v2.2 build"):
        assert tts_onnx._desegment(s) == s


def test_set_speed_clamps_and_ignores_bad():
    from core import tts_onnx
    saved = tts_onnx._speed
    try:
        tts_onnx.set_speed(1.5)
        assert tts_onnx._speed == 1.5
        tts_onnx.set_speed(9)        # clamp high
        assert tts_onnx._speed == 2.0
        tts_onnx.set_speed(0.0)      # clamp low
        assert tts_onnx._speed == 0.5
        tts_onnx.set_speed("nope")   # bad value ignored, left unchanged
        assert tts_onnx._speed == 0.5
    finally:
        tts_onnx._speed = saved


def test_desegment_is_noop_without_wordninja():
    from core import tts_onnx
    saved = tts_onnx._wordninja
    tts_onnx._wordninja = None
    try:
        assert tts_onnx._desegment("partlycloudy") == "partlycloudy"
    finally:
        tts_onnx._wordninja = saved


@pytest.mark.skipif(not _HAVE_MODEL, reason="Kokoro ONNX model not downloaded")
def test_kokoro_onnx_synthesizes():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("misaki")
    import numpy as np

    from core import tts_onnx

    tts_onnx.load_tts(str(_MODEL_DIR))
    tts_onnx.set_voice("af_heart")
    audio = tts_onnx._synth("Hello, this is a short test.")
    assert audio.size > 0 and np.isfinite(audio).all()
    # A voice that isn't on disk falls back to an available one (no crash).
    fb = tts_onnx.set_voice("zz_not_a_real_voice")
    assert fb in tts_onnx.KOKORO_VOICES and tts_onnx._voice_style is not None
    # Streaming/worker path yields (chunk, sr) tuples and terminates.
    chunks, sr = tts_onnx.synthesize("First sentence. Second sentence.", "af_heart")
    assert sr == tts_onnx.SAMPLE_RATE and chunks and all(c.size for c in chunks)
