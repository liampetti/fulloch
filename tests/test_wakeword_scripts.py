"""Offline wakeword dataset tooling regression tests."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "wakeword"))

import generate_positives  # noqa: E402
from augment_dataset import augment  # noqa: E402
from build_manifest import group_key, split_for  # noqa: E402
from common import SAMPLE_RATE, is_target_free, pcm16_valid, write_wav  # noqa: E402
from train_local import _fixed_negatives, _fixed_positive  # noqa: E402


def test_target_filter_rejects_normalised_target_phrase():
    assert not is_target_free("HEY---Atticus!")
    assert is_target_free("hey atlas")


def test_augmentation_is_deterministic_and_wav_is_runtime_format(tmp_path):
    source = np.linspace(-0.1, 0.1, SAMPLE_RATE, dtype=np.float32)
    first, metadata = augment(source, np.zeros_like(source), 42)
    second, _ = augment(source, np.zeros_like(source), 42)
    assert np.array_equal(first, second)
    assert metadata["seed"] == 42
    path = tmp_path / "clip.wav"
    write_wav(path, first)
    assert pcm16_valid(path) == (True, "")


def test_split_is_stable_and_groups_source_variants_together():
    original = {"path": "one.wav", "voice_reference": "voice.wav"}
    variant = {"path": "two.wav", "source": "one.wav", "voice_reference": "voice.wav"}
    assert split_for(group_key(original)) == split_for(group_key(original))
    assert group_key(original, {"one.wav": original}) == group_key(variant, {"one.wav": original})


def test_positive_generator_accepts_runtime_backend_aliases():
    assert generate_positives.parse_backends("pocket-tts-pytorch,qwen") == [
        "pocket-pytorch", "qwen-pytorch"
    ]
    assert generate_positives.parse_backends("auto") == list(generate_positives.BACKENDS)
    assert "kokoro-onnx" not in generate_positives.BACKENDS


def test_positive_generator_dry_run_never_loads_models(tmp_path, monkeypatch, capsys):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "sample.wav").touch()
    monkeypatch.setattr(generate_positives, "backend_skip_reason", lambda _backend, _args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_positives.py",
            "--backends", "pocket-pytorch",
            "--voices", str(voices),
            "--samples-per-source", "2",
            "--dry-run",
        ],
    )

    assert generate_positives.main() == 0
    output = capsys.readouterr().out
    assert "READY pocket-pytorch: 1 source(s), 2 clip(s)" in output
    assert output.count("PLAN pocket-pytorch/sample/") == 2


def test_fixed_positive_ends_near_clip_end_and_negatives_are_segmented(tmp_path):
    path = tmp_path / "positive.wav"
    # One second of phrase-like energy followed by 0.5 seconds of recorded tail.
    write_wav(path, np.concatenate((np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))))
    positive = _fixed_positive({"path": str(path), "trailing_silence_s": 0.5}, SAMPLE_RATE * 2, 7)
    assert positive.shape == (SAMPLE_RATE * 2,)
    assert np.flatnonzero(positive)[-1] >= SAMPLE_RATE * 2 - round(0.2 * SAMPLE_RATE) - 1
    negative_path = tmp_path / "negative.wav"
    write_wav(negative_path, np.ones(SAMPLE_RATE * 5, dtype=np.float32) * 0.1)
    assert len(_fixed_negatives({"path": str(negative_path)}, SAMPLE_RATE * 2)) == 2
