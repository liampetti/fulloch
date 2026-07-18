"""Contract tests for the lightweight Pocket TTS ONNX adapter."""

import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_loads_bundle_local_wrapper_and_uses_voice_wav(tmp_path, monkeypatch):
    import core.tts_pocket_onnx as tts

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "pocket_tts_onnx.py").write_text(
        "class PocketTTSOnnx:\n"
        "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
        "    def stream(self, text, voice): yield [0.1, 0.2]\n"
    )
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "atticus.wav").write_bytes(b"wav")
    (voices / "default.txt").write_text("Reference transcript.\n")
    monkeypatch.setattr(tts, "VOICES_DIR", voices)

    model = tts.load_tts(str(model_dir))
    prompt = tts.set_voice("atticus")
    chunks, sample_rate = tts.synthesize("Hello", prompt)

    assert model.kwargs["language"] == "english_2026-04"
    assert model.kwargs["precision"] == "int8"
    assert prompt == voices / "atticus.wav"
    assert sample_rate == 24000
    assert len(chunks) == 1


def test_play_chunks_emits_websocket_messages():
    import core.tts_pocket_onnx as tts

    sink = queue.Queue()
    tts.play_chunks([[0.1], [0.2]], 24000, sink=sink)

    assert [sink.get_nowait()[0] for _ in range(4)] == ["start", [0.1], [0.2], "end"]
