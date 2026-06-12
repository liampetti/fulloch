"""Regression test for the shared TTS worker surviving a mid-stream barge-in.

`core/tts.py` runs every generation on a single long-lived worker thread.
A barge-in cancels the in-flight job mid-stream; if the consumer walked away
from its bounded output queue, the worker would wedge on its next
`queue.put(...)` (most insidiously the final `None` sentinel in its
`finally`) and — since that one worker serves all future speech — the
assistant would go permanently mute after the first barge-in.

These tests stub torch / sounddevice / qwen_tts so the module imports
without a GPU or the real model, then drive the real `_worker_loop`,
`speak_stream`, and `synthesize` against a fake generator.
"""

import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class _FakeStream:
    """OutputStream stand-in whose write() paces consumption slowly so the
    bounded queue fills and the worker blocks on put — the exact condition
    that wedged the old code on cancel."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, chunk):
        time.sleep(0.02)

    def abort(self):
        pass


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.set_float32_matmul_precision = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    return torch


def _fake_qwen():
    qwen = types.ModuleType("qwen_tts")

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def enable_streaming_optimizations(self, *a, **k):
            pass

        def create_voice_clone_prompt(self, *a, **k):
            return object()

        def stream_generate_voice_clone(self, text, voice_clone_prompt=None, **kw):
            # Overridden per-test via `tts.model.stream_generate_voice_clone`.
            yield (b"chunk", 16000)

    qwen.Qwen3TTSModel = _FakeModel
    return qwen


def _import_tts():
    """Import `core.tts` with torch / qwen_tts faked so no GPU model loads.

    torch and qwen_tts are faked only for the duration of the import, then
    the originals are restored so the rest of the suite keeps the real
    modules. `sounddevice` is real and cheap to import, so we leave it and
    just swap the module-level `sd` on the imported module for a fake whose
    OutputStream doesn't touch a sound card.
    """
    # Other tests (e.g. test_assistant_ack) install a lightweight `core.tts`
    # stub into sys.modules that lacks the worker. Drop it so we load the
    # real module with `_worker_loop`.
    existing = sys.modules.get("core.tts")
    if existing is not None and not hasattr(existing, "_worker_loop"):
        del sys.modules["core.tts"]

    saved = {name: sys.modules.get(name) for name in ("torch", "qwen_tts")}
    sys.modules["torch"] = _fake_torch()
    sys.modules["qwen_tts"] = _fake_qwen()
    try:
        import core.tts as tts  # noqa: E402
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = _FakeStream
    tts.sd = fake_sd
    return tts


def test_barge_in_mid_stream_does_not_wedge_worker():
    """Cancel a long utterance mid-stream, then confirm the next job still runs.

    With the old consumer (which abandoned its queue on cancel) the shared
    worker wedged on `out.put(None)` and the follow-up `synthesize` would
    hang forever. The drain on cancel keeps the worker free.
    """
    tts = _import_tts()

    def fake_gen(text, voice_clone_prompt=None, **kw):
        if text == "long":
            # Yield far more than the queue holds, instantly, so the worker
            # fills the maxsize=5 queue and blocks on a put almost at once.
            for i in range(500):
                yield (f"long-{i}", 16000)
        else:
            for i in range(3):
                yield (f"{text}-{i}", 16000)

    tts.model.stream_generate_voice_clone = fake_gen

    session = tts.TtsSession()
    speak_thread = threading.Thread(
        target=tts.speak_stream,
        args=("long", object()),
        kwargs={"session": session},
        daemon=True,
    )
    speak_thread.start()

    # Let the consumer start playing and the worker fill + block on put,
    # then barge in.
    time.sleep(0.1)
    session.stop()

    speak_thread.join(timeout=5.0)
    assert not speak_thread.is_alive(), "speak_stream did not wind down after cancel"

    # The real assertion: a subsequent job must still be served by the
    # (previously at-risk) shared worker.
    result: list = []

    def run_second():
        result.append(tts.synthesize("second", object()))

    second_thread = threading.Thread(target=run_second, daemon=True)
    second_thread.start()
    second_thread.join(timeout=5.0)
    assert not second_thread.is_alive(), "shared TTS worker wedged after barge-in"
    assert result, "synthesize returned nothing"
    chunks, sample_rate = result[0]
    assert len(chunks) == 3
    assert sample_rate == 16000


def test_repeated_barge_ins_keep_worker_alive():
    """Several barge-ins in a row must each leave the shared worker usable —
    one wedge anywhere kills speech for the rest of the process."""
    tts = _import_tts()

    def fake_gen(text, voice_clone_prompt=None, **kw):
        if text == "long":
            for i in range(500):
                yield (f"long-{i}", 16000)
        else:
            for i in range(3):
                yield (f"{text}-{i}", 16000)

    tts.model.stream_generate_voice_clone = fake_gen

    for attempt in range(3):
        session = tts.TtsSession()
        t = threading.Thread(
            target=tts.speak_stream,
            args=("long", object()),
            kwargs={"session": session},
            daemon=True,
        )
        t.start()
        time.sleep(0.1)
        session.stop()
        t.join(timeout=5.0)
        assert not t.is_alive(), f"speak_stream hung on barge-in #{attempt}"

        result: list = []
        s = threading.Thread(
            target=lambda r=result: r.append(tts.synthesize("ok", object())),
            daemon=True,
        )
        s.start()
        s.join(timeout=5.0)
        assert not s.is_alive(), f"worker wedged after barge-in #{attempt}"
        assert result and len(result[0][0]) == 3
