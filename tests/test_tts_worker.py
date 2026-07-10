"""Regression test for the shared TTS worker surviving a mid-stream barge-in.

`core/tts.py` runs every generation on a single long-lived worker thread.
A barge-in cancels the in-flight job mid-stream; if the consumer walked away
from its bounded output queue, the worker would wedge on its next
`queue.put(...)` (most insidiously the final `None` sentinel in its
`finally`) and — since that one worker serves all future speech — the
assistant would go permanently mute after the first barge-in.

These tests stub torch / qwen_tts so the module imports without a GPU or the
real model, then drive the real `_worker_loop`, `speak_stream`, and
`synthesize` against a fake generator. TTS is WebSocket-only (no satellite
sink registered here), so playback goes through the "no satellite connected"
drain path in `core/tts.py`.
"""

import queue
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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
    modules.
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

    # The model load is deferred behind load_tts() now — call it so the
    # module-global `model` is populated with the fake before the worker runs.
    tts.load_tts()
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


def test_speak_stream_returns_estimated_playback_end():
    """speak_stream must return playback-start + audio-duration, not just
    "now" — chunks are handed to the satellite sink as fast as they're
    generated with no flow control, so generation finishing doesn't mean the
    audio has actually finished playing on the browser. Regression coverage
    for the follow-up-window bug: arming the window at generation-finish
    time let it silently expire while a long reply was still playing."""
    tts = _import_tts()
    sample_rate = 16000

    def fake_gen(text, voice_clone_prompt=None, **kw):
        # 3 chunks of 1 second each of audio at 16kHz.
        for _ in range(3):
            yield ([0.0] * sample_rate, sample_rate)

    tts.model.stream_generate_voice_clone = fake_gen
    sink: "queue.Queue" = queue.Queue()
    t_before = time.monotonic()
    playback_end = tts.speak_stream("hello", object(), session=tts.TtsSession(), sink=sink)
    t_after = time.monotonic()

    # 3 seconds of audio; generation itself (a python loop over lists) takes
    # a negligible fraction of that, so the estimate must land close to 3s
    # in the future — nowhere near "now" (which is what generation-finish
    # time would give).
    assert playback_end >= t_before + 2.9
    assert playback_end <= t_after + 3.1


def test_speak_stream_sends_cancel_not_end_on_barge_in():
    """Regression coverage for the "Always Listen doesn't stop playback" bug:
    a barge-in used to leave "end" as the final message, which the client
    treats as a no-op — already-scheduled audio just kept playing. Mid-stream
    cancellation must now emit "cancel" instead, so the client actually stops
    it."""
    tts = _import_tts()

    def fake_gen(text, voice_clone_prompt=None, **kw):
        for _ in range(50):
            yield ([0.0] * 100, 16000)

    tts.model.stream_generate_voice_clone = fake_gen
    sink: "queue.Queue" = queue.Queue()
    session = tts.TtsSession()
    # Cancel from on_first_audio, which fires synchronously on speak_stream's
    # own thread right after "start" and the first chunk are queued —
    # deterministic, unlike racing a barge-in against a background
    # generation thread from the test.
    t = threading.Thread(
        target=tts.speak_stream,
        args=("hello", object()),
        kwargs={"session": session, "on_first_audio": session.stop, "sink": sink},
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "speak_stream did not wind down after cancel"

    kinds = []
    while True:
        try:
            kinds.append(sink.get_nowait()[0])
        except queue.Empty:
            break
    assert "cancel" in kinds
    assert "end" not in kinds
