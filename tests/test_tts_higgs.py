"""Higgs adapter tests with its native process replaced by a fake."""

import queue
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.tts_higgs as tts  # noqa: E402
from core.tts_session import TtsSession  # noqa: E402


def test_speak_stream_forwards_native_pcm_frames_and_ends(monkeypatch):
    class FakeWorker:
        alive = True

        def synthesize_stream(self, text):
            assert text == "Hello"
            yield np.ones(tts.CHUNK_SAMPLES + 2, dtype=np.float32)

    monkeypatch.setattr(tts, "_start_worker", lambda: FakeWorker())
    sink = queue.Queue()

    tts.speak_stream("Hello", sink=sink)

    assert sink.get() == ("start", tts.SAMPLE_RATE)
    first, marker = sink.get()
    assert len(first) == tts.CHUNK_SAMPLES + 2
    assert marker is None
    assert sink.get() == ("end",)


def test_force_cancel_playback_discards_queued_audio():
    sink = queue.Queue()
    sink.put((np.ones(2, dtype=np.float32), None))

    tts.force_cancel_playback(sink)

    assert sink.get() == ("cancel",)


def test_cancelled_native_stream_close_is_not_an_error(monkeypatch):
    session = TtsSession()

    class FakeWorker:
        alive = True

        def synthesize_stream(self, text):
            session.stop()
            raise ConnectionError("Higgs server closed the connection early")
            yield  # pragma: no cover - make this a generator

    monkeypatch.setattr(tts, "_start_worker", lambda: FakeWorker())

    assert tts.speak_stream("Hello", session=session, sink=queue.Queue()) is not None


def test_long_text_is_split_before_higgs_generation(monkeypatch):
    monkeypatch.setattr(tts, "MAX_SYNTHESIS_CHARS", 10)

    assert list(tts._synthesis_fragments("one two three four")) == ["one two", "three four"]


def test_fragments_prefer_sentences_over_short_comma_clauses(monkeypatch):
    monkeypatch.setattr(tts, "MAX_SYNTHESIS_CHARS", 80)

    assert list(tts._synthesis_fragments("Hello, there friend. How are you?")) == [
        "Hello, there friend.",
        "How are you?",
    ]


def test_weather_forecast_keeps_measurements_in_their_sentence():
    text = (
        "Forecast for home. currently partly cloudy at 14 degrees Celsius. "
        "Today partly cloudy 12 to 18 degrees Celsius. "
        "Tomorrow partly cloudy 11 to 15 degrees Celsius."
    )

    assert list(tts._synthesis_fragments(text)) == [
        "Forecast for home.",
        "currently partly cloudy at 14 degrees Celsius.",
        "Today partly cloudy 12 to 18 degrees Celsius.",
        "Tomorrow partly cloudy 11 to 15 degrees Celsius.",
    ]


def test_leading_delivery_controls_apply_to_every_fragment(monkeypatch):
    monkeypatch.setattr(tts, "MAX_SYNTHESIS_CHARS", 10)
    calls = []

    class FakeWorker:
        alive = True

        def synthesize_stream(self, text):
            calls.append(text)
            yield np.ones(2, dtype=np.float32)

    monkeypatch.setattr(tts, "_start_worker", lambda: FakeWorker())

    tts.speak_stream("<|style:whispering|>one two three four", sink=queue.Queue())

    assert calls == [
        "<|style:whispering|>one two",
        "<|style:whispering|>three four",
    ]


def test_play_chunks_uses_standard_sink_contract():
    sink = queue.Queue()
    tts.play_chunks([np.ones(2, dtype=np.float32)], tts.SAMPLE_RATE, sink=sink)
    assert sink.get() == ("start", tts.SAMPLE_RATE)
    assert sink.get()[1] is None
    assert sink.get() == ("end",)
