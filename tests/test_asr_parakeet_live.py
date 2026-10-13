"""Unit seams for Parakeet's live-frame worker; no NeMo runtime required."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.asr_parakeet import (
    LiveAsrFailure,
    LiveAsrResult,
    ParakeetASRPipelineWrapper,
    ParakeetLiveWorker,
)
from core.audio import AudioCapture
from core.satellite import SatelliteSession


class _Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **_kwargs):
        self.calls.append(audio[0].copy())
        return [type("Hypothesis", (), {"text": f"words-{len(audio[0])}"})()]


def _endpoint(pcm, capture_id, *, early=False):
    return (pcm, 1.0, -20.0, False, "sat-a", 2.0, False, early, None, None, early, None, capture_id)


@pytest.mark.parametrize("scoped", [False, True])
def test_discard_retires_active_buffer_and_already_emitted_results(monkeypatch, scoped):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    model = _Model()
    worker = ParakeetLiveWorker(model)
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-a", frame, capture_id=1)
        worker.feed_frame("sat-a", frame, capture_id=1)
        first = worker.get_event(timeout=1)
        assert first is not None and first.item[12] == 1
        # Put an emitted event back to exercise cancellation before retrieval.
        worker._events.put_nowait(first)
        if scoped:
            worker.discard_capture("sat-a", 1)
        else:
            worker.discard("sat-a")
        assert not worker.is_current_result("sat-a", first.result)
        worker.feed_frame("sat-a", np.zeros(4, dtype=np.float32), capture_id=1)
        assert worker.get_event(timeout=0.05) is None
        assert len(model.calls) == 1
        assert "sat-a" not in worker._buffers
        assert "sat-a" not in worker._active

        worker.start("sat-a", frame, capture_id=2)
        worker.feed_frame("sat-a", frame, capture_id=2)
        new = worker.get_event(timeout=1)
        assert new is not None and new.item[12] == 2
        np.testing.assert_array_equal(new.result.pcm, np.ones(8, dtype=np.float32))
    finally:
        worker.close()


def test_commit_during_inflight_asr_keeps_queued_new_capture(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)

    class BlockingModel(_Model):
        started = threading.Event()
        release = threading.Event()

        def transcribe(self, audio, **kwargs):
            if not self.calls:
                self.started.set()
                assert self.release.wait(timeout=2)
            return super().transcribe(audio, **kwargs)

    model = BlockingModel()
    worker = ParakeetLiveWorker(model)
    try:
        worker.start("sat-a", np.ones(4, dtype=np.float32), capture_id=1)
        worker.feed_frame("sat-a", np.ones(4, dtype=np.float32), capture_id=1)
        assert model.started.wait(timeout=1)
        worker.start("sat-a", np.full(4, 2, dtype=np.float32), capture_id=2)
        worker.feed_frame("sat-a", np.full(4, 2, dtype=np.float32), capture_id=2)
        worker.discard_capture("sat-a", 1)
        # Cancellation must not wait for the blocked model, nor submit the
        # committed endpoint for another expensive transcription.
        assert worker.finish(_endpoint(np.ones(8, dtype=np.float32), 1))
        model.release.set()
        event = worker.get_event(timeout=1)
        assert event is not None and event.item[12] == 2
        np.testing.assert_array_equal(event.result.pcm, np.full(8, 2, dtype=np.float32))
        assert len(model.calls) == 2
    finally:
        model.release.set()
        worker.close()


def test_new_generation_start_racing_discard_cleanup_survives(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    worker = ParakeetLiveWorker(_Model())
    frame = np.ones(4, dtype=np.float32)
    original_discard = worker._discard_queued_frames

    def start_between_invalidation_and_cleanup(satellite_id, generation=None, **kwargs):
        worker.start(satellite_id, frame, capture_id=2)
        worker.feed_frame(satellite_id, frame, capture_id=2)
        original_discard(satellite_id, generation, **kwargs)

    try:
        worker.start("sat-a", frame, capture_id=1)
        worker.feed_frame("sat-a", frame, capture_id=1)
        assert worker.get_event(timeout=1) is not None
        monkeypatch.setattr(worker, "_discard_queued_frames", start_between_invalidation_and_cleanup)
        worker.discard("sat-a")
        event = worker.get_event(timeout=1)
        assert event is not None and event.item[12] == 2
        assert event.result.generation == 1
        assert event.result.pcm.size == 8
    finally:
        worker.close()


def test_delayed_old_final_does_not_clear_active_new_capture(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    worker = ParakeetLiveWorker(_Model())
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-a", frame, capture_id=2)
        worker.feed_frame("sat-a", frame, capture_id=2)
        assert worker.get_event(timeout=1).item[12] == 2
        worker.finish(_endpoint(frame, 1))
        assert worker.get_event(timeout=1).final
        worker.discard_capture("sat-a", 1)
        worker.feed_frame("sat-a", frame, capture_id=2)
        event = worker.get_event(timeout=1)
        assert event is not None and event.item[12] == 2
        assert event.result.pcm.size == 12  # new capture was not restarted/reset
    finally:
        worker.close()


def test_final_priority_preserves_queued_start_of_next_capture(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)

    class BlockingModel(_Model):
        started = threading.Event()
        release = threading.Event()

        def transcribe(self, audio, **kwargs):
            if not self.calls:
                self.started.set()
                assert self.release.wait(timeout=2)
            return super().transcribe(audio, **kwargs)

    model = BlockingModel()
    worker = ParakeetLiveWorker(model)
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-a", frame, capture_id=1)
        worker.feed_frame("sat-a", frame, capture_id=1)
        assert model.started.wait(timeout=1)
        worker.start("sat-a", frame, capture_id=2)
        worker.feed_frame("sat-a", frame, capture_id=2)
        worker.finish(_endpoint(frame, 1))
        model.release.set()
        events = [worker.get_event(timeout=1) for _ in range(3)]
        assert [(event.item[12], event.final) for event in events] == [(1, False), (1, True), (2, False)]
    finally:
        model.release.set()
        worker.close()


def test_early_wake_verification_does_not_clear_live_capture(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    worker = ParakeetLiveWorker(_Model())
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-a", frame, capture_id=1)
        worker.feed_frame("sat-a", frame, capture_id=1)
        assert worker.get_event(timeout=1) is not None
        worker.finish(_endpoint(frame, 1, early=True))
        assert worker.get_event(timeout=1).final
        worker.feed_frame("sat-a", frame, capture_id=1)
        event = worker.get_event(timeout=1)
        assert event is not None and not event.final
        assert event.result.pcm.size == 12
    finally:
        worker.close()


def test_capture_commit_does_not_invalidate_another_satellite(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    worker = ParakeetLiveWorker(_Model())
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-b", frame, capture_id=1)
        worker.feed_frame("sat-b", frame, capture_id=1)
        first = worker.get_event(timeout=1)
        worker.discard_capture("sat-a", 1)
        assert worker.is_current_result("sat-b", first.result)
        worker.feed_frame("sat-b", frame, capture_id=1)
        assert worker.get_event(timeout=1).result.pcm.size == 12
    finally:
        worker.close()


def test_partial_timing_is_unchanged_and_new_capture_has_no_added_cooldown(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("core.asr_parakeet.time", SimpleNamespace(monotonic=lambda: clock.now))
    worker = ParakeetLiveWorker(_Model())
    frame = np.ones(4, dtype=np.float32)
    started = threading.Event()
    processed = threading.Event()
    original_append = worker._append

    def observed_append(satellite_id, pcm):
        original_append(satellite_id, pcm)
        if satellite_id == "sat-a":
            started.set()
        else:
            processed.set()

    worker._append = observed_append
    try:
        worker.start("sat-a", frame, capture_id=1)
        assert started.wait(timeout=1)
        clock.now = 100.74
        worker.feed_frame("sat-a", frame, capture_id=1)
        worker.start("barrier", frame, capture_id=1)
        assert processed.wait(timeout=1)
        assert worker.get_event(timeout=0) is None
        clock.now = 100.75
        worker.feed_frame("sat-a", frame, capture_id=1)
        assert worker.get_event(timeout=1) is not None
        clock.now = 101.74
        worker.feed_frame("sat-a", frame, capture_id=1)
        processed.clear()
        worker.start("barrier", frame, capture_id=2)
        assert processed.wait(timeout=1)
        assert worker.get_event(timeout=0) is None
        clock.now = 101.75
        worker.feed_frame("sat-a", frame, capture_id=1)
        assert worker.get_event(timeout=1) is not None
        worker.discard_capture("sat-a", 1)
        started.clear()
        worker.start("sat-a", frame, capture_id=2)
        assert started.wait(timeout=1)
        clock.now = 102.50
        worker.feed_frame("sat-a", frame, capture_id=2)
        assert worker.get_event(timeout=1).item[12] == 2
    finally:
        worker.close()


def test_audio_capture_commit_retains_newer_work_without_rejecting_wake():
    capture = AudioCapture(use_vad=False)
    session = SatelliteSession(id="sat-a")
    dropped = []
    capture.set_asr_work_dropped_callback(lambda *args: dropped.append(args))
    old = _endpoint(np.zeros(4), 1)
    new = _endpoint(np.zeros(4), 2)
    capture._put_utterance(old, satellite_id="sat-a", kind="final")
    capture._put_utterance(new, satellite_id="sat-a", kind="final")
    capture.commit_capture(session, 1)
    assert session.committed_capture_id == 1
    assert session.drop_results_until == 0
    assert capture.audio_queue.get_nowait() is new
    assert capture.audio_queue.empty()
    assert dropped == []


def test_precomputed_live_result_bypasses_model_transcription():
    model = _Model()
    pipe = ParakeetASRPipelineWrapper(model)
    try:
        result = pipe(LiveAsrResult("already decoded", np.zeros(4), 0.12))[0]
        assert result == {"text": "already decoded"}
        assert model.calls == []
        assert pipe.last_transcribe_seconds == 0.12
    finally:
        pipe.live_worker.close()


def test_context_phrases_enable_nemo_tdt_boosting():
    class _DecodingModel:
        def __init__(self):
            self.cfg = SimpleNamespace(decoding={"strategy": "greedy"})
            self.calls = []

        def change_decoding_strategy(self, config, verbose):
            self.calls.append((config, verbose))

    model = _DecodingModel()
    pipe = ParakeetASRPipelineWrapper(model)
    try:
        pipe.set_context_phrases(
            ["hey atticus", "atticus", "Phoebe Bridgers", "hey atticus"],
        )
        config, verbose = model.calls[0]
        assert config.strategy == "greedy_batch"
        assert config.greedy.boosting_tree.key_phrases_list == [
            "hey atticus", "atticus", "Phoebe Bridgers",
        ]
        assert config.greedy.boosting_tree.context_score == 2.0
        assert config.greedy.boosting_tree_alpha == 2.0
        assert verbose is False
        assert pipe.supports_context_unbias is False
    finally:
        pipe.live_worker.close()


def test_context_phrases_do_not_require_omegaconf_in_unit_environment(monkeypatch):
    class _DecodingModel:
        def __init__(self):
            self.cfg = SimpleNamespace(decoding={"strategy": "greedy"})
            self.calls = []

        def change_decoding_strategy(self, config, verbose):
            self.calls.append((config, verbose))

    real_import = __import__

    def without_omegaconf(name, *args, **kwargs):
        if name == "omegaconf":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_omegaconf)
    model = _DecodingModel()
    pipe = ParakeetASRPipelineWrapper(model)
    try:
        pipe.set_context_phrases(["atticus"])
        config, verbose = model.calls[0]
        assert config.strategy == "greedy_batch"
        assert config.greedy.boosting_tree.key_phrases_list == ["atticus"]
        assert verbose is False
    finally:
        pipe.live_worker.close()


def test_live_worker_emits_partial_then_exact_endpoint_final(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    model = _Model()
    worker = ParakeetLiveWorker(model)
    frame = np.ones(4, dtype=np.float32)
    final = np.full(8, 0.5, dtype=np.float32)
    try:
        worker.start("sat-a", frame)
        worker.feed_frame("sat-a", frame)
        partial = worker.get_event(timeout=1)
        assert partial is not None and not partial.final
        assert partial.result.text == "words-8"
        assert partial.result.transcribe_seconds is not None

        worker.finish((final, 1.0, -20.0, False, "sat-a", 2.0))
        endpoint = worker.get_event(timeout=1)
        assert endpoint is not None and endpoint.final
        np.testing.assert_array_equal(endpoint.result.pcm, final)
        assert endpoint.result.text == "words-8"
        assert endpoint.result.transcribe_seconds is not None
    finally:
        worker.close()


def test_live_worker_discards_abandoned_capture(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 60)
    worker = ParakeetLiveWorker(_Model())
    try:
        worker.start("sat-a", np.ones(4, dtype=np.float32))
        worker.discard("sat-a")
        # Commands are processed in FIFO order by the single model worker.
        time.sleep(0.05)
        assert "sat-a" not in worker._buffers
    finally:
        worker.close()


def test_live_worker_discard_invalidates_queued_and_inflight_partial(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)

    class _BlockingModel:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def transcribe(self, audio, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(timeout=1)
            return [type("Hypothesis", (), {"text": f"call-{self.calls}"})()]

    model = _BlockingModel()
    worker = ParakeetLiveWorker(model)
    frame = np.ones(4, dtype=np.float32)
    try:
        worker.start("sat-a", frame)
        worker.feed_frame("sat-a", frame)
        assert model.started.wait(timeout=1)
        worker.feed_frame("sat-a", frame)  # stale queued partial
        worker.discard("sat-a")
        model.release.set()
        assert worker.get_event(timeout=0.1) is None

        worker.start("sat-a", frame)
        worker.feed_frame("sat-a", frame)
        event = worker.get_event(timeout=1)
        assert event is not None
        assert event.result.text == "call-2"
    finally:
        worker.close()


def test_live_worker_ignores_idle_frames_and_bounds_partial_window(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_WINDOW_SECONDS", 0.001)
    model = _Model()
    worker = ParakeetLiveWorker(model)
    frame = np.ones(10, dtype=np.float32)
    try:
        worker.feed_frame("sat-a", frame)
        time.sleep(0.05)
        assert model.calls == []

        worker.start("sat-a", frame)
        worker.feed_frame("sat-a", frame)
        event = worker.get_event(timeout=1)
        assert event is not None and not event.final
        assert len(model.calls[0]) == 16  # 1 ms rolling window at 16 kHz
    finally:
        worker.close()


def test_live_worker_reports_cuda_failure_once_and_stops(monkeypatch):
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INITIAL_SECONDS", 0)
    monkeypatch.setattr("core.asr_parakeet.PARTIAL_INTERVAL_SECONDS", 0)

    class _FailingModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    worker = ParakeetLiveWorker(_FailingModel())
    try:
        worker.start("sat-a", np.ones(4, dtype=np.float32))
        worker.feed_frame("sat-a", np.ones(4, dtype=np.float32))
        event = worker.get_event(timeout=1)
        assert isinstance(event, LiveAsrFailure)
        assert "illegal memory access" in str(event.error)
        assert worker._closed.is_set()
    finally:
        worker.close()


def test_audio_capture_sends_endpoints_to_live_worker_not_queue():
    class _Worker:
        def __init__(self):
            self.items = []

        def finish(self, item):
            self.items.append(item)
            return True

    capture = AudioCapture(use_vad=False)
    worker = _Worker()
    capture.set_live_asr_backend(worker)
    item = (np.zeros(4), 1.0, -20.0, False, "sat-a", time.monotonic())

    assert capture._put_utterance(item, satellite_id="sat-a", kind="final")
    assert worker.items == [item]
    assert capture.audio_queue.empty()


def test_audio_capture_routes_wake_verification_through_live_worker():
    class _Worker:
        def __init__(self):
            self.items = []

        def finish(self, item):
            self.items.append(item)
            return True

    capture = AudioCapture(use_vad=False)
    worker = _Worker()
    capture.set_live_asr_backend(worker)
    item = (np.zeros(4), 1.0, -20.0, False, "sat-a", time.monotonic(), False, True)

    assert capture._put_utterance(item, satellite_id="sat-a", kind="wake_verification", candidate=True)
    assert worker.items == [item]
    assert capture.audio_queue.empty()
