"""Unit seams for Parakeet's live-frame worker; no NeMo runtime required."""

import time
from types import SimpleNamespace

import numpy as np

from core.asr_parakeet import (
    LiveAsrFailure,
    LiveAsrResult,
    ParakeetASRPipelineWrapper,
    ParakeetLiveWorker,
)
from core.audio import AudioCapture


class _Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **_kwargs):
        self.calls.append(audio[0].copy())
        return [type("Hypothesis", (), {"text": f"words-{len(audio[0])}"})()]


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
