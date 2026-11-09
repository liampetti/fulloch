"""Contract tests for the official Pocket TTS PyTorch adapter."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.tts_pocket as tts  # noqa: E402
from core.tts_session import TtsSession  # noqa: E402


def test_load_tts_uses_selected_mirror_and_device(monkeypatch, tmp_path):
    calls = {}

    class Model:
        @classmethod
        def load_model(cls, **kwargs):
            calls.update(kwargs)
            calls["config_contents"] = Path(kwargs["config"]).read_text()
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            pass

    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=Model))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "french_24l.yaml").write_text(
        "weights_path: hf://kyutai/pocket-tts/languages/french_24l/model.safetensors@revision\n"
        "weights_path_without_voice_cloning: hf://fallback\n"
        "  tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/french_24l/tokenizer.model@revision\n"
    )
    monkeypatch.setattr(tts.importlib.resources, "files", lambda _: tmp_path)

    tts.load_tts(language="french_24l", device="cpu")

    assert calls["temp"] == 0.7 and calls["lsd_decode_steps"] == 1
    assert f"hf://{tts.MIRROR_REPO}/languages/french_24l/model.safetensors@{tts.MIRROR_REVISION}" in calls["config_contents"]
    assert "weights_path_without_voice_cloning" not in calls["config_contents"]
    assert f"hf://{tts.MIRROR_REPO}/languages/french_24l/tokenizer.model@{tts.MIRROR_REVISION}" in calls["config_contents"]


@pytest.mark.parametrize("language", ["english_2026-04", "french_24l", "german", "italian", "portuguese", "spanish"])
def test_bundled_language_configs_use_mirror_for_weights_and_tokenizer(monkeypatch, language):
    from importlib import resources

    pytest.importorskip("pocket_tts")
    configs = resources.files("pocket_tts")
    calls = {}

    class Model:
        @classmethod
        def load_model(cls, **kwargs):
            calls["config"] = Path(kwargs["config"]).read_text()
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            pass

    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(tts.importlib.resources, "files", lambda _: configs)
    tts.load_tts(language=language)
    assert calls["device"] == "cpu"
    assert f"languages/{language}/model.safetensors@{tts.MIRROR_REVISION}" in calls["config"]
    assert f"languages/{language}/tokenizer.model@{tts.MIRROR_REVISION}" in calls["config"]
    assert "weights_path_without_voice_cloning" not in calls["config"]


def test_unsupported_speed_is_ignored_with_explicit_warning(caplog):
    tts.set_speed(1.4)

    assert "has no speed control" in caplog.text
    assert "ignored" in caplog.text


def test_pocket_rollover_keeps_first_sentence_early(monkeypatch):
    monkeypatch.setattr(tts, "_MAX_SYNTHESIS_CHARS", 80)

    fragments = list(
        tts._synthesis_fragments(
            "Hello, there friend. The weather is warm, dry, and clear. It stays pleasant tonight."
        )
    )

    assert fragments == [
        "Hello, there friend.",
        "The weather is warm, dry, and clear. It stays pleasant tonight.",
    ]


def test_pocket_native_stream_forwards_pcm_without_waiting_for_completion(monkeypatch):
    calls = []

    class Audio:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def to(self, **kwargs):
            calls.append(kwargs)
            return self

        def numpy(self):
            return np.asarray(self.values, dtype=np.float32)

    class Model:
        def generate_audio_stream(self, **kwargs):
            assert kwargs == {"model_state": "voice", "text_to_generate": "Hello"}
            yield Audio([0.1, 0.2])
            yield Audio([0.3])

    monkeypatch.setattr(tts, "_model", Model())
    monkeypatch.setattr(tts, "_torch", SimpleNamespace(float32="float32"))

    chunks = list(tts._stream("Hello", "voice"))

    np.testing.assert_allclose(chunks[0], [0.1, 0.2])
    np.testing.assert_allclose(chunks[1], [0.3])
    assert calls == [
        {"device": "cpu", "dtype": "float32"},
        {"device": "cpu", "dtype": "float32"},
    ]


@pytest.mark.parametrize("locale,expected", [
    ("en-US", "english_2026-04"), ("fr-FR", "french_24l"), ("de", "german"),
    ("it", "italian"), ("pt-BR", "portuguese"), ("es", "spanish"), ("sv", None),
])
def test_locale_maps_only_supported_models(locale, expected):
    assert tts._language_model(locale) == expected


def test_uncached_language_falls_back_and_requests_download(monkeypatch, tmp_path):
    from unittest.mock import Mock

    default, default_voice = object(), object()
    monkeypatch.setattr(tts, "_model", default)
    monkeypatch.setattr(tts, "_default_language", "english_2026-04")
    monkeypatch.setattr(tts, "_alternate", None)
    monkeypatch.setattr(tts, "_load_failures", {})
    monkeypatch.setattr(tts, "_weights_cached", lambda _: False)
    download = Mock()
    monkeypatch.setattr(tts, "_request_language_download", download)
    job = tts._Job("bonjour", default_voice, None, TtsSession(), "fr", tmp_path / "voice.wav")

    assert tts._select_job_model(job) == (default, default_voice)
    download.assert_called_once_with("french_24l")


def test_alternate_loads_on_cpu_and_is_reused_without_cuda_model(monkeypatch, tmp_path):
    from unittest.mock import Mock

    default, default_voice = object(), object()
    voice_path = tmp_path / "voice.wav"
    alternate = SimpleNamespace(get_state_for_audio_prompt=Mock(return_value="french voice"))
    load = Mock(return_value=alternate)
    monkeypatch.setattr(tts, "_model", default)
    monkeypatch.setattr(tts, "_default_language", "english_2026-04")
    monkeypatch.setattr(tts, "_alternate", None)
    monkeypatch.setattr(tts, "_load_failures", {})
    monkeypatch.setattr(tts, "_load_options", {"temperature": 0.6})
    monkeypatch.setattr(tts, "_weights_cached", lambda _: True)
    monkeypatch.setattr(tts, "_enough_alternate_ram", lambda: True)
    monkeypatch.setattr(tts, "_load_model", load)
    job = tts._Job("bonjour", default_voice, None, TtsSession(), "fr-FR", voice_path)

    assert tts._select_job_model(job) == (alternate, "french voice")
    assert tts._select_job_model(job) == (alternate, "french voice")
    load.assert_called_once_with("french_24l", "cpu", {"temperature": 0.6})
    alternate.get_state_for_audio_prompt.assert_called_once_with(voice_path)
    assert tts._model is default


def test_new_language_replaces_cpu_alternate_and_custom_voice_falls_back(monkeypatch, tmp_path):
    from unittest.mock import Mock

    default, old_model, voice = object(), object(), object()
    new_model = SimpleNamespace(get_state_for_audio_prompt=Mock(return_value="new voice"))
    old_path = tmp_path / "old.wav"
    load = Mock(return_value=new_model)
    monkeypatch.setattr(tts, "_model", default)
    monkeypatch.setattr(tts, "_default_language", "english_2026-04")
    monkeypatch.setattr(tts, "_alternate", ("german", old_path, old_model, "old voice"))
    monkeypatch.setattr(tts, "_load_failures", {})
    monkeypatch.setattr(tts, "_weights_cached", lambda _: True)
    monkeypatch.setattr(tts, "_enough_alternate_ram", lambda: True)
    monkeypatch.setattr(tts, "_load_model", load)
    custom_job = tts._Job("bonjour", voice, None, TtsSession(), "fr", None)
    assert tts._select_job_model(custom_job) == (default, voice)
    load.assert_not_called()
    job = tts._Job("bonjour", voice, None, TtsSession(), "fr", tmp_path / "new.wav")
    assert tts._select_job_model(job) == (new_model, "new voice")
    assert tts._alternate[:2] == ("french_24l", tmp_path / "new.wav")
    assert load.call_args.args[1] == "cpu"


def test_low_ram_and_load_failure_retain_default(monkeypatch, tmp_path):
    from unittest.mock import Mock

    default, voice = object(), object()
    monkeypatch.setattr(tts, "_model", default)
    monkeypatch.setattr(tts, "_default_language", "english_2026-04")
    monkeypatch.setattr(tts, "_alternate", None)
    monkeypatch.setattr(tts, "_load_failures", {})
    monkeypatch.setattr(tts, "_weights_cached", lambda _: True)
    load = Mock(side_effect=RuntimeError("load failed"))
    monkeypatch.setattr(tts, "_load_model", load)
    monkeypatch.setattr(tts, "_enough_alternate_ram", lambda: False)
    job = tts._Job("bonjour", voice, None, TtsSession(), "fr", tmp_path / "voice.wav")
    assert tts._select_job_model(job) == (default, voice)
    load.assert_not_called()
    monkeypatch.setattr(tts, "_enough_alternate_ram", lambda: True)
    assert tts._select_job_model(job) == (default, voice)
    assert tts._alternate is None
    assert tts._select_job_model(job) == (default, voice)
    load.assert_called_once()


def test_ram_headroom_respects_cgroup_limit(monkeypatch):
    from core import memory_guard

    gib = 1024 ** 3
    monkeypatch.setattr(memory_guard, "snapshot", lambda: memory_guard.MemorySnapshot(12*gib, 11*gib, 10*gib, 0))
    assert not tts._enough_alternate_ram()
    monkeypatch.setattr(memory_guard, "snapshot", lambda: memory_guard.MemorySnapshot(12*gib, 8*gib, 10*gib, 0))
    assert tts._enough_alternate_ram()


def test_language_download_is_single_flight_and_offline_failure_is_throttled(monkeypatch):
    from unittest.mock import Mock

    threads = []
    monkeypatch.setattr(tts, "_download_active", None)
    monkeypatch.setattr(tts, "_download_failures", {})
    monkeypatch.setattr(tts.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(tts.threading, "Thread", lambda **kw: threads.append(kw) or SimpleNamespace(start=Mock()))
    tts._request_language_download("french_24l")
    tts._request_language_download("german")
    assert len(threads) == 1
    monkeypatch.setattr(tts.subprocess, "run", Mock(side_effect=OSError("offline")))
    threads[0]["target"](*threads[0]["args"])
    assert tts._download_active is None
    tts._request_language_download("french_24l")
    assert len(threads) == 1


def test_background_download_uses_pinned_weights_and_tokenizer(monkeypatch):
    from unittest.mock import Mock

    download = Mock()
    monkeypatch.setattr(tts, "_download_active", "french_24l")
    monkeypatch.setattr(tts, "_download_failures", {})
    monkeypatch.setattr(tts, "_weights_cached", lambda _: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(tts.subprocess, "run", download)

    tts._download_language("french_24l")

    args, kwargs = download.call_args
    assert args[0][:2] == [sys.executable, "-c"]
    assert args[0][3:] == [tts.MIRROR_REPO, tts.MIRROR_REVISION,
                           "languages/french_24l/model.safetensors", "languages/french_24l/tokenizer.model"]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "0"
    assert tts.os.environ["HF_HUB_OFFLINE"] == "1"
    assert kwargs["check"] is True
    assert tts._download_active is None


def test_submit_captures_language_and_matching_voice_for_each_job(monkeypatch, tmp_path):
    from unittest.mock import Mock

    voice, path = object(), tmp_path / "voice.wav"
    monkeypatch.setattr(tts, "_voice_binding", (voice, path))
    submit = Mock()
    monkeypatch.setattr(tts, "submit_tts_job", submit)
    session = TtsSession()

    tts._submit("bonjour", voice, session, language="fr-FR")
    job = submit.call_args.args[1]
    assert job.language == "fr-FR"
    assert job.voice_state is voice
    assert job.voice_path == path
    tts._submit("bonjour", object(), session, language="fr")
    assert submit.call_args.args[1].voice_path is None


def test_streaming_worker_uses_alternate_then_default_without_extra_gpu_model(monkeypatch, tmp_path):
    import queue
    from unittest.mock import Mock

    default, voice = object(), object()
    alternate = SimpleNamespace(get_state_for_audio_prompt=Mock(return_value="french state"))
    calls = []
    monkeypatch.setattr(tts, "_model", default)
    monkeypatch.setattr(tts, "_default_language", "english_2026-04")
    monkeypatch.setattr(tts, "_alternate", None)
    monkeypatch.setattr(tts, "_load_failures", {})
    monkeypatch.setattr(tts, "_voice_binding", (voice, tmp_path / "voice.wav"))
    monkeypatch.setattr(tts, "_weights_cached", lambda _: True)
    monkeypatch.setattr(tts, "_enough_alternate_ram", lambda: True)
    load = Mock(return_value=alternate)
    monkeypatch.setattr(tts, "_load_model", load)

    def stream(text, state, model):
        calls.append((text, state, model))
        yield np.array([0.1], dtype=np.float32)

    monkeypatch.setattr(tts, "_stream", stream)
    sink = queue.Queue()
    tts.speak_stream("bonjour", voice, sink=sink, language="fr")
    assert sink.get_nowait() == ("start", tts.SAMPLE_RATE)
    assert sink.get_nowait()[0].shape == (1,)
    assert sink.get_nowait() == ("end",)
    tts.speak_stream("hello", voice, sink=queue.Queue(), language="en")

    assert calls == [("bonjour", "french state", alternate), ("hello", voice, default)]
    assert load.call_args.args[1] == "cpu"
    assert tts._model is default


def test_clear_foreign_output_can_override_english_detection_fallback(monkeypatch):
    from utils import response_language

    monkeypatch.setattr(response_language, "detect_response_language", lambda _: "fr")
    text = "Bonjour, je voudrais pratiquer cette langue avec vous chaque jour."
    assert tts._speech_language(text, "en") == "fr"
    assert tts._speech_language("Bonjour", "en") == "en"
    assert tts._speech_language(text, "de") == "de"
