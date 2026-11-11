"""Backend registry and no-LLM gate.

`core.backends` is the single source of truth mapping `(domain, backend)` to
a loader + metadata. These tests cover the default Qwen stack when `models:`
is absent, error behaviour, and the regex-only
bypass `AgentLoop` takes when `llm.backend: none`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import backends as b  # noqa: E402


def test_defaults_use_the_pytorch_gpu_stack():
    r = b.resolve_models(None)
    assert r["asr"]["backend"] == "qwen"
    assert r["tts"]["backend"] == "qwen"
    assert r["llm"]["backend"] == "llama"
    # model ids / context default from the registry metadata
    assert r["asr"]["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert r["llm"]["model"].endswith("Qwen3.5-9B-UD-Q4_K_XL.gguf")
    assert r["llm"]["spec"].hf_repo == "unsloth/Qwen3.5-9B-MTP-GGUF"
    assert r["llm"]["opts"]["mtp"] is False
    assert r["llm"]["opts"]["flash_attn"] is False
    assert r["llm"]["n_context"] == 12288


def test_empty_block_falls_back_per_domain():
    # A models block that only sets the LLM leaves ASR/TTS on PyTorch GPU defaults.
    r = b.resolve_models({"llm": {"backend": "none"}})
    assert r["asr"]["backend"] == "qwen"
    assert r["llm"]["backend"] == "none"
    assert r["llm"]["model"] is None  # no default_model for the bypass


def test_laya_resolves_as_a_cpu_semantic_command_backend():
    resolved = b.resolve_models({"llm": {"backend": "laya"}})["llm"]

    assert resolved["backend"] == "laya"
    assert resolved["model"] == "./data/models/laya"
    assert resolved["opts"]["confidence_threshold"] == 0.90
    assert resolved["spec"].cpu_ok is True


def test_stats_ui_renders_laya_semantic_routing():
    ui = Path("server/static/js/chat-ui.js").read_text()

    assert "Semantic Routing" in ui
    assert "s.laya" in ui


def test_model_override_and_opts_passthrough():
    r = b.resolve_models({"asr": {"backend": "qwen", "model": "X", "foo": 1}})
    assert r["asr"]["model"] == "X"
    # leftover keys are forwarded to the loader as opts
    assert r["asr"]["opts"] == {"foo": 1}


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        b.get_spec("asr", "does-not-exist")


def test_loaderless_loader_raises():
    # The no-LLM bypass has no loader by design.
    with pytest.raises(ValueError):
        b.get_loader("llm", "none")


def test_backends_are_implemented():
    # The no-LLM bypass is a deliberate loaderless backend; everything else
    # (including the remote OpenAI path, wired in Step 6) has a loader.
    assert b.get_spec("llm", "none").implemented is True
    assert b.get_spec("llm", "openai").implemented is True
    assert b.get_spec("asr", "qwen").implemented is True
    assert b.get_spec("asr", "parakeet").implemented is True
    assert b.get_spec("asr", "parakeet-onnx").implemented is True
    assert b.get_spec("asr", "orukeet").implemented is True
    assert b.get_spec("asr", "moonshine").implemented is True
    assert b.get_spec("tts", "kokoro-onnx").implemented is True
    assert b.get_spec("tts", "pocket-tts-onnx").implemented is True
    assert b.get_spec("tts", "pocket-tts-gguf").implemented is True
    assert b.get_spec("tts", "pocket-tts-pytorch").implemented is True
    assert b.get_spec("tts", "audio8").implemented is True
    assert b.get_loader("llm", "openai")  # resolves to load_openai


def test_gpu_only_flags():
    assert b.get_spec("asr", "qwen").gpu_only is True
    assert b.get_spec("asr", "parakeet").gpu_only is True
    assert b.get_spec("asr", "parakeet-onnx").gpu_only is False
    assert b.get_spec("asr", "orukeet").gpu_only is True
    assert b.get_spec("tts", "qwen").gpu_only is True
    assert b.get_spec("llm", "llama").gpu_only is True
    assert b.get_spec("llm", "gemma").gpu_only is True
    assert b.get_spec("llm", "ornith").gpu_only is True
    assert b.get_spec("llm", "qwen27b").gpu_only is True
    assert b.get_spec("asr", "moonshine").gpu_only is False
    assert b.get_spec("tts", "kokoro-onnx").gpu_only is False
    assert b.get_spec("tts", "pocket-tts-onnx").gpu_only is False
    assert b.get_spec("tts", "pocket-tts-gguf").gpu_only is True
    assert b.get_spec("tts", "pocket-tts-pytorch").gpu_only is False
    assert b.get_spec("tts", "audio8").gpu_only is True
    assert b.get_spec("llm", "openai").gpu_only is False
    assert b.get_spec("asr", "qwen-gguf").gpu_only is True
    assert b.get_spec("tts", "qwen-gguf").gpu_only is True
    assert b.get_spec("asr", "qwen-gguf-small").gpu_only is True
    assert b.get_spec("tts", "qwen-gguf-small").gpu_only is True
    assert b.get_spec("tts", "higgs-gguf").gpu_only is True
    assert b.get_spec("tts", "breeze-tts-2-gguf").gpu_only is True


def test_gpu_crispasr_backend_metadata():
    asr = b.get_spec("asr", "qwen-gguf")
    tts = b.get_spec("tts", "qwen-gguf")
    assert asr.loader == "core.asr_crispasr:load_asr_model"
    assert asr.hf_file == "qwen3-asr-1.7b-q4_k.gguf"
    assert tts.loader == "core.tts_crispasr:load_tts"
    assert len(tts.hf_files) == 2
    assert tts.hf_files[0][1] == "qwen3-tts-12hz-1.7b-base-f16.gguf"
    assert tts.extra == {"gpu": True}
    assert b.resolve_models({"tts": {"backend": "qwen-gguf"}})["tts"]["opts"]["gpu"] is True


def test_higgs_backend_metadata_and_cpu_gating():
    higgs = b.get_spec("tts", "higgs-gguf")
    assert higgs.loader == "core.tts_higgs:load_tts"
    assert higgs.hf_files[0][0] == "liampetti/HiggsTTS3.gguf"
    assert higgs.hf_files[0][1] == "higgs-v3-tts-q4_k.gguf"
    assert higgs.extra["max_actions"] == 256
    assert b.is_offerable(higgs, "gpu") is True
    assert b.is_offerable(higgs, "cpu") is False


def test_breeze_backend_metadata_and_cpu_gating():
    breeze = b.get_spec("tts", "breeze-tts-2-gguf")
    assert breeze.loader == "core.tts_crispasr:load_tts"
    assert breeze.hf_files == (
        ("cstr/breeze-tts-2-GGUF", "breeze-tts-2-q4_k.gguf"),
        ("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf"),
    )
    assert breeze.extra == {"gpu": True, "backend": "bt2-tts"}
    assert b.is_offerable(breeze, "gpu") is True
    assert b.is_offerable(breeze, "cpu") is False


def test_pocket_gguf_backend_metadata_and_cpu_gating():
    pocket = b.get_spec("tts", "pocket-tts-gguf")
    assert pocket.loader == "core.tts_crispasr:load_tts"
    assert pocket.hf_repo == "cstr/pocket-tts-GGUF"
    assert pocket.hf_file == "pocket-tts-english-q8_0.gguf"
    assert pocket.extra == {"gpu": True, "backend": "pocket-tts"}
    assert b.is_offerable(pocket, "gpu") is True
    assert b.is_offerable(pocket, "cpu") is False


def test_pocket_pytorch_backend_metadata_and_cpu_gating():
    pocket = b.get_spec("tts", "pocket-tts-pytorch")
    assert pocket.loader == "core.tts_pocket:load_tts"
    assert pocket.hf_snapshots[0][0] == "lookbe/pocket-tts-6l"
    assert pocket.hf_snapshots[0][1] == ("english/model.safetensors", "english/tokenizer.model")
    assert b.is_offerable(pocket, "gpu") is True
    assert b.is_offerable(pocket, "cpu") is True


def test_orukeet_backend_metadata_and_cpu_gating():
    orukeet = b.get_spec("asr", "orukeet")
    assert orukeet.loader == "core.asr_orukeet:load_asr_model"
    assert orukeet.hf_snapshots[0] == (
        "oruk/orukeet",
        ("orukeet-v0.1.0.nemo",),
        "555136b50265a132d4cea0d35560c26fc4f657ab",
    )
    assert b.is_offerable(orukeet, "gpu") is True
    assert b.is_offerable(orukeet, "cpu") is False


def test_audio8_backend_metadata_and_cpu_gating():
    audio8 = b.get_spec("tts", "audio8")
    assert audio8.loader == "core.tts_audio8:load_tts"
    assert audio8.hf_snapshots[0][0] == "Edge0/Audio8-TTS-Realistic-Air"
    assert audio8.extra["revision"] == audio8.hf_snapshots[0][2]
    assert b.is_offerable(audio8, "gpu") is True
    assert b.is_offerable(audio8, "cpu") is False

    small = b.get_spec("tts", "audio8-small")
    assert small.loader == "core.tts_audio8:load_tts"
    assert small.default_model == "Edge0/Audio8-TTS-Preview-0.1b"
    assert small.extra["revision"] == small.hf_snapshots[0][2]
    assert small.voice_cloning is True
    assert b.is_offerable(small, "gpu") is True
    assert b.is_offerable(small, "cpu") is False
    resolved = b.resolve_models({"tts": {"backend": "audio8-small"}})["tts"]
    assert resolved["model"] == small.default_model
    assert resolved["opts"]["revision"] == small.extra["revision"]


def test_small_gguf_backend_metadata():
    asr = b.get_spec("asr", "qwen-gguf-small")
    tts = b.get_spec("tts", "qwen-gguf-small")
    assert asr.hf_file == "qwen3-asr-0.6b-q4_k.gguf"
    assert tts.hf_files[0][1] == "qwen3-tts-12hz-0.6b-base-q8_0.gguf"
    assert b.resolve_models({"tts": {"backend": "qwen-gguf-small"}})["tts"]["opts"]["gpu"] is True


def test_gemma_backend_metadata():
    # Gemma 4 is an alternative local SLM through the bundled llama-server.
    g = b.get_spec("llm", "gemma")
    assert g.implemented is True
    assert g.loader == "core.slm:load_slm"  # same loader/contract as Qwen
    assert g.hf_repo == "unsloth/gemma-4-12B-it-qat-GGUF"
    assert g.hf_file.endswith(".gguf")
    assert g.default_model.endswith(g.hf_file)
    assert g.n_context and g.n_context > 0
    assert g.extra == {"mtp": False}


def test_gemma_resolves_with_registry_defaults():
    r = b.resolve_models({"llm": {"backend": "gemma"}})
    assert r["llm"]["backend"] == "gemma"
    assert r["llm"]["model"].endswith("gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    assert r["llm"]["opts"]["mtp"] is False


def test_ornith_backend_metadata_and_public_resolution():
    ornith = b.get_spec("llm", "ornith")
    assert ornith.loader == "core.slm:load_slm"
    assert ornith.hf_repo == "ornith-ai/Ornith-1.5-9B-GGUF"
    assert ornith.hf_file == "Ornith-1.5-9B-Q4_K_M.gguf"
    assert ornith.extra == {"mtp": False}

    resolved = b.resolve_models({"llm": {"backend": "local", "local_model": "ornith"}})["llm"]
    assert resolved["backend"] == "ornith"
    assert resolved["model"].endswith(ornith.hf_file)
    assert resolved["opts"]["mtp"] is False


def test_qwen27b_backend_metadata_and_public_resolution():
    qwen27b = b.get_spec("llm", "qwen27b")
    assert qwen27b.loader == "core.slm:load_slm"
    assert qwen27b.hf_repo == "unsloth/Qwen3.8-27B-GGUF"
    assert qwen27b.hf_file == "Qwen3.8-27B-UD-IQ4_XS.gguf"
    assert qwen27b.experimental is True
    assert qwen27b.extra == {"mtp": False}

    resolved = b.resolve_models({"llm": {"backend": "local", "local_model": "qwen27b"}})["llm"]
    assert resolved["backend"] == "qwen27b"
    assert resolved["model"].endswith(qwen27b.hf_file)
    assert resolved["opts"]["mtp"] is False


def test_public_local_llm_modes_resolve_to_their_internal_servers():
    qwen = b.resolve_models({"llm": {"backend": "local", "local_model": "qwen"}})["llm"]
    gemma = b.resolve_models({"llm": {"backend": "local", "local_model": "gemma"}})["llm"]
    ornith = b.resolve_models({"llm": {"backend": "local", "local_model": "ornith"}})["llm"]
    qwen27b = b.resolve_models({"llm": {"backend": "local", "local_model": "qwen27b"}})["llm"]
    external = b.resolve_models({"llm": {"backend": "external", "base_url": "http://llm/v1"}})[
        "llm"
    ]

    assert qwen["backend"] == "llama"
    assert qwen["opts"]["mtp"] is False
    assert qwen["opts"]["flash_attn"] is False
    assert gemma["backend"] == "gemma"
    assert ornith["backend"] == "ornith"
    assert qwen27b["backend"] == "qwen27b"
    assert external["backend"] == "openai"
    assert external["opts"]["base_url"] == "http://llm/v1"


def test_model_docs_and_settings_cover_the_registry():
    root = Path(__file__).parent.parent
    example = (root / "data/config.example.yml").read_text()
    settings = (root / "server/static/js/setup-settings.js").read_text()

    # ASR/TTS selects are registry-driven, so every offerable backend returned
    # by /settings is selectable without a second hard-coded menu.
    assert "optsFor('asr'" in settings
    assert "optsFor('tts'" in settings
    assert "spec?.voice_cloning === true" in settings
    for domain in (b.ASR, b.TTS):
        for spec in b.list_backends(domain):
            assert spec.backend in example

    # Local LLMs have public config aliases rather than their internal server
    # backend names; keep the explicit selector and config docs in sync.
    for local_model in ("qwen", "gemma", "ornith", "qwen27b", "custom"):
        assert f"local_model: {local_model}" in example
    # Model choices are now registry-driven, including image-variant filtering.
    assert "optsFor('llm')" in settings
    assert "state.schema.variant === 'cpu' && b.gpu_only" in settings
    assert "backend: local             # local | external | laya | none" in example
    assert "previous.backend || 'external'" in settings


def test_local_llm_experimental_options_are_explicit_overrides():
    llm = b.resolve_models(
        {"llm": {"backend": "local", "local_model": "qwen", "mtp": True, "flash_attn": True}}
    )["llm"]

    assert llm["opts"]["mtp"] is True
    assert llm["opts"]["flash_attn"] is True


@pytest.mark.parametrize("size", [0, 512, 1024, 2048])
def test_local_prompt_cache_config_is_forwarded(size):
    llm = b.resolve_models({"llm": {"backend": "local", "prompt_cache_mib": size}})["llm"]
    assert llm["opts"]["prompt_cache_mib"] == size


@pytest.mark.parametrize("size", [-1, True, 1.5, "1024", None, 2**31])
def test_local_prompt_cache_rejects_invalid_values(size):
    with pytest.raises(ValueError, match="prompt_cache_mib"):
        b.resolve_models({"llm": {"backend": "local", "prompt_cache_mib": size}})


def test_variant_reads_env(monkeypatch):
    monkeypatch.delenv("FULLOCH_VARIANT", raising=False)
    assert b.variant() == "gpu"
    monkeypatch.setenv("FULLOCH_VARIANT", "cpu")
    assert b.variant() == "cpu"
    monkeypatch.setenv("FULLOCH_VARIANT", "nonsense")
    assert b.variant() == "gpu"  # unknown -> default


def test_is_offerable_by_variant():
    qwen = b.get_spec("asr", "qwen")
    moon = b.get_spec("asr", "moonshine")
    # GPU image offers everything implemented; CPU image hides gpu_only.
    assert b.is_offerable(qwen, "gpu") is True
    assert b.is_offerable(qwen, "cpu") is False
    assert b.is_offerable(moon, "cpu") is True
    assert b.is_offerable(b.get_spec("asr", "parakeet-onnx"), "cpu") is True
    assert b.is_offerable(b.get_spec("asr", "qwen-gguf"), "cpu") is False
    assert b.is_offerable(b.get_spec("llm", "openai"), "cpu") is True


def test_list_backends_covers_each_domain():
    for domain in (b.ASR, b.TTS, b.LLM):
        names = {s.backend for s in b.list_backends(domain)}
        assert names, domain
    assert "none" in {s.backend for s in b.list_backends(b.LLM)}


def test_asr_and_tts_options_have_stable_user_facing_order():
    assert [spec.backend for spec in b.list_backends(b.ASR)] == [
        "parakeet",
        "orukeet",
        "qwen-gguf",
        "qwen-gguf-small",
        "qwen-onnx",
        "qwen-onnx-small",
        "parakeet-onnx",
        "qwen",
        "qwen-small",
        "moonshine",
        "moonshine-tiny",
    ]
    assert [spec.backend for spec in b.list_backends(b.TTS)] == [
        "higgs-gguf",
        "breeze-tts-2-gguf",
        "qwen-gguf",
        "qwen-gguf-small",
        "pocket-tts-pytorch",
        "audio8",
        "audio8-small",
        "pocket-tts-gguf",
        "qwen",
        "qwen-small",
        "kokoro-onnx",
        "pocket-tts-onnx",
    ]
    assert b.get_spec("asr", "qwen").display_name == "Qwen3 1.7B PyTorch (default GPU)"


# --- no-LLM bypass in the agent loop ---------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ("Lights on.", "Lights on."),
        (
            "Reactive question: Couldn't find that room. Ask the user which room.",
            "Couldn't find that room.",
        ),
        ("User question: raw search data", "Basic commands only. Not completed: example tool."),
        (None, "Basic commands only. Not completed: example tool."),
    ],
)
def test_regex_only_dispatch_never_calls_model(monkeypatch, result, expected):
    import core.agent_loop as al

    action = {"intent": "example_tool", "args": ["kitchen"]}
    monkeypatch.setattr(al, "catchAll", lambda prompt: {"actions": [action]})
    dispatch = Mock(return_value=result)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    history = []
    host = SimpleNamespace(
        llm_enabled=False,
        _history_for=lambda satellite: history,
        _compact_completed_turns=Mock(),
        _trim_history=Mock(),
        _emit_agent_event=Mock(),
        _record_spoken=Mock(),
        _speak_no_ai_fallback=Mock(return_value="Basic commands only."),
        _generate_with_context_recovery=Mock(),
    )

    assert al.AgentLoop(host, source="text").run("A command") == expected
    dispatch.assert_called_once_with(action)
    host._generate_with_context_recovery.assert_not_called()
    assert history[-1] == {
        "role": "tool",
        "name": "example_tool",
        "content": result if result is not None else "<error>",
    }
