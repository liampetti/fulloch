"""Backend registry + no-LLM gate (v2.2 Step 1).

`core.backends` is the single source of truth mapping `(domain, backend)` to
a loader + metadata. These tests lock in the default resolution (the v2.1.9
Qwen stack when `models:` is absent), error behaviour, and the regex-only
bypass `AgentLoop` takes when `llm.backend: none`.
"""

import inspect
import sys
from pathlib import Path

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
    assert r["llm"]["opts"]["mtp"] is True
    assert r["llm"]["n_context"] == 12288


def test_empty_block_falls_back_per_domain():
    # A models block that only sets the LLM leaves ASR/TTS on PyTorch GPU defaults.
    r = b.resolve_models({"llm": {"backend": "none"}})
    assert r["asr"]["backend"] == "qwen"
    assert r["llm"]["backend"] == "none"
    assert r["llm"]["model"] is None  # no default_model for the bypass


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
    assert b.get_spec("asr", "moonshine").implemented is True
    assert b.get_spec("tts", "kokoro-onnx").implemented is True
    assert b.get_spec("tts", "pocket-tts-onnx").implemented is True
    assert b.get_loader("llm", "openai")  # resolves to load_openai


def test_gpu_only_flags():
    assert b.get_spec("asr", "qwen").gpu_only is True
    assert b.get_spec("tts", "qwen").gpu_only is True
    assert b.get_spec("llm", "llama").gpu_only is True
    assert b.get_spec("llm", "gemma").gpu_only is True
    assert b.get_spec("asr", "moonshine").gpu_only is False
    assert b.get_spec("tts", "kokoro-onnx").gpu_only is False
    assert b.get_spec("tts", "pocket-tts-onnx").gpu_only is False
    assert b.get_spec("llm", "openai").gpu_only is False
    assert b.get_spec("asr", "qwen-gguf").gpu_only is True
    assert b.get_spec("tts", "qwen-gguf").gpu_only is True
    assert b.get_spec("asr", "qwen-gguf-small").gpu_only is True
    assert b.get_spec("tts", "qwen-gguf-small").gpu_only is True
    assert b.get_spec("tts", "higgs-gguf").gpu_only is True


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


def test_public_local_llm_modes_resolve_to_their_internal_servers():
    qwen = b.resolve_models({"llm": {"backend": "local", "local_model": "qwen"}})["llm"]
    gemma = b.resolve_models({"llm": {"backend": "local", "local_model": "gemma"}})["llm"]
    external = b.resolve_models({"llm": {"backend": "external", "base_url": "http://llm/v1"}})["llm"]

    assert qwen["backend"] == "llama"
    assert qwen["opts"]["mtp"] is True
    assert gemma["backend"] == "gemma"
    assert external["backend"] == "openai"
    assert external["opts"]["base_url"] == "http://llm/v1"


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
    assert b.is_offerable(b.get_spec("asr", "qwen-gguf"), "cpu") is False
    assert b.is_offerable(b.get_spec("llm", "openai"), "cpu") is True


def test_list_backends_covers_each_domain():
    for domain in (b.ASR, b.TTS, b.LLM):
        names = {s.backend for s in b.list_backends(domain)}
        assert names, domain
    assert "none" in {s.backend for s in b.list_backends(b.LLM)}


def test_asr_and_tts_options_have_stable_user_facing_order():
    assert [spec.backend for spec in b.list_backends(b.ASR)] == [
        "qwen-gguf",
        "qwen-gguf-small",
        "qwen-onnx",
        "qwen-onnx-small",
        "qwen",
        "qwen-small",
        "moonshine",
        "moonshine-tiny",
    ]
    assert [spec.backend for spec in b.list_backends(b.TTS)] == [
        "higgs-gguf",
        "qwen-gguf",
        "qwen-gguf-small",
        "qwen",
        "qwen-small",
        "kokoro-onnx",
        "pocket-tts-onnx",
    ]
    assert b.get_spec("asr", "qwen").display_name == "Qwen3 1.7B PyTorch (default GPU)"


# --- no-LLM bypass in the agent loop ---------------------------------------


def _import_agent_loop():
    """Import core.agent_loop.

    It's self-contained (imports only utils/tools/leaf-core), and its leaf
    deps are importable on the dev box / conftest-stubbed in CI, so no module
    stubbing is needed here — stubbing core.slm would pollute sys.modules for
    other test files that need the real generate_slm/load_slm.
    """
    import core.agent_loop as agent_loop  # noqa: E402

    return agent_loop


def test_run_bypasses_slm_when_llm_disabled():
    al = _import_agent_loop()
    src = inspect.getsource(al.AgentLoop._run)
    # The bypass is taken before the SLM agent loop body.
    assert "if not host.llm_enabled:" in src
    bypass_pos = src.index("host.llm_enabled")
    loop_pos = src.index("for iteration in range")
    assert bypass_pos < loop_pos, "no-LLM bypass must precede the SLM loop"


def test_run_without_llm_never_calls_slm():
    al = _import_agent_loop()
    src = inspect.getsource(al.AgentLoop._run_without_llm)
    # The regex-only path must not invoke any SLM generation.
    assert "_generate_with_context_recovery" not in src
    assert "generate_slm" not in src
    # A non-recoverable step (web search / deep_think / unresolved entity)
    # falls back rather than replanning.
    assert "should_replan" in src
    assert "_speak_no_ai_fallback" in src
