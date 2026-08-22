"""Schema-validated config-store reads and writes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import config_store as cs  # noqa: E402
from server.config_schema import SCHEMA, WAKEWORD_PRESETS, field_for  # noqa: E402


def _write(tmp_path, text):
    p = tmp_path / "config.yml"
    p.write_text(text)
    return str(p)


def test_every_field_has_unique_path_and_valid_group():
    from server.config_schema import GROUPS

    paths = [f.path for f in SCHEMA]
    assert len(paths) == len(set(paths)), "duplicate field paths"
    for f in SCHEMA:
        assert f.group in GROUPS
        if f.type == "enum":
            assert f.choices, f.path


def test_update_writes_clean_comment_free_config(tmp_path):
    # Writes drop comments so the active values are easy to read (docs live in
    # config.example.yml). Values are preserved; comments are not.
    path = _write(tmp_path, "general:\n  wakeword: hey atticus  # the phrase\n")
    cs.update_config({"general.wakeword": "computer"}, path)
    text = Path(path).read_text()
    assert "computer" in text
    assert "#" not in text, "comments must be stripped on write"


def test_update_coerces_types(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.update_config(
        {
            "general.use_vad": "false",
            "general.vad_threshold": "0.7",
            "general.vad_min_speech_ms": "400",
            "general.persistent_logging_enabled": "true",
            "general.telemetry_enabled": "true",
        },
        path,
    )
    cfg = cs.read_config(path)
    assert cfg["general"]["use_vad"] is False
    assert cfg["general"]["vad_threshold"] == 0.7
    assert cfg["general"]["vad_min_speech_ms"] == 400
    assert cfg["general"]["persistent_logging_enabled"] is True
    assert cfg["general"]["telemetry_enabled"] is True


def test_persistent_logging_defaults_to_disabled():
    field = field_for("general.persistent_logging_enabled")
    assert field is not None
    assert field.default is False


def test_update_coerces_voice_satellite_limit(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.update_config({"general.max_voice_satellites": "3"}, path)
    assert cs.read_config(path)["general"]["max_voice_satellites"] == 3


def test_satellite_uplink_channels_is_a_constrained_integer_enum(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.update_config({"satellite.uplink_channels": "2"}, path)
    assert cs.read_config(path)["satellite"]["uplink_channels"] == 2
    with pytest.raises(cs.ConfigValidationError):
        cs.update_config({"satellite.uplink_channels": 3}, path)


def test_satellite_audio_defaults_preserve_mono_best_channel_fallback():
    uplink_channels = field_for("satellite.uplink_channels")
    processing = field_for("satellite.dual_mic_processing")
    assert uplink_channels is not None and uplink_channels.default == 1
    assert processing is not None and processing.default == "best_channel"


def test_unknown_key_raises_before_writing(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    with pytest.raises(cs.ConfigValidationError) as ei:
        cs.update_config({"general.bogus": "x", "general.wakeword": "ok"}, path)
    assert "general.bogus" in ei.value.errors
    # Nothing applied — the valid key in the same batch wasn't written.
    assert cs.read_config(path)["general"]["wakeword"] == "hi"


def test_enum_validation(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    with pytest.raises(cs.ConfigValidationError):
        cs.update_config({"general.barge_in": "maybe"}, path)
    cs.update_config({"general.barge_in": "off"}, path)
    assert cs.read_config(path)["general"]["barge_in"] == "off"


def test_empty_value_unsets_key(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n  voice_clone: atticus\n")
    cs.update_config({"general.voice_clone": ""}, path)
    assert "voice_clone" not in cs.read_config(path)["general"]


def test_list_coercion_from_csv(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.update_config({"general.asr_context_terms": "phoebe bridgers, downstairs office"}, path)
    terms = cs.read_config(path)["general"]["asr_context_terms"]
    assert terms == ["phoebe bridgers", "downstairs office"]


def test_dict_coercion_accepts_json_and_rejects_non_objects(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.update_config({"obsidian.path_translation": '{"/host": "/vault"}'}, path)
    assert cs.read_config(path)["obsidian"]["path_translation"] == {"/host": "/vault"}
    with pytest.raises(cs.ConfigValidationError):
        cs.update_config({"obsidian.path_translation": "not json"}, path)


def test_write_models_block(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.write_models(
        {
            "asr": {"backend": "moonshine"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "none"},
        },
        path,
    )
    cfg = cs.read_config(path)
    assert cfg["models"]["asr"]["backend"] == "moonshine"
    assert cfg["models"]["llm"]["backend"] == "none"


def test_write_models_validates_public_llm_modes(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")

    with pytest.raises(ValueError, match="custom local model"):
        cs.write_models({"llm": {"backend": "local", "local_model": "custom", "model": "model.bin"}}, path)
    with pytest.raises(ValueError, match="external LLM needs a base_url"):
        cs.write_models({"llm": {"backend": "external"}}, path)

    cs.write_models({"llm": {"backend": "local", "local_model": "qwen"}}, path)
    assert cs.read_config(path)["models"]["llm"]["local_model"] == "qwen"


def test_write_models_validates_generation_timeout(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")

    with pytest.raises(ValueError, match="generation_timeout"):
        cs.write_models(
            {"llm": {"backend": "local", "local_model": "qwen", "generation_timeout": 0}}, path
        )

    cs.write_models(
        {"llm": {"backend": "local", "local_model": "qwen", "generation_timeout": 120}}, path
    )
    assert cs.read_config(path)["models"]["llm"]["generation_timeout"] == 120


def test_write_models_validates_and_preserves_openwakeword_config(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hey atticus\n")
    models = {
        "llm": {"backend": "none"},
        "wakeword": {
            "backend": "openwakeword",
            "model": "data/models/wakeword/hey_atticus.onnx",
            "threshold": 0.5,
            "smoothing_frames": 3,
            "cooldown_ms": 1500,
        },
    }
    cs.write_models(models, path)
    assert cs.read_config(path)["models"]["wakeword"]["backend"] == "openwakeword"

    models["wakeword"]["threshold"] = 2
    with pytest.raises(ValueError, match="threshold"):
        cs.write_models(models, path)


def test_wakeword_presets_only_offer_bundled_atticus_model():
    assert [preset.wakeword for preset in WAKEWORD_PRESETS] == ["hey atticus"]
    assert WAKEWORD_PRESETS[0].model == "data/models/wakeword/hey_atticus_v0.3.onnx"
    assert WAKEWORD_PRESETS[0].model_options == (
        ("data/models/wakeword/hey_atticus_v0.3.onnx", "Hey Atticus v0.3 (Recommended)"),
    )


def test_set_llm_model_name_preserves_rest_of_block(tmp_path):
    path = _write(
        tmp_path,
        "models:\n  llm:\n    backend: openai\n    base_url: http://x/v1\n    model: old\n",
    )
    cs.set_llm_model_name("new-model", path)
    cfg = cs.read_config(path)
    assert cfg["models"]["llm"]["model"] == "new-model"
    assert cfg["models"]["llm"]["backend"] == "openai"  # untouched
    assert cfg["models"]["llm"]["base_url"] == "http://x/v1"  # untouched


def test_set_llm_model_name_raises_without_llm_block(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    with pytest.raises(KeyError):
        cs.set_llm_model_name("m", path)


def test_settings_view_merges_values(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hey atticus\n")
    view = cs.settings_view(path)
    by_path = {f["path"]: f for f in view["fields"]}
    assert by_path["general.wakeword"]["value"] == "hey atticus"
    assert by_path["general.wakeword"]["set"] is True
    # an unset key reports its default + set=False
    assert by_path["search.searxng_url"]["set"] is False
    assert view["wakeword_presets"] and view["tier_presets"]
    # Backend options carry the experimental flag for the wizard tag.
    asr = {x["backend"]: x for x in view["backends"]["asr"]}
    assert asr["qwen-onnx"]["experimental"] is False
    assert asr["qwen-onnx-small"]["experimental"] is False
    assert asr["qwen"]["display_name"] == "Qwen3 1.7B PyTorch (default GPU)"
    assert asr["moonshine"]["experimental"] is True


def test_settings_view_reads_legacy_higgs_personality(tmp_path):
    path = _write(tmp_path, "general:\n  higgs_personality: wry\n")
    by_path = {f["path"]: f for f in cs.settings_view(path)["fields"]}
    assert by_path["general.personality"]["value"] == "wry"
    assert by_path["general.personality"]["set"] is False



def test_seeds_from_example_when_absent(tmp_path):
    example = tmp_path / "config.example.yml"
    example.write_text("general:\n  wakeword: hey atticus  # seeded\n")
    target = str(tmp_path / "config.yml")
    cs.update_config({"general.voice_clone": "atticus"}, target)
    text = Path(target).read_text()
    # The example's *values* seed the new config, but its comments are stripped.
    assert "hey atticus" in text and "atticus" in text
    assert "#" not in text


def test_field_for_lookup():
    assert field_for("general.wakeword").type == "str"
    assert field_for("nope.nope") is None


def test_settings_view_marks_offerable_by_variant(tmp_path, monkeypatch):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")

    monkeypatch.delenv("FULLOCH_VARIANT", raising=False)
    gpu = cs.settings_view(path)
    assert gpu["variant"] == "gpu"
    assert {t["id"]: t["offerable"] for t in gpu["tier_presets"]} == {
        "full": True,
        "cpu_server": True,
        "cpu_local": True,
    }

    monkeypatch.setenv("FULLOCH_VARIANT", "cpu")
    cpu = cs.settings_view(path)
    assert cpu["variant"] == "cpu"
    tiers = {t["id"]: t["offerable"] for t in cpu["tier_presets"]}
    # Full needs GPU-only backends; the two CPU stacks run on the CPU image.
    assert tiers == {"full": False, "cpu_server": True, "cpu_local": True}
    llm = {x["backend"]: x["offerable"] for x in cpu["backends"]["llm"]}
    assert llm["none"] is True and llm["openai"] is True and llm["llama"] is False
    asr = {x["backend"]: x["offerable"] for x in cpu["backends"]["asr"]}
    assert asr["moonshine"] is True and asr["qwen"] is False
