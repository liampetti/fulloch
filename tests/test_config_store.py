"""Config store: schema-validated read/write, clean comment-free output (v2.2 Step 4)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import config_store as cs  # noqa: E402
from server.config_schema import SCHEMA, field_for  # noqa: E402


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
    cs.update_config({
        "general.use_vad": "false",
        "general.vad_threshold": "0.7",
        "general.vad_min_speech_ms": "400",
    }, path)
    cfg = cs.read_config(path)
    assert cfg["general"]["use_vad"] is False
    assert cfg["general"]["vad_threshold"] == 0.7
    assert cfg["general"]["vad_min_speech_ms"] == 400


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


def test_write_models_block(tmp_path):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    cs.write_models(
        {"asr": {"backend": "moonshine"}, "tts": {"backend": "kokoro-onnx"},
         "llm": {"backend": "none"}},
        path,
    )
    cfg = cs.read_config(path)
    assert cfg["models"]["asr"]["backend"] == "moonshine"
    assert cfg["models"]["llm"]["backend"] == "none"


def test_set_llm_model_name_preserves_rest_of_block(tmp_path):
    path = _write(
        tmp_path,
        "models:\n  llm:\n    backend: openai\n    base_url: http://x/v1\n    model: old\n",
    )
    cs.set_llm_model_name("new-model", path)
    cfg = cs.read_config(path)
    assert cfg["models"]["llm"]["model"] == "new-model"
    assert cfg["models"]["llm"]["backend"] == "openai"      # untouched
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
    assert asr["qwen-onnx-small"]["experimental"] is True


def test_settings_view_reports_llm_key_in_env(tmp_path, monkeypatch):
    path = _write(tmp_path, "general:\n  wakeword: hi\n")
    monkeypatch.delenv("FULLOCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cs.settings_view(path)["llm_api_key_in_env"] is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
    assert cs.settings_view(path)["llm_api_key_in_env"] is True


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
        "full": True, "cpu_server": True, "cpu_local": True}

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
