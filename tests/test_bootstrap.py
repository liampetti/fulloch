"""First-run data-directory scaffolding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bootstrap import _SUBDIRS, ensure_scaffolding  # noqa: E402


def _make_seed(tmp_path):
    seed = tmp_path / "seed"
    (seed / "grammars").mkdir(parents=True)
    (seed / "grammars" / "agent.gbnf").write_text('root ::= "x"')
    (seed / "wakeword").mkdir()
    (seed / "wakeword" / "hey_atticus_v0.3.onnx").write_bytes(b"wakeword model")
    (seed / "wakeword" / "hey_atticus_v0.6.onnx").write_bytes(b"new wakeword model")
    (seed / "config.example.yml").write_text("general:\n  wakeword: hey atticus\n")
    return seed


def test_creates_data_subtree(tmp_path):
    data = tmp_path / "data"
    ensure_scaffolding(str(data), seed_dir=str(tmp_path / "noseed"))
    for sub in _SUBDIRS:
        assert (data / sub).is_dir()


def test_seeds_config_and_grammar_on_first_run(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"
    assert ensure_scaffolding(str(data), seed_dir=str(seed)) is True
    assert (data / "config.yml").read_text().startswith("general:")
    assert (data / "models" / "grammars" / "agent.gbnf").is_file()
    # Later starts must remain distinguishable from the first seed.
    assert ensure_scaffolding(str(data), seed_dir=str(seed)) is False


def test_seeds_wakeword_models_without_overwriting_a_replacement(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"

    ensure_scaffolding(str(data), seed_dir=str(seed))

    wakeword = data / "models" / "wakeword" / "hey_atticus_v0.3.onnx"
    assert wakeword.read_bytes() == b"wakeword model"
    assert (data / "models" / "wakeword" / "hey_atticus_v0.6.onnx").read_bytes() == b"new wakeword model"

    wakeword.write_bytes(b"user replacement")
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert wakeword.read_bytes() == b"user replacement"


def test_seeds_english_intents_without_overwriting_existing(tmp_path):
    seed = _make_seed(tmp_path)
    (seed / "intents" / "en").mkdir(parents=True)
    (seed / "intents" / "en" / "pack.yaml").write_text("seed")
    data = tmp_path / "data"
    ensure_scaffolding(str(data), seed_dir=str(seed))
    target = data / "intents" / "en" / "pack.yaml"
    assert target.read_text() == "seed"
    target.write_text("existing")
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert target.read_text() == "existing"


def test_response_defaults_seed_all_languages_preserving_edits_and_packs(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"
    for language in ("en", "fr", "de"):
        source = seed / "intents" / language / "fulloch-responses.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"default-{language}")
        target = data / "intents" / language
        target.mkdir(parents=True)
        (target / "2025.6.23.yaml").write_text("downloaded-pack")
    (data / "intents" / "fr" / "fulloch-responses.yaml").write_text("custom")

    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert (data / "intents" / "en" / "fulloch-responses.yaml").read_text() == "default-en"
    assert (data / "intents" / "de" / "fulloch-responses.yaml").read_text() == "default-de"
    assert (data / "intents" / "fr" / "fulloch-responses.yaml").read_text() == "custom"
    (seed / "intents" / "en" / "fulloch-responses.yaml").write_text("updated-default")
    ensure_scaffolding(str(data), seed_dir=str(seed))
    for language in ("en", "fr", "de"):
        assert (data / "intents" / language / "2025.6.23.yaml").read_text() == "downloaded-pack"
    assert (data / "intents" / "en" / "fulloch-responses.yaml").read_text() == "default-en"


def test_native_response_seed_fallback(tmp_path):
    from core.bootstrap import _seed_response_files

    data = tmp_path / "data"
    _seed_response_files(data, tmp_path / "absent", native_fallback=True)
    for language in ("en", "fr", "de"):
        assert (data / "intents" / language / "fulloch-responses.yaml").is_file()


def test_response_seed_race_does_not_overwrite_user_file(tmp_path, monkeypatch):
    from core import bootstrap

    source = tmp_path / "seed" / "intents" / "en" / "fulloch-responses.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("default")
    data = tmp_path / "data"
    target = data / "intents" / "en" / "fulloch-responses.yaml"
    real_link = bootstrap.os.link

    def concurrent_edit(src, dst):
        target.write_text("user edit")
        return real_link(src, dst)

    monkeypatch.setattr(bootstrap.os, "link", concurrent_edit)
    bootstrap._seed_response_files(data, tmp_path / "seed", native_fallback=False)
    assert target.read_text() == "user edit"
    assert not list(target.parent.glob(".response-*"))


def test_seeds_bundled_voice_references_without_overwriting_user_voices(tmp_path):
    seed = _make_seed(tmp_path)
    (seed / "voices").mkdir()
    (seed / "voices" / "atticus.wav").write_bytes(b"seed voice")
    data = tmp_path / "data"
    (data / "voices").mkdir(parents=True)
    (data / "voices" / "custom.wav").write_bytes(b"user voice")

    ensure_scaffolding(str(data), seed_dir=str(seed))

    assert (data / "voices" / "atticus.wav").read_bytes() == b"seed voice"
    assert (data / "voices" / "custom.wav").read_bytes() == b"user voice"


def test_generates_https_cert_and_wires_it_on_first_run(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert (data / "certs" / "dashboard.crt").is_file()
    assert (data / "certs" / "dashboard.key").is_file()
    text = (data / "config.yml").read_text()
    assert "dashboard_ssl_certfile" in text
    assert "dashboard_ssl_keyfile" in text
    # The seed's other keys (e.g. wakeword) must survive the ruamel round-trip.
    assert "wakeword: hey atticus" in text


def test_does_not_overwrite_existing(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"
    (data / "models" / "grammars").mkdir(parents=True)
    (data / "config.yml").write_text("general:\n  wakeword: custom\n")
    (data / "models" / "grammars" / "agent.gbnf").write_text("MINE")
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert "custom" in (data / "config.yml").read_text()
    assert (data / "models" / "grammars" / "agent.gbnf").read_text() == "MINE"
    # An already-existing install must never be silently flipped to HTTPS.
    assert not (data / "certs" / "dashboard.crt").exists()


def test_noop_when_no_seed_dir(tmp_path):
    data = tmp_path / "data"
    # Missing seed dir: dirs created, nothing seeded, no error.
    ensure_scaffolding(str(data), seed_dir=str(tmp_path / "absent"))
    assert (data / "voices").is_dir()
    assert not (data / "config.yml").exists()


def test_seed_dir_from_env(tmp_path, monkeypatch):
    seed = _make_seed(tmp_path)
    monkeypatch.setenv("FULLOCH_SEED_DIR", str(seed))
    data = tmp_path / "data"
    ensure_scaffolding(str(data))  # seed_dir defaults to the env var
    assert (data / "config.yml").is_file()
def test_fastpath_upgrade_merges_missing_defaults_without_overwriting_customizations(tmp_path):
    from core.bootstrap import _merge_fastpath_defaults

    source = tmp_path / "default.yaml"
    target = tmp_path / "user.yaml"
    source.write_text("patterns:\n  play: default\n  stop: stop-default\nhassil:\n  local:\n    clock: [default]\n")
    target.write_text("# my config\npatterns:\n  play: custom\ncorrections: []\n")
    _merge_fastpath_defaults(source, target)
    from ruamel.yaml import YAML

    updated = YAML(typ="safe").load(target.read_text())
    assert updated["patterns"] == {"play": "custom", "stop": "stop-default"}
    assert updated["corrections"] == []
    assert updated["hassil"]["local"]["clock"] == ["default"]
    assert "# my config" in target.read_text()
