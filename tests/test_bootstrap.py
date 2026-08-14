"""First-run scaffolding for the precompiled image (v2.2 Step 7)."""

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
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert (data / "config.yml").read_text().startswith("general:")
    assert (data / "models" / "grammars" / "agent.gbnf").is_file()


def test_seeds_default_wakeword_model_without_overwriting_a_replacement(tmp_path):
    seed = _make_seed(tmp_path)
    data = tmp_path / "data"

    ensure_scaffolding(str(data), seed_dir=str(seed))

    wakeword = data / "models" / "wakeword" / "hey_atticus_v0.3.onnx"
    assert wakeword.read_bytes() == b"wakeword model"

    wakeword.write_bytes(b"user replacement")
    ensure_scaffolding(str(data), seed_dir=str(seed))
    assert wakeword.read_bytes() == b"user replacement"


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
