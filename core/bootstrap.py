"""First-run data-directory scaffolding."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEED_DIR = "/app/seed"
_SUBDIRS = (
    "models/grammars",
    "models/hub",
    "models/wakeword",
    "intents/en",
    "voices",
    "notes",
    "certs",
    "wav",
)


def _merge_fastpath_defaults(source: Path, target: Path) -> None:
    """Add missing mapping keys without replacing any user-defined values."""
    from ruamel.yaml import YAML

    try:
        yaml = YAML()
        defaults = yaml.load(source.read_text(encoding="utf-8"))
        original = target.read_text(encoding="utf-8")
        existing = yaml.load(original)
        if not isinstance(defaults, dict) or not isinstance(existing, dict):
            logger.warning("Cannot merge invalid fast-path file %s", target)
            return

        def merge(current, baseline):
            changed = False
            for key, value in baseline.items():
                if key not in current:
                    current[key] = value
                    changed = True
                elif isinstance(current[key], dict) and isinstance(value, dict):
                    changed = merge(current[key], value) or changed
            return changed

        if merge(existing, defaults):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                                 prefix=".fastpath-", delete=False) as output:
                    temporary = Path(output.name)
                    yaml.dump(existing, output)
                if target.read_text(encoding="utf-8") != original:
                    logger.warning("Fast-path file changed during merge; leaving it unchanged: %s", target)
                    return
                os.chmod(temporary, target.stat().st_mode & 0o777)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            logger.info("Added missing fast-path defaults to %s", target)
    except Exception:  # noqa: BLE001 — preserve startup if an edited file is invalid
        logger.exception("Could not merge fast-path defaults into %s", target)


def _seed_response_files(data: Path, seed: Path, *, native_fallback: bool) -> None:
    """Publish bundled defaults only when missing, without replacing edits."""
    source_root = seed / "intents"
    if native_fallback and not source_root.is_dir():
        source_root = Path(__file__).resolve().parent.parent / "data" / "intents"
    for language, filename in ((language, filename) for language in ("en", "fr", "de")
                               for filename in ("fulloch-responses.yaml", "fulloch-fastpath.yaml")):
        source = source_root / language / filename
        target = data / "intents" / language / filename
        if not source.is_file():
            continue
        if target.exists():
            if filename == "fulloch-fastpath.yaml" and source.resolve() != target.resolve():
                _merge_fastpath_defaults(source, target)
            continue
        temporary = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".response-", delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as source_file:
                    shutil.copyfileobj(source_file, output)
            # Atomic no-replace publication even across simultaneous startups.
            os.link(temporary, target)
            logger.info("Seeded response file into %s", target)
        except FileExistsError:
            pass  # Another startup or user created it first.
        except OSError as exc:
            logger.warning("Could not seed response file %s: %s", target, exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _seed_https_cert(data: Path) -> None:
    """Generate a self-signed cert and write its paths into the fresh config.yml.

    Only called for a genuinely new install (see caller) — never touches an
    existing config, so it can't retroactively flip an already-running install
    from HTTP to HTTPS. Uses ruamel directly (not config_store.update_config)
    so the template's inline documentation comments survive; update_config
    strips comments by design for the settings-console write path, which would
    otherwise blank out the freshly-seeded config before the wizard even runs.
    Best-effort: a failure here logs and leaves the dashboard on plain HTTP
    rather than blocking startup.
    """
    try:
        from ruamel.yaml import YAML

        from .tls_certs import ensure_self_signed_cert

        cert_path, key_path = ensure_self_signed_cert(str(data / "certs"))

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 4096
        config_path = data / "config.yml"
        with config_path.open() as f:
            doc = yaml.load(f)
        general = doc.setdefault("general", {})
        general["dashboard_ssl_certfile"] = cert_path
        general["dashboard_ssl_keyfile"] = key_path
        with config_path.open("w") as f:
            yaml.dump(doc, f)
        logger.info("HTTPS enabled by default for this new install (self-signed cert)")
    except Exception:
        logger.exception("Self-signed HTTPS cert setup failed; dashboard will serve over HTTP")


def ensure_scaffolding(data_dir: str = "./data", seed_dir: str = None) -> bool:
    """Create the data subtree and seed config + grammar on first run.

    Return whether this call created ``config.yml`` from the bundled template.
    The caller uses that signal to distinguish a newly seeded install from an
    existing configuration whose selected assets happen to be present.
    """
    seed = Path(seed_dir or os.environ.get("FULLOCH_SEED_DIR", DEFAULT_SEED_DIR))
    data = Path(data_dir)

    for sub in _SUBDIRS:
        (data / sub).mkdir(parents=True, exist_ok=True)

    # The agent grammar can't be downloaded (it's ours), so seed it from the
    # image. Without it, detect_setup_state would keep reporting setup-needed.
    grammar = data / "models" / "grammars" / "agent.gbnf"
    seed_grammar = seed / "grammars" / "agent.gbnf"
    if not grammar.is_file() and seed_grammar.is_file():
        shutil.copy2(seed_grammar, grammar)
        logger.info("Seeded agent grammar into %s", grammar)

    # Bundle every versioned Hey Atticus classifier. They are app assets rather
    # than setup downloads, and existing user files must never be overwritten.
    seed_wakewords = seed / "wakeword"
    if seed_wakewords.is_dir():
        for src in seed_wakewords.glob("hey_atticus_v*.onnx"):
            dst = data / "models" / "wakeword" / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
                logger.info("Seeded wakeword model into %s", dst)

    # English sentence packs are image assets; preserve any existing pack.
    seed_intents = seed / "intents" / "en"
    if seed_intents.is_dir():
        for src in seed_intents.glob("*.yaml"):
            if src.name == "fulloch-responses.yaml":
                continue
            dst = data / "intents" / "en" / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info("Seeded English intent pack into %s", dst)

    _seed_response_files(data, seed, native_fallback=seed_dir is None and not os.environ.get("FULLOCH_SEED_DIR"))

    # Timer alert tone (core/assistant.py: ALARM_WAV_PATH) — same reasoning
    # as the grammar above: it's an app asset, not something the wizard
    # downloads, so it must be copied into the volume here.
    seed_wav = seed / "wav"
    if seed_wav.is_dir():
        for src in seed_wav.iterdir():
            dst = data / "wav" / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
                logger.info("Seeded %s into %s", src.name, dst)

    # Public starter voice references ship with the image so a fresh Pocket or
    # Qwen voice-clone install can speak immediately. Never overwrite a voice
    # a user generated or added to the persistent volume.
    seed_voices = seed / "voices"
    if seed_voices.is_dir():
        for src in seed_voices.iterdir():
            dst = data / "voices" / src.name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
                logger.info("Seeded voice reference %s into %s", src.name, dst)

    # The optional Obsidian bridge is distributed separately, but retain a
    # static archive for dashboard download during the transition.
    plugin_zip = data / "fulloch-obsidian-plugin.zip"
    seed_plugin_zip = seed / "fulloch-obsidian-plugin.zip"
    if not plugin_zip.is_file() and seed_plugin_zip.is_file():
        shutil.copy2(seed_plugin_zip, plugin_zip)
        logger.info("Seeded Obsidian plugin archive into %s", plugin_zip)

    # First-run config from the bundled template; the wizard writes the rest.
    config = data / "config.yml"
    seed_config = seed / "config.example.yml"
    is_new_install = not config.is_file()
    if is_new_install and seed_config.is_file():
        shutil.copy2(seed_config, config)
        logger.info("Created %s from the bundled template", config)

    if is_new_install and config.is_file():
        _seed_https_cert(data)

    return is_new_install and config.is_file()
