"""First-run scaffolding for the precompiled image (v2.2 Step 7).

The container ships application code only — weights and config live in the
single persisted `./data` volume, which is empty on first run. This seeds the
minimum so the app boots into the setup wizard with no host-side steps:

  - creates the `data/` subtree (models/grammars, models/hub, voices, notes, certs)
  - seeds `data/config.yml` from the bundled template (the wizard fills it in)
  - seeds `data/models/grammars/agent.gbnf` — the app's own grammar, which the
    wizard's downloader can't fetch (it's not on HF), so it must ship in the
    image and be copied into the volume here, or first-run setup would loop.
  - generates a self-signed HTTPS cert and wires it into the fresh config, so
    the browser satellite's microphone access works from a phone/other device
    on the LAN out of the box (see core.tls_certs).

Seed files live at `FULLOCH_SEED_DIR` (default `/app/seed`, populated by the
Dockerfile). In a native dev checkout that dir doesn't exist and `data/` is
already populated, so this is a no-op there.
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEED_DIR = "/app/seed"
_SUBDIRS = ("models/grammars", "models/hub", "voices", "notes", "certs", "wav")


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


def ensure_scaffolding(data_dir: str = "./data", seed_dir: str = None) -> None:
    """Create the data subtree and seed config + grammar on first run."""
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

    # First-run config from the bundled template; the wizard writes the rest.
    config = data / "config.yml"
    seed_config = seed / "config.example.yml"
    is_new_install = not config.is_file()
    if is_new_install and seed_config.is_file():
        shutil.copy2(seed_config, config)
        logger.info("Created %s from the bundled template", config)

    if is_new_install and config.is_file():
        _seed_https_cert(data)
