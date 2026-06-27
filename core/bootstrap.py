"""First-run scaffolding for the precompiled image (v2.2 Step 7).

The container ships application code only — weights and config live in the
single persisted `./data` volume, which is empty on first run. This seeds the
minimum so the app boots into the setup wizard with no host-side steps:

  - creates the `data/` subtree (models/grammars, models/hub, voices, notes, certs)
  - seeds `data/config.yml` from the bundled template (the wizard fills it in)
  - seeds `data/models/grammars/agent.gbnf` — the app's own grammar, which the
    wizard's downloader can't fetch (it's not on HF), so it must ship in the
    image and be copied into the volume here, or first-run setup would loop.

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
_SUBDIRS = ("models/grammars", "models/hub", "voices", "notes", "certs")


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

    # First-run config from the bundled template; the wizard writes the rest.
    config = data / "config.yml"
    seed_config = seed / "config.example.yml"
    if not config.is_file() and seed_config.is_file():
        shutil.copy2(seed_config, config)
        logger.info("Created %s from the bundled template", config)
