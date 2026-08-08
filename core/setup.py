"""First-run / existing-install detection for the two-phase startup.

`detect_setup_state(config)` decides whether the wizard should run:
  - no config (or no `general:` block)        → first run, NEEDS_SETUP
  - config present but missing required keys   → config_error (update needed)
  - config present, required keys ok, but the selected backends' model assets
    aren't on disk                             → NEEDS_SETUP (download needed)
  - everything present                         → existing install, proceed

Import-light (stdlib + the import-light `core.backends`); never touches torch
or the model stack, so it runs in Phase A before anything heavy loads.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .backends import DOMAINS, LLM, resolve_models, variant

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "./data/config.yml"
DEFAULT_MODELS_DIR = "./data/models"
# Dropped by the dashboard's "Re-run setup" action; forces the wizard on the
# next start regardless of config/assets. The wizard clears it on completion.
DEFAULT_RESET_MARKER = "./data/.setup_pending"
SNAPSHOT_COMPLETE_MARKER = ".fulloch_complete"

# Keys the app hard-requires to construct the assistant. A populated config
# that lacks one of these is a version mismatch — surfaced as a clear update
# error rather than silently defaulting. Extend as the schema grows.
REQUIRED_KEYS = (("general", "wakeword"),)


@dataclass
class SetupDecision:
    """Outcome of `detect_setup_state`."""

    needs_setup: bool
    config_present: bool
    reason: str
    config_error: Optional[str] = None
    missing_assets: list = field(default_factory=list)


def _looks_like_path(model: str) -> bool:
    """True for a local filesystem model (gguf / explicit path), not an HF repo.

    HF repo ids ("Qwen/Qwen3-ASR-1.7B") also contain "/", so distinguish by
    a local prefix or a weights extension rather than the slash alone.
    """
    s = str(model)
    return (
        s.endswith((".gguf", ".bin", ".safetensors"))
        or s.startswith(("./", "../", "/", ".\\"))
        or os.path.isabs(s)
    )


def _hub_dir(model_id: str, models_dir: str) -> Path:
    """Local huggingface_hub cache dir for a repo id."""
    return Path(models_dir) / "hub" / f"models--{model_id.replace('/', '--')}"


def _hub_snapshot_complete(model_id: str, models_dir: str) -> bool:
    """Check Hugging Face's completed ref/snapshot layout, not just its cache dir."""
    root = _hub_dir(model_id, models_dir)
    if (root / SNAPSHOT_COMPLETE_MARKER).is_file():
        return True
    # Backward compatibility for caches created before Fulloch wrote its own
    # marker. A valid HF cache has a main ref pointing at a non-empty snapshot.
    ref = root / "refs" / "main"
    try:
        revision = ref.read_text().strip()
    except OSError:
        return False
    snapshot = root / "snapshots" / revision
    return bool(revision) and snapshot.is_dir() and any(snapshot.iterdir())


def _asset_present(domain_cfg: dict, models_dir: str) -> bool:
    """Whether the model asset for one resolved domain is on disk.

    Domains with nothing to download — the no-LLM bypass, or a remote
    OpenAI-compatible endpoint (no HF repo/file, just a remote model *name*) —
    count as present, so a remote-LLM config isn't stuck reporting setup-needed.
    """
    spec = domain_cfg.get("spec")
    if spec is not None and not (spec.hf_repo or spec.hf_file or spec.hf_files):
        return True
    model = domain_cfg.get("model")
    if not model:
        return True
    if spec is not None and spec.hf_files:
        root = Path(model)
        return root.is_dir() and all((root / filename).is_file() for _, filename in spec.hf_files)
    if _looks_like_path(model):
        p = Path(model)
        if p.suffix in (".gguf", ".bin", ".safetensors"):
            return p.is_file()
        # Directory-style model (e.g. the ONNX bundle) — present if it exists
        # and isn't empty.
        return p.is_dir() and any(p.iterdir())
    return _hub_snapshot_complete(model, models_dir)


def _missing_required_keys(config: dict) -> list:
    missing = []
    for path in REQUIRED_KEYS:
        node = config
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node in (None, ""):
            missing.append(".".join(path))
    return missing


def detect_setup_state(
    config: Optional[dict],
    models_dir: str = DEFAULT_MODELS_DIR,
    reset_marker: str = DEFAULT_RESET_MARKER,
) -> SetupDecision:
    """Classify the install so `main` can route to setup vs. run."""
    config_present = bool(config) and isinstance(config.get("general"), dict)

    # An explicit reset (dashboard "Re-run setup wizard") wins over everything:
    # re-show the wizard even though config + models are still on disk. Keeping
    # them means a reconfigure re-downloads nothing unless backends change.
    if reset_marker and Path(reset_marker).exists():
        return SetupDecision(
            needs_setup=True,
            config_present=config_present,
            reason="setup reset requested",
        )

    if not config_present:
        return SetupDecision(
            needs_setup=True,
            config_present=False,
            reason="no config — first run",
        )

    missing_keys = _missing_required_keys(config)
    if missing_keys:
        msg = "config is missing required keys: " + ", ".join(missing_keys)
        return SetupDecision(
            needs_setup=True,
            config_present=True,
            reason="config update needed",
            config_error=msg,
        )

    resolved = resolve_models(config.get("models"))

    # Variant guard: the slim CPU image can't load gpu_only backends (no
    # qwen_asr / flash-attn / llama-cpp). This bites when a config resolves to
    # the qwen defaults — e.g. a freshly-seeded config with no `models:` block,
    # or a GPU install's config carried over via a shared ./data that still has
    # the GPU model assets on disk. Without this, the assets look "present" and
    # the transcriber thread crashes on `import qwen_asr`. Force the wizard so
    # the user picks CPU backends instead.
    if variant() == "cpu":
        gpu_only = [
            f"{d}:{resolved[d]['backend']}"
            for d in DOMAINS
            if getattr(resolved[d].get("spec"), "gpu_only", False)
        ]
        if gpu_only:
            return SetupDecision(
                needs_setup=True,
                config_present=True,
                reason="GPU-only backends can't run on the CPU image: " + ", ".join(gpu_only),
            )

    missing_assets = []
    for domain in DOMAINS:
        if not _asset_present(resolved[domain], models_dir):
            cfg = resolved[domain]
            missing_assets.append(f"{domain}:{cfg['backend']} ({cfg['model']})")
    # Local llama-server backends use the shipped GBNF grammar; external OpenAI
    # endpoints may not recognise it and fall back to JSON mode.
    if resolved[LLM]["backend"] in {"llama", "gemma"}:
        if not (Path(models_dir) / "grammars" / "agent.gbnf").is_file():
            missing_assets.append("grammar (agent.gbnf)")

    needs = bool(missing_assets)
    return SetupDecision(
        needs_setup=needs,
        config_present=True,
        reason="missing model assets" if needs else "ready",
        missing_assets=missing_assets,
    )
