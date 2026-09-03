#!/usr/bin/env python3
"""
Fulloch - The Fully Local Home Voice Assistant

A fully local, privacy-focused AI voice home assistant.
Runs speech recognition (Qwen3 ASR), text-to-speech (Qwen3 TTS),
and a small language model (Qwen3.5 9B) entirely on GPU.

Usage:
    python app.py
"""

import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

# Load env before anything reads os.environ. Root .env covers compose-level
# vars (FULLOCH_VERSION, etc.); credentials.json covers runtime
# secrets (ha_token, llm_api_key, dashboard_password, …) persisted in the Docker
# volume so a data-folder copy transfers everything to a new machine.
from dotenv import load_dotenv

from server.credentials_store import inject_env

load_dotenv()              # root .env — compose / shell vars; never overrides
inject_env()               # ./data/credentials.json → os.environ (setdefault)

# Initialise Torch before setting the optional TORCH_LOGS defaults below. Torch
# 2.8 treats an empty TORCH_LOGS value during first import as malformed state.
import torch  # noqa: F401

# Load configuration early so the log level (general.log_level) is available
# before logging is configured. A missing/empty config is tolerated — the app
# boots into first-run setup mode rather than crashing (see detect_setup_state).
_CONFIG_PATH = "./data/config.yml"
try:
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}

_GENERAL = config.get("general") or {}

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_log_level_name = str(_GENERAL.get("log_level", "info")).lower()
LOG_LEVEL = _LOG_LEVELS.get(_log_level_name, logging.INFO)
os.environ["FULLOCH_PERSISTENT_LOGGING_ENABLED"] = (
    "1" if _GENERAL.get("persistent_logging_enabled", False) else "0"
)
# Developer-only switch for diagnosing OpenAI-compatible HTTP traffic. Leave
# this off for normal debug sessions: the SDK logs full request options.
DEBUG_LLM_HTTP_LOGS = False
APP_LOG_PATH = Path("./data/logs/fulloch.log")
TELEMETRY_LOG_PATH = Path("./data/logs/telemetry.jsonl")
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


class _ConfiguredTimezoneFormatter(logging.Formatter):
    """Render log timestamps in the assistant's configured local timezone."""

    def __init__(self, *args, tz_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self._tz = ZoneInfo(tz_name) if tz_name else timezone.utc
        except ZoneInfoNotFoundError:
            self._tz = timezone.utc

    def formatTime(self, record, datefmt=None):
        timestamp = datetime.fromtimestamp(record.created, tz=self._tz)
        if datefmt:
            return timestamp.strftime(datefmt)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S") + f",{timestamp.microsecond // 1000:03d}"


_log_formatter = _ConfiguredTimezoneFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tz_name=_GENERAL.get("timezone"),
)
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_formatter)
_log_handlers: list[logging.Handler] = [_log_handler]
if _GENERAL.get("persistent_logging_enabled", False):
    try:
        APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _app_log_handler = RotatingFileHandler(
            APP_LOG_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _app_log_handler.setFormatter(_log_formatter)
        _log_handlers.append(_app_log_handler)
    except OSError as exc:
        # Keep the assistant usable when its persistent data directory is unavailable.
        _log_handler.emit(logging.makeLogRecord({
            "name": __name__, "levelno": logging.WARNING, "levelname": "WARNING",
            "msg": "Could not enable application file logging: %s", "args": (exc,),
        }))
logging.basicConfig(level=LOG_LEVEL, handlers=_log_handlers)
if _GENERAL.get("telemetry_enabled", False):
    try:
        _telemetry_handler = RotatingFileHandler(
            TELEMETRY_LOG_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _telemetry_handler.setFormatter(logging.Formatter("%(message)s"))
        _telemetry_logger = logging.getLogger("fulloch.telemetry")
        _telemetry_logger.setLevel(logging.INFO)
        _telemetry_logger.addHandler(_telemetry_handler)
        logging.getLogger(__name__).info("Telemetry file logging enabled: %s", TELEMETRY_LOG_PATH)
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not enable telemetry file logging: %s", exc)
else:
    logging.getLogger(__name__).info("Telemetry file logging disabled by general.telemetry_enabled")
logger = logging.getLogger(__name__)
if _log_level_name not in _LOG_LEVELS:
    logger.warning(f"Unknown general.log_level {_log_level_name!r}; using info")

# Capture first-party INFO logs into a ring buffer the loading screen streams,
# so model load / warmup progress shows as a live terminal instead of a bare
# spinner. Import-light (stdlib only); see server/lifecycle.py.
from server.lifecycle import install_log_capture  # noqa: E402

install_log_capture(max(LOG_LEVEL, logging.INFO))

# Hush third-party startup chatter so it never drowns our own logs: capped at
# WARNING even when log_level is debug, and raised in lockstep when log_level is
# stricter (e.g. error). Project-side loggers (core, utils, tools, audio) follow
# LOG_LEVEL via the root logger, so general.log_level: debug surfaces them.
_third_party_level = max(LOG_LEVEL, logging.WARNING)
for _noisy in (
    "transformers",
    "huggingface_hub",
    "accelerate",
    "sentence_transformers",
    "datasets",
    "filelock",
    "urllib3",
    # Spotipy logs complete API responses, including bearer tokens, at DEBUG.
    # Never allow those payloads into normal application logs.
    "spotipy",
    # The OpenAI SDK logs full request options and HTTP lifecycle events at
    # DEBUG. Keep those payload-sized lines opt-in while preserving Fulloch's
    # own debug output.
    "openai",
    "httpx",
    "httpcore",
    "qwen_tts",
    # Pocket TTS emits a line for every 80 ms decoder frame at DEBUG, plus
    # per-fragment timers at INFO. Fulloch emits its own concise TTS lifecycle.
    "pocket_tts",
    "torch",
    # torch.compile / CUDA graph capture chatter from
    # `enable_streaming_optimizations(use_compile=True)` in core/tts.py.
    # The parent `torch` logger doesn't propagate to these — torch sets up
    # its own log levels via torch._logging — so they need explicit overrides.
    "torch._dynamo",
    "torch._inductor",
    "torch._functorch",
    "torch.fx",
    "flash_attn",
    # numba (pulled in by librosa/audio deps) floods DEBUG with bytecode/type
    # inference traces — numba.core.ssa, numba.core.byteflow, etc. Capping the
    # parent `numba` logger covers all the numba.core.* children.
    "numba",
    # torchaudio's I/O backend probes FFmpeg extensions newest-first on import;
    # the older-FFmpeg-in-container case logs a full traceback per failed probe
    # at DEBUG before falling back to the version that loads. Harmless noise.
    "torio",
    "torchaudio",
):
    logging.getLogger(_noisy).setLevel(_third_party_level)
if DEBUG_LLM_HTTP_LOGS:
    for _noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(_noisy).setLevel(LOG_LEVEL)
# "Setting pad_token_id to eos_token_id" fires at WARNING on every
# transformers.generate call — silence just that submodule.
logging.getLogger("transformers.generation.utils").setLevel(max(LOG_LEVEL, logging.ERROR))

# Point HF cache to models folder
models_dir = Path("./data/models").resolve()
os.environ["HF_HOME"] = str(models_dir)

# Disable telemetry. NOTE: HF_HUB_OFFLINE is deliberately NOT set here — the
# first-run wizard needs huggingface_hub online to download models, and hf reads
# offline as an import-time constant. main() sets it (offline for an existing
# install, online when setup will download) before anything imports hf.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Quiet torch.compile / dynamo / inductor C++-side logging that the Python
# logger overrides above can't touch. Must be set before torch is imported
# (which happens via the core.assistant import below).
os.environ.setdefault("TORCH_LOGS", "")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCHINDUCTOR_VERBOSE", "0")

# Optional general settings forwarded to Assistant. Only keys actually present
# (and non-null) in config are passed, so every default lives in exactly one
# place — the Assistant / AudioCapture signature — and can't drift between here
# and there. Tests construct those classes directly and get the same defaults.
# The values are read fresh in `_assistant_args()` at Phase B (so a config the
# wizard just wrote is picked up); this is just the key list.
_ASSISTANT_OPTION_KEYS = (
    "wakeword_pattern",
    "voice_clone",
    "tts_speed",
    "whisper_gain",
    "barge_in",
    "conversation_mode_default",
    "max_voice_satellites",
    "barge_in_threshold_dbfs",
    "follow_up_time",
    "asr_language",
    "asr_context_hint",
    "asr_context_terms",
    "use_vad",
    "vad_threshold",
    "vad_endpoint_silence_ms",
    "vad_min_speech_ms",
    "vad_soft_endpoint_silence_ms",
    "save_wakeword_wavs",
    "persistent_logging_enabled",
    "personality",
    "personality_custom",
)

from core.assistant import Assistant
from core.bootstrap import ensure_scaffolding
from core.setup import detect_setup_state
from server.lifecycle import ERROR, LOADING, NEEDS_SETUP, AppContext, Lifecycle

# Port used to serve the wizard during first-run setup when none is configured.
_SETUP_FALLBACK_PORT = 8765


def _read_config():
    """Read config.yml fresh from disk (used after first-run seeding and at
    Phase B, so a config the wizard just wrote is picked up without a restart)."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _assistant_args(cfg):
    """Extract the Assistant constructor args from a fresh config dict."""
    general = cfg.get("general") or {}
    options = {k: general[k] for k in _ASSISTANT_OPTION_KEYS if general.get(k) is not None}
    # Map deprecated Higgs settings to personality settings.
    if "personality" not in options and general.get("higgs_personality") is not None:
        options["personality"] = general["higgs_personality"]
    if "personality_custom" not in options and general.get("higgs_personality_custom") is not None:
        options["personality_custom"] = general["higgs_personality_custom"]
    return general.get("wakeword"), cfg.get("models"), options


def _start_dashboard(context, host, port, certfile, keyfile, http_redirect_port=None):
    """Start the shared setup/dashboard server. The deprecated redirect-port argument is ignored."""
    try:
        from server.dashboard import start_dashboard

        start_dashboard(
            host=host,
            port=int(port),
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
            context=context,
            http_redirect_port=http_redirect_port,
        )
        scheme = "https" if (certfile and keyfile) else "http"
        logger.info("=" * 60)
        logger.info(" Fulloch — open the setup wizard / dashboard at:")
        logger.info("   %s://%s:%s", scheme, host, port)
        if scheme == "https":
            logger.info("   (typing http:// will 308-redirect here on the same port)")
        logger.info(" (first run boots the wizard; everything persists in ./data)")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Could not start dashboard: {e}")


def main():
    # First-run scaffolding: create the data subtree + seed config/grammar from
    # the bundled template so an empty ./data boots straight into the wizard.
    ensure_scaffolding()

    # Read config fresh (it may have just been seeded) and derive dashboard
    # settings + the setup decision from it.
    cfg = _read_config()
    general = cfg.get("general") or {}
    dash_port = general.get("dashboard_port")
    dash_host = os.environ.get("DASHBOARD_HOST", general.get("dashboard_host", "127.0.0.1"))
    if tz := general.get("timezone"):
        from utils.local_time import set_tz
        set_tz(tz)
    dash_cert = general.get("dashboard_ssl_certfile")
    dash_key = general.get("dashboard_ssl_keyfile")
    dash_http_redirect_port = general.get("dashboard_http_redirect_port")
    integration_api_enabled = general.get("integration_api_enabled", True)
    integration_api_port = general.get("integration_api_port", 8766)

    # Phase A: decide whether first-run setup is needed before loading anything.
    decision = detect_setup_state(cfg)

    # HF offline policy, set BEFORE anything imports huggingface_hub (it reads
    # this as an import-time constant): online when setup will download models so
    # the wizard can fetch them; offline for an existing install (privacy + no
    # network at runtime). hf isn't imported until the downloader / model load,
    # both of which happen after this point.
    os.environ["HF_HUB_OFFLINE"] = "0" if decision.needs_setup else "1"

    if decision.needs_setup:
        phase = ERROR if decision.config_error else NEEDS_SETUP
        detail = decision.config_error or decision.reason
        lifecycle = Lifecycle(phase=phase, detail=detail, missing_assets=decision.missing_assets)
    else:
        lifecycle = Lifecycle(phase=LOADING)

    context = AppContext(lifecycle=lifecycle, config_path=_CONFIG_PATH)

    if integration_api_enabled:
        from server.integration_api import start_integration_api

        start_integration_api(context, dash_host, int(integration_api_port))

    # The dashboard is one long-lived server shared by setup and run: the
    # wizard's install endpoint downloads models then releases the block below,
    # and the assistant is attached to this same server (no second server).
    if decision.needs_setup:
        _start_dashboard(
            context,
            dash_host,
            dash_port or _SETUP_FALLBACK_PORT,
            dash_cert,
            dash_key,
            http_redirect_port=dash_http_redirect_port,
        )
    elif dash_port:
        _start_dashboard(
            context,
            dash_host,
            dash_port,
            dash_cert,
            dash_key,
            http_redirect_port=dash_http_redirect_port,
        )

    if decision.needs_setup:
        logger.warning("Setup required: %s", detail)
        if decision.missing_assets:
            logger.warning("Missing model assets: %s", ", ".join(decision.missing_assets))
        if decision.config_error:
            # Fatal config mismatch — stay up to surface the error; don't run.
            logger.error("Config needs updating; fix it and restart.")
            try:
                lifecycle.proceed.wait()  # never signalled — Ctrl+C to exit
            except KeyboardInterrupt:
                pass
            return
        if decision.auto_download:
            # A completed setup with missing assets can resume downloading automatically.
            from server.dashboard import start_auto_download

            logger.info("Completed setup is missing model assets — downloading now.")
            start_auto_download(context)
        else:
            logger.info("Waiting in setup mode — open the dashboard to configure.")
        try:
            lifecycle.proceed.wait()
        except KeyboardInterrupt:
            logger.info("Stopping...")
            return

    # Phase B: build + run the assistant (config re-read fresh in case the
    # wizard just wrote it). Attaching it to the context flips the running
    # dashboard from setup page to live dashboard.
    wakeword, models, options = _assistant_args(_read_config())
    if not wakeword:
        logger.error("No wakeword configured; cannot start. Re-run setup.")
        return
    thinking = _read_config().get("thinking") or {}
    assistant = Assistant(
        wakeword=wakeword,
        models=models,
        thinking=thinking,
        lifecycle=lifecycle,
        **options,
    )
    context.set_assistant(assistant)
    assistant.run()


if __name__ == "__main__":
    main()
