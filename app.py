#!/usr/bin/env python3
"""
Fulloch - The Fully Local Home Voice Assistant

A fully local, privacy-focused AI voice home assistant.
Runs speech recognition (Qwen3 ASR), text-to-speech (Qwen3 TTS),
and a small language model (Qwen3.5 9B) entirely on GPU.

Usage:
    python app.py
"""

import ctypes
import logging
import os
import sys
from pathlib import Path

import yaml

# Load env before anything reads os.environ. Root .env covers compose-level
# vars (FULLOCH_VERSION, DASHBOARD_PORT, etc.); credentials.json covers runtime
# secrets (ha_token, llm_api_key, dashboard_password, …) persisted in the Docker
# volume so a data-folder copy transfers everything to a new machine.
from dotenv import load_dotenv

from server.credentials_store import inject_env

load_dotenv()              # root .env — compose / shell vars; never overrides
inject_env()               # ./data/credentials.json → os.environ (setdefault)

# torch must be imported before llama_cpp (load-order side effect, not used here)
import torch  # noqa: F401

# llama_cpp is optional — the CPU image ships without it (regex + remote only).
# When present, its log callback is installed below to silence ggml chatter.
try:
    import llama_cpp
except ImportError:
    llama_cpp = None

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

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    "qwen_tts",
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
# "Setting pad_token_id to eos_token_id" fires at WARNING on every
# transformers.generate call — silence just that submodule.
logging.getLogger("transformers.generation.utils").setLevel(max(LOG_LEVEL, logging.ERROR))

# Drop llama-cpp-python's INFO chatter (model loader, CUDA init, KV-cache
# layout, and per-call perf timings — the dashboard stats panel reports those
# now). Warnings and errors always pass through. Installed before any
# `Llama(...)` instantiation. Skipped on the CPU image (no llama_cpp).
if llama_cpp is not None:

    @llama_cpp.llama_log_callback
    def _filtered_llama_log(level, text, user_data):
        # ggml log levels: 1=INFO, 2=WARN, 3=ERROR, 4=DEBUG, 5=CONT.
        if level in (1, 4, 5):
            return
        sys.stderr.write(text.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    llama_cpp.llama_log_set(_filtered_llama_log, ctypes.c_void_p(0))

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
    "barge_in",
    "barge_in_threshold_dbfs",
    "follow_up_time",
    "input_device",
    "output_device",
    "asr_language",
    "asr_context_hint",
    "asr_context_terms",
    "use_vad",
    "vad_threshold",
    "vad_endpoint_silence_ms",
    "vad_min_speech_ms",
    "vad_soft_endpoint_silence_ms",
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
    return general.get("wakeword"), cfg.get("models"), options


def _start_dashboard(context, host, port, certfile, keyfile):
    """Start the single dashboard server (setup + run share it). Never fatal."""
    try:
        from server.dashboard import start_dashboard

        start_dashboard(
            host=host,
            port=int(port),
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
            context=context,
        )
        scheme = "https" if (certfile and keyfile) else "http"
        logger.info("=" * 60)
        logger.info(" Fulloch — open the setup wizard / dashboard at:")
        logger.info("   %s://%s:%s", scheme, host, port)
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

    # The dashboard is one long-lived server shared by setup and run: the
    # wizard's install endpoint downloads models then releases the block below,
    # and the assistant is attached to this same server (no second server).
    if decision.needs_setup:
        _start_dashboard(context, dash_host, dash_port or _SETUP_FALLBACK_PORT, dash_cert, dash_key)
    elif dash_port:
        _start_dashboard(context, dash_host, dash_port, dash_cert, dash_key)

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
        if decision.config_present and decision.reason == "missing model assets":
            # An already-configured install (e.g. the user picked a new model in
            # Settings and restarted) is just missing the assets on disk — the
            # wizard has nothing left to ask, so skip straight to downloading
            # instead of re-showing it.
            from server.dashboard import start_auto_download

            logger.info("Config is up to date but missing model assets — downloading now.")
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
    assistant = Assistant(wakeword=wakeword, models=models, lifecycle=lifecycle, **options)
    context.set_assistant(assistant)
    assistant.run()


if __name__ == "__main__":
    main()
