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

# Load .env before anything reads os.environ (e.g. tools/home_assistant.py reads
# FULLOCH_HA_TOKEN, server/dashboard.py reads FULLOCH_DASHBOARD_TOKEN). In Docker
# these are already injected via compose `env_file`, so load_dotenv is a no-op
# there (it never overrides existing vars); this covers native `python app.py`.
from dotenv import load_dotenv

load_dotenv()

# torch must be imported before llama_cpp (load-order side effect, not used here)
import torch  # noqa: F401
import llama_cpp

# Load configuration early so the log level (general.log_level) is available
# before logging is configured.
with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

_GENERAL = config['general']

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_log_level_name = str(_GENERAL.get('log_level', 'info')).lower()
LOG_LEVEL = _LOG_LEVELS.get(_log_level_name, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
if _log_level_name not in _LOG_LEVELS:
    logger.warning(f"Unknown general.log_level {_log_level_name!r}; using info")

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
logging.getLogger("transformers.generation.utils").setLevel(
    max(LOG_LEVEL, logging.ERROR)
)

# Drop llama-cpp-python's INFO chatter (model loader, CUDA init, KV-cache
# layout, and per-call perf timings — the dashboard stats panel reports those
# now). Warnings and errors always pass through. Installed before any
# `Llama(...)` instantiation.
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

# Set environment variables for offline mode and disabling telemetry
os.environ["HF_HUB_OFFLINE"] = "1"
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

# Configuration
WAKEWORD = _GENERAL['wakeword']

# Optional general settings forwarded to Assistant. Only keys actually present
# (and non-null) in config are passed, so every default lives in exactly one
# place — the Assistant / AudioCapture signature — and can't drift between here
# and there. Tests construct those classes directly and get the same defaults.
_ASSISTANT_OPTION_KEYS = (
    'wakeword_pattern', 'voice_clone', 'barge_in', 'barge_in_threshold_dbfs',
    'follow_up_time', 'input_device', 'output_device', 'asr_language',
    'asr_context_hint', 'asr_context_terms', 'use_vad', 'vad_threshold',
    'vad_endpoint_silence_ms', 'vad_min_speech_ms',
    'vad_soft_endpoint_silence_ms',
)
_ASSISTANT_OPTIONS = {
    k: _GENERAL[k] for k in _ASSISTANT_OPTION_KEYS if _GENERAL.get(k) is not None
}

DASHBOARD_PORT = _GENERAL.get('dashboard_port', None)
DASHBOARD_HOST = _GENERAL.get('dashboard_host', '127.0.0.1')
DASHBOARD_SSL_CERTFILE = _GENERAL.get('dashboard_ssl_certfile', None)
DASHBOARD_SSL_KEYFILE = _GENERAL.get('dashboard_ssl_keyfile', None)

from core.assistant import Assistant


def main():
    assistant = Assistant(wakeword=WAKEWORD, **_ASSISTANT_OPTIONS)
    if DASHBOARD_PORT:
        try:
            from server.dashboard import start_dashboard
            start_dashboard(
                assistant,
                host=DASHBOARD_HOST,
                port=int(DASHBOARD_PORT),
                ssl_certfile=DASHBOARD_SSL_CERTFILE,
                ssl_keyfile=DASHBOARD_SSL_KEYFILE,
            )
        except Exception as e:
            logger.error(f"Could not start dashboard: {e}")
    assistant.run()


if __name__ == "__main__":
    main()
