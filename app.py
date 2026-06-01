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
import os
import logging
import sys
import yaml
from pathlib import Path

# torch must be imported before llama_cpp
import torch
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
WAKEWORD_PATTERN = _GENERAL.get('wakeword_pattern', None)  # optional regex override
VOICE_CLONE = _GENERAL.get('voice_clone', None)
BARGE_IN = _GENERAL.get('barge_in', 'off')
FOLLOW_UP_TIME = _GENERAL.get('follow_up_time', '0s')
INPUT_DEVICE = _GENERAL.get('input_device', None)
OUTPUT_DEVICE = _GENERAL.get('output_device', None)
ASR_LANGUAGE = _GENERAL.get('asr_language', None)
DASHBOARD_PORT = _GENERAL.get('dashboard_port', None)
DASHBOARD_HOST = _GENERAL.get('dashboard_host', '127.0.0.1')

from core.assistant import Assistant

def main():
    assistant = Assistant(
        wakeword=WAKEWORD,
        wakeword_pattern=WAKEWORD_PATTERN,
        voice_clone=VOICE_CLONE,
        barge_in=BARGE_IN,
        follow_up_time=FOLLOW_UP_TIME,
        input_device=INPUT_DEVICE,
        output_device=OUTPUT_DEVICE,
        asr_language=ASR_LANGUAGE,
    )
    if DASHBOARD_PORT:
        try:
            from server.dashboard import start_dashboard
            start_dashboard(assistant, host=DASHBOARD_HOST, port=int(DASHBOARD_PORT))
        except Exception as e:
            logger.error(f"Could not start dashboard: {e}")
    assistant.run()


if __name__ == "__main__":
    main()
