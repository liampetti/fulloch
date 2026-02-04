#!/usr/bin/env python3
"""
Fulloch - The Fully Local Home Voice Assistant

A fully local, privacy-focused AI voice home assistant.
Runs speech recognition (Moonshine ASR), text-to-speech (Kokoro TTS),
and a small language model (Qwen 3 4B) entirely on-device.

Usage:
    python app.py
"""

import os
import yaml
from pathlib import Path

# Load configuration
with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

# Point HF cache to models folder
models_dir = Path("./data/models").resolve()
os.environ["HF_HOME"] = str(models_dir)

# Set environment variables for offline mode and disabling telemetry
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.assistant import Assistant

# Configuration
WAKEWORD = config['general']['wakeword']
USE_AI = config['general']['use_ai']
USE_TINY_ASR = config['general'].get('use_tiny_asr', False)

def main():
    """Main entry point for the voice assistant."""
    assistant = Assistant(wakeword=WAKEWORD, use_ai=USE_AI, use_tiny_asr=USE_TINY_ASR)
    assistant.run()


if __name__ == "__main__":
    main()
