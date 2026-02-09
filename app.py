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
import logging
import yaml
from pathlib import Path

# Suppress noisy "Setting pad_token_id to eos_token_id" from transformers
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from core.assistant import Assistant

# Configuration
WAKEWORD = config['general']['wakeword']
USE_AI = config['general']['use_ai']
USE_TINY_ASR = config['general'].get('use_tiny_asr', False)
USE_TINY_TTS = config['general'].get('use_tiny_tts', False)
VOICE_CLONE = config['general'].get('voice_clone', None)

def main():
    """Main entry point for the voice assistant."""
    assistant = Assistant(wakeword=WAKEWORD, use_ai=USE_AI, use_tiny_asr=USE_TINY_ASR, use_tiny_tts=USE_TINY_TTS, voice_clone=VOICE_CLONE)
    assistant.run()


if __name__ == "__main__":
    main()
