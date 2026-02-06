"""
Text-to-Speech module using Kokoro TTS.

Handles loading and running the Kokoro text-to-speech model.
"""

import logging
import re

import sounddevice as sd
import torch
from kokoro import KPipeline

logger = logging.getLogger(__name__)

# Model configuration
TTS_MODEL_NAME = "hexgrad/Kokoro-82M"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Emoji removal pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "]+",
    flags=re.UNICODE,
)

# Thinking removal pattern
THINK_PATTERN = r"<think>.*?</think>"

# Create global pipeline (loads model once)
logger.info(f"Loading {TTS_MODEL_NAME} on {DEVICE}...")
kpipeline = KPipeline(
    repo_id=TTS_MODEL_NAME,
    lang_code="a",  # "a" = auto
    device=DEVICE
)


def remove_emoji(text: str, rem_think: bool = True) -> str:
    """Remove emoji characters and thinking from text."""
    if rem_think:
        text = re.sub(THINK_PATTERN, "", text, flags=re.DOTALL)
        text = text.strip()

    return EMOJI_PATTERN.sub("", text)


def speak_stream(text: str, prompt=None, voice: str = "af_bella", speed: float = 1.2):
    """
    Generate speech from text using Kokoro and stream to speakers.

    Args:
        text: Text to synthesize
        prompt: Not used
        voice: Voice model to use (default: af_bella)
        speed: Speech speed multiplier (default: 1.2)
    """
    generator = kpipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r"\n+",
    )

    sample_rate = 24000  # Kokoro uses 24 kHz

    for _, _, audio in generator:
        sd.play(audio, samplerate=sample_rate, blocking=True)
