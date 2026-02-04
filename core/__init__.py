"""
Core package for Fulloch voice assistant.

This package contains the core components:
- audio: Audio capture and silence detection
- asr: Automatic speech recognition (Moonshine)
- tts: Text-to-speech synthesis (Kokoro)
- slm: Small language model inference (Qwen)
- assistant: Main orchestration and wakeword detection
"""

from .audio import (
    AudioCapture,
    is_silent,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
)
from .asr import load_asr_model, stream_generator
from .tts import speak_stream
from .slm import load_slm, generate_slm
from .assistant import Assistant

__all__ = [
    # Audio
    "AudioCapture",
    "is_silent",
    "SAMPLE_RATE",
    "SILENCE_THRESHOLD",
    # ASR
    "load_asr_model",
    "stream_generator",
    # TTS
    "speak_stream",
    # SLM
    "load_slm",
    "generate_slm",
    # Assistant
    "Assistant",
]
