"""
Automatic Speech Recognition module using Moonshine ASR.

Handles loading and running the Moonshine speech recognition model.
"""

import logging
from typing import Generator

import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration, pipeline

logger = logging.getLogger(__name__)

# Model configuration
ASR_MODEL_NAME = "UsefulSensors/moonshine-tiny"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_asr_model():
    """
    Load the Moonshine ASR model and create a pipeline.

    Returns:
        A Hugging Face pipeline configured for automatic speech recognition
    """
    logger.info(f"Loading {ASR_MODEL_NAME} on {DEVICE}...")

    processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)
    asr_model = MoonshineForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1 if DEVICE == "cuda" else 0,
        dtype=DTYPE,
    )

    return asr_pipe


def stream_generator(queue) -> Generator:
    """
    Generator that yields audio from a queue for the ASR pipeline.

    Args:
        queue: Queue containing audio numpy arrays. None signals stop.

    Yields:
        Audio numpy arrays until None is received
    """
    while True:
        audio = queue.get()
        if audio is None:
            break
        yield audio
