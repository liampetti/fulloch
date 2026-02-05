"""
Text-to-Speech module using Optimised Qwen3 TTS with two-phase latency & Hann crossfade for streaming: https://github.com/rekuenkdr/Qwen3-TTS-streaming

Handles loading and running the Kokoro text-to-speech model.
"""
import logging
import re
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

REF_AUDIO = "./data/voices/cori.wav"
REF_TEXT = "./data/voices/cori.txt"

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

def remove_emoji(text: str, rem_think: bool = True) -> str:
    """Remove emoji characters and thinking from text."""
    if rem_think:
        text = re.sub(THINK_PATTERN, "", text, flags=re.DOTALL)
        text = text.strip()

    return EMOJI_PATTERN.sub("", text)

# Get reference audio text
with open(REF_TEXT) as f:
    ref_text = f.read()
 
# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    attn_implementation="flash_attention_2",
)

# Enable optimizations (recommended)
model.enable_streaming_optimizations(
    decode_window_frames=80,         # Must match streaming parameter
    use_compile=True,                # torch.compile the decoder
    compile_mode="reduce-overhead",  # Includes CUDA graphs automatically
)

# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text=ref_text,
)

def speak_stream(text: str, voice: str = "cori", speed: float = 1.0):
    """
    Generate speech from text using stream optimised Qwen3 TTS from rekuenkdr

    Args:
        text: Text to synthesize
        voice: Not used
        speed: Not used
    """
    # Stream audio with two-phase settings
    for audio_chunk, sample_rate in model.stream_generate_voice_clone(
        text=text,
        language="english",
        voice_clone_prompt=prompt,
        overlap_samples=512,
        # Phase 2 settings (stable)
        emit_every_frames=12,
        decode_window_frames=80,
        # Phase 1 settings (fast first chunk)
        first_chunk_emit_every=5,
        first_chunk_decode_window=48,
        first_chunk_frames=48,
    ):
        
        sd.play(audio_chunk, samplerate=sample_rate, blocking=True)