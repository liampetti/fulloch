"""Voice-clone reference generation, refactored out of the interactive CLI
(`scripts/voice_design.py`) into importable functions for the wizard (v2.2
Step 4).

The dashboard drives generate → preview (audio bytes the browser plays) → save
(`data/voices/<name>.{wav,txt}`), which the runtime Qwen Base clone picks up via
`general.voice_clone`. The heavy Qwen3-TTS VoiceDesign model is imported and
loaded lazily (and cached) so importing this module costs nothing.
"""

import io
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

VOICES_DIR = "./data/voices"
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

DEFAULT_PHRASE = (
    "A rainbow is a meteorological phenomenon that is caused by reflection, "
    "refraction and dispersion of light in water droplets resulting in a "
    "spectrum of light appearing in the sky."
)
PHRASE_MIN = 20
PHRASE_MAX = 400

_model = None
_model_lock = threading.Lock()
# Last generated (audio, sr, phrase) — saved by `save_last` so the previewed
# clip is exactly what's persisted (regenerating would differ).
_last: Optional[tuple] = None


def list_voices(voices_dir: Optional[str] = None) -> list:
    """Voice names usable as a clone under `voices_dir` (sorted).

    A name needs a .wav plus a transcript — either its own `<name>.txt` or the
    shared `default.txt` fallback (so bundled sample clips, all recorded with the
    same transcript, need no per-voice .txt). `default` itself is never a voice.
    """
    d = Path(voices_dir or VOICES_DIR)
    if not d.is_dir():
        return []
    wavs = {p.stem for p in d.glob("*.wav")}
    if (d / "default.txt").is_file():
        return sorted(wavs)
    return sorted(wavs & {p.stem for p in d.glob("*.txt")})


def safe_name(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (raw or "").strip().lower()).strip("-")
    return slug or "voice"


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode float32 audio to in-memory 16-bit PCM WAV for browser playback."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from qwen_tts import Qwen3TTSModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading %s on %s...", MODEL_NAME, device)
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
            if device == "cuda":
                kwargs["attn_implementation"] = "flash_attention_2"
            _model = Qwen3TTSModel.from_pretrained(MODEL_NAME, **kwargs)
        return _model


def generate(instruct: str, phrase: Optional[str] = None) -> tuple:
    """Generate a voice-design sample; cache it for `save_last`. Returns (audio, sr)."""
    global _last
    phrase = (phrase or "").strip() or DEFAULT_PHRASE
    if not (PHRASE_MIN <= len(phrase) <= PHRASE_MAX):
        raise ValueError(f"phrase must be {PHRASE_MIN}-{PHRASE_MAX} characters")
    if not (instruct or "").strip():
        raise ValueError("a voice description is required")

    model = _load_model()
    wavs, sr = model.generate_voice_design(
        text=phrase, instruct=instruct, language="english"
    )
    audio = wavs[0]
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    sr = int(sr)
    _last = (audio, sr, phrase)
    return audio, sr


def save_voice(name: str, audio: np.ndarray, sr: int, phrase: str,
               voices_dir: Optional[str] = None) -> tuple:
    """Write a `<name>.{wav,txt}` reference pair. Returns (wav_path, txt_path)."""
    import soundfile as sf
    name = safe_name(name)
    d = Path(voices_dir or VOICES_DIR)
    d.mkdir(parents=True, exist_ok=True)
    wav_path = d / f"{name}.wav"
    txt_path = d / f"{name}.txt"
    sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    txt_path.write_text(phrase.strip() + "\n", encoding="utf-8")
    return wav_path, txt_path


def save_last(name: str, voices_dir: Optional[str] = None) -> str:
    """Persist the most recently generated clip under `name`. Returns the name."""
    if _last is None:
        raise RuntimeError("no generated voice to save — generate one first")
    audio, sr, phrase = _last
    safe = safe_name(name)
    save_voice(safe, audio, sr, phrase, voices_dir=voices_dir)
    return safe
