"""Thread-safe, non-blocking WAV playback for notification sounds."""

import threading
from pathlib import Path

import sounddevice as sd
import soundfile as sf

_WAV_DIR = Path(__file__).resolve().parent.parent / "wav"


class BeepManager:
    """Plays WAVs from the project's `wav/` directory on a daemon thread."""

    def _play_beep(self, filename: str):
        data, samplerate = sf.read(_WAV_DIR / filename, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()

    def play_beep(self, filename: str = "activation.wav"):
        """Play a WAV without blocking the caller."""
        threading.Thread(target=self._play_beep, args=(filename,), daemon=True).start()
