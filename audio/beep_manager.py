"""Thread-safe, non-blocking WAV playback for notification sounds."""

import threading
from pathlib import Path

import sounddevice as sd
import soundfile as sf

_WAV_DIR = Path(__file__).resolve().parent.parent / "wav"


class BeepManager:
    """Plays WAVs from the project's `wav/` directory on a daemon thread."""

    def __init__(self, device=None):
        # PortAudio device for playback (index, name, or None = system
        # default). Set by the Assistant from general.output_device so beeps
        # follow the same speaker as TTS.
        self.device = device

    def set_output_device(self, device) -> None:
        self.device = device

    def _play_beep(self, filename: str):
        data, samplerate = sf.read(_WAV_DIR / filename, dtype="float32")
        sd.play(data, samplerate, device=self.device)
        sd.wait()

    def play_beep(self, filename: str = "activation.wav"):
        """Play a WAV without blocking the caller."""
        threading.Thread(target=self._play_beep, args=(filename,), daemon=True).start()
