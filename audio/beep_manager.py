"""
BeepManager module for the voice assistant.

Handles playing beep/notification sounds in a thread-safe, non-blocking way.
"""

import threading
import sounddevice as sd
import soundfile as sf
import os

class BeepManager:
    """Manages beep/notification playback."""
    def __init__(self):
        # Get the path to the controller directory
        self.audio_dir = os.path.dirname(os.path.abspath(__file__))
        self.controller_dir = os.path.dirname(self.audio_dir)
        self.wav_dir = os.path.join(self.controller_dir, "wav")

    def _get_wav_path(self, filename: str) -> str:
        """Get full path to wav file in controller directory."""
        return os.path.join(self.wav_dir, filename)

    def _play_beep(self, filename: str = "activation.wav"):
        wav_path = self._get_wav_path(filename)
        data, samplerate = sf.read(wav_path, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()

    def play_beep(self, filename: str = "activation.wav"):
        """
        Play a beep sound in a non-blocking way.
        
        Args:
            filename: Name of wav file in controller/wav directory
        """
        threading.Thread(target=self._play_beep, args=(filename,), daemon=True).start()
