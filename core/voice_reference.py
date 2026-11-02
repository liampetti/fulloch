"""Resolve bundled voice names and user-supplied WAV/transcript pairs."""

from pathlib import Path


def voice_reference(voice: str, voices_dir: str | Path) -> tuple[Path, Path]:
    root = Path(voices_dir)
    value = Path(voice or "")
    custom = value.suffix.lower() == ".wav"
    audio = value if custom else root / f"{voice}.wav"
    transcript = audio.with_suffix(".txt")
    if not custom and not transcript.is_file():
        transcript = root / "default.txt"
    # Custom clips must carry their own transcript: the bundled default text
    # describes a particular recording, not arbitrary user audio.
    if not audio.is_file():
        raise FileNotFoundError(f"Voice reference not found: {audio}")
    if not transcript.is_file():
        raise FileNotFoundError(f"Voice transcript not found: {transcript}")
    return audio, transcript
