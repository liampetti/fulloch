"""Interactive Voice Designer.

Generate a `data/voices/<name>.{wav,txt}` reference pair from a natural-
language voice description using Qwen3-TTS-12Hz-1.7B-VoiceDesign. The
runtime Base clone (`core/tts.py`) picks that pair up via the
`general.voice_clone` config key on next launch.

Standalone CLI alternative to the in-app voice designer (the setup wizard's
`/setup/voice`); run with the project venv: `python scripts/voice_design.py`.
"""

import os
import re
import sys
from pathlib import Path

# Point HF cache at our local data/models, matching app.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / "data" / "models"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

DEFAULT_PHRASE = (
    "A rainbow is a meteorological phenomenon that is caused by reflection, "
    "refraction and dispersion of light in water droplets resulting in a "
    "spectrum of light appearing in the sky."
)
DEFAULT_INSTRUCT = (
    "A warm, friendly Australian woman in her 30s, speaking at a relaxed pace."
)

PHRASE_MIN = 20
PHRASE_MAX = 400

VOICES_DIR = _REPO_ROOT / "data" / "voices"
CONFIG_PATH = _REPO_ROOT / "data" / "config.yml"

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def _hr(msg: str) -> None:
    print()
    print("─" * 64)
    print(msg)
    print("─" * 64)


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def _ask_yn(prompt: str, default: bool = False) -> bool:
    yn = "(Y/n)" if default else "(y/N)"
    ans = input(f"{prompt} {yn}: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def get_phrase() -> str:
    _hr("Choose a phrase to speak")
    print("  The model performs better with a varied, complete sentence.")
    print(f"  Example: {DEFAULT_PHRASE}")
    print(f"  ({PHRASE_MIN}-{PHRASE_MAX} chars; blank = use the example)")
    while True:
        text = input("Phrase: ").strip()
        if not text:
            return DEFAULT_PHRASE
        if len(text) < PHRASE_MIN:
            print(f"  Too short — at least {PHRASE_MIN} chars please.")
            continue
        if len(text) > PHRASE_MAX:
            print(f"  Too long — keep it under {PHRASE_MAX} chars.")
            continue
        return text


def get_instruct() -> str:
    _hr("Describe the voice you want")
    print(f"  Example: {DEFAULT_INSTRUCT}")
    while True:
        text = input("Voice description: ").strip()
        if text:
            return text
        if _ask_yn("  Use the example?", default=True):
            return DEFAULT_INSTRUCT


def load_model():
    print()
    print(f"⏳ Loading {MODEL_NAME}...")
    from qwen_tts import Qwen3TTSModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not detected — generation will be very slow.")
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
    if device == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3TTSModel.from_pretrained(MODEL_NAME, **kwargs)
    print("✅ Model loaded.")
    return model


def generate(model, phrase: str, instruct: str):
    wavs, sr = model.generate_voice_design(
        text=phrase, instruct=instruct, language="english"
    )
    audio = wavs[0]
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def play(audio: np.ndarray, sr: int) -> None:
    duration = len(audio) / sr if sr else 0
    print(f"🔊 Playing back ({duration:.1f}s)... (Ctrl-C to stop)")
    try:
        sd.play(audio, samplerate=sr, blocking=True)
    except KeyboardInterrupt:
        sd.stop()
        print("  Playback stopped.")


def save_voice(audio: np.ndarray, sr: int, phrase: str, name: str):
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = VOICES_DIR / f"{name}.wav"
    txt_path = VOICES_DIR / f"{name}.txt"
    sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    txt_path.write_text(phrase.strip() + "\n", encoding="utf-8")
    return wav_path, txt_path


def safe_name(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return slug or "voice"


def update_config(*, voice_clone: str = None, wakeword: str = None) -> bool:
    """Replace `voice_clone:` and/or `wakeword:` in data/config.yml.

    Operates on the first matching line in the `general:` block. Preserves
    surrounding indentation and comments.
    """
    if not CONFIG_PATH.exists():
        print(f"⚠️  {CONFIG_PATH} not found; skipping config update.")
        return False
    original = CONFIG_PATH.read_text(encoding="utf-8")
    new = original

    def _replace(text: str, key: str, value: str) -> tuple[str, bool]:
        pattern = re.compile(
            rf'^(?P<indent>\s*){key}:\s*["\']?[^"\'\n#]*["\']?',
            flags=re.MULTILINE,
        )
        replaced = [False]

        def _sub(m: re.Match) -> str:
            replaced[0] = True
            return f'{m.group("indent")}{key}: "{value}"'

        out = pattern.sub(_sub, text, count=1)
        return out, replaced[0]

    changed = False
    if voice_clone is not None:
        new, ok = _replace(new, "voice_clone", voice_clone)
        if not ok:
            print("⚠️  Couldn't find `voice_clone:` in config.yml.")
        changed = changed or ok
    if wakeword is not None:
        new, ok = _replace(new, "wakeword", wakeword)
        if not ok:
            print("⚠️  Couldn't find `wakeword:` in config.yml.")
        changed = changed or ok

    if not changed:
        return False
    CONFIG_PATH.write_text(new, encoding="utf-8")
    print(f"✅ Updated {CONFIG_PATH.relative_to(_REPO_ROOT)}.")
    return True


def review_menu() -> str:
    _hr("What next?")
    print("  s = save this voice and quit")
    print("  r = regenerate (same phrase & description)")
    print("  v = change voice description only (keep phrase)")
    print("  p = change phrase only (keep voice description)")
    print("  a = start over (new phrase + description)")
    print("  q = quit without saving")
    while True:
        choice = input("Choice [s/r/v/p/a/q]: ").strip().lower() or "s"
        if choice in ("s", "r", "v", "p", "a", "q"):
            return choice
        print("  Unknown — pick one of s / r / v / p / a / q.")


def post_save_prompts(name: str) -> None:
    if _ask_yn(
        f"Update data/config.yml to use voice_clone=\"{name}\"?", default=True
    ):
        update_config(voice_clone=name)
    if _ask_yn("Set a new wakeword for this voice?", default=False):
        while True:
            wake = _ask("New wakeword", default="").strip().lower()
            if not wake:
                print("  Skipping wakeword update.")
                break
            if not re.fullmatch(r"[a-z][a-z0-9 \-]{1,30}", wake):
                print("  Letters/digits/spaces/hyphens only, 2-31 chars.")
                continue
            update_config(wakeword=wake)
            break


def main() -> int:
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected.")
        if not _ask_yn("Continue anyway (very slow)?", default=False):
            return 1

    model = load_model()

    phrase = None
    instruct = None

    while True:
        if phrase is None:
            phrase = get_phrase()
        if instruct is None:
            instruct = get_instruct()

        print("\n🎙️  Generating...")
        try:
            audio, sr = generate(model, phrase, instruct)
        except Exception as e:
            print(f"❌ Generation failed: {type(e).__name__}: {e}")
            if not _ask_yn("Try again with a new prompt?", default=True):
                return 1
            phrase = instruct = None
            continue

        play(audio, sr)

        choice = review_menu()
        if choice == "s":
            default_name = safe_name(instruct.split(",")[0]) or "my-voice"
            while True:
                name = safe_name(_ask("Save as (voice name)", default=default_name))
                wav_path = VOICES_DIR / f"{name}.wav"
                if wav_path.exists() and not _ask_yn(
                    f"  {wav_path.name} already exists. Overwrite?", default=False
                ):
                    continue
                break
            wav_path, txt_path = save_voice(audio, sr, phrase, name)
            print(f"✅ Saved {wav_path.relative_to(_REPO_ROOT)}")
            print(f"✅ Saved {txt_path.relative_to(_REPO_ROOT)}")
            post_save_prompts(name)
            print("\n🎉 Done. Restart Fulloch to pick up the new voice.")
            return 0
        if choice == "r":
            continue
        if choice == "v":
            instruct = None
            continue
        if choice == "p":
            phrase = None
            continue
        if choice == "a":
            phrase = instruct = None
            continue
        if choice == "q":
            print("Quit without saving.")
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
