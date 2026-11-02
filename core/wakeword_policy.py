"""Lightweight ASR verification policy shared by runtime and setup metadata."""

# These decoders cannot reliably recover unusual names such as Atticus.
# Trust the acoustic detector instead of rejecting a correctly spoken wakeword.
ACOUSTIC_ONLY_ASR = frozenset({"moonshine", "moonshine-tiny", "parakeet-onnx"})


def verify_asr_wakeword(asr_backend: str, wakeword: dict) -> bool:
    # ASR-only activation still needs its transcript matcher. Never disable
    # the sole gate. Explicit config-file overrides remain available for KWS.
    if wakeword.get("backend") != "openwakeword":
        return True
    return bool(wakeword.get("verify_asr", asr_backend not in ACOUSTIC_ONLY_ASR))
