"""Shared text cleanup for the TTS pipeline."""

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "☀-⛿"          # misc symbols
    "✀-➿"          # dingbats
    "]+",
    flags=re.UNICODE,
)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

# Split on whitespace that immediately follows a sentence terminator.
# Lookbehind keeps the punctuation with the preceding sentence so each
# returned string is itself a valid utterance.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_for_tts(text: str, strip_think: bool = True) -> str:
    """Prepare raw SLM output for speech: strip quotes, asterisks, <think>, emoji."""
    text = text.replace('"', '').replace('*', '')
    if strip_think:
        text = _THINK_PATTERN.sub("", text).strip()
    return _EMOJI_PATTERN.sub("", text)


def split_sentences(text: str) -> list:
    """Split `text` into sentences on `.`, `!`, `?` (punctuation preserved).

    Returns at least one element if `text` has any content. Edge cases
    like "Mr. Smith" will mis-split — fine for the assistant's own
    short outputs where abbreviations don't appear.
    """
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
