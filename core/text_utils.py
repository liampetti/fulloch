"""Shared text cleanup for the TTS pipeline."""

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "☀-⛿"  # misc symbols
    "✀-➿"  # dingbats
    "]+",
    flags=re.UNICODE,
)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
# Gemma 4 wraps its reasoning in a "thought" channel: `<|channel>thought ... <channel|>`.
# Its template also emits an empty `<|channel>thought\n<channel|>` ghost block even
# when thinking is off, so a stray one can surface in output — strip both. A lone
# unmatched closing `<channel|>` is mopped up too.
_GEMMA_THINK_PATTERN = re.compile(r"<\|channel>thought.*?<channel\|>", flags=re.DOTALL)
_GEMMA_CHANNEL_STRAY = re.compile(r"</?\|?channel\|?>")

# Split on whitespace that immediately follows a sentence terminator.
# Lookbehind keeps the punctuation with the preceding sentence so each
# returned string is itself a valid utterance.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Break at sentence + clause punctuation, keeping the delimiter on the left
# part. Shared by every TTS backend that overlaps synthesis with playback
# (core/tts_onnx.py, core/tts_crispasr.py — non-autoregressive/chunked
# backends where one fragment plays while the next renders) and by
# utils/reply_stream.py's incremental reply parser (A1a), which emits
# clause-level fragments as the SLM streams so TTS can start on the first
# clause instead of waiting for the whole reply.
CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+")


def split_clauses(text: str):
    """Yield clause/sentence fragments of `text`, dropping whitespace-only
    pieces. One fragment per clause so the first can start playing/rendering
    while the rest streams in or renders."""
    for fragment in CLAUSE_SPLIT.split(text.strip()):
        fragment = fragment.strip()
        if fragment:
            yield fragment


def clean_for_tts(text: str, strip_think: bool = True) -> str:
    """Prepare raw SLM output for speech: strip quotes, asterisks, <think>, emoji."""
    text = text.replace('"', "").replace("*", "")
    if strip_think:
        text = _THINK_PATTERN.sub("", text)
        text = _GEMMA_THINK_PATTERN.sub("", text)
        text = _GEMMA_CHANNEL_STRAY.sub("", text).strip()
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
