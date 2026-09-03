"""
Main assistant orchestration module.

Handles wakeword detection, intent processing, and response generation.
"""

import inspect
import json
import logging
import queue
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tools import notes, notes_root
from tools.capabilities import native_capabilities
from tools.thinking_context import reset_artifacts, set_artifacts
from tools.thinking_playbooks import matching_playbooks
from tools.tool_registry import ThinkingResult, thinking_result_error, tool_registry
from utils import intents
from utils.completeness import should_commit_provisional
from utils.intent_catch import catchAll
from utils.local_time import now as local_now
from utils.prompts import (
    CACHE_PRIMING_USER_PROMPT,
    get_agent_system_prompt,
    get_greeting_system_prompt,
    get_greeting_user_prompt,
    get_partial_thinking_summary_prompt,
    get_thinking_report_answer_prompt,
    get_thinking_report_prompt,
    get_thinking_worker_prompt,
    get_web_summary_system_prompt,
)

from .agent_loop import (
    _PROMPT_STRIP_CHARS,
    AgentLoop,
)
from .audio import AudioCapture
from .backends import ASR, LLM, TTS, get_module, resolve_models
from .background_jobs import BackgroundJob, BackgroundJobManager
from .higgs_controls import apply_delivery, extract_delivery_request
from .satellite import SatelliteSession
from .satellite_context import current_satellite_id as _current_satellite_id
from .slm import ContextExhaustedError, RemoteUnreachable, generate_slm, load_slm
from .telemetry import event as telemetry_event
from .text_utils import clean_for_tts, split_sentences, spoken_for_tts
from .tts_session import TtsSession, parse_barge_time
from .turn_arbiter import TurnArbiter
from .turn_stats import TurnStats, set_model_labels

logger = logging.getLogger(__name__)


class ConversationModeUnavailable(RuntimeError):
    """Raised when an exclusive Conversation mode session cannot be opened."""


# Names accepted by general.log_level, for the settings-console hot-apply
# (mirrors the map in app.py; the root logger level is the live handle).
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# Pre-rendered alert tone played over the satellite before a timer's spoken
# reminder (see Assistant._play_alarm_tone). Regenerate with dev/gen_sound.py.
ALARM_WAV_PATH = "./data/wav/alarm.wav"

# Background deliberate work needs room to investigate, revise its plan from
# tool observations, and still produce a final synthesis.
MAX_THINKING_WORKER_CALLS = 12
MAX_THINKING_CAPABILITY_CALLS = 3
# Background investigations can wait through a busy remote model queue without
# applying those longer waits to foreground conversation turns.
DEEP_THINK_READ_TIMEOUT_S = 600.0
DEEP_THINK_GENERATION_TIMEOUT_S = 900.0
# Planning actions can include structured multi-leg evidence. Leave room for
# larger models to finish those JSON arguments before the bounded worker loop
# advances to its next step.
DEEP_THINK_STEP_MAX_TOKENS = 1024
DEEP_THINK_TRANSCRIPT_MAX_CHARS = 12_000
DEEP_THINK_OBSERVATION_MAX_CHARS = 3_000

_THINKING_OBSERVATION_RE = re.compile(r"\n\n(?=\[[^\]\n]+\]\n)")

# Strips every non-word character. The self-echo check uses this so ASR's
# "1254" / "am" still match the assistant's spoken "12 54" / "a m" — the
# get_time tool spells digits out with spaces and ASR re-concatenates them.
_NON_WORD_RE = re.compile(r"\W+")


def _thinking_excerpt(text: str, limit: int) -> str:
    """Keep an observation useful without letting one result crowd out the rest."""
    if len(text) <= limit:
        return text
    marker = "\n...[compacted]...\n"
    remaining = limit - len(marker)
    if remaining <= 0:
        return text[:limit]
    head = remaining * 2 // 3
    return text[:head].rstrip() + marker + text[-(remaining - head) :].lstrip()


def _compact_thinking_transcript(transcript: str) -> str:
    """Bound ReAct observations while retaining every prior tool's context."""
    transcript = transcript.strip()
    if len(transcript) <= DEEP_THINK_TRANSCRIPT_MAX_CHARS:
        return transcript
    entries = _THINKING_OBSERVATION_RE.split(transcript)
    if len(entries) == 1:
        return _thinking_excerpt(transcript, DEEP_THINK_TRANSCRIPT_MAX_CHARS)
    labels = [entry.split("\n", 1)[0] for entry in entries]
    overhead = sum(len(label) + 2 for label in labels) + 2 * (len(entries) - 1)
    per_entry = max(160, (DEEP_THINK_TRANSCRIPT_MAX_CHARS - overhead) // len(entries))
    compacted = []
    for label, entry in zip(labels, entries, strict=True):
        body = entry[len(label) :].lstrip("\n")
        compacted.append(f"{label}\n{_thinking_excerpt(body, per_entry)}")
    return "\n\n".join(compacted)[:DEEP_THINK_TRANSCRIPT_MAX_CHARS]


def _append_thinking_observation(transcript: str, label: str, observation: str) -> str:
    """Append one bounded observation in a format safe to resume after a pause."""
    observation = _thinking_excerpt(observation.strip(), DEEP_THINK_OBSERVATION_MAX_CHARS)
    entry = f"[{label}]\n{observation}"
    combined = f"{transcript.strip()}\n\n{entry}" if transcript.strip() else entry
    return _compact_thinking_transcript(combined)


def _describe_thinking_capability(name: str, schema) -> str:
    """Render the callable shape a small ReAct worker needs, without prompt bloat."""
    args = []
    for param in schema.params:
        if param.required:
            args.append(param.name)
        else:
            args.append(f"{param.name}={param.default!r}")
    description = " ".join(schema.description.split())
    if len(description) > 240:
        description = description[:237].rstrip() + "..."
    return f"- {name}({', '.join(args)}): {description}"


def _thinking_action_key(name: str, args: list) -> str:
    """Identify equivalent worker actions independent of JSON object key order."""
    return f"{name}:{json.dumps(args, sort_keys=True, separators=(',', ':'), default=str)}"

# Short tokens that Qwen3-ASR (and Whisper) commonly hallucinate from
# background noise when English is enforced. None of these can match a real
# wakeword, so dropping them globally is safe. Checked against the
# punctuation-stripped lowercase transcription.
_ASR_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "yeah",
        "yep",
        "yup",
        "okay",
        "ok",
        "sure",
        "hmm",
        "hm",
        "uh",
        "um",
        "uh huh",
        "thanks",
        "thank you",
        "thank you very much",
        "you're welcome",
        "welcome",
        "bye",
        "goodbye",
        "see you",
        "see ya",
        "alright",
        "all right",
        "right",
        "oh",
        "so",
        "cough",
        # ASR prompt-echo: on non-speech (a cough), the context-biased decoder
        # regurgitates its context label instead of transcribing. Drop it before
        # routing so it can't open a spurious follow-up window. See
        # `asr_context_hint` / asr.py:context.
        "technical terms",
    }
)

# Bare stop commands that interrupt an active turn without any wakeword.
# Kept separate from _barge_re so the idle wakeword detector stays strict.
# These fire in _check_barge_in only (already gated by _turn_active), so
# the false-positive risk is low; _is_self_echo still runs first to suppress
# "stop" spoken by the assistant's own TTS.
_BARGE_STOP_RE = re.compile(
    r"\b(?:stop|halt|cancel|pause|quiet|enough)\b",
    re.IGNORECASE,
)

# Innocuous tokens allowed alongside a stop word without turning a bare stop
# into a redirect. "Atticus, stop talking now please" is still a pure stop —
# an instruction to cease entirely — not a new command. Used by
# `_is_pure_stop` to decide whether a barge-in should open a follow-up window.
_STOP_FILLER_TOKENS = frozenset(
    {
        "talking",
        "speaking",
        "now",
        "please",
        "it",
        "that",
        "this",
        "thanks",
        "thank",
        "you",
        "just",
        "right",
        "ok",
        "okay",
        "be",
    }
)


def _build_wakeword_pattern(wakeword: str) -> str:
    """Compile a tolerant wakeword matcher pattern.

    Tolerates two ASR foibles:
      - Punctuation / whitespace runs between tokens ("Hey, Atticus") via
        a `\\W+` join between words.
      - The s↔z swap that names like "Atticus" / "Atticuz" trigger — any
        `s` or `z` in the wakeword is matched by `[sz]`, so the same
        configured wakeword catches either pronunciation.
    Wraps the whole thing in word boundaries so "atticus" doesn't fire
    inside a longer word like "atticuses".
    """

    def _tolerant_word(word: str) -> str:
        return "".join("[sz]" if c in "sz" else re.escape(c) for c in word)

    tokens = [_tolerant_word(w) for w in wakeword.split()]
    return r"\b" + r"\W+".join(tokens) + r"\b"


# Greeting tokens that conventionally lead a wakeword ("hey atticus"). Users
# drop them when interrupting mid-speech — "Atticus, stop" is far more natural
# than "Hey Atticus, stop" — so the barge-in matcher accepts the bare name too.
_WAKE_GREETINGS = ("hey", "hay", "hi", "hello", "ok", "okay")


def _build_barge_pattern(wakeword: str, base_pattern: str) -> str:
    """Compile a prefix-tolerant matcher for the barge-in check only.

    Returns `base_pattern` OR-ed with a bare-name matcher built by stripping
    any leading greeting tokens off `wakeword` and applying the same tolerant
    builder as `_build_wakeword_pattern`. So while the assistant is speaking,
    "Hey Atticus" still matches via `base_pattern` and "Atticus" alone matches
    via the bare-name branch — but the strict prefix-requiring `base_pattern`
    stays in force for *idle* wakeword detection, keeping idle false-positives
    low. If the wakeword has no greeting prefix the bare name equals the full
    name and the extra branch is a harmless duplicate.
    """
    tokens = wakeword.split()
    while len(tokens) > 1 and tokens[0] in _WAKE_GREETINGS:
        tokens.pop(0)
    bare_name = _build_wakeword_pattern(" ".join(tokens))
    return f"(?:{base_pattern})|(?:{bare_name})"


# Spoken phrase pools live in `utils.phrases` — edit them there.
from utils.phrases import (  # noqa: E402
    ACK_CACHE_ATTRS,
    ACK_PHRASES,
    BUSY_PHRASES,
    GREETING_TOPICS,
    LLM_ERROR_PHRASES,
    NO_AI_PHRASES,
    REMINDER_PREFIX_PHRASES,
    STARTUP_CACHE_SPECS,
    TOOL_UNAVAILABLE_PHRASES,
)

# TTL on the partial-thinking capture. Past this, a 'summarise your
# thoughts' request returns a graceful 'no recent thoughts' message
# rather than dredging up a stale reasoning trace.
THINKING_PARTIAL_TTL_S = 60.0

# Token cap for the free-text *spoken* summaries (web-search result + cancelled
# thinking). Both prompts ask for a few sentences, but they run WITHOUT the agent
# GBNF grammar, so on a remote endpoint nothing bounds length — Qwen3.5 was seen
# generating 3000+ tokens (~45s) "summarising" web snippets. A spoken answer of a
# few sentences is well under this; truncating free text past it is harmless
# (unlike the JSON agent call, where a cap could break parsing). The local path
# was unaffected only because its grammar stops generation almost immediately.
SUMMARY_MAX_NEW_TOKENS = 256

# Unified conversation memory cap. Holds user / assistant (raw agent JSON) /
# tool entries so the agent sees the full turn trace on each replan. Sized
# generously — agent prompt cache stays warm; user / agent / tool messages
# average ~80 tokens each, so 50 entries ≈ 4k tokens of history.
HISTORY_MAX_MESSAGES = 50
# Floor for context-overflow recovery: keep at least this many of the most
# recent history entries (≈ the in-flight turn plus a little) when shedding old
# history to fit the context window. Only if even this won't fit do we fall back
# to clearing entirely.
CONTEXT_TRIM_KEEP_MIN = 4
# Silence timeout — past this gap with no new turn, history is cleared so a
# stale "you said earlier..." context can't bleed into a fresh conversation.
CHAT_SESSION_TIMEOUT_S = 1800.0
# Cap on a compacted tool-result entry's length (chars). Raw tool payloads
# (a web summary, a note dump, a state list) can be many KB; once a turn is
# complete, only a short trace survives compaction — see
# _compact_completed_turns.
COMPACTED_TOOL_TRACE_MAX_CHARS = 160

# Spoken only when the current request cannot fit an otherwise empty context.
# Normal overflow sheds old history, then clears it and retries silently.
CONTEXT_EXHAUSTED_REPLY = "I've lost some conversation context. Could you ask that again?"

# Time window after a barge-in cancel during which incoming ASR results
# are discarded. ASR inference is ~200ms on this GPU, so 500ms reliably
# catches anything already in flight at the moment of cancel; a real
# user reply takes ≥1.5s (silence detection + min utterance) to be
# enqueued by the recorder, so it lands well clear of this window.
DROP_AFTER_CANCEL_S = 0.5

# A failed remote endpoint should not be retried for every utterance in a
# noisy room. The next ordinary request after this delay probes for recovery.
REMOTE_LLM_RETRY_COOLDOWN_S = 30.0

# Conversation mode is permissive by design, so give overlapping speakers a
# brief chance to finish before sending only the latest request to the agent.
CONVERSATION_TURN_SETTLE_S = 0.75
DEFAULT_MAX_VOICE_SATELLITES = 6

# A bare wakeword (no command following) is the shape the ASR decoder most
# readily fabricates from voiced non-speech: the `asr_context_hint` prompt primes
# the wakeword spelling with nothing to anchor it, so a cough/sigh/TV burst
# decodes to "hey atticus". `_verify_bare_wakeword` gates opening the follow-up
# window on one. The loudness gate rejects a bare wakeword that sits at or below
# the background-speech baseline by this margin (dBFS) — a genuine wakeword is
# the user, closer/louder than ambient media. 0.0 = must be at least as loud as
# the baseline; raise it to demand a clearer margin over background.
BARE_WAKEWORD_MIN_OVER_BASELINE_DB = 0.0

# User-facing estimates for the reference 16GB GPU. They are intentionally
# broad: disk speed, CUDA cache state, and model choice materially affect load.
_MODEL_LOAD_ESTIMATES = {
    (ASR, "qwen"): "usually 10-20 seconds",
    (TTS, "qwen"): "usually 30-60 seconds",
    (LLM, "llama"): "usually 10-25 seconds",
    (ASR, "qwen-gguf"): "usually 5-15 seconds",
    (TTS, "qwen-gguf"): "usually 5-15 seconds",
    (TTS, "pocket-tts-gguf"): "usually 5-15 seconds",
    (TTS, "pocket-tts-pytorch"): "usually 5-15 seconds",
}


class _GainSink:
    """Apply a per-turn gain without changing the shared TTS sink protocol."""

    def __init__(self, sink, gain: float):
        self._sink = sink
        self._gain = gain

    def put(self, item, *args, **kwargs) -> None:
        if isinstance(item, tuple) and item and isinstance(item[0], np.ndarray):
            item = (item[0] * self._gain, *item[1:])
        self._sink.put(item, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sink, name)


class _NonBlockingSink:
    """Apply transport backpressure without delaying cancellation controls.

    TTS PCM must remain contiguous: dropping frames when a transport queue fills
    causes audible garbling and truncates fast backends such as Pocket TTS.
    Producers therefore wait for audio space. Start, cancel, and stop still
    replace queued audio immediately so a new or cancelled turn is responsive.
    """

    def __init__(self, sink: "queue.Queue"):
        self._sink = sink

    def put(self, item, *args, **kwargs) -> None:
        kind = item[0] if isinstance(item, tuple) and item else None
        if not isinstance(kind, str):
            self._sink.put(item, *args, **kwargs)
            return
        try:
            self._sink.put_nowait(item)
            return
        except queue.Full:
            pass
        if kind in {"start", "cancel", "stop"}:
            self._drain()
        else:  # Never displace PCM merely to enqueue a completion marker.
            self._sink.put(item, *args, **kwargs)
            return
        try:
            self._sink.put_nowait(item)
        except queue.Full:
            pass

    def put_nowait(self, item) -> None:
        self.put(item)

    def get_nowait(self):
        return self._sink.get_nowait()

    def _drain(self) -> None:
        while True:
            try:
                self._sink.get_nowait()
            except queue.Empty:
                return


class _BroadcastSink:
    """Fan out one TTS stream to every active proactive-speech recipient."""

    def __init__(self, sinks: list):
        self._sinks = sinks

    def put(self, item, *args, **kwargs) -> None:
        for sink in self._sinks:
            sink.put(item, *args, **kwargs)


class _BroadcastEvent:
    """Mirror a TTS backend's active/inactive transition across satellites."""

    def __init__(self, events: list[threading.Event]):
        self._events = events

    def set(self) -> None:
        for event in self._events:
            event.set()

    def clear(self) -> None:
        for event in self._events:
            event.clear()


def _safe_sink(sink: Optional["queue.Queue"]):
    # Both browser and native transports use bounded hand-off queues. Preserve
    # every PCM frame by letting their send loops pace the TTS producer.
    return _NonBlockingSink(sink) if sink is not None and sink.maxsize > 0 else sink


def _whisper_gain(value) -> float:
    """Normalise the configured PCM gain for explicit quiet delivery."""
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        logger.warning("Invalid whisper_gain %r; using 0.30", value)
        return 0.30


class Assistant:
    """Owns the audio capture and runs the wakeword → intent → TTS loop."""

    def __init__(
        self,
        wakeword: str,
        wakeword_pattern: Optional[str] = None,
        voice_clone: Optional[str] = None,
        tts_speed: Optional[float] = None,
        whisper_gain: float = 0.30,
        personality: str = "balanced",
        personality_custom: str = "",
        barge_in: str = "off",
        conversation_mode_default: bool = False,
        max_voice_satellites: int = DEFAULT_MAX_VOICE_SATELLITES,
        barge_in_threshold_dbfs: Optional[float] = None,
        follow_up_time: str = "0s",
        asr_language: Optional[str] = None,
        asr_context_hint: bool = True,
        asr_context_terms: list = None,
        use_vad: bool = True,
        vad_threshold: Optional[float] = None,
        vad_endpoint_silence_ms: Optional[int] = None,
        vad_min_speech_ms: Optional[int] = None,
        vad_soft_endpoint_silence_ms: Optional[int] = None,
        save_wakeword_wavs: bool = False,
        persistent_logging_enabled: bool = False,
        models: Optional[dict] = None,
        thinking: Optional[dict] = None,
        lifecycle=None,
    ):
        """
        Initialize the assistant.

        Args:
            wakeword: Activation phrase to listen for
            wakeword_pattern: Optional explicit regex (compiled case-insensitive)
                that overrides the auto-built tolerant matcher. Lets the config
                express phonetic swaps the auto-builder can't — e.g. f↔t for
                "hey fulloch" / "hey tulloch". An invalid pattern logs a warning
                and falls back to the auto-built pattern for `wakeword`.
            voice_clone: Name of a `data/voices/<name>.{wav,txt}` reference
                pair used to clone the speaking voice. The Base Qwen3-TTS
                model conditions every generation on this clone, so the
                speaker stays consistent across turns.
            whisper_gain: PCM amplitude used for explicit whisper/quiet requests,
                from 0.0 (silent) to 1.0 (normal volume).
            barge_in: "off" (half-duplex) or "wakeword" (interrupt with wakeword)
            conversation_mode_default: Let the first connected satellite enter
                exclusive Conversation mode automatically, disconnecting other
                voice satellites until it is disabled.
            max_voice_satellites: Maximum concurrent voice clients. Each one
                owns stateful VAD and optional wake-word models.
            barge_in_threshold_dbfs: Silence floor while TTS plays, in dBFS
                (the unit transcription volume is logged in). An interrupting
                voice must exceed it to be captured — lower = easier to barge
                in. None uses the built-in default (BARGE_IN_THRESHOLD_DBFS,
                -48 dBFS). Useful on a speakerphone that ducks its mic during
                playback (so interrupts arrive quiet); raise it if the assistant
                self-interrupts.
            follow_up_time: "0s" or "<N>s" — window after TTS ends during
                which a wakeword-free utterance counts as a follow-up turn.
        """
        self.wakeword = wakeword.lower()
        self.max_voice_satellites = max(1, int(max_voice_satellites))
        # Spoken name for self-introduction: the bare wakeword with any
        # leading greeting token stripped ("hey atticus" -> "Atticus"), so
        # the greeting introduces it by what the user actually calls it.
        _name_tokens = self.wakeword.split()
        while len(_name_tokens) > 1 and _name_tokens[0] in _WAKE_GREETINGS:
            _name_tokens.pop(0)
        self.wakeword_name = " ".join(_name_tokens).title() or "Fulloch"
        # An explicit regex in config wins over the auto-built matcher — lets
        # you express phonetic swaps the builder can't (f↔t, u↔a). A bad
        # pattern from a hand-edited config degrades to the auto-builder rather
        # than crashing startup.
        pattern = wakeword_pattern or _build_wakeword_pattern(self.wakeword)
        try:
            self._wakeword_re = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(
                f"Invalid wakeword_pattern {pattern!r} ({e}); "
                f"falling back to auto-built pattern for {self.wakeword!r}"
            )
            pattern = _build_wakeword_pattern(self.wakeword)
            self._wakeword_re = re.compile(pattern, re.IGNORECASE)
        # Barge-in matcher: same pattern, but with the greeting prefix made
        # optional so a mid-speech "Atticus, stop" interrupts. Used ONLY in
        # `_check_barge_in`; idle detection keeps the strict `_wakeword_re`.
        self._barge_re = re.compile(
            _build_barge_pattern(self.wakeword, pattern),
            re.IGNORECASE,
        )
        self.voice_clone = voice_clone
        self.tts_speed = tts_speed
        self.whisper_gain = _whisper_gain(whisper_gain)
        self.personality = personality
        self.personality_custom = personality_custom
        self.asr_language = asr_language
        self.asr_context_hint = asr_context_hint
        self.asr_context_terms = asr_context_terms or []
        # Flat pool of every token that appears in any `asr_context_terms`
        # entry (multi-word terms split into their words). A wakeword command
        # built *only* from these tokens is almost certainly the ASR decoder
        # echoing its bias prompt from non-speech — see `_is_context_echo`.
        self._asr_context_tokens = frozenset(
            tok
            for term in self.asr_context_terms
            for tok in re.sub(r"[^\w\s]", "", term.lower()).split()
        )
        # Leading scaffolding of the `asr_context_hint` prompt (e.g. "technical
        # terms"); set when the context is built in `_load_models`. The decoder
        # echoes it verbatim off non-speech, so the transcriber drops any result
        # containing it. Empty when the hint is off.
        self._context_prompt_marker = ""
        self.use_vad = use_vad
        self.persistent_logging_enabled = bool(persistent_logging_enabled)
        # Built in `_load_models()` once the TTS model is in memory.
        self.voice_clone_prompt = None
        self.audio_capture = AudioCapture(
            barge_in_threshold_dbfs=barge_in_threshold_dbfs,
            use_vad=use_vad,
            vad_threshold=vad_threshold,
            vad_endpoint_silence_ms=vad_endpoint_silence_ms,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_soft_endpoint_silence_ms=vad_soft_endpoint_silence_ms,
            save_wakeword_wavs=save_wakeword_wavs,
        )
        self.audio_capture.set_wakeword_detected_callback(self._on_wakeword_model_match)
        self.audio_capture.set_asr_work_dropped_callback(self._on_asr_work_dropped)
        # Mic stays muted through model load and the opening greeting;
        # `_warm_and_announce`'s `finally` flips it on once warmup ends. This
        # is the genuinely global HA-switch-facing flag (see
        # AudioCapture.mic_globally_enabled's docstring) — every satellite's
        # recorder also gates on its own SatelliteSession.transcribing, which
        # is what the half-duplex self-mute uses.
        self.audio_capture.mic_globally_enabled = False
        self.tts_session = TtsSession()

        if barge_in not in ("off", "wakeword"):
            logger.warning(f"Invalid barge_in: {barge_in!r}; defaulting to 'off'")
            barge_in = "off"
        self.barge_in = barge_in
        self.conversation_mode_default = bool(conversation_mode_default)
        self.follow_up_seconds = parse_barge_time(follow_up_time)
        self.tts_start_time = 0.0
        # Written by the ASR stream generator with the speech-onset time of
        # the buffer currently being transcribed. The follow-up window
        # measures against this (when the user started talking) rather than
        # when the transcription lands, so a long reply isn't penalised.
        self._asr_onset: dict = {"t": 0.0}
        # Written by the ASR stream generator with the dBFS loudness of the
        # buffer currently being transcribed (voiced-window RMS where VAD is
        # active). Logged alongside each transcription and fed into the
        # background-noise baseline; nothing acts on it yet.
        self._asr_loudness: dict = {"db": None}
        # Written by the ASR stream generator: True when the buffer currently
        # being transcribed is a *provisional* soft-endpoint snapshot of an
        # unfinished utterance (the speculative early-commit path). Such a
        # result may only commit a turn at the dispatch gate — never trigger the
        # follow-up/stand-down/baseline side effects — and is ignored mid-turn.
        self._asr_provisional: dict = {"flag": False}
        # Written by the ASR stream generator: the raw audio buffer currently
        # being transcribed, kept so a bare wakeword can be re-transcribed
        # without the context bias to confirm it isn't a bias-prompt echo (see
        # `_verify_bare_wakeword`).
        self._asr_audio: dict = {"buf": None}

        # Unified conversation memory — every user / assistant / tool entry
        # for the agent's view of the session. Mutated only from the active
        # turn (serialised under `_turn_lock` for the SLM portion; a turn's
        # own history writes are done by the time it releases the lock) so no
        # extra lock is needed. Shared across every satellite ("one assistant
        # mind heard from multiple rooms") — see `_history_for`.
        # Each entry: {"role": "user"|"assistant"|"tool", "content": str,
        #              "name": <intent>?}  (assistant content is raw agent
        # JSON; tool content is the tool's return string).
        self._history: list = []
        # Completed background reports stay outside compacted history as durable
        # references. Follow-up answers re-read the saved report rather than
        # asking the conversational model to reconstruct its findings.
        self._completed_thinking_reports: dict[str, dict[str, object]] = {}
        self._pending_thinking_tasks: dict[str, str] = {}

        # Partial-thinking capture for the interrupt-and-summarise path.
        # Set when a thinking-mode chat call is cancelled mid-stream;
        # consumed by `summarize_thinking` on the next turn. TTL'd via
        # `THINKING_PARTIAL_TTL_S` so a stale trace doesn't leak forward.
        self._last_thinking_partial: Optional[str] = None
        self._last_thinking_question: Optional[str] = None
        self._last_thinking_cancelled_at: float = 0.0

        # Reminder poll deduplication — keyed on (summary, YYYY-MM-DD).
        # Value is the monotonic time after which the entry expires (2 hours),
        # so the same reminder can re-fire on a different day or after a restart.
        self._spoken_reminders: dict[tuple[str, str], float] = {}

        # Backend selection. `models` is the parsed `models:` config block
        # (or None — defaults to the Qwen stack). Resolved once through the
        # registry so `_load_models` and the no-LLM gate share one source of
        # truth. `llm_enabled` is False for `llm.backend: none`, which runs a
        # regex-only assistant that never touches the SLM.
        self._models_raw = models or {}
        thinking = thinking or {}
        self.thinking_enabled = True
        try:
            requested_slots = int(thinking.get("server_slots", 1))
        except (TypeError, ValueError):
            requested_slots = 1
        self.thinking_server_slots = min(max(requested_slots, 1), 2)
        if self.thinking_server_slots != requested_slots:
            logger.warning("Invalid thinking.server_slots %r; using %d", requested_slots, self.thinking_server_slots)
        self._models = resolve_models(models)
        self.thinking_jobs = (
            BackgroundJobManager(
                self._run_background_thinking_job,
                self._on_thinking_job_status,
                self._background_job_admitted,
            )
        )
        self.llm_enabled = self._models[LLM]["backend"] != "none"
        # The resolved LLM backend name (e.g. "llama"/"openai"/"none"). The
        # dashboard reads this to swap to the "Parloch" branding when the model
        # runs off-device over a remote OpenAI-compatible endpoint.
        self.llm_backend = self._models[LLM]["backend"]
        # Remote-LLM reachability, for the dashboard's "Parloch unreachable" banner.
        # Only meaningful when llm_backend == "openai". None = not yet probed;
        # True/False = last known reachability. Set by a startup probe and updated
        # on every turn (a RemoteUnreachable catch flips it False, a successful
        # remote generation flips it True). The branding (Parloch) keys off the
        # *configured* backend, not this — an off-device LLM that's down is still
        # Parloch, just degraded to regex/fast-path only.
        self._llm_remote_ok: bool | None = None
        self._llm_remote_error: str = ""
        self._llm_remote_retry_at = 0.0
        self._llm_remote_outage_announced = False
        self._llm_remote_lock = threading.Lock()
        # Optional startup-lifecycle handle (server.lifecycle.Lifecycle). When
        # present, `_load_models` advances it LOADING -> READY so the dashboard
        # can show a loading screen and hand off to the assistant when the
        # greeting fires. None in tests / non-server use.
        self.lifecycle = lifecycle
        logger.info(
            "Backends: asr=%s tts=%s llm=%s",
            self._models["asr"]["backend"],
            self._models["tts"]["backend"],
            self._models[LLM]["backend"],
        )

        # Models loaded lazily in transcriber thread
        self.asr_pipe = None
        self.asr_stream_generator = None
        self.slm_model = None
        self.grammar = None
        self.greeting_prompt = None
        self.web_summary_prompt = None
        # Pre-rendered fallback clips, populated in `_warm_and_announce`.
        self.no_ai_cache: list = []
        self.llm_error_cache: list = []
        self.tool_unavailable_cache: list = []
        self.conversation_listening_cache: list = []
        # Pre-rendered opening-greeting clips. `_warm_and_announce` synthesises
        # and attempts to play these during startup — before any browser could
        # possibly be connected (the WebSocket satellite only exists once the
        # dashboard is already serving, which is after warmup) — so that first
        # attempt silently drops the audio (no sink yet). Cached here and
        # replayed once by `replay_greeting()` on the first satellite
        # connection instead. `_greeting_delivered` guards against replaying
        # on every reconnect (tab refresh, follow-up browser tab, etc).
        self.greeting_cache: list = []
        self.greeting_text = ""
        # Kept outside conversation history solely for an immediate, explicit
        # "tell me more about that" response to the startup greeting.
        self._startup_greeting_response: str | None = None
        self._greeting_delivered = False

        # Serializes SLM access. Voice turns run on the transcriber or barge-in
        # worker; text turns from the dashboard run on the FastAPI thread —
        # the local model server and history are shared, so all `_handle_wakeword`
        # callers acquire this. Voice barge-in cancels mid-stream so the lock releases
        # quickly even while a text turn waits behind it.
        self._turn_lock = threading.Lock()
        # Exclusive ownership of "whose turn is it" across every satellite
        # and the dashboard's "dashboard-text" pseudo-satellite — spans the
        # whole turn (dispatch decision through TTS), not just the SLM call
        # `_turn_lock` guards. Closes the text/voice turn race Phase 2 left
        # open: two sources could each run a turn concurrently (serialised
        # only at the SLM call), each safely bookkept on its own
        # SatelliteSession, but still interleaving _history writes.
        self._turn_arbiter = TurnArbiter()

        # Dashboard / external observers subscribe here to receive every
        # finalised user/assistant exchange. Listeners must be cheap (they
        # run on the calling thread) and exceptions are swallowed.
        self._turn_listeners: list = []
        self._turn_listeners_lock = threading.Lock()

        # Satellite (browser WebSocket, and from Phase 5 onward /ws/satellite-v2)
        # sessions, keyed by satellite_id. Each connect mints a SatelliteSession
        # (core/satellite.py) holding that connection's own chunk_q/tts_sink and
        # turn/echo/follow-up/noise-baseline state, so two satellites connected
        # at once don't clobber each other's queues *or* each other's turns —
        # a barge-in / follow-up / self-echo decision for satellite B is judged
        # purely against B's own history, never A's.
        self.satellites: dict[str, SatelliteSession] = {}
        # Conversation mode is intentionally exclusive. Full-duplex audio from
        # more than one room would make turn attribution and echo control unsafe.
        self._conversation_lock = threading.Lock()
        self._conversation_owner_id: Optional[str] = None
        # Dashboard text turns (`handle_text_turn`) aren't a satellite at all,
        # but need the same per-turn state (active_session, last_turn_end for
        # the history-timeout check, etc.) as a real one — a reserved,
        # never-disconnected pseudo-session gets that for free instead of a
        # parallel set of bare `self._text_*` fields. `chunk_q=None` since it
        # never receives audio; `tts_sink=None` since text turns never speak.
        self.satellites["dashboard-text"] = SatelliteSession(id="dashboard-text", chunk_q=None)
        # Retained for legacy callers that need a default single satellite.
        # Untargeted proactive speech broadcasts to every connected satellite.
        self._last_connected_satellite_id: Optional[str] = None
        # Per-thread scratch space for whichever turn is running on *this*
        # thread — `.sink` / `.tts_active_event`, resolved fresh at the top of
        # `_run_half_duplex`/`_run_turn` from the turn's own satellite session
        # and read by the `play_chunks`/`speak_stream` wrapper methods below.
        # A plain instance attribute would be wrong here: in barge-in mode two
        # satellites' turns can genuinely run concurrently (each on its own
        # `_turn_thread`, serialised only at the SLM call via `_turn_lock`),
        # so a shared field would let one satellite's sink leak into the
        # other's `play_chunks` call. threading.local() gives each turn's
        # thread its own slot for free.
        self._turn_local = threading.local()
        # Written by the ASR stream generator with the id of the satellite
        # that recorded the buffer currently being transcribed — mirrors
        # `_asr_onset`/`_asr_loudness`/`_asr_provisional` above. Lets the
        # transcriber loop route a turn's reply back to the satellite that
        # actually spoke, even though `_run_transcriber_loop` still drains one
        # shared `audio_queue` fed by every connected satellite's recorder.
        self._asr_satellite_id: dict = {"id": None}
        # Written by the ASR stream generator (A2): seconds the buffer
        # currently being transcribed sat queued between the recorder
        # detecting its endpoint and ASR dequeuing it. None when the
        # recorder didn't tag an endpoint time (old-style queue item).
        # Snapshotted into a turn's `TurnStats.endpoint_wait_seconds`
        # alongside `stt_seconds` at dispatch.
        self._asr_endpoint_wait: dict = {"s": None}
        # True only for the one early, speech-onset ASR probe per utterance.
        # It can confirm a wakeword to the satellite, never dispatch a command.
        self._asr_wake_probe: dict = {"flag": False}
        self._asr_kws_candidate: dict = {"flag": False}
        # Path of the complete candidate utterance, attached to the queued audio
        # so recorder cleanup cannot lose it before ASR verifies the wakeword.
        self._asr_kws_wav_path: dict = {"path": None}
        self._asr_wake_generation: dict = {"value": None}
        self.wakeword_metrics = {"accepted": 0, "rejected": 0, "status": "asr"}
        # A2: the last completed turn's stats payload (`TurnStats.to_payload()`),
        # for `GET /status` to expose without the caller needing to scrape SSE
        # history. None until the first turn finishes.
        self._last_turn_stats: Optional[dict] = None

        # Obsidian plugin state — set by the dashboard when the plugin connects.
        # vault_current_file: metadata about the note currently open in Obsidian.
        self._vault_current_file: Optional[dict] = None
        self._vault_path: Optional[str] = None

        # Set once `_load_models()` finishes — text turns from the dashboard
        # wait on this before calling into the pipeline.
        self.models_ready = threading.Event()

    @staticmethod
    def _llm_stat_label(llm_cfg: dict) -> str:
        """A concise LLM label for the dashboard stats panel."""
        backend = llm_cfg["backend"]
        if backend == "none":
            return "none (regex-only)"
        if backend == "openai":
            return f"OpenAI: {llm_cfg.get('model') or '?'}"
        # Local llama.cpp — show the gguf filename without the extension.
        import os

        name = os.path.basename(str(llm_cfg.get("model") or ""))
        return name[:-5] if name.endswith(".gguf") else (name or llm_cfg["spec"].display_name)

    def _diagnose_failure(self, exc, spec=None) -> tuple:
        """Return ``(fatal, message)`` for a model load/runtime failure.

        Turns an opaque backend exception into a plain-language line for the
        setup screen's red alert, so the user can pick a working configuration
        without trawling debug logs. ``fatal`` is True for GPU / out-of-memory
        class failures — the model or CUDA context is poisoned and every later
        turn would fail too, which is the signal to bounce back to setup.

        The common failure on this hardware is simply too large a model for the
        card, so when VRAM is readable we name the free / required amounts.
        """
        # Markers of an unrecoverable GPU failure. llama.cpp reports an OOM at
        # context creation as the opaque "Failed to create llama_context";
        # torch / CUDA say "out of memory" or raise a CUDA error.
        markers = (
            "out of memory",
            "failed to create llama_context",
            "cudamalloc",
            "cuda error",
            "cublas",
            "cudnn",
            "device-side assert",
            "illegal memory access",
            "memory fault",
            "mmu fault",
            "xid",
            "ggml_cuda",
        )
        name = getattr(spec, "display_name", None) or "the model"
        msg = (str(exc) or exc.__class__.__name__).strip()
        low = msg.lower()

        # A missing or incomplete model file — download was interrupted or the
        # model directory is only partially populated. Allow re-download by
        # clearing the offline flag before bouncing back to the wizard.
        is_missing = (
            exc.__class__.__name__ == "LocalEntryNotFoundError"
            or "localentrynotfounderror" in low
            or ("hf_hub_offline" in low and "set" in low)
            or (isinstance(exc, (FileNotFoundError, OSError)) and "No such file" in msg)
        )
        if is_missing:
            import os as _os

            _os.environ["HF_HUB_OFFLINE"] = "0"
            return True, (
                f"{name} model file is missing or incomplete — the download may have been "
                "interrupted. Click 'Re-run setup wizard' to re-download."
            )
        fatal = any(m in low for m in markers)
        looks_oom = any(m in low for m in ("out of memory", "llama_context", "cudamalloc"))

        vram = ""
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                free_gb, total_gb = free / 1e9, total / 1e9
                need = getattr(spec, "vram_gb", 0) or 0
                vram = f" ({free_gb:.1f} GB of {total_gb:.1f} GB GPU memory free"
                if need:
                    vram += f"; {name} needs about {need:.0f} GB"
                    if free_gb + 0.3 < need:  # short of the model's own estimate
                        fatal = looks_oom = True
                vram += ")."
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            pass

        if looks_oom:
            return True, (
                f"Out of GPU memory — the current setup is too large for your "
                f"hardware.{vram} Choose a lighter tier or a smaller model below."
            )
        if fatal:
            if any(m in low for m in ("illegal memory access", "memory fault", "mmu fault", "xid")):
                return True, (
                    f"{name} triggered a GPU memory fault. This is not a normal out-of-memory error; "
                    "disable experimental MTP and Flash Attention, then retry. "
                    "For an NVIDIA Xid, inspect the kernel log on the system that owns the GPU. "
                    f"Server details: {msg}{vram}"
                )
            return True, (
                f"{name} hit an unrecoverable GPU error and stopped: {msg}.{vram} "
                f"Try a lighter configuration below."
            )
        return False, f"{name} error: {msg}"

    def _enter_error_state(self, detail: str) -> None:
        """Flip the app to the ERROR lifecycle phase with a user-facing detail.

        The web UI polls ``/status``; on ERROR the dashboard redirects to the
        setup screen, which shows ``detail`` in a red alert and drops into the
        wizard so the user can choose a working configuration. Safe no-op when
        there's no lifecycle (tests / headless runs).
        """
        if self.lifecycle is not None:
            self.lifecycle.set("ERROR", detail)

    def _note_runtime_error(self, exc) -> None:
        """Escalate a fatal mid-turn crash to the ERROR state.

        Per-turn errors are otherwise just logged so one bad turn can't take the
        assistant down. But a GPU / out-of-memory class failure poisons the
        model for every following turn, so we bounce the user back to setup with
        an explanation instead of failing silently turn after turn.
        """
        fatal, detail = self._diagnose_failure(exc)
        if fatal:
            logger.error("Fatal runtime error — switching to setup: %s", detail)
            self._enter_error_state(detail)

    def _load_models(self):
        """Load the configured ASR, TTS and (optionally) LLM backends.

        Backends are resolved through `core.backends`: each domain's module is
        imported from the registry and its load + helper functions are pulled
        from it, so swapping `models.<domain>.backend` swaps the implementation
        with no orchestrator change. `llm.backend: none` skips the SLM entirely.
        """
        if self.lifecycle is not None:
            self.lifecycle.set("LOADING", "loading models")

        # Backend currently being loaded — read by `_diagnose_failure` so a load
        # crash names the culprit (and its VRAM estimate) on the setup alert.
        self._loading_backend = None

        asr_cfg = self._models[ASR]
        tts_cfg = self._models[TTS]
        llm_cfg = self._models[LLM]

        # Make the dashboard stats panel show the backends actually in use.
        set_model_labels(
            stt=asr_cfg["spec"].display_name,
            tts=tts_cfg["spec"].display_name,
            llm=self._llm_stat_label(llm_cfg),
        )

        # --- ASR -----------------------------------------------------------
        self._loading_backend = asr_cfg["spec"]
        self._set_loading_detail(ASR, asr_cfg)
        asr_mod = get_module(ASR, asr_cfg["backend"])
        logger.info(f"Using {asr_cfg['spec'].display_name}")
        self.asr_pipe = asr_mod.load_asr_model(
            model_name=asr_cfg["model"],
            language=self.asr_language,
            **asr_cfg["opts"],
        )
        if self.asr_context_hint:
            _MAX_CONTEXT_TERMS = 10
            extras = [t for t in self.asr_context_terms if t]
            if len(extras) > _MAX_CONTEXT_TERMS:
                logger.warning(
                    f"asr_context_terms has {len(extras)} entries; only the first "
                    f"{_MAX_CONTEXT_TERMS} will be used (more dilutes the bias)"
                )
                extras = extras[:_MAX_CONTEXT_TERMS]
            terms = [self.wakeword] + extras
            self.asr_pipe.context = "Technical terms: " + ", ".join(terms)
            logger.info(f"ASR context hint enabled: {self.asr_pipe.context!r}")
            # The decoder sometimes regurgitates the prompt scaffolding verbatim
            # off non-speech ("Technical terms: hey atticus, <garbage>") — that
            # carries the wakeword plus a non-term tail, so neither
            # _is_context_echo nor the bare-wakeword guard catches it. The leading
            # marker (everything before the first colon) is an unambiguous tell a
            # user never utters; the transcriber drops any result containing it.
            self._context_prompt_marker = self.asr_pipe.context.split(":", 1)[0].strip().lower()
        # KWS candidates still need the decoder's proper-name and domain-term
        # bias. The classifier is the acoustic gate; ASR verifies its text.
        self._asr_verification_context = (
            "Technical terms: " + ", ".join(terms) if self.asr_context_hint else ""
        )
        wake_cfg = (self._models_raw or {}).get("wakeword", {})
        if wake_cfg.get("backend", "asr") == "openwakeword":
            wakeword_settings = {
                key: wake_cfg[key]
                for key in ("model", "threshold", "smoothing_frames", "cooldown_ms")
                if key in wake_cfg
            }
            self.wakeword_metrics["settings"] = wakeword_settings
            try:
                from .wakeword_openwakeword import OpenWakeWordBackend

                backend = OpenWakeWordBackend(
                    wake_cfg.get("model", ""),
                    wake_cfg.get("threshold", 0.7),
                    wake_cfg.get("smoothing_frames", 3),
                    wake_cfg.get("cooldown_ms", 1500),
                )
                backend.self_test()
                self.audio_capture.set_wakeword_backend(backend)
                self.wakeword_metrics["status"] = "openwakeword"
                logger.info("Experimental openWakeWord idle gate enabled: %s", wakeword_settings)
            except Exception as exc:  # A gate must never make voice control unavailable.
                self.wakeword_metrics["status"] = f"asr fallback: {exc}"
                logger.warning(
                    "openWakeWord unavailable; using ASR-only wakeword detection: %s", exc
                )
        self.asr_stream_generator = asr_mod.stream_generator
        # Pay the ONNX cold-start (ORT kernel/arena init) now, not on the user's
        # first command. No-op on backends that don't implement it (e.g. GPU Qwen).
        if hasattr(self.asr_pipe, "warmup"):
            self.asr_pipe.warmup()

        # --- TTS -----------------------------------------------------------
        self._loading_backend = tts_cfg["spec"]
        self._set_loading_detail(TTS, tts_cfg)
        tts_mod = get_module(TTS, tts_cfg["backend"])
        # Kept for the settings-console hot-apply (set_voice/set_speed live).
        # The backend gates which of those can apply without a restart: Kokoro
        # swaps voice/speed instantly; Qwen's clone needs a warmup + phrase-cache
        # re-render, so it stays restart-only (see apply_hot_config).
        self._tts_module = tts_mod
        self._tts_backend = tts_cfg["backend"]
        logger.info(f"Using {tts_cfg['spec'].display_name} with voice clone: {self.voice_clone}")
        tts_mod.load_tts(model_id=tts_cfg["model"], **tts_cfg["opts"])
        if self.tts_speed is not None and hasattr(tts_mod, "set_speed"):
            tts_mod.set_speed(self.tts_speed)
        self.voice_clone_prompt = tts_mod.set_voice(self.voice_clone)
        tts_mod.warmup_model(self.voice_clone_prompt)

        # play_chunks/speak_stream are real bound methods (defined below),
        # not raw rebindings of the tts module's functions — they resolve
        # `self._turn_local` (this thread's sink/tts_active_event slot)
        # explicitly on every call so callers (agent_loop.py, the stall/ack
        # helpers below) need no changes: they keep calling
        # `self.play_chunks(...)` / `self.speak_stream(...)` exactly as before.
        self.synthesize = tts_mod.synthesize
        self.web_search_stall_cache: list = []
        self.note_write_stall_cache: list = []
        self.music_search_stall_cache: list = []
        self.pre_thinking_stall_cache: list = []
        self.thinking_stall_cache: list = []
        self.ack_cache: list = []
        self.barge_ack_cache: list = []
        self.busy_cache: list = []
        self.replan_stall_cache: list = []

        # --- LLM -----------------------------------------------------------
        if self.llm_enabled:
            self._loading_backend = llm_cfg["spec"]
            self._set_loading_detail(LLM, llm_cfg)
            logger.info("Using %s", self._loading_display_name(llm_cfg))
            backend = llm_cfg["backend"]
            if backend == "openai":
                from .llm_openai import load_openai

                self.grammar, self.slm_model = load_openai(
                    model=llm_cfg["model"], **llm_cfg["opts"]
                )
            else:  # local llama.cpp (default)
                load_kwargs = {
                    "model_path": llm_cfg["model"],
                }
                if llm_cfg["n_context"]:
                    load_kwargs["n_ctx"] = llm_cfg["n_context"]
                load_kwargs["server_slots"] = self.thinking_server_slots
                self.grammar, self.slm_model = load_slm(
                    **load_kwargs,
                    **llm_cfg["opts"],
                    persistent_logging_enabled=self.persistent_logging_enabled,
                )
            self.greeting_prompt = get_greeting_system_prompt(
                self.wakeword_name, self._personality_for_prompt()
            )
            self.web_summary_prompt = get_web_summary_system_prompt()
        else:
            logger.info("No LLM backend — running regex-only (basic commands)")

        self.models_ready.set()
        # Probe the remote LLM (if any) in the background so a down/misconfigured
        # endpoint surfaces the dashboard banner without waiting for a first turn.
        if self.llm_backend == "openai":
            threading.Thread(target=self._probe_remote_llm, daemon=True).start()
        if self.lifecycle is not None:
            self.lifecycle.set("LOADING", "warming up prompts (usually 10-20 seconds)")
        self._warm_and_announce()
        # The opening greeting has played — the assistant is live. Flip the
        # lifecycle so the dashboard hands off from the loading screen.
        if self.lifecycle is not None:
            self.lifecycle.set("READY")
        self._start_reminder_poll()
        try:
            from tools.time_tools import set_speak_callback

            set_speak_callback(self.speak_proactive)
        except ImportError:
            pass

    def _set_loading_detail(self, domain: str, cfg: dict) -> None:
        """Expose the current model and a conservative load estimate to the UI."""
        if self.lifecycle is None:
            return
        estimate = _MODEL_LOAD_ESTIMATES.get((domain, cfg["backend"]))
        detail = f"loading {self._loading_display_name(cfg)}"
        if estimate:
            detail += f" ({estimate})"
        self.lifecycle.set("LOADING", detail)

    @staticmethod
    def _loading_display_name(cfg: dict) -> str:
        """Name a configured custom LLM rather than its shared runtime backend."""
        spec = cfg["spec"]
        if cfg.get("model") != spec.default_model and cfg.get("backend") in {"llama", "gemma"}:
            return f"Custom local model ({Path(cfg['model']).name})"
        return spec.display_name

    def _note_llm_remote_status(self, ok: bool, error: str = "") -> None:
        """Record the last-known reachability of the remote LLM endpoint.

        No-op unless the LLM is the remote OpenAI backend. Called from the agent
        loop (failure on a RemoteUnreachable catch, success after a generation)
        and the startup probe, so the dashboard reflects live reachability.
        """
        if self.llm_backend != "openai":
            return
        with self._llm_remote_lock:
            self._llm_remote_ok = ok
            self._llm_remote_error = "" if ok else (error or "unreachable")
            if ok:
                self._llm_remote_retry_at = 0.0
                self._llm_remote_outage_announced = False
            else:
                self._llm_remote_retry_at = time.monotonic() + REMOTE_LLM_RETRY_COOLDOWN_S

    def _remote_llm_retry_blocked(self) -> bool:
        """Whether a recent remote failure is still in its retry cooldown."""
        if self.llm_backend != "openai":
            return False
        with self._llm_remote_lock:
            return self._llm_remote_ok is False and time.monotonic() < self._llm_remote_retry_at

    def _remote_llm_unavailable_fallback(
        self, session: Optional[TtsSession], source: str, satellite_id: Optional[str] = None
    ) -> str:
        """Announce a remote outage once per voice outage episode.

        The dashboard remains visibly degraded through `_llm_remote_ok`. Voice
        requests after the first notice stay quiet until a later request probes
        and restores the endpoint; text callers still receive a visible reply.
        """
        if source != "voice":
            return self._speak_llm_error_fallback(session, source, satellite_id=satellite_id)
        with self._llm_remote_lock:
            if self._llm_remote_outage_announced:
                return ""
            self._llm_remote_outage_announced = True
        return self._speak_llm_error_fallback(session, source, satellite_id=satellite_id)

    @property
    def remote_llm_unreachable(self) -> bool:
        """True when the off-device LLM is configured but not reachable — the
        dashboard shows a red 'regex/fast-path only' banner. False while
        reachable or not yet probed (None), so the banner never flashes on a
        healthy or still-starting endpoint."""
        return self.llm_backend == "openai" and self._llm_remote_ok is False

    def _probe_remote_llm(self) -> None:
        """One-shot reachability probe so the dashboard banner reflects a down
        endpoint *before* the user speaks. Runs on a daemon thread (the 1-token
        ping can block on a dead connect)."""
        client = self.slm_model
        if client is None or not getattr(client, "_fulloch_remote", False):
            return
        try:
            ok, error = client.ping()
        except Exception as e:  # noqa: BLE001 — never let the probe crash startup
            ok, error = False, f"{type(e).__name__}: {e}"
        self._note_llm_remote_status(ok, error)
        if not ok:
            logger.warning("Remote LLM probe failed (%s) — regex/fast-path only", error)

    def _personality_for_prompt(self) -> str | None:
        """Return the configured conversational personality."""
        if self.personality == "custom":
            return self.personality_custom.strip() or None
        return self.personality

    def _warm_and_announce(self):
        """Prime every cache, then speak the opening greeting.

        Order matters for the user experience: all silent warmup work
        (agent KV-cache prime, every TTS pre-render, the notes index)
        runs *before* the greeting, so the greeting becomes the last
        audible event of startup. `audio_capture.mic_globally_enabled` is
        re-armed in the `finally` immediately after greeting playback
        ends, meaning the mic comes alive precisely when the speaker
        goes quiet — no awkward silent gap between "online and ready"
        and the assistant actually listening.
        """
        self.audio_capture.mic_globally_enabled = False
        try:
            self._render_phrase_caches()

            if self.llm_enabled:
                # Prime the agent prompt's KV-cache so the first real user
                # request reuses the prefilled prefix. A remote endpoint that's
                # down at startup mustn't crash boot — skip the prime and carry
                # on (turns degrade to regex-only until it's reachable).
                logger.info("Priming agent cache...")
                try:
                    generate_slm(
                        self.slm_model,
                        user_prompt=CACHE_PRIMING_USER_PROMPT,
                        grammar=self.grammar,
                        system_prompt=get_agent_system_prompt(
                            self.wakeword_name,
                            personality=self._personality_for_prompt(),
                            higgs_tts=self._tts_backend == "higgs-gguf",
                            obsidian_edit_enabled=notes._obsidian_edit_allowed(),
                        ),
                    )
                except RemoteUnreachable:
                    logger.warning("Remote LLM unreachable at startup — skipping cache prime")

                # Pre-load the BGE embedding model + restore the persisted notes
                # index so the first semantic-search query doesn't pay the
                # SentenceTransformer load + initial scan latency. Semantic
                # note search is only reachable via the SLM agent, so skip the
                # (heavy) warmup with no LLM — it loads lazily if ever needed.
                try:
                    from tools.notes import warm_index

                    logger.info("Warming notes index...")
                    if warm_index():
                        logger.info("Notes index ready.")
                except Exception:
                    logger.exception("Notes index warmup failed; semantic search will load lazily")

            # Greeting comes last — last audible event of startup.
            cleaned = None
            if self.llm_enabled:
                topic = random.choice(GREETING_TOPICS)
                logger.info(f"Preparing startup prompt (topic: {topic})...")
                try:
                    greeting = generate_slm(
                        self.slm_model,
                        user_prompt=get_greeting_user_prompt(topic),
                        system_prompt=self.greeting_prompt,
                        temperature=1.0,
                    )
                    cleaned = clean_for_tts(greeting)
                except RemoteUnreachable:
                    logger.warning("Remote LLM unreachable — using a basic greeting")
            if not cleaned:
                # No SLM (or remote down) to compose a greeting — speak a fixed
                # line that also sets expectations about basic-commands-only mode.
                cleaned = (
                    "Hello, I'm running in basic mode "
                    "without an AI model, so I can handle simple commands like "
                    "lights, timers, and music."
                )
            self.greeting_text = cleaned
            self._startup_greeting_response = cleaned if self.llm_enabled else None

            # Splitting the greeting into individual sentences and
            # synthesising each as its own worker job populates the
            # reduce-overhead CUDA-graph cache with extra prefill shapes.
            sentences = split_sentences(cleaned) or [cleaned]
            logger.info(f"Rendering warm-up prompts ({len(sentences)} sentence(s))...")
            greeting_parts = [self.synthesize(s, self.voice_clone_prompt) for s in sentences]
            # Cached for replay_greeting() — see greeting_cache's docstring
            # above for why the live attempt below almost never actually
            # reaches a listener.
            self.greeting_cache = greeting_parts
            # Synthesis above already did the real warmup work (CUDA-graph /
            # ORT session priming); playback is just for the user to hear it,
            # so a missing local output device (headless/no-PulseAudio Docker
            # host) shouldn't abort startup — same fallback as the recorder.
            try:
                for chunks, sr in greeting_parts:
                    if chunks:
                        self.play_chunks(chunks, sr, session=self.tts_session)
            except Exception as e:
                logger.warning(
                    "Local audio playback unavailable (%s) — greeting not played, "
                    "use the browser satellite or dashboard for voice.",
                    e,
                )

            logger.info("Warmup complete")
        finally:
            self.audio_capture.mic_globally_enabled = True

    def _reminder_poll_loop(self) -> None:
        """Daemon thread: speak upcoming Fulloch calendar events via speak_proactive.

        Polls every 60 seconds. Fires speak_proactive for any event on the
        configured reminder calendar that starts within the next 15 minutes and
        hasn't already been spoken this session.
        """
        try:
            from tools.home_assistant import get_upcoming_events
        except ImportError:
            return

        _REMINDER_EXPIRY_S = 7200.0  # 2 hours — prevents re-fire within same day

        logger.info("Reminder poll thread started (60s interval)")
        while True:
            time.sleep(60)
            try:
                mono_now = time.monotonic()
                # Evict expired entries.
                expired = [k for k, exp in self._spoken_reminders.items() if mono_now >= exp]
                for k in expired:
                    del self._spoken_reminders[k]

                for event in get_upcoming_events(window_seconds=15 * 60):
                    # Key on summary + date only — avoids any timezone/format
                    # variation in the start string returned by HA.
                    date_part = event["start"][:10]
                    key = (event["summary"], date_part)
                    if key not in self._spoken_reminders:
                        self._spoken_reminders[key] = mono_now + _REMINDER_EXPIRY_S
                        prefix = random.choice(REMINDER_PREFIX_PHRASES)
                        text = f"{prefix} {event['summary']}"
                        logger.info(f"Speaking reminder: {text!r}")
                        self.speak_proactive(text)
                    else:
                        logger.debug(f"Reminder suppressed (already spoken): {event['summary']!r}")
            except Exception:
                logger.exception("Reminder poll iteration failed")

    def _start_reminder_poll(self) -> None:
        """Start the reminder poll thread if calendar and polling are configured."""
        # Respect the same config gate as the tool loader (tools/__init__.py):
        # with `home_assistant:` absent from config.yml, never import
        # tools.home_assistant. Importing it has side effects — it registers the
        # HA tools into the global registry and connects to HA (default URL + the
        # credentials.json token) — so importing it here would silently re-enable HA for the
        # agent despite it being disabled in config.
        from tools._config import config as _cfg

        if "home_assistant" not in _cfg:
            return
        try:
            from tools.home_assistant import HA_CONFIG, _reminder_calendar_entity
        except ImportError:
            return
        if not _reminder_calendar_entity():
            return
        if not HA_CONFIG.get("reminder_poll", True):
            logger.info("Reminder polling disabled via config (reminder_poll: false)")
            return
        t = threading.Thread(target=self._reminder_poll_loop, daemon=True)
        t.start()

    def _maybe_reset_session(self) -> None:
        """Clear unified history if every satellite has been silent past the
        session timeout.

        `_history` is shared across every satellite ("one assistant mind
        heard from multiple rooms"), so the timeout is gated on the most
        recently active satellite, not the one triggering this check —
        otherwise satellite B merely being idle for a while would wipe
        satellite A's still-fresh conversation the moment B took a turn.
        """
        last_ends = [s.last_turn_end for s in self.satellites.values()]
        most_recent = max(last_ends) if last_ends else 0.0
        if most_recent == 0.0:
            return
        if not self._history:
            return
        if time.monotonic() - most_recent > CHAT_SESSION_TIMEOUT_S:
            logger.info(
                f"Session timeout ({CHAT_SESSION_TIMEOUT_S:.0f}s) — "
                f"clearing {len(self._history)} history entries"
            )
            self._history.clear()
            self._last_thinking_partial = None
            self._last_thinking_question = None
            for s in self.satellites.values():
                s.skip_followup_self_echo = False

    def _history_for(self, session: Optional[SatelliteSession]) -> list:
        """The conversation history a turn's prompt should be built from.

        Returns the one shared `_history` today regardless of `session` —
        per-satellite history is out of scope for this refactor (see the
        plan's "Out of scope" list). The indirection lands now so a future
        per-satellite history would be a one-method change here instead of
        touching every prompt-building call site in `agent_loop.py` again.
        """
        return self._history

    def _trim_history(self) -> None:
        """Cap `_history` at HISTORY_MAX_MESSAGES with FIFO eviction."""
        if len(self._history) > HISTORY_MAX_MESSAGES:
            self._history = self._history[-HISTORY_MAX_MESSAGES:]

    def _compact_completed_turns(self) -> None:
        """Shrink tool results and drop bare planning emissions from `_history`.

        Called at the start of each turn, when every tool / `{"actions": ...}`
        entry belongs to an already-completed turn. The agent needs a tool
        result in full only while composing that turn's reply; keeping the raw
        dumps afterwards just bloats the context (a single note read or search
        payload can be many KB) and evicts the real conversation.

        Tool entries are NOT dropped entirely, though — only truncated to
        `COMPACTED_TOOL_TRACE_MAX_CHARS`. A dropped-entirely tool entry left
        history with only `{"reply": ...}` assistant turns, which is
        indistinguishable from a fact the model merely asserted out loud: a
        relative follow-up ("brighten them now") had no way to tell "I really
        set this to 30%" from "I once said 30% out loud", so the model would
        sometimes invent a new number instead of dispatching a fresh tool call.
        Keeping a short trace of which tool ran gives it that signal back
        without re-bloating history with full payloads.
        """
        kept = []
        for msg in self._history:
            role = msg.get("role")
            if role == "tool":
                content = msg.get("content") or ""
                if len(content) > COMPACTED_TOOL_TRACE_MAX_CHARS:
                    msg = {
                        **msg,
                        "content": content[:COMPACTED_TOOL_TRACE_MAX_CHARS].rstrip() + "…",
                    }
            elif role == "assistant":
                try:
                    emission = json.loads(msg.get("content") or "")
                except Exception:
                    emission = None
                # Drop a bare action-emission (planning scaffolding); keep
                # replies (what Fulloch actually said) and anything unparseable.
                if isinstance(emission, dict) and "reply" not in emission and "actions" in emission:
                    continue
            kept.append(msg)
        if len(kept) != len(self._history):
            logger.debug(
                "Compacted history: dropped %d scaffolding entries",
                len(self._history) - len(kept),
            )
        self._history = kept

    def _shed_oldest_history(self) -> bool:
        """Drop the oldest history entries to recover from a context overflow.

        Returns True if entries were shed (worth retrying the SLM call), False
        if `_history` is already down to the recent floor — at which point a
        single oversized message is to blame and the caller falls back to the
        full clear + apology. Sheds roughly half the over-floor surplus per call
        (so a few retries converge), then drops forward to the next `user`
        message so the trimmed history still starts on a turn boundary rather
        than an orphaned tool/assistant entry.
        """
        n = len(self._history)
        if n <= CONTEXT_TRIM_KEEP_MIN:
            return False
        drop = max(2, (n - CONTEXT_TRIM_KEEP_MIN + 1) // 2)
        del self._history[:drop]
        while len(self._history) > CONTEXT_TRIM_KEEP_MIN and self._history[0].get("role") != "user":
            del self._history[0]
        logger.info(
            "Context overflow: shed %d oldest history entries, %d remain",
            drop,
            len(self._history),
        )
        return True

    def _generate_with_context_recovery(self, **kwargs) -> str:
        """`generate_slm` wrapper that degrades gracefully on context overflow.

        On `ContextExhaustedError`, sheds the oldest history (preserving the
        in-flight turn) and retries, so a long conversation loses its tail end
        instead of being wiped. If the recent floor still cannot fit, clears
        history and retries the current request once. Re-raises only if that
        request cannot fit an otherwise empty conversation.
        """
        cleared_history = False
        # A foreground timeout must not restart the shared server underneath a
        # durable background request. The foreground can use its normal fallback
        # and the thinking worker remains able to finish on its own slot.
        thinking_jobs = getattr(self, "thinking_jobs", None)
        kwargs.setdefault(
            "recover_on_failure", thinking_jobs is None or thinking_jobs.active() is None
        )
        while True:
            try:
                return generate_slm(self.slm_model, **kwargs)
            except ContextExhaustedError:
                if self._shed_oldest_history():
                    continue
                if cleared_history:
                    raise
                logger.warning(
                    "Context overflow persists at history floor; clearing history and retrying turn"
                )
                # Agent calls carry the current question in history (rather
                # than `user_prompt`), so retain it while dropping prior turns.
                current_user = next(
                    (
                        message.copy()
                        for message in reversed(self._history)
                        if message.get("role") == "user"
                    ),
                    None,
                )
                self._history.clear()
                if current_user is not None:
                    self._history.append(current_user)
                cleared_history = True

    def register_turn_listener(self, callback) -> None:
        """Subscribe a callable to per-turn user/assistant events.

        callback(event: dict) is invoked synchronously from the turn's
        thread once for the user message and once for the assistant
        response. Event shape:
            {"role": "user"|"assistant", "content": str,
             "ts": float (unix seconds), "source": "voice"|"text"}
        Listeners should be quick and non-blocking — they're called
        inside the turn lock for text turns. Exceptions are logged
        and swallowed.
        """
        with self._turn_listeners_lock:
            self._turn_listeners.append(callback)

    def unregister_turn_listener(self, callback) -> None:
        """Remove a short-lived protocol listener when its satellite disconnects."""
        with self._turn_listeners_lock:
            try:
                self._turn_listeners.remove(callback)
            except ValueError:
                pass

    def _emit_turn_event(
        self,
        role: str,
        content: str,
        source: str,
        stats: Optional[TurnStats] = None,
        satellite_id: Optional[str] = None,
        artifact: Optional[dict] = None,
    ) -> float:
        """Emit a `user`/`assistant` chat-bubble event to `/stream` subscribers.

        `satellite_id` tags which satellite (or `"dashboard-text"`) the turn
        belongs to, so the dashboard can show a "from <label>" tag on a voice
        turn (#14 6a). Threaded explicitly by the caller rather than read off
        `_current_satellite_id` (as `_emit_agent_event` does) — half of this
        method's call sites fire before `AgentLoop.run()` sets that ctxvar
        (the "user" event) or after it's already reset (the final "assistant"
        reply, emitted once `AgentLoop.run()` has returned), so the ctxvar
        wouldn't be in scope for them.
        """
        ts = time.time()
        sat = self.satellites.get(satellite_id) if satellite_id else None
        event = {
            "role": role,
            "content": content,
            "ts": ts,
            "source": source,
            "satellite_id": satellite_id,
            "satellite_label": (sat.label or sat.ha_area_name) if sat is not None else None,
        }
        if role == "assistant" and source == "voice":
            event["tts_backend"] = getattr(self, "_tts_backend", None)
        if stats is not None:
            event["stats"] = stats.to_payload()
        if artifact is not None:
            event["artifact"] = artifact
        self._dispatch_event(event)
        return ts

    def _emit_stats_patch(self, ref_ts: float, source: str, patch: dict) -> None:
        """Patch a stat onto an already-emitted assistant message (keyed by its
        ts). Used for TTS time, which is only known after playback starts."""
        self._dispatch_event(
            {
                "role": "stats",
                "ts": time.time(),
                "ref_ts": ref_ts,
                "source": source,
                "patch": patch,
            }
        )

    def _emit_agent_event(
        self,
        kind: str,
        payload: dict,
        source: str = "voice",
        replan: bool = False,
    ) -> None:
        """Emit a `plan` / `step` / `observation` event to listeners.

        kind:
          - "plan"        — payload = {"actions": [{...}]}  (or {"reply": "..."})
          - "step"        — payload = {"intent": str, "args": list}
          - "observation" — payload = {"intent": str, "result": str}

        `replan=True` marks a `plan` emitted on a re-call (the agent re-decided
        from new observations); the prior plan was deliberately superseded, not
        failed. Surfaced so dashboards can label it instead of looking like the
        previous plan silently failed. Kept off `payload` (which mirrors the raw
        emission shape the frontend parses) — it's event metadata, not the plan.

        Tags the event with `satellite_id`/`satellite_label` by reading
        `AgentLoop`'s `_current_satellite_id` contextvar rather than taking a
        parameter — that ctxvar is already scoped to exactly the duration of
        the `AgentLoop.run` call this fires from, so reusing it here avoids
        threading a second, redundant satellite-identity parameter through
        every one of the ~8 `host._emit_agent_event(...)` call sites in
        `agent_loop.py`. Today the dashboard's `/stream` consumer ignores
        these fields; Phase 6a's SSE tagging just reads them off the event
        dict that's already flowing.
        """
        satellite_id = _current_satellite_id.get()
        sat = self.satellites.get(satellite_id) if satellite_id else None
        event = {
            "role": "agent",
            "kind": kind,
            "ts": time.time(),
            "source": source,
            "payload": payload,
            "satellite_id": satellite_id,
            "satellite_label": (sat.label or sat.ha_area_name) if sat is not None else None,
        }
        if replan:
            event["replan"] = True
        self._dispatch_event(event)

    def _dispatch_event(self, event: dict) -> None:
        with self._turn_listeners_lock:
            listeners = list(self._turn_listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Turn listener failed: {e}")

    def run_thinking_task(
        self,
        task: str,
        origin_satellite_id: str | None = None,
        origin_source: str = "integration",
    ) -> dict:
        """Queue a bounded, non-speaking thinking task from the integration API."""
        if self.thinking_jobs is None:
            raise RuntimeError("thinking is not ready")
        pending_task = (
            self._pending_thinking_tasks.get(origin_satellite_id)
            if origin_source == "conversation" and origin_satellite_id is not None
            else None
        )
        if pending_task:
            task = f"{pending_task}\n\nAdditional user detail: {task}"
        job_id = self.thinking_jobs.submit(
            task,
            conversation=self._history_for(None),
            notes=notes.recall_facts(),
            origin_satellite_id=origin_satellite_id,
            origin_source=origin_source,
        )
        if pending_task:
            self._pending_thinking_tasks.pop(origin_satellite_id, None)
        return self.thinking_jobs.status(job_id) or {"id": job_id}

    def thinking_task_status(self, job_id: str) -> dict | None:
        return self.thinking_jobs.status(job_id) if self.thinking_jobs is not None else None

    def cancel_thinking_task(self, job_id: str) -> bool:
        return self.thinking_jobs.cancel(job_id) if self.thinking_jobs is not None else False

    def active_thinking_task(self) -> dict | None:
        return self.thinking_jobs.active() if self.thinking_jobs is not None else None

    def consume_completed_thinking_report(self, satellite_id: str | None) -> str | None:
        """Deliver a voice summary first, then the complete report on request."""
        pending = self._completed_thinking_report_entry(satellite_id)
        if pending is None:
            return None
        report = self._read_completed_thinking_report(pending)
        if report is None:
            return "I can't retrieve that completed report right now."
        if pending.get("summary_delivered"):
            return "Here is the full report. " + report
        pending["summary_delivered"] = True
        return (
            "Here's the short version. "
            + self._report_summary(report)
            + " The full report is saved in Fulloch Reports. Would you like me to read the full report?"
        )

    def _completed_thinking_report_entry(self, satellite_id: str | None) -> dict[str, object] | None:
        if satellite_id is None:
            return None
        pending = self._completed_thinking_reports.get(satellite_id)
        if pending is not None:
            return pending
        # Browser sessions receive a new server-generated ID after a reconnect.
        # Carry over the sole report whose originating session is gone.
        disconnected = [
            sid for sid in self._completed_thinking_reports if sid not in self.satellites
        ]
        if len(disconnected) != 1:
            return None
        pending = self._completed_thinking_reports.pop(disconnected[0])
        self._completed_thinking_reports[satellite_id] = pending
        return pending

    @staticmethod
    def _read_completed_thinking_report(pending: dict[str, object]) -> str | None:
        note_id = pending.get("note_id")
        if not isinstance(note_id, str) or not re.fullmatch(
            r"fulloch-reports/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", note_id
        ):
            return None
        path = notes_root.get_notes_root() / f"{note_id}.md"
        try:
            path.resolve().relative_to((notes_root.get_notes_root() / "fulloch-reports").resolve())
            return path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_completed_thinking_evidence(pending: dict[str, object]) -> dict:
        """Read the bounded artifact ledger paired with a completed report."""
        note_id = pending.get("note_id")
        if not isinstance(note_id, str) or not re.fullmatch(
            r"fulloch-reports/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", note_id
        ):
            return {}
        path = notes_root.get_notes_root() / f"{note_id}.evidence.json"
        try:
            path.resolve().relative_to((notes_root.get_notes_root() / "fulloch-reports").resolve())
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def answer_completed_thinking_report(
        self, satellite_id: str | None, question: str, cancel_check, stats=None
    ) -> str | None:
        """Answer a report follow-up without exposing it to normal agent context."""
        pending = self._completed_thinking_report_entry(satellite_id)
        if pending is None:
            return None
        report = self._read_completed_thinking_report(pending)
        if report is None:
            return "I can't retrieve that completed report right now."
        evidence = self._read_completed_thinking_evidence(pending)
        answer = generate_slm(
            self.slm_model,
            user_prompt=question,
            system_prompt=get_thinking_report_answer_prompt(report, evidence),
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            temperature=0.0,
            cancel_check=cancel_check,
            stats=stats,
        )
        return (answer or "The report does not answer that.").strip()

    @staticmethod
    def _report_summary(report: str) -> str:
        match = re.search(r"(?ims)^## Summary\s*\n+(.*?)(?=^##\s|\Z)", report)
        if match:
            summary = " ".join(match.group(1).split())
            if summary:
                return summary[:450].rstrip()
        return Assistant._spoken_report_summary(report)

    @staticmethod
    def _spoken_report_summary(report: str) -> str:
        """Quote the report's conclusion without another model call."""
        text = re.sub(r"(?m)^#{1,6}\s+", "", report)
        sentences = split_sentences(text)
        selected = []
        total = 0
        for sentence in reversed(sentences):
            if len(selected) == 2 or total + len(sentence) > 450:
                break
            selected.append(sentence)
            total += len(sentence) + 1
        return " ".join(reversed(selected)) or "I completed the research report."

    def _run_background_thinking_job(
        self, job: BackgroundJob, cancelled: Callable[[], bool]
    ) -> tuple[str, str]:
        """Run a bounded, tool-using investigation without altering foreground history."""
        if self.slm_model is None:
            raise RuntimeError("local language model is not loaded")
        capabilities = {
            name: capability
            for name, capability in native_capabilities().items()
            if capability.access_class == "read"
        }
        playbooks = matching_playbooks(job.snapshot.task, capabilities)
        descriptions = []
        for name in capabilities:
            schema = tool_registry._schemas.get(name)
            if schema is not None:
                descriptions.append(_describe_thinking_capability(name, schema))
        findings = job.state
        capability_calls: dict[str, int] = {}
        attempted_actions: set[str] = set()
        needs_input: str | None = None

        def fallback_report() -> str:
            if findings.strip():
                return (
                    "I gathered source material but couldn't complete a reliable final report. "
                    "I have not drawn a conclusion from those incomplete results."
                )
            return "I couldn't retrieve enough source material to complete this report."

        def run_capability(name: str, args: list) -> bool:
            nonlocal findings, needs_input
            stop_for_preliminary_report = False
            capability = capabilities.get(name)
            if capability is None:
                findings = _append_thinking_observation(
                    findings, "worker", f"Unavailable capability: {name}"
                )
                job.state = findings
                return False
            schema = tool_registry._schemas.get(name)
            if schema is not None:
                required = sum(param.required for param in schema.params)
                if not required <= len(args) <= len(schema.params):
                    findings = _append_thinking_observation(
                        findings,
                        "worker",
                        f"Invalid arguments for {name}; expected {required}-{len(schema.params)} positional values.",
                    )
                    job.state = findings
                    return False
            if capability_calls.get(name, 0) >= MAX_THINKING_CAPABILITY_CALLS:
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    f"Capability budget reached for {name}; synthesise the evidence collected so far.",
                )
                job.state = findings
                return False
            capability_calls[name] = capability_calls.get(name, 0) + 1
            self.thinking_jobs.update_stage(
                job,
                "Searching flights"
                if name == "search_flights"
                else "Comparing accommodation"
                if name == "search_hotels"
                else "Searching sources"
                if name in {"external_information", "search_papers"}
                else "Reviewing information",
            )
            logger.debug("Deep-think job %s dispatching %s with args=%r", job.id, name, args)
            artifact_token = set_artifacts(job.artifacts)
            try:
                result = capability.invoke(args, {})
            except Exception as exc:
                result = (
                    ThinkingResult(
                        f"Tool {name} failed: {type(exc).__name__}: {exc}",
                        status="failed",
                        scope="The requested tool operation raised an exception.",
                    )
                    if schema is not None and schema.thinking_outcome
                    else f"Tool {name} failed: {type(exc).__name__}: {exc}"
                )
            finally:
                reset_artifacts(artifact_token)
            if schema is not None and schema.thinking_outcome:
                error = (
                    "The tool did not provide its required typed evidence envelope."
                    if not isinstance(result, ThinkingResult)
                    else thinking_result_error(result)
                )
                if error:
                    result = ThinkingResult(
                        f"Tool {name} returned an invalid deep-think outcome: {error}",
                        status="failed",
                        scope="The tool result was rejected before it entered the evidence ledger.",
                    )
            if isinstance(result, ThinkingResult):
                artifact_id = job.record_outcome(
                    name,
                    result.thinking_status,
                    result.scope,
                    result.evidence,
                    result.next_actions,
                    result.artifact,
                )
                if artifact_id:
                    result = ThinkingResult(
                        f"{result}\nArtifact reference: {artifact_id}",
                        status=result.thinking_status,
                        evidence=result.evidence,
                        scope=result.scope,
                        next_actions=result.next_actions,
                        artifact=result.artifact,
                    )
                if result.thinking_status == "needs_input":
                    if any(item.get("status") == "evidence" for item in job.evidence):
                        stop_for_preliminary_report = True
                        result = str(result)
                    else:
                        needs_input = "Reactive question: " + str(result).removeprefix(
                            "Reactive question:"
                        ).strip()
                elif result.thinking_status in {"failed", "unavailable"}:
                    result = str(result)
            step = intents.classify_step(result)
            if step.kind is intents.StepKind.REACTIVE:
                # A capability rejected malformed or incomplete input before
                # producing evidence. Let a corrected action use its budget.
                capability_calls[name] -= 1
            if step.artifact is not None and job.artifact is None:
                job.artifact = step.artifact
            if step.kind is intents.StepKind.WEB_SEARCH:
                summary = self._summarise_search_result(step.text, cancelled)
                if summary and summary.strip():
                    result = summary
                else:
                    logger.warning("Deep-think job %s received an empty web summary", job.id)
            logger.info(
                "Deep-think job %s received %d characters from %s", job.id, len(result), name
            )
            findings = _append_thinking_observation(findings, f"tool:{name}", result)
            if name == "evaluate_itinerary" and "not feasible" in result.lower():
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "This rejects one evaluated itinerary only. Inspect untested retrieved combinations or take a materially different available search before making a broader feasibility conclusion.",
                )
            job.state = findings
            if stop_for_preliminary_report:
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "Further input would refine the result, but sufficient evidence exists for a preliminary scoped report now.",
                )
                job.state = findings
                return False
            return True

        for _ in range(MAX_THINKING_WORKER_CALLS):
            if cancelled():
                return "", findings
            self.thinking_jobs.update_stage(job, "Analysing findings")
            prompt = get_thinking_worker_prompt(
                job.snapshot.task,
                list(job.snapshot.conversation),
                notes=job.snapshot.notes,
                job_state=findings,
                capabilities="\n".join(descriptions),
                capability_playbooks="\n\n".join(playbook.render() for playbook in playbooks),
            )

            def generate(worker_prompt=prompt) -> str:
                return generate_slm(
                    self.slm_model,
                    user_prompt=worker_prompt,
                    grammar=self.grammar,
                    # One tool-selection step must yield promptly when a
                    # foreground turn arrives; the report synthesis below is
                    # deliberately allowed a larger budget.
                    max_new_tokens=DEEP_THINK_STEP_MAX_TOKENS,
                    temperature=0.4,
                    cancel_check=cancelled,
                    # Qwen's reasoning channel and llama.cpp's JSON grammar do
                    # not compose: reasoning tokens can consume the completion
                    # while leaving `content` empty. Deliberation comes from the
                    # bounded tool/reasoning loop, so keep structured output on
                    # the normal channel.
                    thinking_mode=False,
                    read_timeout=DEEP_THINK_READ_TIMEOUT_S,
                    generation_timeout=DEEP_THINK_GENERATION_TIMEOUT_S,
                    recover_on_failure=False,
                )

            response = (
                generate() if self.thinking_server_slots == 2 else self._with_turn_lock(generate)
            )
            if cancelled():
                return "", findings
            response = (response or "").strip()
            if not response:
                logger.warning(
                    "Deep-think job %s returned an empty worker response; synthesising collected findings",
                    job.id,
                )
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "Worker stopped without a next action; synthesise only the evidence collected so far.",
                )
                job.state = findings
                break
            try:
                emission = intents.parse_agent_emission(response)
            except (TypeError, ValueError):
                logger.warning(
                    "Deep-think job %s produced an invalid worker emission; synthesising collected findings",
                    job.id,
                )
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "Worker returned an invalid planning response; select the next capability or reply sufficient findings collected.",
                )
                job.state = findings
                continue
            reply = emission.get("reply")
            if isinstance(reply, str) and reply.strip():
                reply = reply.strip()
                if reply.startswith("Reactive question:"):
                    return (
                        reply,
                        findings,
                    )
                findings = _append_thinking_observation(findings, "worker", reply)
                break
            plan = emission.get("plan")
            if isinstance(plan, str) and plan.strip():
                logger.debug("Deep-think job %s plan: %s", job.id, plan.strip())
            actions = emission.get("actions") or []
            if not actions and isinstance(plan, str) and plan.strip():
                fallback_name = next(
                    (
                        playbook.fallback_capability
                        for playbook in playbooks
                        if playbook.fallback_capability in capabilities
                    ),
                    None,
                )
                if fallback_name is not None:
                    action_key = _thinking_action_key(fallback_name, [job.snapshot.task])
                    if action_key not in attempted_actions:
                        attempted_actions.add(action_key)
                        findings = _append_thinking_observation(
                            findings,
                            "worker",
                            f"Worker supplied a plan without an action; using {fallback_name}.",
                        )
                        run_capability(fallback_name, [job.snapshot.task])
                        if needs_input:
                            return needs_input, findings
                        continue
            if len(actions) != 1 or not isinstance(actions[0], dict):
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "Worker did not select exactly one next capability; synthesise current findings.",
                )
                break
            action = actions[0]
            name = tool_registry.canonical_name(action.get("intent", ""))
            capability = capabilities.get(name or "")
            if capability is None:
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    "Blocked unavailable capability: " + str(action.get("intent")),
                )
                job.state = findings
                break
            args, kwargs = intents.coerce_args(action.get("args"))
            if kwargs:
                findings = _append_thinking_observation(
                    findings, "worker", f"Unsupported keyword arguments for {name}"
                )
                job.state = findings
                break
            action_key = _thinking_action_key(name, args)
            if action_key in attempted_actions:
                findings = _append_thinking_observation(
                    findings,
                    "worker",
                    f"Duplicate capability request for {name}; synthesise the evidence collected so far.",
                )
                job.state = findings
                break
            attempted_actions.add(action_key)
            if not run_capability(name, args):
                break
            if needs_input:
                return needs_input, findings
        if not findings.strip():
            return "I couldn't retrieve enough source material to complete this report.", findings

        self.thinking_jobs.update_stage(job, "Synthesising report")
        report_prompt = get_thinking_report_prompt(
            job.snapshot.task,
            findings if not job.evidence else "",
            json.dumps(job.evidence, ensure_ascii=True, sort_keys=True),
        )

        def synthesise() -> str:
            return generate_slm(
                self.slm_model,
                user_prompt="Produce the final report now.",
                system_prompt=report_prompt,
                max_new_tokens=8192,
                temperature=0.4,
                cancel_check=cancelled,
                thinking_mode=True,
                read_timeout=DEEP_THINK_READ_TIMEOUT_S,
                generation_timeout=DEEP_THINK_GENERATION_TIMEOUT_S,
                recover_on_failure=False,
            )

        report = synthesise() if self.thinking_server_slots == 2 else self._with_turn_lock(synthesise)
        if cancelled():
            return "", findings
        report = (report or "").strip()
        if report:
            logger.info("Deep-think job %s produced a %d-character report", job.id, len(report))
            return report, findings
        return fallback_report(), findings

    def _on_thinking_job_status(self, job: BackgroundJob) -> None:
        """Publish job transitions for dashboard and Home Assistant SSE clients."""
        if job.status == "NEEDS_INPUT":
            question = job.summary.removeprefix("Reactive question:").strip()
            if job.snapshot.origin_satellite_id:
                self._pending_thinking_tasks[job.snapshot.origin_satellite_id] = job.snapshot.task
            self._history.append(
                {"role": "tool", "name": "deep_think", "content": job.summary}
            )
            self._trim_history()
            self._dispatch_event(
                {
                    "role": "thinking",
                    "ts": time.time(),
                    "job_id": job.id,
                    "status": job.status,
                    "summary": question[:500],
                    "error": "",
                    "note_id": "",
                    "task": job.snapshot.task,
                    "stage": job.stage,
                }
            )
            if job.snapshot.origin_source == "conversation" and question:
                self._emit_turn_event(
                    "assistant",
                    question,
                    "proactive",
                    satellite_id=job.snapshot.origin_satellite_id,
                )
                satellite_id = job.snapshot.origin_satellite_id
                if satellite_id and satellite_id != "dashboard-text":
                    threading.Thread(
                        target=self.speak_proactive,
                        args=(question,),
                        kwargs={
                            "emit_event": False,
                            "satellite_id": satellite_id,
                            "follow_up": True,
                        },
                        daemon=True,
                        name=f"thinking-input-{job.id[:8]}",
                    ).start()
            return
        if job.status == "READY" and not job.note_id:
            job.note_id = self._save_thinking_report(job)
        if job.status == "READY":
            if job.snapshot.origin_satellite_id:
                self._pending_thinking_tasks.pop(job.snapshot.origin_satellite_id, None)
            report_summary = self._report_summary(job.summary)
            self._history.append(
                {
                    "role": "tool",
                    "name": "deep_think",
                    "content": (
                        f"Completed deep-think report for: {job.snapshot.task}\n"
                        f"Summary: {report_summary}\n"
                        f"Report note: {job.note_id or 'unavailable'}"
                    ),
                }
            )
            self._trim_history()
        self._dispatch_event(
            {
                "role": "thinking",
                "ts": time.time(),
                "job_id": job.id,
                "status": job.status,
                "summary": job.summary[:500],
                "error": job.error,
                "note_id": job.note_id,
                "task": job.snapshot.task,
                "stage": job.stage,
            }
        )
        artifact = (
            {
                **job.artifact,
                "type": "flight_plan",
                "note_id": job.note_id,
                "report_url": f"/reports/{job.note_id}" if job.note_id else "",
                "prices_can_change": True,
            }
            if job.status == "READY" and job.artifact is not None and job.artifact.get("type") == "flight_search"
            else {
                **job.artifact,
                "note_id": job.note_id,
                "report_url": f"/reports/{job.note_id}" if job.note_id else "",
            }
            if job.status == "READY" and job.artifact is not None
            else {
                "type": "generated_report",
                "title": job.snapshot.task[:160] or "Research report",
                "created_at": job.created_at,
                "summary": job.summary[:600],
                "report_url": f"/reports/{job.note_id}",
            }
            if job.status == "READY" and job.note_id
            else None
        )
        if job.status == "READY" and job.snapshot.origin_source == "conversation":
            if job.snapshot.origin_satellite_id:
                self._completed_thinking_reports[job.snapshot.origin_satellite_id] = (
                    {
                        "note_id": job.note_id,
                        "task": job.snapshot.task,
                        "summary_delivered": False,
                    }
                )
            self._emit_turn_event(
                "assistant",
                "I've finished looking into that. Would you like a short summary?",
                "proactive",
                satellite_id=job.snapshot.origin_satellite_id,
                artifact=artifact,
            )
            satellite_id = job.snapshot.origin_satellite_id
            if satellite_id and satellite_id != "dashboard-text":
                threading.Thread(
                    target=self.speak_proactive,
                    args=("I've finished looking into that. Would you like a short summary?",),
                    kwargs={
                        "emit_event": False,
                        "satellite_id": satellite_id,
                        "follow_up": True,
                    },
                    daemon=True,
                    name=f"thinking-complete-{job.id[:8]}",
                ).start()
        elif artifact is not None:
            self._emit_turn_event(
                "assistant",
                (
                    "I found a recommended flight option and saved the full comparison."
                    if artifact["type"] == "flight_plan"
                    else "I've completed the research report and saved the full version."
                ),
                "proactive",
                artifact=artifact,
            )

    def _save_thinking_report(self, job: BackgroundJob) -> str:
        """Append a completed worker result to a durable, retrievable Markdown report."""
        note_id = f"fulloch-reports/{local_now().strftime('%Y-%m-%d')}-{job.id[:8]}"
        path = notes_root.get_notes_root() / f"{note_id}.md"
        timestamp = local_now().strftime("%Y-%m-%d %H:%M")
        findings = job.summary or "No report was produced."
        if not re.search(r"(?im)^## Summary\s*$", findings):
            first_sentence = split_sentences(findings)
            answer = first_sentence[0] if first_sentence else "No reliable conclusion was produced."
            findings = (
                "## Summary\n\n"
                f"{answer} Scope: this preliminary report is limited to the retrieved evidence. "
                "Caveat: details may remain incomplete.\n\n"
                + findings
            )
        report = (
            f"# Deep Think Report\n\n"
            f"**Completed:** {timestamp}\n\n"
            f"**Objective:** {job.snapshot.task}\n\n"
            f"## Findings\n\n{findings}\n"
        )
        evidence_path = path.with_suffix(".evidence.json")
        evidence = {
            "task": job.snapshot.task,
            "evidence": job.evidence,
            "artifacts": job.artifacts,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            evidence_path.write_text(
                json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8"
            )
            notes._after_write(path)
            logger.info("Saved deep-think report %s to %s", job.id, path)
            return note_id
        except Exception:
            logger.exception("Failed to persist thinking report %s", job.id)
            return ""

    def _with_turn_lock(self, action: Callable[[], str]) -> str:
        with self._turn_lock:
            return action()

    def _background_job_admitted(self) -> bool:
        """One-slot jobs run only when no foreground turn or speech is live."""
        if self.thinking_server_slots == 2:
            return True
        if self._turn_arbiter.owner is not None:
            return False
        return not any(
            s.turn_active or s.active_session is not None for s in self.satellites.values()
        )

    def _yield_background_for_foreground(self) -> None:
        if self.thinking_jobs is not None:
            self.thinking_jobs.pause_active()
            self.thinking_jobs.wake()

    def _emit_satellite_state(self, satellite_id: str, state: str, **extra) -> None:
        """Publish an authoritative lifecycle transition for a native satellite."""
        event = {
            "type": "assistant.state",
            "state": state,
            "satellite_id": satellite_id,
            **extra,
        }
        dispatch = getattr(self, "_dispatch_event", None)
        if dispatch is not None:
            dispatch(event)

    def _emit_satellite_stand_down_after_tts(self, sat: SatelliteSession) -> None:
        """Close a short server-generated response after its final PCM drains."""
        if sat.protocol_turn_id is not None:
            self._dispatch_event(
                {
                    "type": "assistant.stand_down_after_tts",
                    "satellite_id": sat.id,
                    "turn_id": sat.protocol_turn_id,
                }
            )

    def _begin_satellite_turn(self, sat: SatelliteSession) -> None:
        """Establish a new native-satellite turn when its wakeword is accepted."""
        sat.protocol_state_generation += 1
        if sat.protocol_follow_up_timer is not None:
            sat.protocol_follow_up_timer.cancel()
            sat.protocol_follow_up_timer = None
        sat.protocol_turn_id = uuid.uuid4().hex
        self._emit_satellite_state(sat.id, "wake_detected", turn_id=sat.protocol_turn_id)
        self._emit_satellite_state(sat.id, "listening", turn_id=sat.protocol_turn_id)

    def _set_pending_satellite_wake(self, sat: SatelliteSession) -> None:
        """Bound optimistic wake feedback until its final ASR verification completes."""
        if sat.protocol_wake_timer is not None:
            sat.protocol_wake_timer.cancel()
        sat.protocol_wake_pending = True
        generation = sat.protocol_state_generation

        def expire() -> None:
            if sat.protocol_wake_pending and sat.protocol_state_generation == generation:
                logger.warning("Wake verification timed out (%s)", sat.id)
                self.audio_capture.flush(sat.id)
                self._reject_pending_satellite_wake(sat)

        timer = threading.Timer(10, expire)
        timer.daemon = True
        sat.protocol_wake_timer = timer
        timer.start()

    @staticmethod
    def _clear_pending_satellite_wake(sat: SatelliteSession) -> None:
        sat.protocol_wake_pending = False
        if sat.protocol_wake_timer is not None:
            sat.protocol_wake_timer.cancel()
            sat.protocol_wake_timer = None

    def _on_wakeword_model_match(self, satellite_id: str) -> None:
        """Give native satellites immediate feedback while ASR verifies the KWS."""
        sat = self.satellites.get(satellite_id)
        if sat is None or sat.protocol_turn_id is not None:
            return
        self._begin_satellite_turn(sat)
        self._set_pending_satellite_wake(sat)

    def _on_asr_work_dropped(
        self, satellite_id: str, kind: str, candidate: bool, reason: str
    ) -> None:
        """Close optimistic native wake feedback when its ASR work is discarded."""
        sat = self.satellites.get(satellite_id)
        if sat is None or not candidate or not sat.protocol_wake_pending:
            return
        logger.info(
            "Standing down pending wake after %s ASR work was %s (%s)", kind, reason, satellite_id
        )
        self._reject_pending_satellite_wake(sat)

    def _begin_satellite_follow_up_turn(self, sat: SatelliteSession) -> None:
        """Replace an expiring follow-up window with the spoken continuation."""
        sat.protocol_state_generation += 1
        if sat.protocol_follow_up_timer is not None:
            sat.protocol_follow_up_timer.cancel()
            sat.protocol_follow_up_timer = None
        sat.protocol_turn_id = uuid.uuid4().hex
        self._emit_satellite_state(sat.id, "listening", turn_id=sat.protocol_turn_id)

    def _reject_pending_satellite_wake(self, sat: SatelliteSession) -> None:
        """Close a turn opened by a soft transcript that the final ASR rejected."""
        if sat.protocol_wake_pending:
            self._clear_pending_satellite_wake(sat)
            self._satellite_idle(sat)

    def _satellite_thinking(self, sat: SatelliteSession) -> None:
        """The recorder endpoint was accepted and processing is about to begin."""
        if sat.protocol_turn_id is not None:
            self._clear_pending_satellite_wake(sat)
            self._emit_satellite_state(sat.id, "thinking", turn_id=sat.protocol_turn_id)

    def _schedule_satellite_follow_up(
        self, sat: SatelliteSession, playback_end: float, window: float
    ) -> None:
        """Publish follow-up only once the last scheduled PCM should be silent."""
        sat.protocol_state_generation += 1
        generation = sat.protocol_state_generation
        if sat.protocol_follow_up_timer is not None:
            sat.protocol_follow_up_timer.cancel()

        def open_window() -> None:
            if sat.protocol_state_generation != generation or sat.protocol_turn_id is None:
                return
            expires_at_ms = int((time.time() + window) * 1000)
            self._emit_satellite_state(
                sat.id, "follow_up", turn_id=sat.protocol_turn_id, expires_at_ms=expires_at_ms
            )

            def expire() -> None:
                if sat.protocol_state_generation != generation:
                    return
                sat.protocol_turn_id = None
                self._emit_satellite_state(sat.id, "idle", turn_id=None)

            expiry = threading.Timer(window, expire)
            expiry.daemon = True
            sat.protocol_follow_up_timer = expiry
            expiry.start()

        timer = threading.Timer(max(0.0, playback_end - time.monotonic()), open_window)
        timer.daemon = True
        sat.protocol_follow_up_timer = timer
        timer.start()

    def _satellite_idle(self, sat: SatelliteSession) -> None:
        sat.protocol_state_generation += 1
        self._clear_pending_satellite_wake(sat)
        if sat.protocol_follow_up_timer is not None:
            sat.protocol_follow_up_timer.cancel()
            sat.protocol_follow_up_timer = None
        sat.protocol_turn_id = None
        self._emit_satellite_state(sat.id, "idle", turn_id=None)

    def set_llm_model(self, model: str) -> dict:
        """Hot-swap the remote LLM model on the live handle — no restart.

        Only valid when the running LLM backend is the remote OpenAI client: the
        model is a per-request string there (`core/llm_openai.py`), so switching
        it needs no reload, just a guarded attribute swap. A local llama / none
        backend loads its weights at startup and can't switch live, so we refuse
        and the caller tells the user a restart is required.

        Serialised under `_turn_lock` (the local model server and the
        handle is shared with the transcriber / barge-in worker) so an in-flight
        turn never reads a half-swapped model.
        """
        model = (model or "").strip()
        if not model:
            return {"ok": False, "error": "model name required"}
        handle = self.slm_model
        if not getattr(handle, "_fulloch_remote", False):
            return {"ok": False, "error": "live LLM backend is not OpenAI; restart to apply"}
        with self._turn_lock:
            handle.set_model(model)
        logger.info("Live LLM model switched to %s", model)
        return {"ok": True, "model": model}

    def connect_satellite(
        self,
        satellite_id: str,
        conversation_mode: Optional[bool] = None,
        *,
        label: Optional[str] = None,
        ha_area: Optional[str] = None,
        ha_area_name: Optional[str] = None,
        server_vad: bool = True,
        auth_token: Optional[str] = None,
        device_id: Optional[str] = None,
        initial_grace: bool = False,
    ) -> "queue.Queue":
        """Start a satellite session (browser `/ws/satellite` or a
        `/ws/satellite-v2` client). Returns the audio chunk queue.

        `satellite_id` is minted by the caller (the WS handler), not here —
        the caller needs to know it synchronously at connect time (to key
        `set_satellite_sink`/`disconnect_satellite` and, from Phase 4, to
        tell the browser its own id). The WebSocket handler feeds float32
        16 kHz mono chunks into the returned queue; the satellite_recorder_thread
        drains it and pushes utterances to the ASR pipeline. Conversation mode is
        exclusive, disconnects any other voice satellites, and bypasses the
        wakeword; a None request uses the configured default. Normal mode opens
        a 60 s initial wakeword-free grace window.

        `label`/`server_vad`/`auth_token`/`device_id` are the satellite-v2 forward-compat
        fields (#12/#13) — the browser path leaves them at their defaults
        (`None`/`True`/`None`/`None`); only the `/ws/satellite-v2` handler passes
        real values. `initial_grace` is browser-only: the user explicitly enters
        Voice mode there, while an always-on native satellite must start gated by
        its wakeword. `ha_area` (#14, 6b) IS set by the browser path too, from
        the `?area=` query param on `/ws/satellite` — the user's one-time
        room picker choice, persisted client-side in `localStorage`.
        `ha_area_name` is that same choice's display name (`?area_name=`),
        used only for the `satellite_label` fallback in `_emit_turn_event`/
        `_emit_agent_event` — never for area-scoping logic, which always goes
        through `ha_area`'s id.
        """
        # A reboot can establish a new WebSocket before the previous TCP
        # connection is declared dead. A native device ID has one active
        # session, so stop the old recorder and sender before replacing it.
        if device_id:
            prior_ids = [
                sid for sid, existing in self.satellites.items() if existing.device_id == device_id
            ]
            for prior_id in prior_ids:
                logger.info(
                    "Replacing existing satellite session %s for device %s", prior_id, device_id
                )
                self.disconnect_satellite(prior_id)

        chunk_q: "queue.Queue" = queue.Queue(maxsize=100)
        session = SatelliteSession(
            id=satellite_id,
            chunk_q=chunk_q,
            label=label,
            ha_area=ha_area,
            ha_area_name=ha_area_name,
            server_vad=server_vad,
            auth_token=auth_token,
            device_id=device_id,
        )
        # Check and register under one lock. Otherwise a satellite could pass
        # the ownership check just as another connection enables Conversation
        # mode, then appear after that mode's disconnect snapshot.
        with self._conversation_lock:
            if self._conversation_owner_id is not None:
                raise ConversationModeUnavailable("Conversation mode is active on another device.")
            voice_sessions = [sid for sid in self.satellites if sid != "dashboard-text"]
            if len(voice_sessions) >= self.max_voice_satellites:
                raise ConversationModeUnavailable(
                    "The maximum number of voice satellites is connected."
                )
            self.satellites[satellite_id] = session
        requested_mode = (
            self.conversation_mode_default if conversation_mode is None else conversation_mode
        )
        if requested_mode:
            enabled, reason = self.set_satellite_conversation_mode(satellite_id, True)
            if not enabled:
                self.satellites.pop(satellite_id, None)
                raise ConversationModeUnavailable(reason)
        self._last_connected_satellite_id = satellite_id
        session.recorder_thread = threading.Thread(
            target=self.audio_capture.satellite_recorder_thread,
            args=(session,),
            daemon=True,
            name="satellite-recorder",
        )
        session.recorder_thread.start()
        # A browser user explicitly clicked into Voice mode and expects to speak
        # immediately. Native satellites are always listening, so a reconnect
        # must not turn nearby conversation into a wakeword-free follow-up.
        if initial_grace and not session.conversation_mode:
            session.last_turn_end = time.monotonic()
            self.audio_capture.arm_follow_up(session, 60)
        logger.info(
            "Satellite %s connected (conversation_mode=%s)", satellite_id, session.conversation_mode
        )
        return chunk_q

    def disconnect_satellite(self, satellite_id: str) -> None:
        """Tear down one satellite session; stop its recorder thread.

        Only touches `satellite_id`'s own session — a second connected
        satellite's queue/sink/recorder thread is untouched.
        """
        self.set_satellite_conversation_mode(satellite_id, False)
        # A reconnect receives a new server session ID. Discard any utterances
        # queued by the retired recorder so the shared ASR worker cannot process
        # stale audio after its replacement is already active.
        self.audio_capture.flush(satellite_id)
        session = self.satellites.pop(satellite_id, None)
        if session is not None:
            # Cancel the active turn before losing its session/sink. The recorder
            # gets a sentinel immediately; a short join releases its per-session
            # VAD/KWS models promptly without delaying a slow model turn.
            if session.active_session is not None:
                session.active_session.stop()
            if session.turn_session is not None:
                session.turn_session.stop()
            self._clear_pending_conversation_turn(session)
            self._satellite_idle(session)
            if session.tts_sink is not None:
                sink = _safe_sink(session.tts_sink)
                tts_mod = getattr(self, "_tts_module", None)
                if tts_mod is not None:
                    tts_mod.force_cancel_playback(sink)
                sink.put(("stop",))
            try:
                session.chunk_q.put_nowait(None)  # sentinel → recorder exits
            except queue.Full:
                # Drop stale microphone data to make shutdown observable now.
                try:
                    session.chunk_q.get_nowait()
                    session.chunk_q.put_nowait(None)
                except queue.Empty:
                    pass
            if (
                session.recorder_thread is not None
                and session.recorder_thread is not threading.current_thread()
            ):
                session.recorder_thread.join(timeout=1.0)
        # A mid-turn disconnect must release the arbiter like a stop would —
        # otherwise a satellite that vanished mid-reply would wedge every
        # other satellite's turns behind a lock nobody will ever release.
        self._turn_arbiter.release(satellite_id)
        logger.info("Satellite %s disconnected", satellite_id)

    @property
    def conversation_owner_id(self) -> Optional[str]:
        with self._conversation_lock:
            return self._conversation_owner_id

    def set_satellite_conversation_mode(self, satellite_id: str, enabled: bool) -> tuple[bool, str]:
        """Enable or disable exclusive Conversation mode for one satellite."""
        satellites_to_disconnect: list[str] = []
        with self._conversation_lock:
            session = self.satellites.get(satellite_id)
            if session is None:
                return False, "Satellite is no longer connected."
            if not enabled:
                session.conversation_mode = False
                session.conversation_listening_pending = False
                self._clear_pending_conversation_turn(session)
                if self._conversation_owner_id == satellite_id:
                    self._conversation_owner_id = None
                self.audio_capture.clear_follow_up(session)
                self._satellite_idle(session)
                return True, ""
            if self._conversation_owner_id not in (None, satellite_id):
                return False, "Conversation mode is active on another device."
            was_enabled = session.conversation_mode
            self._conversation_owner_id = satellite_id
            session.conversation_mode = True
            if not was_enabled:
                session.conversation_listening_pending = True
            self.audio_capture.arm_follow_up(session, 3600)
            # Native satellites require a lifecycle turn ID before they can
            # accept a wakeword-free request. Dashboard activation previously
            # changed only the mode flag, leaving the ESP32 idle with no turn.
            if session.device_id and not was_enabled:
                self._begin_satellite_follow_up_turn(session)
            satellites_to_disconnect = [
                sid for sid in self.satellites if sid not in ("dashboard-text", satellite_id)
            ]

        # Tear down other voice paths outside the conversation lock: disconnect
        # clears their own conversation state and takes the same lock.
        for other_id in satellites_to_disconnect:
            self.request_stop(other_id)
            self.disconnect_satellite(other_id)
        self._play_conversation_listening_phrase(session)
        logger.info("Conversation mode enabled for satellite %s", satellite_id)
        return True, ""

    def set_satellite_user_muted(self, satellite_id: str, muted: bool) -> tuple[bool, str]:
        """Persistently pause or resume listening for one connected satellite."""
        session = self.satellites.get(satellite_id)
        if session is None or satellite_id == "dashboard-text":
            return False, "Satellite is no longer connected."
        session.user_muted = muted
        if muted:
            self.audio_capture.clear_follow_up(session)
        return True, ""

    def send_satellite_control(self, satellite_id: str, event: dict) -> None:
        """Best-effort delivery of a dashboard-originated control event."""
        session = self.satellites.get(satellite_id)
        if session is None or session.control_sink is None:
            return
        try:
            session.control_sink.put_nowait(event)
        except queue.Full:
            logger.warning("Satellite %s control queue is full", satellite_id)

    def set_satellite_sink(self, satellite_id: str, q: Optional["queue.Queue"]) -> None:
        """Route TTS output to `satellite_id`'s WebSocket queue (or None to clear it)."""
        session = self.satellites.get(satellite_id)
        if session is not None:
            session.tts_sink = q
            self._play_conversation_listening_phrase(session)

    def _play_conversation_listening_phrase(self, sat: SatelliteSession) -> None:
        """Play the one-time acknowledgement when Conversation mode starts."""
        if not sat.conversation_listening_pending or sat.tts_sink is None:
            return
        if not self.conversation_listening_cache:
            return
        sat.conversation_listening_pending = False
        chunks, sr = self.conversation_listening_cache[0]
        prev_sink = getattr(self._turn_local, "sink", None)
        prev_active = getattr(self._turn_local, "tts_active_event", None)
        self._turn_local.sink = _safe_sink(sat.tts_sink)
        self._turn_local.tts_active_event = sat.tts_active
        try:
            self.play_chunks(chunks, sr, session=self.tts_session)
            # Cached playback bypasses speak_proactive(), so explicitly open
            # the native satellite's follow-up state after this acknowledgement.
            playback_end = time.monotonic() + sum(len(chunk) for chunk in chunks) / sr
            self._mark_turn_end(sat.id, playback_end)
        except Exception:
            logger.exception("Conversation-mode acknowledgement playback failed")
        finally:
            self._turn_local.sink = prev_sink
            self._turn_local.tts_active_event = prev_active

    def _sink_for(self, satellite_id: Optional[str]) -> Optional["queue.Queue"]:
        session = self.satellites.get(satellite_id) if satellite_id else None
        return _safe_sink(session.tts_sink) if session is not None else None

    def _tts_lock_for_sink(self, sink) -> Optional[threading.RLock]:
        """Find the owning satellite's stream lock through sink decorators."""
        raw_sink = sink
        while isinstance(raw_sink, (_GainSink, _NonBlockingSink)):
            raw_sink = raw_sink._sink
        for sat in self.satellites.values():
            if sat.tts_sink is raw_sink:
                return sat.tts_stream_lock
        return None

    def play_chunks(
        self,
        chunks,
        sample_rate: int,
        session: Optional[TtsSession] = None,
        sink: Optional["queue.Queue"] = None,
        tts_active_event: Optional[threading.Event] = None,
    ):
        """Route pre-synthesized chunks to whichever satellite the current
        turn belongs to (`self._turn_local`, set by whichever of
        `_run_half_duplex`/`_run_turn`/`speak_proactive` is running on *this*
        thread). A thin bound wrapper around the active TTS backend's
        `play_chunks` — kept so the ~15 call sites in agent_loop.py (stall/ack
        phrases, fallback replies) need no changes to pass a sink explicitly;
        the sink is resolved once per turn instead.

        `sink` / `tts_active_event` can be passed explicitly for callers on a
        different thread (ThinkingWatchdog, daemon ack threads) that don't have
        `self._turn_local` set.
        """
        if sink is None:
            sink = getattr(self._turn_local, "sink", None)
        if tts_active_event is None:
            tts_active_event = getattr(self._turn_local, "tts_active_event", None)
        gain = getattr(self._turn_local, "tts_gain", 1.0)
        if sink is not None and gain != 1.0:
            sink = _GainSink(sink, gain)
        lock = self._tts_lock_for_sink(sink)
        if lock is None:
            return self._tts_module.play_chunks(
                chunks, sample_rate, session=session, sink=sink, tts_active_event=tts_active_event
            )
        with lock:
            return self._tts_module.play_chunks(
                chunks, sample_rate, session=session, sink=sink, tts_active_event=tts_active_event
            )

    def speak_stream(
        self,
        text: str,
        prompt,
        session: Optional[TtsSession] = None,
        stats: Optional[TurnStats] = None,
        on_first_audio: Optional[Callable[[], None]] = None,
    ):
        """Synthesise and speak on whichever satellite the current turn
        belongs to (`self._turn_local`). See `play_chunks` for why this is a
        bound wrapper rather than a raw rebinding of the tts module function.
        """
        sink = getattr(self._turn_local, "sink", None)
        gain = getattr(self._turn_local, "tts_gain", 1.0)
        if sink is not None and gain != 1.0:
            sink = _GainSink(sink, gain)
        kwargs = {
            "session": session,
            "stats": stats,
            "on_first_audio": on_first_audio,
            "sink": sink,
            "tts_active_event": getattr(self._turn_local, "tts_active_event", None),
        }
        lock = self._tts_lock_for_sink(sink)
        if lock is None:
            return self._tts_module.speak_stream(spoken_for_tts(text), prompt, **kwargs)
        with lock:
            return self._tts_module.speak_stream(spoken_for_tts(text), prompt, **kwargs)

    def _prepare_delivery_request(self, user_prompt: str, sat: SatelliteSession) -> str:
        """Extract delivery controls and retain the cross-backend quiet setting."""
        previous = sat.higgs_delivery if self._tts_backend == "higgs-gguf" else ""
        agent_prompt, delivery = extract_delivery_request(user_prompt, previous)
        sat.tts_gain = self.whisper_gain if delivery == "<|style:whispering|>" else 1.0

        if self._tts_backend == "higgs-gguf":
            sat.higgs_delivery = delivery
            return agent_prompt

        # Other backends cannot honour Higgs pacing tags. Only remove explicit
        # whisper wording when it has the equivalent, model-agnostic gain.
        sat.higgs_delivery = ""
        return agent_prompt if sat.tts_gain != 1.0 else user_prompt

    def replay_greeting(self, satellite_id: str) -> None:
        """Mark the greeting as delivered without playing it.

        The opening greeting was already synthesised during startup by
        `_warm_and_announce` to warm the TTS model and the LLM cache — the
        actual playback attempt during startup silently dropped because no
        satellite was connected. Now that one is, we mark it delivered so
        the next (real) user turn doesn't trigger a second greeting, but
        playing it back here would inject several seconds of irrelevant
        audio into the mic, which gets transcribed as user input while the
        mic is still live between connect and the first `tts_start` frame.

        The warming effect is already done; no audible output is needed.

        No-op (no delivered mark) when cache is empty — startup may have
        failed, and a later replay_greeting on reconnect would have nothing
        to skip.
        """
        if not self.greeting_cache:
            return
        self._greeting_delivered = True

    def _play_alarm_tone(self, session: Optional[TtsSession] = None) -> float:
        """Push the pre-rendered alarm tone straight to the current turn's sink.

        Uses the same ("start", sr) / (chunk, None) / ("end",) sink protocol
        as speak_stream, so it queues seamlessly ahead of a spoken reminder on
        the browser (satPlayAt in index.js chains consecutive tts_start/end
        pairs back-to-back rather than resetting). Returns the tone's
        duration in seconds (0.0 if it couldn't be played) so the caller can
        add it to the reminder's own playback-end estimate.
        """
        sink = getattr(self._turn_local, "sink", None)
        if sink is None or (session is not None and session.cancelled):
            return 0.0
        try:
            import soundfile as sf

            data, sr = sf.read(ALARM_WAV_PATH, dtype="float32", always_2d=False)
        except Exception as e:
            logger.warning(f"Alarm tone failed to load from {ALARM_WAV_PATH}: {e}")
            return 0.0
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        sink.put(("start", sr))
        sink.put((data, None))
        sink.put(("end",))
        return len(data) / sr

    def set_vault_context(
        self,
        current_file: Optional[dict],
        vault_path: Optional[str] = None,
    ) -> None:
        """Update the currently-open Obsidian note context injected into the agent prompt."""
        self._vault_current_file = current_file
        if vault_path is not None:
            self._vault_path = vault_path

    # Config paths the running assistant can apply without a restart. Voice
    # options are conditional on the active TTS backend — see apply_hot_config.
    _HOT_CONFIG_PATHS = frozenset(
        {
            "general.log_level",
            "general.barge_in",
            "general.conversation_mode_default",
            "general.follow_up_time",
            "general.tts_speed",
            "general.whisper_gain",
            "general.voice_clone",
            "general.use_vad",
            "general.barge_in_threshold_dbfs",
            "general.vad_threshold",
            "general.vad_endpoint_silence_ms",
            "general.vad_min_speech_ms",
            "general.vad_soft_endpoint_silence_ms",
        }
    )

    def _render_phrase_caches(self) -> None:
        acknowledgements = [
            self.synthesize(phrase, self.voice_clone_prompt) for phrase in ACK_PHRASES
        ]
        for attr in ACK_CACHE_ATTRS:
            setattr(self, attr, acknowledgements)
        rendered = len(acknowledgements)
        for attr, phrases, llm_only in STARTUP_CACHE_SPECS:
            if llm_only and not self.llm_enabled:
                continue
            setattr(
                self, attr, [self.synthesize(phrase, self.voice_clone_prompt) for phrase in phrases]
            )
            rendered += len(getattr(self, attr))
        logger.info("Cached %d reusable startup phrase clips", rendered)

    def _rerender_phrase_caches(self) -> None:
        """Re-synthesise the pre-rendered phrase clips in the current voice.

        Each cache is rebuilt fully then rebound (atomic), so the transcriber
        thread reading a cache mid-rebuild sees the old or new list, never a
        partial one — same discipline as the HA deny-list swap.
        """
        self._render_phrase_caches()
        logger.info("Phrase caches re-rendered for voice %r", self.voice_clone)

    def apply_hot_config(self, changes: list) -> set:
        """Apply config changes to the running assistant where possible.

        `changes` is `[{"path", "value"}, ...]` of already-coerced values (from
        update_config). Returns the set of dotted-paths actually applied live;
        anything not in that set still needs a restart, which is how the route
        decides `restart_required`. Each branch is best-effort and isolated so
        one failure doesn't block the others.
        """
        applied: set = set()
        live_voice = self._tts_backend in {
            "kokoro-onnx",
            "pocket-tts-onnx",
            "pocket-tts-gguf",
            "pocket-tts-pytorch",
        }
        for ch in changes:
            path, value = ch["path"], ch["value"]
            if path not in self._HOT_CONFIG_PATHS:
                continue
            try:
                if path == "general.log_level":
                    lvl = _LOG_LEVELS.get(str(value).lower())
                    if lvl is None:
                        continue
                    logging.getLogger().setLevel(lvl)
                elif path == "general.barge_in":
                    self.barge_in = value
                elif path == "general.conversation_mode_default":
                    self.conversation_mode_default = bool(value)
                elif path == "general.follow_up_time":
                    self.follow_up_seconds = parse_barge_time(value)
                elif path == "general.tts_speed":
                    # Qwen and Pocket TTS have no speed knob — the change is a
                    # no-op there, so a restart would not help either.
                    if live_voice and hasattr(self._tts_module, "set_speed"):
                        self._tts_module.set_speed(value)
                        self.tts_speed = value
                elif path == "general.whisper_gain":
                    self.whisper_gain = _whisper_gain(value)
                elif path == "general.voice_clone":
                    # Kokoro and Pocket TTS can swap their cached voice prompt
                    # immediately; Qwen needs a clone warmup, so leave it restart-only.
                    if not live_voice:
                        continue
                    self.voice_clone_prompt = self._tts_module.set_voice(value)
                    self.voice_clone = value
                    threading.Thread(
                        target=self._rerender_phrase_caches,
                        daemon=True,
                        name="voice-rerender",
                    ).start()
                elif path == "general.use_vad":
                    if not self.audio_capture.set_use_vad(bool(value)):
                        continue  # VAD model wasn't loaded — needs a restart
                elif path == "general.barge_in_threshold_dbfs":
                    self.audio_capture.set_barge_in_threshold_dbfs(value)
                elif path == "general.vad_threshold":
                    self.audio_capture.set_vad_params(threshold=value)
                elif path == "general.vad_endpoint_silence_ms":
                    self.audio_capture.set_vad_params(endpoint_silence_ms=value)
                elif path == "general.vad_min_speech_ms":
                    self.audio_capture.set_vad_min_speech_ms(value)
                elif path == "general.vad_soft_endpoint_silence_ms":
                    self.audio_capture.set_vad_params(soft_endpoint_silence_ms=value)
                applied.add(path)
            except Exception as e:  # noqa: BLE001 — one bad field mustn't block the rest
                logger.warning("Hot-apply of %s failed: %s", path, e)
        return applied

    def handle_text_turn(self, text: str) -> str:
        """Run a typed input through the full intent/SLM/tool pipeline.

        Same routing as a voice turn — wakeword strip + regex catch +
        SLM intent + chat fallback — but the answer is returned to the
        caller instead of spoken. Suitable for the dashboard's POST
        /chat endpoint.

        Waits for models to finish loading and serialises against
        in-flight voice turns via `_turn_lock`.
        """
        prompt = (text or "").strip()
        if not prompt:
            return ""

        if not self.models_ready.wait(timeout=120):
            return "The assistant is still starting up. Try again in a moment."

        # Strip a leading wakeword if the user typed "Computer, ..." — keeps
        # parity with the voice path's pre-handle wakeword stripping. Uses
        # the tolerant regex so "Hey, Atticus ..." strips the same as
        # "Hey Atticus ...".
        lowered = prompt.lower()
        leading = self._wakeword_re.match(lowered)
        if leading is not None:
            prompt = prompt[leading.end() :].lstrip(" ,.:;-")
            if not prompt:
                return ""

        self._emit_turn_event("user", prompt, "text", satellite_id="dashboard-text")
        # Text turns get the same per-turn bookkeeping a real satellite would,
        # via the reserved "dashboard-text" pseudo-session (see __init__).
        text_session = self.satellites["dashboard-text"]
        if not self._turn_arbiter.try_acquire("dashboard-text"):
            # A satellite's voice turn already owns the arbiter — text turns
            # have no TTS, so the bounce is just a spoken-style text reply
            # rather than an audio clip.
            busy = random.choice(BUSY_PHRASES)
            self._emit_turn_event("assistant", busy, "text", satellite_id="dashboard-text")
            return busy
        self._yield_background_for_foreground()
        # Per-turn cancel handle so the dashboard Stop button can abort the SLM
        # stream mid-turn (request_stop signals `active_session`).
        session = TtsSession()
        text_session.active_session = session
        try:
            self._maybe_reset_session()
            stats = TurnStats()  # text turns have no ASR/TTS
            answer = self._handle_wakeword(
                prompt, session=session, source="text", stats=stats, satellite_id="dashboard-text"
            )
            if session.cancelled:
                # Stopped from the dashboard — stand down silently, no bubble.
                return ""
            cleaned = clean_for_tts(answer)
            self._emit_turn_event(
                "assistant", cleaned, "text", stats=stats, satellite_id="dashboard-text"
            )
            logger.info(stats.log_line())
            self._last_turn_stats = stats.to_payload()
            return cleaned
        except Exception as e:
            logger.error(f"Text turn error: {e}")
            self._note_runtime_error(e)
            err = "Sorry, something went wrong handling that."
            self._emit_turn_event("assistant", err, "text", satellite_id="dashboard-text")
            return err
        finally:
            text_session.active_session = None
            # Feeds `_maybe_reset_session`'s "most recently active satellite"
            # scan — without this, dashboard-text never contributes to it and
            # active text chatting alone couldn't keep the shared history alive.
            text_session.last_turn_end = time.monotonic()
            self._turn_arbiter.release("dashboard-text")

    def _record_spoken(self, spoken: str) -> None:
        """Replace the most recent assistant entry with one whose `reply` is
        the spoken text the user actually heard. Called after a successful
        dispatch turn where the spoken answer is joined tool outputs (not
        the agent's raw JSON). Keeps `_history` representative of what the
        user experienced for downstream SLM calls.
        """
        if not spoken:
            return
        self._history.append(
            {
                "role": "assistant",
                "content": json.dumps({"reply": spoken}),
            }
        )
        self._trim_history()

    def _play_random_ack(
        self,
        session: Optional[TtsSession] = None,
        cache: Optional[list] = None,
        sink: Optional["queue.Queue"] = None,
        tts_active_event: Optional[threading.Event] = None,
    ) -> None:
        """Play one pre-rendered ack chunk. Safe no-op if the cache is empty
        (e.g. an early failure before `_warm_and_announce` populated it).
        Pass `cache` to use an alternate phrase pool (e.g. replan_stall_cache).

        `sink` and `tts_active_event` are accepted explicitly because this is
        almost always called from a daemon thread spawned by the SLM worker;
        `self._turn_local` is a `threading.local()`, so that thread otherwise
        sees no sink and the ack falls through to "TTS chunks dropped on the
        floor". When omitted we fall back to the caller's thread-local.
        """
        c = cache if cache is not None else self.ack_cache
        if not c:
            return
        chunks, sr = random.choice(c)
        prev_sink = getattr(self._turn_local, "sink", None)
        prev_active = getattr(self._turn_local, "tts_active_event", None)
        if sink is not None:
            self._turn_local.sink = sink
        if tts_active_event is not None:
            self._turn_local.tts_active_event = tts_active_event
        try:
            self.play_chunks(chunks, sr, session=session or self.tts_session)
        except Exception:
            logger.exception("ack playback failed")
        finally:
            self._turn_local.sink = prev_sink
            self._turn_local.tts_active_event = prev_active

    def _play_busy_phrase(self, sat: SatelliteSession) -> bool:
        """Play a `BUSY_PHRASES` clip on `sat`'s own sink — the audible
        bounce when `sat` loses the `TurnArbiter` to another satellite (or
        the dashboard's text turn). Not wrapped in a turn, so resolves
        `self._turn_local` explicitly against `sat` rather than relying on
        the usual per-turn setup in `_run_half_duplex`/`_run_turn`.
        """
        if not self.busy_cache:
            return False
        chunks, sr = random.choice(self.busy_cache)
        self._emit_satellite_stand_down_after_tts(sat)
        self._turn_local.sink = _safe_sink(sat.tts_sink)
        self._turn_local.tts_active_event = sat.tts_active
        try:
            self.play_chunks(chunks, sr, session=self.tts_session)
        except Exception:
            logger.exception("busy-phrase playback failed")
        finally:
            self._turn_local.sink = None
            self._turn_local.tts_active_event = None
        return True

    def _speak_no_ai_fallback(
        self, session: Optional[TtsSession], source: str, satellite_id: Optional[str] = None
    ) -> str:
        """No-LLM miss handler: surface a 'basic commands only' phrase.

        Records the phrase in `_history`. For a voice turn with a populated
        cache it plays the matching pre-rendered clip and emits the assistant
        bubble itself, returning "" so the voice caller doesn't speak it again
        (see the `if cleaned:` guards in `_run_turn` / `_run_half_duplex`).
        Otherwise it returns the phrase for the caller to speak/show.
        """
        phrase = NO_AI_PHRASES[0]
        self._record_spoken(phrase)
        if source == "voice" and self.no_ai_cache:
            self._emit_turn_event("assistant", phrase, "voice", satellite_id=satellite_id)
            chunks, sr = self.no_ai_cache[0]
            try:
                self.play_chunks(chunks, sr, session=session or self.tts_session)
            except Exception:
                logger.exception("no-AI fallback playback failed")
            return ""
        return phrase

    def _speak_llm_error_fallback(
        self, session: Optional[TtsSession], source: str, satellite_id: Optional[str] = None
    ) -> str:
        """Remote LLM failure handler: surface an 'AI server unreachable' phrase.

        Used when `RemoteUnreachable` fires during an agent turn and the regex
        fast-path didn't catch the request. Same contract as
        `_speak_no_ai_fallback`: records phrase in `_history`; voice path plays
        the pre-rendered clip and returns "" so the caller doesn't double-speak.
        """
        phrase = LLM_ERROR_PHRASES[0]
        self._record_spoken(phrase)
        if source == "voice" and self.llm_error_cache:
            self._emit_turn_event("assistant", phrase, "voice", satellite_id=satellite_id)
            chunks, sr = self.llm_error_cache[0]
            try:
                self.play_chunks(chunks, sr, session=session or self.tts_session)
            except Exception:
                logger.exception("LLM-error fallback playback failed")
            return ""
        return phrase

    def _speak_tool_unavailable_fallback(
        self, session: Optional[TtsSession], source: str, satellite_id: Optional[str] = None
    ) -> str:
        """Hallucinated-tool guard handler: the agent named a tool that isn't
        loaded. Speak a canned 'can't do that' instead of letting the model
        fabricate an answer from priors.

        Same contract as `_speak_no_ai_fallback`: records the phrase in
        `_history`; for a voice turn with a populated cache it plays the matching
        pre-rendered clip, emits the assistant bubble, and returns "" so the
        caller doesn't speak it again. Otherwise returns the phrase to speak/show.
        """
        phrase = TOOL_UNAVAILABLE_PHRASES[0]
        self._record_spoken(phrase)
        if source == "voice" and self.tool_unavailable_cache:
            self._emit_turn_event("assistant", phrase, "voice", satellite_id=satellite_id)
            chunks, sr = self.tool_unavailable_cache[0]
            try:
                self.play_chunks(chunks, sr, session=session or self.tts_session)
            except Exception:
                logger.exception("tool-unavailable fallback playback failed")
            return ""
        return phrase

    def _summarise_search_result(self, payload: str, cancel_check, stats=None) -> str:
        """Compress a `User question:` web-search payload into a short spoken answer.

        `payload` is the raw tool return: snippets + the rephrased query.
        Runs a free-text SLM call (no agent grammar) with the focused
        web-summary prompt. Result replaces the raw snippets in `_history`
        so the agent's next call sees a clean 2-4 sentence summary instead
        of kilobytes of HTML.
        """
        return generate_slm(
            self.slm_model,
            user_prompt=payload,
            system_prompt=self.web_summary_prompt,
            cancel_check=cancel_check,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            stats=stats,
        )

    def _summarise_partial_thinking(self, cancel_check, stats=None) -> str:
        """Spoken summary of the last cancelled thinking turn.

        Consumes the captured partial state regardless of outcome — one
        successful summary uses it; a stale/missing one returns a graceful
        message and clears it so the user doesn't get prompted again.

        Runs as a free-text call (no agent grammar) since we want a direct
        spoken summary, not a JSON-wrapped agent emission.
        """
        partial = self._last_thinking_partial
        question = self._last_thinking_question
        cancelled_at = self._last_thinking_cancelled_at
        self._last_thinking_partial = None
        self._last_thinking_question = None

        if not partial or not question:
            return "I don't have any recent thoughts to summarise."
        if time.monotonic() - cancelled_at > THINKING_PARTIAL_TTL_S:
            return "Those thoughts were too long ago — I've let them go."

        return generate_slm(
            self.slm_model,
            user_prompt=get_partial_thinking_summary_prompt(question, partial),
            system_prompt=self.greeting_prompt,
            cancel_check=cancel_check,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            stats=stats,
        )

    def _context_exhausted_reply(self) -> str:
        """Reset the conversation after a context overflow and apologise.

        Wipes `_history` so the next turn starts within budget, then returns
        the spoken/returned apology. Called from the agent loop when an SLM
        call raises `ContextExhaustedError` — otherwise the turn would fail
        silently (the caller's generic `except` only logs).
        """
        logger.warning(
            "SLM context exhausted; clearing %d history entries and apologising",
            len(self._history),
        )
        self._history.clear()
        return CONTEXT_EXHAUSTED_REPLY

    def _handle_wakeword(
        self,
        user_prompt: str,
        session: Optional[TtsSession] = None,
        source: str = "voice",
        on_slm_start: Optional[Callable[[], None]] = None,
        stats: Optional[TurnStats] = None,
        satellite_id: Optional[str] = None,
    ) -> str:
        """Run a user prompt through the unified agent loop.

        Regex fast-path catches common phrasings and synthesises the first
        agent emission. Otherwise the SLM is called with the full history.
        The loop dispatches `actions` emissions in order, replans on
        sentinel/error step results, and exits when the agent emits
        `{"reply": "..."}` or all actions complete without replan.

        `session` (if given) carries a `cancelled` flag that barge-in flips
        to abort mid-stream; pass None for warmup paths. May return an
        empty or partial string on cancel.

        `on_slm_start` fires exactly once, just before the first SLM call —
        skipped entirely when the regex fast-path satisfies the turn.
        Voice callers use it to kick off a parallel "got it" ack.

        `satellite_id` identifies the calling satellite (or `"dashboard-text"`
        for a typed turn) — forwarded to `AgentLoop` so tools can later read
        the calling satellite's identity via the `_current_satellite_id`
        contextvar (only HA's per-satellite area default, #14, uses this so far).

        Serialised under `_turn_lock` so local model turns and history remain ordered
        and dashboard text turns share this method with voice turns.
        """
        self._yield_background_for_foreground()
        with self._turn_lock:
            return AgentLoop(
                self,
                session=session,
                source=source,
                stats=stats,
                on_slm_start=on_slm_start,
                satellite_id=satellite_id,
                satellite=self.satellites.get(satellite_id),
            ).run(user_prompt)

    def _run_half_duplex(
        self,
        user_prompt: str,
        satellite_id: Optional[str] = None,
        stt_seconds=None,
        endpoint_wait_seconds=None,
        endpoint_kind=None,
    ) -> None:
        """Synchronous handle + speak, with mic muted during TTS playback."""
        sat = self.satellites.get(satellite_id)
        if sat is None:
            # The satellite disconnected between this utterance being
            # captured and dispatch — nowhere to reply, drop the turn.
            logger.info("Dropping turn: satellite %s no longer connected", satellite_id)
            return
        if not self._turn_arbiter.try_acquire(satellite_id):
            # Another satellite (or the dashboard's text turn) already owns
            # the arbiter — bounce audibly on this satellite's own sink
            # rather than interleaving into the winner's turn.
            logger.info("Turn bounced (busy): satellite %s", satellite_id)
            if not self._play_busy_phrase(sat):
                self._satellite_idle(sat)
            return
        self._satellite_thinking(sat)
        # Resolve which satellite's sink/tts_active this turn's TTS goes to
        # before anything downstream (ack, fallback phrases, the final
        # reply) can call play_chunks/speak_stream — those read
        # self._turn_local (this thread's slot) rather than taking a sink
        # argument, so it must be set first.
        self._turn_local.sink = _safe_sink(sat.tts_sink)
        self._turn_local.tts_active_event = sat.tts_active
        # A real turn is starting — close the prior follow-up window; it's
        # re-armed at the end of this method via `_mark_turn_end`.
        self.audio_capture.clear_follow_up(sat)
        self._maybe_reset_session()
        self._emit_turn_event("user", user_prompt, "voice", satellite_id=satellite_id)
        agent_prompt = self._prepare_delivery_request(user_prompt, sat)
        self._turn_local.tts_gain = sat.tts_gain
        if agent_prompt != user_prompt:
            logger.debug(
                "Delivery extracted: original=%r agent=%r tag=%s gain=%s",
                user_prompt,
                agent_prompt,
                sat.higgs_delivery,
                sat.tts_gain,
            )
        stats = TurnStats(
            stt_seconds=stt_seconds,
            endpoint_wait_seconds=endpoint_wait_seconds,
            endpoint_kind=endpoint_kind,
        )

        # Per-turn cancel handle so the dashboard Stop button can abort the SLM
        # stream and TTS (request_stop signals `active_session`).
        session = TtsSession()
        sat.active_session = session

        ack_thread: list = []

        def _start_ack() -> None:
            # Cached PCM cannot inherit a per-turn Higgs delivery control. Skip
            # it rather than breaking a whisper/slow/fast request with a loud,
            # unstyled acknowledgement immediately before the answer.
            if sat.tts_gain != 1.0 or (self._tts_backend == "higgs-gguf" and sat.higgs_delivery):
                return
            # Mute this satellite's recorder before the ack plays so it
            # doesn't re-enter the ASR queue. Half-duplex relies on this
            # flag, not on AEC.
            sat.transcribing = False
            t = threading.Thread(
                target=self._play_random_ack,
                args=(session,),
                kwargs={
                    "sink": getattr(self._turn_local, "sink", None),
                    "tts_active_event": getattr(self._turn_local, "tts_active_event", None),
                },
                daemon=True,
            )
            t.start()
            ack_thread.append(t)

        try:
            answer = self._handle_wakeword(
                agent_prompt,
                session=session,
                source="voice",
                on_slm_start=_start_ack,
                stats=stats,
                satellite_id=satellite_id,
            )
            cleaned = clean_for_tts(answer)
            if self._tts_backend == "higgs-gguf":
                cleaned = apply_delivery(cleaned, sat.higgs_delivery)
            sat.transcribing = False
            try:
                if ack_thread:
                    ack_thread[0].join()
                # Stopped mid-SLM or during the ack: stand down silently — no
                # answer bubble, no speech.
                if session.cancelled:
                    return
                # Empty answer = the no-LLM bypass already played a pre-rendered
                # fallback clip (and emitted its bubble), so don't speak again.
                playback_end = None
                if cleaned:
                    # Emit before speaking so the dashboard can reveal the text while
                    # TTS plays, not only after playback finishes.
                    ts = self._emit_turn_event(
                        "assistant", cleaned, "voice", stats=stats, satellite_id=satellite_id
                    )
                    # Emit the TTS stat as soon as the first audio chunk is ready, so the
                    # panel doesn't wait for the whole reply to finish playing.
                    playback_end = self.speak_stream(
                        cleaned,
                        self.voice_clone_prompt,
                        session=session,
                        stats=stats,
                        on_first_audio=lambda: self._emit_stats_patch(
                            ts,
                            "voice",
                            {"tts": stats.tts_payload(), "total": stats.total_with_tts()},
                        ),
                    )
                # A2: one structured line per turn — the measurement base for
                # A0's/A1's acceptance criteria and the seed of the deferred
                # turn-trace dashboard.
                logger.info(stats.log_line())
                self._last_turn_stats = stats.to_payload()
            finally:
                sat.transcribing = True
        finally:
            sat.active_session = None
            self._turn_local.sink = None
            self._turn_local.tts_active_event = None
            self._turn_local.tts_gain = 1.0
            self._turn_arbiter.release(satellite_id)
        # A stop stands down without a follow-up window.
        if not session.cancelled:
            self._mark_turn_end(satellite_id, playback_end)

    def _start_turn(
        self,
        user_prompt: str,
        satellite_id: Optional[str] = None,
        stt_seconds=None,
        endpoint_wait_seconds=None,
        endpoint_kind=None,
    ) -> None:
        """Kick off a barge-in turn (handle + speak) on a worker thread."""
        sat = self.satellites.get(satellite_id)
        if sat is None:
            logger.info("Dropping turn: satellite %s no longer connected", satellite_id)
            return
        # Wait briefly for any prior turn *on this satellite* to wind down so
        # its session can't leak forward and abort the new turn, *before*
        # trying the arbiter — a same-satellite re-trigger (e.g. barge-in
        # redirect) should win its own turn back, not get bounced by its own
        # not-yet-released prior attempt.
        if sat.turn_thread is not None and sat.turn_thread.is_alive():
            sat.turn_thread.join(timeout=2.0)
        if not self._turn_arbiter.try_acquire(satellite_id):
            # A different satellite (or the dashboard's text turn) owns the
            # arbiter — bounce audibly rather than interleaving into it.
            logger.info("Turn bounced (busy): satellite %s", satellite_id)
            if not self._play_busy_phrase(sat):
                self._satellite_idle(sat)
            return
        self._satellite_thinking(sat)
        # A real turn is starting — close the prior follow-up window; the new
        # turn re-arms it at its end via `_mark_turn_end`.
        self.audio_capture.clear_follow_up(sat)
        session = TtsSession()
        sat.turn_session = session
        sat.active_session = session
        sat.turn_active = True
        sat.turn_thread = threading.Thread(
            target=self._run_turn,
            args=(
                user_prompt,
                session,
                satellite_id,
                stt_seconds,
                endpoint_wait_seconds,
                endpoint_kind,
            ),
            daemon=True,
        )
        sat.turn_thread.start()

    def _clear_pending_conversation_turn(self, sat: SatelliteSession) -> None:
        """Cancel an unstarted conversation-mode request for `sat`."""
        with sat.conversation_turn_lock:
            sat.conversation_turn_generation += 1
            if sat.pending_conversation_turn is not None:
                sat.pending_conversation_turn.cancel()
                sat.pending_conversation_turn = None

    def _queue_conversation_turn(
        self,
        user_prompt: str,
        satellite_id: str,
        stt_seconds=None,
        endpoint_wait_seconds=None,
        endpoint_kind=None,
    ) -> None:
        """Dispatch only the latest conversation transcript after a short settle."""
        sat = self.satellites.get(satellite_id)
        if sat is None:
            return
        with sat.conversation_turn_lock:
            sat.conversation_turn_generation += 1
            generation = sat.conversation_turn_generation
            if sat.pending_conversation_turn is not None:
                sat.pending_conversation_turn.cancel()

            def dispatch() -> None:
                with sat.conversation_turn_lock:
                    if generation != sat.conversation_turn_generation or not sat.conversation_mode:
                        return
                    sat.pending_conversation_turn = None
                self._start_turn(
                    user_prompt,
                    satellite_id=satellite_id,
                    stt_seconds=stt_seconds,
                    endpoint_wait_seconds=endpoint_wait_seconds,
                    endpoint_kind=endpoint_kind,
                )

            timer = threading.Timer(CONVERSATION_TURN_SETTLE_S, dispatch)
            timer.daemon = True
            sat.pending_conversation_turn = timer
            timer.start()
        logger.info(
            "Conversation request queued for %.2fs settle: %r",
            CONVERSATION_TURN_SETTLE_S,
            user_prompt,
        )

    def _run_turn(
        self,
        user_prompt: str,
        session: TtsSession,
        satellite_id: Optional[str] = None,
        stt_seconds=None,
        endpoint_wait_seconds=None,
        endpoint_kind=None,
    ) -> None:
        """Worker-thread body: handle the prompt, then speak the answer."""
        sat = self.satellites.get(satellite_id)
        if sat is None:
            logger.info("Dropping turn: satellite %s no longer connected", satellite_id)
            return
        # See _run_half_duplex: must be resolved before any nested
        # play_chunks/speak_stream call (ack, fallback phrases, the reply).
        # This is thread-local, so a concurrent turn on another satellite's
        # own worker thread has its own independent slot.
        self._turn_local.sink = _safe_sink(sat.tts_sink)
        self._turn_local.tts_active_event = sat.tts_active
        try:
            self._maybe_reset_session()
            self._emit_turn_event("user", user_prompt, "voice", satellite_id=satellite_id)
            agent_prompt = self._prepare_delivery_request(user_prompt, sat)
            self._turn_local.tts_gain = sat.tts_gain
            if agent_prompt != user_prompt:
                logger.debug(
                    "Delivery extracted: original=%r agent=%r tag=%s gain=%s",
                    user_prompt,
                    agent_prompt,
                    sat.higgs_delivery,
                    sat.tts_gain,
                )
            stats = TurnStats(
                stt_seconds=stt_seconds,
                endpoint_wait_seconds=endpoint_wait_seconds,
                endpoint_kind=endpoint_kind,
            )

            ack_thread: list = []

            def _start_ack() -> None:
                if sat.tts_gain != 1.0 or (
                    self._tts_backend == "higgs-gguf" and sat.higgs_delivery
                ):
                    return
                # Barge-in mode: recorder stays live; ack shares the turn's
                # session so a user interruption stops it mid-utterance.
                t = threading.Thread(
                    target=self._play_random_ack,
                    args=(session,),
                    kwargs={
                        "sink": getattr(self._turn_local, "sink", None),
                        "tts_active_event": getattr(self._turn_local, "tts_active_event", None),
                    },
                    daemon=True,
                )
                t.start()
                ack_thread.append(t)

            answer = self._handle_wakeword(
                agent_prompt,
                session=session,
                source="voice",
                on_slm_start=_start_ack,
                stats=stats,
                satellite_id=satellite_id,
            )
            if session.cancelled:
                logger.info("Turn cancelled before TTS")
                return
            if ack_thread:
                ack_thread[0].join()
                if session.cancelled:
                    logger.info("Turn cancelled during ack")
                    return
            cleaned = clean_for_tts(answer)
            if self._tts_backend == "higgs-gguf":
                cleaned = apply_delivery(cleaned, sat.higgs_delivery)
            # Empty answer = the no-LLM bypass already played a pre-rendered
            # fallback clip (and emitted its bubble), so don't speak again.
            playback_end = None
            if cleaned:
                # Recorded for self-echo suppression in `_check_barge_in`.
                sat.last_spoken_text = cleaned.lower()
                self.tts_start_time = time.monotonic()
                # Emit before speaking so the dashboard reveals the text while TTS
                # plays. Barge-in may cut speech short; the full text still shows,
                # while _history is reconciled to the spoken text via _record_spoken.
                ts = self._emit_turn_event(
                    "assistant", cleaned, "voice", stats=stats, satellite_id=satellite_id
                )
                # Emit the TTS stat as soon as the first audio chunk is ready, so the
                # panel doesn't wait for the whole reply to finish playing.
                playback_end = self.speak_stream(
                    cleaned,
                    self.voice_clone_prompt,
                    session=session,
                    stats=stats,
                    on_first_audio=lambda: self._emit_stats_patch(
                        ts,
                        "voice",
                        {"tts": stats.tts_payload(), "total": stats.total_with_tts()},
                    ),
                )
            # A2: one structured line per turn — the measurement base for
            # A0's/A1's acceptance criteria and the seed of the deferred
            # turn-trace dashboard.
            logger.info(stats.log_line())
            self._last_turn_stats = stats.to_payload()
            if not session.cancelled:
                # Mic stays live during barge-in TTS, so the queue contains
                # audio of our own voice captured while we were speaking.
                # Flush it before opening the follow-up window — same treatment
                # as a barge-in cancel — otherwise the tail chunk arrives with
                # onset ≈ last_turn_end (speech_onset_t was never set because
                # the residue stayed below the barge-in floor) and falls
                # inside the follow-up window as a spurious new turn.
                self.audio_capture.flush(satellite_id)
                sat.drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                self._mark_turn_end(satellite_id, playback_end)
        except Exception as e:
            logger.error(f"Turn error: {e}")
            self._note_runtime_error(e)
        finally:
            sat.turn_active = False
            sat.active_session = None
            self._turn_local.sink = None
            self._turn_local.tts_active_event = None
            self._turn_local.tts_gain = 1.0
            self._turn_arbiter.release(satellite_id)

    def _cancel_turn(self, satellite_id: str) -> None:
        """Signal `satellite_id`'s active turn to abort and wait briefly for
        it to wind down. Barge-in is always self-referential — a satellite's
        mic can't hear another's speaker — so this only ever cancels the
        satellite whose own utterance triggered the barge-in.
        """
        sat = self.satellites.get(satellite_id)
        if sat is None:
            return
        # sat.tts_sink is a plain attribute (not the turn's thread-local), so
        # it's safe to read here even though this runs on the transcriber
        # thread while the turn itself runs on its own worker thread.
        sink = _safe_sink(sat.tts_sink)
        if sat.turn_session is not None:
            sat.turn_session.stop()
        if sat.turn_thread is not None:
            sat.turn_thread.join(timeout=2.0)
        # Belt-and-suspenders: on fast (GPU) hardware, generation can outrun
        # playback enough that speak_stream's send loop already exited
        # before the flag above was set — nothing is left to emit the
        # mid-stream "cancel" message, and the browser keeps playing
        # everything already queued. Force it regardless of turn state.
        tts_mod = getattr(self, "_tts_module", None)
        if tts_mod is not None:
            tts_mod.force_cancel_playback(sink)

    def request_stop(self, satellite_id: Optional[str] = None) -> None:
        """Stop whatever the agent is doing right now — SLM generation, TTS
        playback, an in-flight turn — silently, and stand down with no
        follow-up window. Safe no-op when idle. This is the dashboard Stop
        button's effect; the voice equivalent is a pure-stop barge-in. An
        in-flight tool dispatch runs to completion (see Known Gaps); the turn
        cancels at the next checkpoint.

        `satellite_id` (or `"dashboard-text"`) picks which turn to stop.
        Omit it to fall back to whichever satellite currently owns the
        `TurnArbiter` — with the arbiter enforcing exclusivity, that's the
        only turn that could possibly be running. A "stop everywhere"
        admin action is out of scope (TODO #14 stretch) — wire it later as
        `for sid in self.satellites: self.request_stop(sid)`.
        """
        target = satellite_id if satellite_id is not None else self._turn_arbiter.owner
        active_job = self.active_thinking_task()
        if (
            active_job is not None
            and active_job.get("origin_source") == "conversation"
            and (target is None or active_job.get("origin_satellite_id") == target)
        ):
            self.cancel_thinking_task(active_job["id"])
        if target is None:
            # Proactive speech (dashboard replay, alarms) does not own an
            # agent turn, but still exposes its TtsSession on its output satellite.
            target = next(
                (sid for sid, sat in self.satellites.items() if sat.active_session is not None),
                None,
            )
        if target is None:
            # A conversation request may still be settling and therefore not
            # own the arbiter or expose an active session yet.
            for sat in self.satellites.values():
                self._clear_pending_conversation_turn(sat)
        if target is not None:
            sat = self.satellites.get(target)
            if sat is not None:
                self._clear_pending_conversation_turn(sat)
            if sat is not None and sat.active_session is not None:
                sat.active_session.stop()
                # On fast (GPU) hardware, generation can outrun playback
                # enough that speak_stream's send loop already exited — and
                # its mid-stream "cancel" path with it — before either flag
                # above was ever checked, so the browser keeps playing
                # everything already queued. Force a cancel unconditionally
                # rather than relying on those flags alone.
                tts_mod = getattr(self, "_tts_module", None)
                if tts_mod is not None:
                    tts_mod.force_cancel_playback(_safe_sink(sat.tts_sink))
                if sat.turn_active:
                    # Barge-in worker: flush the mic and drop in-flight ASR
                    # so our own speech captured mid-turn doesn't replay,
                    # mirroring a voice barge-in.
                    self.audio_capture.flush(target)
                    sat.drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                # A stop means stand down — no wakeword-free follow-up window.
                sat.last_turn_end = 0.0
                self.audio_capture.clear_follow_up(sat)
                self._satellite_idle(sat)
            self._turn_arbiter.release(target)
        # Belt-and-suspenders for any out-of-band playback on the shared
        # session (greeting replay / warmup — not a satellite's own turn).
        self.tts_session.stop()
        self._dispatch_event({"role": "stopped", "ts": time.time()})
        logger.info("Stop requested — standing down")

    def shutdown(self) -> None:
        """Stop active work and owned native subprocesses before process exit."""
        self.request_stop()
        tts_mod = getattr(self, "_tts_module", None)
        if tts_mod is not None and hasattr(tts_mod, "shutdown"):
            tts_mod.shutdown()
        asr_worker = getattr(self.asr_pipe, "worker", None)
        if asr_worker is not None and hasattr(asr_worker, "close"):
            asr_worker.close()
        process = getattr(self.slm_model, "_fulloch_local_process", None)
        if process is not None:
            from .slm import _stop_local_server

            _stop_local_server(process)
        self.audio_capture.stop()

    def _mark_turn_end(
        self, satellite_id: Optional[str], playback_end: Optional[float] = None
    ) -> None:
        """Record `satellite_id`'s turn-end time and open its wakeword-free
        follow-up window.

        `playback_end` is the estimated monotonic-clock time the TTS audio
        actually finishes playing on the browser (`speak_stream`'s return
        value). Generation can finish before playback does, so the follow-up
        window is armed from that estimate rather than generation completion.
        Without this, the follow-up window was armed (and
        could silently expire) while the reply was still audibly playing, so
        a user who replied the instant the voice stopped could still land
        outside the window. Falls back to now() when no estimate is available
        (e.g. no satellite connected).

        The recorder reads the armed window to accept short replies (which the
        full min-utterance length would otherwise drop); the transcriber gauges
        the window itself from the satellite's own `last_turn_end` against
        each utterance's onset. In satellite always-listen mode the recorder
        window is kept open indefinitely so short utterances are never dropped.
        """
        sat = self.satellites.get(satellite_id)
        if sat is None:
            return
        now = time.monotonic()
        sat.last_turn_end = max(now, playback_end) if playback_end is not None else now
        window = 3600.0 if sat.conversation_mode else self.follow_up_seconds
        if window > 0:
            # The recorder's short-utterance exemption must start only after
            # browser playback, matching the transcript-side onset gate below.
            self.audio_capture.arm_follow_up(sat, window, start_at=sat.last_turn_end)
            self._schedule_satellite_follow_up(sat, sat.last_turn_end, window)
        else:
            self._satellite_idle(sat)

    def _is_context_echo(self, command: str) -> bool:
        """True if `command` is built solely from ASR context-bias tokens.

        The Qwen3-ASR context prompt biases the decoder toward the wakeword and
        `asr_context_terms`. Fed a few seconds of contentless voiced noise (a
        sigh or cough Silero scored as speech), the decoder regurgitates those
        bias words verbatim — e.g. "Hey Atticus, <term>" from a cough. A genuine
        command always carries non-bias words, so a wakeword command made
        *entirely* of context tokens is treated as an echo hallucination and
        dropped. Works for any number of terms (single/multi-word): all are
        pooled into `_asr_context_tokens` at init.
        """
        if not self._asr_context_tokens:
            return False
        tokens = re.sub(r"[^\w\s]", "", command.lower()).split()
        return bool(tokens) and all(t in self._asr_context_tokens for t in tokens)

    def _verify_bare_wakeword(self, loudness_db, baseline_db) -> bool:
        """Guard a bare (command-less) wakeword against context-bias hallucination.

        A wakeword with no command is the shape the ASR decoder most readily
        invents from voiced non-speech (a cough/sigh/TV burst): the
        `asr_context_hint` prompt primes the wakeword spelling with nothing to
        anchor it. Two cheap checks gate opening the follow-up window on one:

        1. **Loudness** — a genuine wakeword spoken to the device sits above the
           background-speech baseline (the user is closer/louder than ambient
           media). A bare wakeword at or below baseline is almost certainly the
           room, so reject it for free. (In a quiet room the baseline floors low,
           so most utterances clear this and fall through to check 2.)
        2. **Unbiased re-transcribe** — re-run ASR on the same buffer with the
           context bias removed. If the wakeword no longer appears, the bias
           prompt fabricated it. Only runs when a bias is actually active
           (skipped for non-biasing backends / hint off), so the extra transcribe
           is paid rarely and during the natural pause after a bare wakeword.

        Returns True to proceed (genuine), False to drop.
        """
        # (1) Loudness gate — free, catches the loud-background case.
        if (
            loudness_db is not None
            and loudness_db < baseline_db + BARE_WAKEWORD_MIN_OVER_BASELINE_DB
        ):
            logger.info(
                f"Bare wakeword rejected (at/below background baseline: "
                f"{loudness_db:.1f} < {baseline_db:.1f} dBFS) — likely noise"
            )
            return False

        # (2) Unbiased re-transcribe — only meaningful when a context bias is
        # actually steering the decoder.
        if self.asr_context_hint and getattr(self.asr_pipe, "context", ""):
            buf = self._asr_audio.get("buf")
            if buf is not None and not self._wakeword_in_unbiased_pass(buf):
                logger.info(
                    "Bare wakeword rejected (absent in unbiased re-transcribe) — context-bias echo"
                )
                return False

        return True

    def _wakeword_in_unbiased_pass(self, buf) -> bool:
        """Re-transcribe `buf` with the ASR context bias dropped; True if the
        wakeword still appears.

        Tells a genuine bare wakeword from a bias-prompt hallucination: if the
        decoder only produced the wakeword because the "Technical terms" prompt
        steered it there, an unbiased pass over the same audio won't reproduce it.
        Best-effort — on any ASR error, assume genuine (True) so a transient
        failure never silently swallows real wakewords. Runs in the transcriber
        thread (the only caller of `self.asr_pipe`), so mutating `.context` here
        is safe.
        """
        pipe = self.asr_pipe
        saved = pipe.context
        try:
            pipe.context = ""
            results = pipe(buf, batch_size=1, generate_kwargs={"max_new_tokens": 256})
            # Backends normally return a list for a single buffer, but some
            # streaming wrappers return an iterator here. Verification must not
            # turn a false wakeword into a transcriber error.
            if not isinstance(results, (list, tuple)):
                results = list(results)
        except Exception:  # noqa: BLE001 — verification must never break routing
            logger.debug("Unbiased re-transcribe failed; assuming genuine", exc_info=True)
            return True
        finally:
            pipe.context = saved
        text = (results[0].get("text") if results else "") or ""
        appears = bool(self._wakeword_re.search(text.lower()))
        logger.debug(
            f"Unbiased re-transcribe: {text!r} (wakeword {'present' if appears else 'absent'})"
        )
        return appears

    def _is_self_echo(
        self, satellite_id: Optional[str], text_lower: str, short_only: bool = False
    ) -> bool:
        """True if `text_lower` looks like AEC residue of `satellite_id`'s own
        assistant voice.

        AEC suppresses the speaker-into-mic energy but isn't a phoneme filter,
        so ASR can still re-transcribe parts of the assistant's own answer —
        and if the answer happened to contain the wakeword (or a homophone),
        the assistant would interrupt itself. We sidestep that by checking
        whether the incoming utterance is mostly drawn from the words we just
        spoke *on this satellite*. Real user speech rarely overlaps the
        assistant's prior answer by more than half.
        """
        sat = self.satellites.get(satellite_id)
        spoken = sat.last_spoken_text if sat is not None else ""
        if not spoken:
            return False
        # Strip *all* non-word chars (punctuation and whitespace) from
        # both sides. Per-word substring match against the compacted
        # spoken side handles ASR concatenations like "1254" ↔ "12 54".
        spoken_norm = _NON_WORD_RE.sub("", spoken)
        words = [_NON_WORD_RE.sub("", w) for w in text_lower.split()]
        words = [w for w in words if w]
        if not words:
            return False
        # In the follow-up window, only short utterances (≤ 3 words) are
        # plausibly AEC reverb; longer ones are deliberate user speech even
        # if their words happen to overlap the assistant's prior reply.
        if short_only and len(words) > 3:
            return False
        if len(words) == 1:
            # Single-word transcript — too short to score by overlap.
            # Treat as echo if the word (or the wakeword) appears in what
            # we just said. Catches single-letter residue like the "A"
            # tail of a spoken time ("...12:28 A M."). The wakeword check
            # runs against the un-stripped `spoken` so it still resolves
            # multi-word wakewords like "hey atticus" (spoken_norm has no
            # whitespace, so a substring check there would always miss).
            return words[0] in spoken_norm or self._wakeword_re.search(spoken) is not None
        # AEC residue always starts from a fragment the assistant actually spoke.
        # If the first word isn't in the spoken text at all, this is user speech
        # anchored by their own word (e.g. "yes, save a recipe..." after the
        # assistant offered to save one) — not an echo.
        if words[0] not in spoken_norm:
            return False
        overlap = sum(1 for w in words if w in spoken_norm) / len(words)
        return overlap >= 0.6

    def _is_speaking(self, satellite_id: Optional[str]) -> bool:
        """True while `satellite_id`'s own turn is generating, or its TTS is
        still estimated to be playing on the browser.

        Judged purely against this satellite's own state — a turn on A never
        makes B read as "speaking". `last_turn_end` holds the *estimated*
        playback-end timestamp (see `_mark_turn_end`), not when generation
        handed off its last chunk — on fast (GPU) hardware, generation can
        finish handing chunks to the satellite sink well before the browser
        actually finishes playing them. Gating barge-in on `turn_active`
        alone leaves a dead zone between those two moments where the reply is
        still audibly playing but nothing treats it as interruptible:
        `turn_active` has already flipped False, and the follow-up window
        hasn't opened yet either (its own gate requires the elapsed time
        since `last_turn_end` to be *non-negative*). Checking `last_turn_end`
        here too closes that gap.
        """
        sat = self.satellites.get(satellite_id)
        if sat is None:
            return False
        return sat.turn_active or time.monotonic() < sat.last_turn_end

    def _check_barge_in(self, satellite_id: Optional[str], text_lower: str) -> bool:
        """Return True if `text_lower` should interrupt `satellite_id`'s own
        active turn."""
        if not self._is_speaking(satellite_id):
            return False
        if self._is_self_echo(satellite_id, text_lower):
            return False
        sat = self.satellites.get(satellite_id)
        if sat is not None and sat.conversation_mode:
            return bool(re.search(r"\w", text_lower))
        if self.barge_in == "off":
            return False
        return (
            self._barge_re.search(text_lower) is not None
            or _BARGE_STOP_RE.search(text_lower) is not None
        )

    def _is_bare_barge_wakeword(self, text_lower: str) -> bool:
        """True when a barge-in contains only the assistant's name."""
        stripped = text_lower.strip(_PROMPT_STRIP_CHARS)
        return (
            _BARGE_STOP_RE.search(stripped) is None
            and self._barge_re.fullmatch(stripped) is not None
        )

    def _verify_strict_barge_wakeword(
        self, text_lower: str, loudness_db, baseline_db: float
    ) -> bool:
        """Reject a strict wakeword barge-in fabricated by playback echo.

        Wakeword barge-in deliberately leaves the microphone live. A fast reply
        can therefore leave a short playback/AEC residue whose biased ASR result
        starts with the wakeword but has meaningless trailing text. Explicit stop
        commands and bare-name redirects keep their current low-latency path;
        only the strict "hey atticus ..." shape pays an unbiased re-transcribe.
        """
        if _BARGE_STOP_RE.search(text_lower) is not None:
            return True
        if self._wakeword_re.search(text_lower) is None:
            return True
        return self._verify_bare_wakeword(loudness_db, baseline_db)

    def _is_pure_stop(self, text_lower: str) -> bool:
        """True if a barge-in utterance is *only* a stop/cancel command.

        A bare "stop" (or "Atticus, stop") means *terminate and stand down* —
        the user wants the assistant to cease entirely, not to acknowledge or
        wait for more. So we cancel the turn and return to idle without opening
        a follow-up window (the wakeword is required again). A redirect like
        "Atticus, stop — play jazz instead" carries a real command after the
        stop word, so it keeps the follow-up behaviour.

        We require a stop word to be present, then strip the wakeword/bare-name
        occurrences and the stop words themselves; if every remaining token is a
        greeting or innocuous filler (`_STOP_FILLER_TOKENS`), nothing meaningful
        was asked and it's a pure stop.
        """
        if _BARGE_STOP_RE.search(text_lower) is None:
            return False
        stripped = self._barge_re.sub(" ", text_lower)
        stripped = _BARGE_STOP_RE.sub(" ", stripped)
        tokens = re.sub(r"[^\w\s]", "", stripped).split()
        return all(t in _WAKE_GREETINGS or t in _STOP_FILLER_TOKENS for t in tokens)

    def _transcriber_thread(self):
        """ASR results → wakeword/intent/TTS. Owns model loading on first call."""
        try:
            self._load_models()
        except Exception as exc:  # noqa: BLE001 — surface any load failure to the UI
            _, detail = self._diagnose_failure(exc, spec=getattr(self, "_loading_backend", None))
            logger.exception("Model loading failed — switching to setup")
            self._enter_error_state(detail)
            return
        logger.info("Transcriber started")

        try:
            self._run_transcriber_loop()
        except Exception as exc:  # noqa: BLE001 — the ASR pipeline itself died
            _, detail = self._diagnose_failure(exc)
            logger.exception("Transcriber loop crashed — switching to setup")
            self._enter_error_state(detail)

    def _run_transcriber_loop(self):
        """Drain the ASR pipeline, routing each utterance through a turn."""
        stream_args = (
            self.audio_capture.audio_queue,
            self._asr_onset,
            self._asr_loudness,
            self._asr_provisional,
            self._asr_audio,
            self._asr_satellite_id,
            self._asr_endpoint_wait,
            self._asr_wake_probe,
        )
        # Third-party/test ASR stream generators retain the original eight-arg
        # contract. The KWS metadata is optional for them, and available to the
        # bundled queue drainer without forcing a compatibility break.
        stream_params = inspect.signature(self.asr_stream_generator).parameters
        stream_kwargs = {}
        if "verification_context" in stream_params:
            stream_kwargs["verification_context"] = self._asr_verification_context
        if "kws_candidate_sink" in stream_params:
            stream_kwargs["kws_candidate_sink"] = self._asr_kws_candidate
        if "kws_wav_path_sink" in stream_params:
            stream_kwargs["kws_wav_path_sink"] = self._asr_kws_wav_path
        if "wake_generation_sink" in stream_params:
            stream_kwargs["wake_generation_sink"] = self._asr_wake_generation
        for result in self.asr_pipe(
            self.asr_stream_generator(*stream_args, **stream_kwargs),
            batch_size=1,
            generate_kwargs={"max_new_tokens": 256},
        ):
            try:
                text = result.get("text", "").strip()
                # Which satellite recorded this utterance — set by the ASR
                # stream generator just before yielding it, so it's still
                # correct for this exact result. Resolved up front since
                # every gating check below (barge-in, follow-up, echo,
                # noise-baseline) is judged against this satellite's own
                # state, never another connected satellite's.
                turn_satellite_id = self._asr_satellite_id.get("id")
                sat = self.satellites.get(turn_satellite_id)
                if sat is None:
                    # The satellite disconnected between recording this
                    # utterance and it reaching the transcriber.
                    self.audio_capture.mark_wakeword_wav(
                        self._asr_kws_wav_path.get("path"), accepted=False
                    )
                    logger.debug(
                        "Dropping ASR result for disconnected satellite %s: %r",
                        turn_satellite_id,
                        text,
                    )
                    continue

                wake_generation = self._asr_wake_generation.get("value")
                if wake_generation is not None and wake_generation != sat.protocol_state_generation:
                    self.audio_capture.mark_wakeword_wav(
                        self._asr_kws_wav_path.get("path"), accepted=False
                    )
                    logger.debug(
                        "Dropping expired wake verification result (%s)", turn_satellite_id
                    )
                    continue

                provisional = self._asr_provisional.get("flag", False)
                wake_probe = self._asr_wake_probe.get("flag", False)
                kws_candidate = self._asr_kws_candidate.get("flag", False)
                kws_wav_path = self._asr_kws_wav_path.get("path")
                audio = self._asr_audio.get("buf")
                telemetry_event(
                    "asr_complete",
                    satellite_id=turn_satellite_id,
                    seconds=getattr(self.asr_pipe, "last_transcribe_seconds", None),
                    audio_seconds=round(float(audio.size) / 16000, 3)
                    if isinstance(audio, np.ndarray)
                    else None,
                    kws_candidate=kws_candidate,
                    provisional=provisional,
                    has_text=bool(text),
                )
                if not text:
                    if kws_candidate:
                        self.wakeword_metrics["rejected"] += 1
                        self.audio_capture.mark_wakeword_wav(kws_wav_path, accepted=False)
                    if not provisional:
                        self._reject_pending_satellite_wake(sat)
                    continue

                if kws_candidate:
                    # The recorder only tags an utterance after openWakeWord
                    # fires; this transcript used the normal ASR term context.
                    if self._wakeword_re.search(text.lower()) is None:
                        self.wakeword_metrics["rejected"] += 1
                        self.audio_capture.mark_wakeword_wav(kws_wav_path, accepted=False)
                        logger.info("openWakeWord activation rejected by ASR: %r", text)
                        telemetry_event("wakeword_rejected", satellite_id=turn_satellite_id)
                        self._reject_pending_satellite_wake(sat)
                        continue
                    self.wakeword_metrics["accepted"] += 1
                    self.audio_capture.mark_wakeword_wav(kws_wav_path, accepted=True)
                    telemetry_event("wakeword_accepted", satellite_id=turn_satellite_id)

                if time.monotonic() < sat.drop_results_until:
                    logger.debug(f"Dropping post-cancel ASR result: {text}")
                    continue

                # Drop known ASR noise hallucinations before any routing.
                # Enforcing English causes taps and ambient sounds to surface as
                # short filler tokens ("Yeah.", "Okay.") that would otherwise
                # trigger the follow-up window. Strip punctuation before matching
                # so "Yeah." and "yeah" both hit.
                _text_bare = re.sub(r"[^\w\s]", "", text.lower()).strip()
                if _text_bare in _ASR_NOISE_TOKENS:
                    if not provisional:
                        self._reject_pending_satellite_wake(sat)
                    logger.debug(f"Dropped noise token: {text!r}")
                    continue

                # Context-prompt echo: the decoder occasionally regurgitates the
                # asr_context_hint scaffolding verbatim off non-speech
                # ("Technical terms: hey atticus, <garbage>"). That carries the
                # wakeword plus a non-term tail, so neither _is_context_echo nor
                # the bare-wakeword guard catches it — but the leading marker is
                # an unambiguous tell a user never utters. Drop the whole result.
                if self._context_prompt_marker and self._context_prompt_marker in text.lower():
                    if not provisional:
                        self._reject_pending_satellite_wake(sat)
                    logger.debug(f"Dropped context-prompt echo: {text!r}")
                    continue

                if wake_probe:
                    # This is deliberately feedback-only. The final endpointed
                    # transcript remains responsible for accepting and routing
                    # the command, so an incomplete phrase cannot execute early.
                    wake_match = self._wakeword_re.search(text.lower())
                    probe_command = (
                        text[wake_match.end() :].strip(_PROMPT_STRIP_CHARS)
                        if wake_match is not None
                        else ""
                    )
                    if (
                        wake_match is not None
                        and not self._is_context_echo(probe_command)
                        and not self._is_speaking(turn_satellite_id)
                        and sat.protocol_turn_id is None
                    ):
                        self._begin_satellite_turn(sat)
                        self._set_pending_satellite_wake(sat)
                    continue

                # Provisional (soft-endpoint) snapshot of an unfinished
                # utterance. It may only *commit* a turn at the dispatch gate
                # below; the side-effecting special cases (stand-down, follow-up
                # open, baseline feed) all guard on `provisional` and drop it so
                # the real hard endpoint stays authoritative. Mid-turn a partial
                # is ignored outright — barge-in waits for the hard endpoint.
                if (
                    provisional
                    and sat.turn_active
                    and not (sat.conversation_mode or self.barge_in == "wakeword")
                ):
                    logger.debug(f"Ignoring provisional during active turn: {text!r}")
                    continue

                # Drop the hard VAD endpoint when it's the same utterance as a
                # soft endpoint we already committed. The soft endpoint fires
                # ~200ms before the hard endpoint, and flush()+drop_results_until
                # (0.5s) doesn't cover the ~1.5-2s ASR processing time of the
                # hard endpoint's audio. Without this guard the duplicate
                # transcription matches the wakeword and is treated as a
                # self-barge-in, cancelling the turn that just started.
                if not provisional and sat.provisional_committed_onset > 0:
                    onset = self._asr_onset.get("t", 0.0)
                    hard_text = re.sub(r"[^\w\s]", "", text.lower()).strip()
                    same_committed_text = (
                        sat.provisional_committed_text
                        and hard_text.endswith(sat.provisional_committed_text)
                        and time.monotonic() - sat.provisional_committed_at < 3.0
                    )
                    if abs(onset - sat.provisional_committed_onset) < 0.5 or same_committed_text:
                        logger.debug(
                            f"Dropping hard endpoint (duplicate of committed soft endpoint): {text!r}"
                        )
                        sat.provisional_committed_onset = 0.0
                        sat.provisional_committed_text = ""
                        sat.provisional_committed_at = 0.0
                        continue
                    sat.provisional_committed_onset = 0.0
                    sat.provisional_committed_text = ""
                    sat.provisional_committed_at = 0.0

                text_lower = text.lower()
                wakeword_match = self._wakeword_re.search(text_lower)
                wakeword_present = wakeword_match is not None
                in_follow_up = False
                just_barged_in = False
                bare_barge_wakeword = False

                # Tag every transcription with its dBFS volume and the current
                # background-speech baseline (logged for tuning; the baseline
                # also gates bare wakewords — see `_verify_bare_wakeword`). It is
                # fed only by *genuine background* — see
                # the background-suppression branch below, which is the single
                # feed site. Non-wakeword alone isn't enough: an accepted
                # follow-up is the user (loud, close) and our own TTS bleed
                # mid-turn is us, not the room — feeding either would poison the
                # "background is quieter than the user" estimate.
                loudness_db = self._asr_loudness.get("db")
                baseline_db = sat.noise_baseline.value()
                _vol = f"{loudness_db:.1f}" if loudness_db is not None else "n/a"
                logger.debug(
                    f"Transcribed: {text} [volume={_vol} dBFS, baseline={baseline_db:.1f} dBFS]"
                )

                if self._is_speaking(turn_satellite_id):
                    # Mid-turn (or still-audibly-playing-back) transcriptions
                    # are only acted on as potential barge-ins; otherwise
                    # they're echo of our own voice (or cross-talk during SLM
                    # thinking) and must be ignored.
                    if not self._check_barge_in(turn_satellite_id, text_lower):
                        logger.debug(f"Suppressed (mid-turn, no barge-in match): {text!r}")
                        continue

                    if not self._verify_strict_barge_wakeword(text_lower, loudness_db, baseline_db):
                        logger.info("Strict wakeword barge-in rejected as playback/context echo")
                        continue

                    logger.info(f"BARGE-IN: {text}")
                    bare_barge_wakeword = self._is_bare_barge_wakeword(text_lower)
                    self._cancel_turn(turn_satellite_id)
                    # Buffer contains TTS-bleed accumulated during the
                    # cancelled turn; flush so the next user utterance is
                    # captured cleanly. Then open a short drop window so
                    # any ASR inference still finishing on contaminated
                    # audio gets discarded — without swallowing the user's
                    # real reply, which can't arrive for ≥1.5s.
                    if not bare_barge_wakeword:
                        self.audio_capture.flush(turn_satellite_id)
                        sat.drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                    just_barged_in = True
                elif not wakeword_present:
                    # Follow-up window: a brief grace period after TTS ends
                    # during which wakeword-free input is treated as a reply
                    # to the assistant's last answer. Measured from when this
                    # utterance's *speech began* (`_asr_onset`), not now — so
                    # the window means "started replying within N seconds",
                    # and a long answer isn't disqualified by its own length
                    # plus the silence-detection tail and ASR latency.
                    onset_gap = self._asr_onset["t"] - sat.last_turn_end
                    in_follow_up = (
                        self.follow_up_seconds > 0
                        and sat.last_turn_end > 0
                        and 0 <= onset_gap < self.follow_up_seconds
                    ) or sat.conversation_mode
                    if not in_follow_up:
                        self._reject_pending_satellite_wake(sat)
                        # Genuine background: non-wakeword, not an accepted
                        # follow-up, and not mid-turn (that path is handled by
                        # the `turn_active` branch above). This is the only
                        # site that feeds the noise baseline — the room/TV
                        # talking to itself, which is exactly what we estimate.
                        # A provisional is not a final utterance — never feed it
                        # into the background-noise baseline estimate.
                        if loudness_db is not None and not provisional:
                            sat.noise_baseline.add(loudness_db)
                        logger.debug(
                            f"Suppressed (no wakeword, outside follow-up window "
                            f"[gap={onset_gap:.1f}s, window={self.follow_up_seconds:.0f}s]): {text!r}"
                        )
                        continue
                    if sat.skip_followup_self_echo:
                        # One-shot bypass after a wakeword-alone barge-in:
                        # this utterance is the user's deliberate question,
                        # even if every word also appears in the cancelled
                        # response. Consume the flag here.
                        sat.skip_followup_self_echo = False
                    elif self._is_self_echo(turn_satellite_id, text_lower, short_only=True):
                        # Suppress brief AEC reverb after TTS ends (typically
                        # 1–3 word fragments). Longer utterances are user
                        # speech — even if the words overlap the assistant's
                        # reply (e.g. user picks an option Atticus just offered).
                        logger.debug(f"Suppressed (echo in follow-up window): {text!r}")
                        continue
                    logger.info(f"FOLLOW-UP: {text}")

                if just_barged_in:
                    if bare_barge_wakeword:
                        # A short barge-in capture can split "Atticus, stop"
                        # into separate ASR results. Stop TTS now, then keep the
                        # queued tail available as the follow-up command.
                        self._mark_turn_end(turn_satellite_id)
                        sat.skip_followup_self_echo = True
                        logger.info("Barge-in wakeword alone — awaiting follow-up")
                        continue
                    if self._is_pure_stop(text_lower):
                        # Pure "stop"/"cancel": the turn is already cancelled.
                        # Stand down to idle — no follow-up window, no spoken
                        # acknowledgement. A bare stop is an instruction to
                        # cease entirely, not a preface to a new request, so the
                        # wakeword is required again before the next turn.
                        # Reset last_turn_end so `_is_speaking()` (and thus
                        # barge-in eligibility) doesn't keep reading True off
                        # a stale future playback-end estimate now that the
                        # audio has been force-cancelled — same as
                        # `request_stop()`.
                        sat.last_turn_end = 0.0
                        self._clear_pending_conversation_turn(sat)
                        self.audio_capture.clear_follow_up(sat)
                        self._satellite_idle(sat)
                        logger.info("Barge-in (pure stop) — standing down")
                        continue
                    # A completed non-stop barge-in is the user's replacement
                    # request. Keep processing its transcript below rather than
                    # making them repeat it in a follow-up window.
                    logger.info("Barge-in — dispatching replacement request")

                # Pure stop in the follow-up window: cease, don't issue a new
                # command. Stand down silently and close the window (wakeword
                # required again). Active turns use the barge-in path above; a
                # fresh wakeword turn while idle still routes "stop" to the agent.
                if not wakeword_present and self._is_pure_stop(text_lower):
                    # Don't stand down on a partial — wait for the hard endpoint
                    # to confirm the user really only said "stop".
                    if provisional:
                        continue
                    sat.last_turn_end = 0.0
                    self._clear_pending_conversation_turn(sat)
                    self.audio_capture.clear_follow_up(sat)
                    self._satellite_idle(sat)
                    logger.info("Pure stop in follow-up — standing down")
                    continue

                if wakeword_present:
                    logger.debug(f"WAKEWORD: {text}")
                    user_prompt = (
                        text_lower[wakeword_match.end() :]
                        .strip(_PROMPT_STRIP_CHARS)
                        .replace('"', "")
                    )
                else:
                    # Follow-up: no wakeword required (within follow_up window).
                    # Strip a leading wakeword if the user habitually said it anyway.
                    stripped = text_lower.strip(_PROMPT_STRIP_CHARS)
                    leading_ww = self._wakeword_re.match(stripped)
                    if leading_ww:
                        stripped = stripped[leading_ww.end() :].lstrip(" ,.:;-")
                    user_prompt = stripped.replace('"', "")

                # Context-bias echo guard: a wakeword utterance whose command is
                # nothing but ASR context terms (e.g. "Hey Atticus, <term>" from
                # a cough) is the decoder echoing its bias prompt off non-speech.
                # Drop it. Wakeword-only (empty command) is handled below; the
                # follow-up path is exempt — a one-word context-term reply there
                # can be a legitimate answer.
                if wakeword_present and self._is_context_echo(user_prompt):
                    if not provisional:
                        self._reject_pending_satellite_wake(sat)
                    logger.debug(f"Suppressed (context-bias echo, likely non-speech): {text!r}")
                    continue

                # A soft endpoint is an in-progress transcript, not a turn
                # commit. Once it contains a credible wakeword, notify the
                # satellite immediately and keep the recorder buffer intact so
                # the hard endpoint can supply the complete command.
                onset_gap = self._asr_onset["t"] - sat.last_turn_end
                started_in_follow_up = (
                    self.follow_up_seconds > 0
                    and sat.last_turn_end > 0
                    and 0 <= onset_gap < self.follow_up_seconds
                ) or sat.conversation_mode
                if (
                    provisional
                    and wakeword_present
                    and not just_barged_in
                    and not started_in_follow_up
                    and not sat.protocol_wake_pending
                ):
                    self._begin_satellite_turn(sat)
                    self._set_pending_satellite_wake(sat)

                if not user_prompt:
                    # A provisional with no command yet (e.g. "Hey Atticus…"
                    # mid-pause) must not open the follow-up window early; let
                    # the hard endpoint decide.
                    if provisional:
                        continue
                    if wakeword_present:
                        # A bare wakeword is the ASR's favourite context-bias
                        # hallucination (a cough decoded as "hey atticus"). Gate
                        # it on loudness + an unbiased re-transcribe before
                        # opening the follow-up window; a rejected one is noise.
                        if not self._verify_bare_wakeword(loudness_db, baseline_db):
                            self._reject_pending_satellite_wake(sat)
                            continue
                        # Wakeword said alone (often to barge in and then
                        # pause before asking) — open the follow-up window
                        # from this moment so the next utterance is heard
                        # without re-saying the wakeword. Also arm the
                        # one-shot bypass so the user's question isn't
                        # suppressed as self-echo when it shares words
                        # with the response we just cancelled.
                        if sat.protocol_wake_pending:
                            self._clear_pending_satellite_wake(sat)
                        else:
                            self._begin_satellite_turn(sat)
                        self._mark_turn_end(turn_satellite_id)
                        sat.skip_followup_self_echo = True
                        logger.info("Wakeword alone — awaiting follow-up")
                    else:
                        logger.debug("Nothing in utterance")
                    continue

                # Early-commit gate: a provisional reaches here only as a genuine
                # wakeword/follow-up command. Commit the turn before the hard
                # endpoint only if the partial reads as a complete and
                # speculation-safe command (lights/time/etc.); otherwise drop it
                # and let the buffer grow to the hard endpoint, which re-runs
                # this path on the full utterance.
                if provisional:
                    if not should_commit_provisional(user_prompt, catchAll(user_prompt)):
                        logger.debug(f"Provisional held (awaiting hard endpoint): {user_prompt!r}")
                        continue
                    # Commit: drop the in-progress buffer and drain the queue so
                    # the hard endpoint's duplicate of this same utterance (still
                    # accumulating) is discarded, and briefly guard any in-flight
                    # result on the contaminated tail.
                    self.audio_capture.flush(turn_satellite_id)
                    sat.drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                    sat.provisional_committed_onset = self._asr_onset.get("t", 0.0)
                    sat.provisional_committed_text = re.sub(
                        r"[^\w\s]", "", user_prompt.lower()
                    ).strip()
                    sat.provisional_committed_at = time.monotonic()
                    logger.info(f"Committing early on soft endpoint: {user_prompt!r}")

                # Snapshot the ASR latency of the utterance that triggered this
                # turn before later (echo/cross-talk) transcriptions overwrite it.
                stt_seconds = getattr(self.asr_pipe, "last_transcribe_seconds", None)
                # A2: how long the utterance sat queued after the recorder
                # detected its endpoint, before ASR dequeued it — and whether
                # this turn committed on the 500ms soft endpoint or the full
                # 1.5s hard endpoint (see A0). Both snapshotted here for the
                # same reason as stt_seconds above.
                endpoint_wait_seconds = self._asr_endpoint_wait.get("s")
                endpoint_kind = "soft" if provisional else "hard"
                # A soft wake probe already established the turn ID while the
                # recorder kept listening. Without one, the hard endpoint is
                # the first authoritative wakeword-acceptance point.
                if wakeword_present and not in_follow_up:
                    if sat.protocol_wake_pending:
                        self._clear_pending_satellite_wake(sat)
                    else:
                        self._begin_satellite_turn(sat)
                elif in_follow_up:
                    self._begin_satellite_follow_up_turn(sat)
                if sat.conversation_mode:
                    self._satellite_thinking(sat)
                    self._queue_conversation_turn(
                        user_prompt,
                        turn_satellite_id,
                        stt_seconds=stt_seconds,
                        endpoint_wait_seconds=endpoint_wait_seconds,
                        endpoint_kind=endpoint_kind,
                    )
                elif self.barge_in == "off":
                    self._run_half_duplex(
                        user_prompt,
                        satellite_id=turn_satellite_id,
                        stt_seconds=stt_seconds,
                        endpoint_wait_seconds=endpoint_wait_seconds,
                        endpoint_kind=endpoint_kind,
                    )
                else:
                    self._start_turn(
                        user_prompt,
                        satellite_id=turn_satellite_id,
                        stt_seconds=stt_seconds,
                        endpoint_wait_seconds=endpoint_wait_seconds,
                        endpoint_kind=endpoint_kind,
                    )

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                self._note_runtime_error(e)

    def get_state(self) -> str:
        """Return the current assistant state: idle | thinking | speaking.

        Scans every satellite (plus the "dashboard-text" pseudo-session) —
        with only one turn normally active at a time (until Phase 3's
        arbiter), this reports the same thing the old single global flags
        did, just found by looking rather than read directly.
        """
        if any(s.tts_active.is_set() for s in self.satellites.values()):
            return "speaking"
        if any(s.turn_active or s.active_session is not None for s in self.satellites.values()):
            return "thinking"
        return "idle"

    def speak_proactive(
        self,
        text: str,
        alarm: bool = False,
        emit_event: bool = True,
        satellite_id: Optional[str] = None,
        follow_up: bool = False,
    ) -> None:
        """Speak `text` through the cloned voice without a user turn.

        Waits for any in-progress voice turn (SLM + TTS) to fully complete
        before speaking. `_turn_lock` covers only the SLM phase; TTS runs
        without it, so we also spin on every satellite's `turn_active`
        (barge-in worker) and `tts_active` (half-duplex TTS) to avoid two
        audio streams overlapping. Mutes every satellite's mic for the
        duration. Blocks until playback finishes.

        `alarm=True` (timer completions) plays the pre-rendered alert tone
        immediately before the spoken text. `satellite_id` targets one connected
        voice satellite; an explicit target never falls back to another device.
        `follow_up` is only for proactive questions that expect an immediate
        spoken response, such as a completed deep-think report prompt.
        """
        if not self.models_ready.wait(timeout=30):
            logger.warning("speak_proactive: models not ready")
            return
        deadline = time.monotonic() + 60.0
        while any(s.turn_active or s.tts_active.is_set() for s in self.satellites.values()):
            if time.monotonic() > deadline:
                logger.warning("speak_proactive: timed out waiting for active turn")
                break
            time.sleep(0.05)
        with self._turn_lock:
            if satellite_id is None:
                proactive_sats = [
                    sat for sat in self.satellites.values() if sat.tts_sink is not None
                ]
            else:
                proactive_sat = self.satellites.get(satellite_id)
                if proactive_sat is None or proactive_sat.tts_sink is None:
                    logger.info(
                        "speak_proactive: target satellite %s is no longer connected", satellite_id
                    )
                    return
                proactive_sats = [proactive_sat]
            for s in self.satellites.values():
                s.transcribing = False
            if len(proactive_sats) == 1:
                self._turn_local.sink = _safe_sink(proactive_sats[0].tts_sink)
                self._turn_local.tts_active_event = proactive_sats[0].tts_active
            elif proactive_sats:
                self._turn_local.sink = _BroadcastSink(
                    [_safe_sink(sat.tts_sink) for sat in proactive_sats]
                )
                self._turn_local.tts_active_event = _BroadcastEvent(
                    [sat.tts_active for sat in proactive_sats]
                )
            else:
                self._turn_local.sink = None
                self._turn_local.tts_active_event = None
            session = TtsSession()
            playback_end = None
            for sat in proactive_sats:
                sat.active_session = session
            proactive_lifecycle_started = False
            try:
                cleaned = clean_for_tts(text)
                if not cleaned:
                    return
                if proactive_sats:
                    # Proactive speech is not a conversational reply: establish
                    # a short lifecycle turn which explicitly returns to idle
                    # after its final PCM frame instead of opening follow-up.
                    for sat in proactive_sats:
                        sat.protocol_state_generation += 1
                        if sat.protocol_follow_up_timer is not None:
                            sat.protocol_follow_up_timer.cancel()
                            sat.protocol_follow_up_timer = None
                        self.audio_capture.clear_follow_up(sat)
                        sat.protocol_turn_id = uuid.uuid4().hex
                        self._emit_satellite_state(sat.id, "thinking", turn_id=sat.protocol_turn_id)
                    proactive_lifecycle_started = True
                if alarm:
                    self._play_alarm_tone(session)
                if emit_event:
                    self._emit_turn_event(
                        "assistant",
                        cleaned,
                        "proactive",
                        satellite_id=satellite_id,
                    )
                playback_end = self.speak_stream(cleaned, self.voice_clone_prompt, session=session)
            finally:
                for sat in proactive_sats:
                    sat.active_session = None
                    if proactive_lifecycle_started:
                        if follow_up:
                            self._mark_turn_end(sat.id, playback_end)
                        else:
                            self._emit_satellite_stand_down_after_tts(sat)
                            sat.protocol_turn_id = None
                for s in self.satellites.values():
                    s.transcribing = True
                self._turn_local.sink = None
                self._turn_local.tts_active_event = None

    def run(self):
        """Start the transcriber thread and block until Ctrl+C.

        Voice I/O is exclusively through the WebSocket satellite
        (`/ws/satellite`); the satellite recorder thread is started per
        session inside `connect_satellite`, so there's no per-process
        recorder thread here.
        """
        trans_thread = threading.Thread(target=self._transcriber_thread, daemon=True)
        trans_thread.start()

        logger.info("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.shutdown()
            trans_thread.join(timeout=2)
