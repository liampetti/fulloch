"""
Main assistant orchestration module.

Handles wakeword detection, intent processing, and response generation.
"""

import json
import logging
import random
import re
import threading
import time
from typing import Callable, Optional

# Sentinels returned by the thinking tools — kept as module-level
# constants so renames only happen in one place.
from utils.prompts import (
    CACHE_PRIMING_USER_PROMPT,
    get_agent_system_prompt,
    get_greeting_system_prompt,
    get_greeting_user_prompt,
    get_partial_thinking_summary_prompt,
    get_web_summary_system_prompt,
)

from .agent_loop import (
    _PROMPT_STRIP_CHARS,
    AgentLoop,
)
from .audio import AudioCapture
from .slm import ContextExhaustedError, generate_slm, load_slm
from .text_utils import clean_for_tts, split_sentences
from .tts_session import TtsSession, parse_barge_time
from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

# Strips every non-word character. The self-echo check uses this so ASR's
# "1254" / "am" still match the assistant's spoken "12 54" / "a m" — the
# get_time tool spells digits out with spaces and ASR re-concatenates them.
_NON_WORD_RE = re.compile(r"\W+")

# Short tokens that Qwen3-ASR (and Whisper) commonly hallucinate from
# background noise when English is enforced. None of these can match a real
# wakeword, so dropping them globally is safe. Checked against the
# punctuation-stripped lowercase transcription.
_ASR_NOISE_TOKENS: frozenset[str] = frozenset({
    "yeah", "yep", "yup",
    "okay", "ok",
    "sure",
    "hmm", "hm", "uh", "um", "uh huh",
    "thanks", "thank you", "thank you very much",
    "you're welcome", "welcome",
    "bye", "goodbye", "see you", "see ya",
    "alright", "all right",
    "right",
    "oh", "so",
    "cough",
    # ASR prompt-echo: on non-speech (a cough), the context-biased decoder
    # regurgitates its context label instead of transcribing. Drop it before
    # routing so it can't open a spurious follow-up window. See
    # `asr_context_hint` / asr.py:context.
    "technical terms"
})

# Bare stop commands that interrupt an active turn without any wakeword.
# Kept separate from _barge_re so the idle wakeword detector stays strict.
# These fire in _check_barge_in only (already gated by _turn_active), so
# the false-positive risk is low; _is_self_echo still runs first to suppress
# "stop" spoken by the assistant's own TTS.
_BARGE_STOP_RE = re.compile(
    r"\b(?:stop|halt|cancel|pause|quiet|enough)\b",
    re.IGNORECASE,
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
    ACK_PHRASES,
    GREETING_TOPICS,
    NOTE_WRITE_STALL_PHRASES,
    PRE_THINKING_STALL_PHRASES,
    REMINDER_PREFIX_PHRASES,
    REPLAN_STALL_PHRASES,
    THINKING_STALL_PHRASES,
    WEB_SEARCH_STALL_PHRASES,
)

# TTL on the partial-thinking capture. Past this, a 'summarise your
# thoughts' request returns a graceful 'no recent thoughts' message
# rather than dredging up a stale reasoning trace.
THINKING_PARTIAL_TTL_S = 60.0

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
CHAT_SESSION_TIMEOUT_S = 600.0

# Spoken only when a turn overflows the context window AND even the recent
# history floor won't fit (a single oversized message) — the rare case where we
# do clear. Normal overflow now sheds the oldest history and retries silently.
CONTEXT_EXHAUSTED_REPLY = (
    "Sorry, that was too much for me to hold in mind. "
    "I've cleared our conversation — what would you like to do?"
)

# Time window after a barge-in cancel during which incoming ASR results
# are discarded. ASR inference is ~200ms on this GPU, so 500ms reliably
# catches anything already in flight at the moment of cancel; a real
# user reply takes ≥1.5s (silence detection + min utterance) to be
# enqueued by the recorder, so it lands well clear of this window.
DROP_AFTER_CANCEL_S = 0.5


class Assistant:
    """Owns the audio capture and runs the wakeword → intent → TTS loop."""

    def __init__(
        self,
        wakeword: str,
        wakeword_pattern: Optional[str] = None,
        voice_clone: Optional[str] = None,
        barge_in: str = "off",
        follow_up_time: str = "0s",
        input_device: Optional[str] = None,
        output_device: Optional[str] = None,
        asr_language: Optional[str] = None,
        asr_context_hint: bool = True,
        asr_context_terms: list = None,
        use_vad: bool = True,
        vad_threshold: Optional[float] = None,
        vad_endpoint_silence_ms: Optional[int] = None,
        vad_min_speech_ms: Optional[int] = None,
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
            barge_in: "off" (half-duplex) or "wakeword" (interrupt with wakeword)
            follow_up_time: "0s" or "<N>s" — window after TTS ends during
                which a wakeword-free utterance counts as a follow-up turn.
            input_device: PortAudio/PulseAudio source name for the mic (None =
                system default). Set to "echocancel_source" to route through
                PulseAudio's module-echo-cancel.
            output_device: PortAudio/PulseAudio sink name for TTS playback
                (None = system default). Set to "echocancel_sink" to pair with
                module-echo-cancel so the assistant's voice becomes the AEC
                reference signal.
        """
        self.wakeword = wakeword.lower()
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
        self.asr_language = asr_language
        self.asr_context_hint = asr_context_hint
        self.asr_context_terms = asr_context_terms or []
        self.use_vad = use_vad
        # Built in `_load_models()` once the TTS model is in memory.
        self.voice_clone_prompt = None
        self.output_device = output_device
        self.audio_capture = AudioCapture(
            input_device=input_device,
            use_vad=use_vad,
            vad_threshold=vad_threshold,
            vad_endpoint_silence_ms=vad_endpoint_silence_ms,
            vad_min_speech_ms=vad_min_speech_ms,
        )
        # Mic stays muted through model load and the opening greeting;
        # `_warm_and_announce`'s `finally` flips it on once warmup ends.
        self.audio_capture.transcribing = False
        self.tts_session = TtsSession()

        if barge_in not in ("off", "wakeword"):
            logger.warning(f"Invalid barge_in: {barge_in!r}; defaulting to 'off'")
            barge_in = "off"
        self.barge_in = barge_in
        self.follow_up_seconds = parse_barge_time(follow_up_time)
        self.tts_start_time = 0.0
        self._last_turn_end = 0.0
        # Written by the ASR stream generator with the speech-onset time of
        # the buffer currently being transcribed. The follow-up window
        # measures against this (when the user started talking) rather than
        # when the transcription lands, so a long reply isn't penalised.
        self._asr_onset: dict = {'t': 0.0}

        # Turn state (only meaningful in barge-in mode). A "turn" runs the SLM
        # intent/chat path + stall + final TTS in a worker thread; the
        # transcriber loop can cancel it via _cancel_turn().
        self._turn_active = False
        self._turn_session: Optional[TtsSession] = None
        self._turn_thread: Optional[threading.Thread] = None
        # Monotonic deadline: ASR results returned before this are dropped.
        # Set briefly after a barge-in so any inference already on the GPU
        # at the moment of cancel doesn't get treated as a fresh turn. The
        # window is short (see DROP_AFTER_CANCEL_S) so a user reply spoken
        # ~1s+ after the cancel isn't accidentally swallowed.
        self._drop_results_until: float = 0.0

        # Lower-cased text the assistant last spoke. Self-echo suppression
        # in `_check_barge_in` compares incoming transcripts against it so
        # AEC residue containing the wakeword (or near-homophones) doesn't
        # interrupt mid-answer. Updated just before each `speak_stream`.
        self._last_spoken_text: str = ""

        # Unified conversation memory — every user / assistant / tool entry
        # for the agent's view of the session. Mutated only from the active
        # turn (one at a time, under `_turn_lock`) so no lock is needed.
        # Each entry: {"role": "user"|"assistant"|"tool", "content": str,
        #              "name": <intent>?}  (assistant content is raw agent
        # JSON; tool content is the tool's return string).
        self._history: list = []

        # One-shot flag set when the user said the wakeword alone (often a
        # barge-in followed by "OK, now I'll ask my actual question"). The
        # follow-up branch consumes it to skip the self-echo check — the
        # user just explicitly opened a turn, and their question will often
        # share topic words with the just-cancelled response.
        self._skip_followup_self_echo = False

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

        # Models loaded lazily in transcriber thread
        self.asr_pipe = None
        self.asr_stream_generator = None
        self.slm_model = None
        self.grammar = None
        self.greeting_prompt = None
        self.web_summary_prompt = None

        # Serializes SLM access. Voice turns run on the transcriber or barge-in
        # worker; text turns from the dashboard run on the FastAPI thread —
        # llama-cpp-python isn't thread-safe, so all `_handle_wakeword` callers
        # acquire this. Voice barge-in cancels mid-stream so the lock releases
        # quickly even while a text turn waits behind it.
        self._turn_lock = threading.Lock()

        # Dashboard / external observers subscribe here to receive every
        # finalised user/assistant exchange. Listeners must be cheap (they
        # run on the calling thread) and exceptions are swallowed.
        self._turn_listeners: list = []
        self._turn_listeners_lock = threading.Lock()

        # Set once `_load_models()` finishes — text turns from the dashboard
        # wait on this before calling into the pipeline.
        self.models_ready = threading.Event()

    def _load_models(self):
        """Load ASR, TTS and SLM models."""
        from .asr import load_asr_model, stream_generator
        logger.info("Using Qwen ASR")

        self.asr_pipe = load_asr_model(language=self.asr_language)
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
        self.asr_stream_generator = stream_generator

        from .tts import (
            play_chunks,
            set_output_device,
            set_tts_active_event,
            set_voice,
            speak_stream,
            synthesize,
            warmup_model,
        )
        set_output_device(self.output_device)
        # The recorder uses this to switch to a stricter silence threshold
        # while we're talking, so AEC residue doesn't keep utterances alive.
        set_tts_active_event(self.audio_capture.tts_active)
        logger.info(f"Using Qwen TTS Base with voice clone: {self.voice_clone}")
        self.voice_clone_prompt = set_voice(self.voice_clone)
        warmup_model(self.voice_clone_prompt)

        self.speak_stream = speak_stream
        self.synthesize = synthesize
        self.play_chunks = play_chunks
        self.web_search_stall_cache: list = []
        self.note_write_stall_cache: list = []
        self.pre_thinking_stall_cache: list = []
        self.thinking_stall_cache: list = []
        self.ack_cache: list = []
        self.replan_stall_cache: list = []

        self.grammar, self.slm_model = load_slm()
        self.greeting_prompt = get_greeting_system_prompt()
        self.web_summary_prompt = get_web_summary_system_prompt()

        self.models_ready.set()
        self._warm_and_announce()
        self._start_reminder_poll()
        try:
            from tools.time_tools import set_speak_callback
            set_speak_callback(self.speak_proactive)
        except ImportError:
            pass

    def _warm_and_announce(self):
        """Prime every cache, then speak the opening greeting.

        Order matters for the user experience: all silent warmup work
        (agent KV-cache prime, every TTS pre-render, the notes index)
        runs *before* the greeting, so the greeting becomes the last
        audible event of startup. `audio_capture.transcribing` is
        re-armed in the `finally` immediately after greeting playback
        ends, meaning the mic comes alive precisely when the speaker
        goes quiet — no awkward silent gap between "online and ready"
        and the assistant actually listening.
        """
        self.audio_capture.transcribing = False
        try:
            # Prime the agent prompt's KV-cache so the first real user
            # request reuses the prefilled prefix.
            logger.info("Priming agent cache...")
            generate_slm(
                self.slm_model,
                user_prompt=CACHE_PRIMING_USER_PROMPT,
                grammar=self.grammar,
                system_prompt=get_agent_system_prompt(),
            )

            # Pre-render the context-specific stall phrases so each plays
            # instantly and in the cloned voice (synthesizing them live each
            # turn made them too short for the clone to lock in).
            logger.info("Caching web-search stall phrases...")
            self.web_search_stall_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in WEB_SEARCH_STALL_PHRASES
            ]
            logger.info("Caching note-write stall phrases...")
            self.note_write_stall_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in NOTE_WRITE_STALL_PHRASES
            ]
            logger.info("Caching pre-thinking stall phrases...")
            self.pre_thinking_stall_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in PRE_THINKING_STALL_PHRASES
            ]
            logger.info("Caching thinking stall phrases...")
            self.thinking_stall_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in THINKING_STALL_PHRASES
            ]
            logger.info("Caching ack phrases...")
            self.ack_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in ACK_PHRASES
            ]
            logger.info("Caching replan stall phrases...")
            self.replan_stall_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in REPLAN_STALL_PHRASES
            ]

            # Pre-load the BGE embedding model + restore the persisted notes
            # index so the first semantic-search query doesn't pay the
            # SentenceTransformer load + initial scan latency.
            try:
                from tools.notes import warm_index
                logger.info("Warming notes index...")
                if warm_index():
                    logger.info("Notes index ready.")
            except Exception:
                logger.exception("Notes index warmup failed; semantic search will load lazily")

            # Greeting comes last — last audible event of startup.
            topic = random.choice(GREETING_TOPICS)
            logger.info(f"Generating opening greeting (topic: {topic})...")
            greeting = generate_slm(
                self.slm_model,
                user_prompt=get_greeting_user_prompt(topic),
                system_prompt=self.greeting_prompt,
                temperature=1.0,
            )
            cleaned = clean_for_tts(greeting)

            # Splitting the greeting into individual sentences and
            # synthesising each as its own worker job populates the
            # reduce-overhead CUDA-graph cache with extra prefill shapes.
            sentences = split_sentences(cleaned) or [cleaned]
            logger.info(f"Synthesising greeting ({len(sentences)} sentence(s))...")
            greeting_parts = [
                self.synthesize(s, self.voice_clone_prompt) for s in sentences
            ]
            for chunks, sr in greeting_parts:
                if chunks:
                    self.play_chunks(chunks, sr, session=self.tts_session)

            logger.info("Warmup complete")
        finally:
            self.audio_capture.transcribing = True

    def _reminder_poll_loop(self) -> None:
        """Daemon thread: speak upcoming Fulloch calendar events via speak_proactive.

        Polls every 60 seconds. Fires speak_proactive for any event on the
        configured reminder calendar that starts within the next 90 seconds and
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

                for event in get_upcoming_events(window_seconds=90):
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
        """Clear unified history if the silence gap exceeds the session timeout."""
        if self._last_turn_end == 0.0:
            return
        if not self._history:
            return
        if time.monotonic() - self._last_turn_end > CHAT_SESSION_TIMEOUT_S:
            logger.info(
                f"Session timeout ({CHAT_SESSION_TIMEOUT_S:.0f}s) — "
                f"clearing {len(self._history)} history entries"
            )
            self._history.clear()
            self._last_thinking_partial = None
            self._last_thinking_question = None
            self._skip_followup_self_echo = False

    def _trim_history(self) -> None:
        """Cap `_history` at HISTORY_MAX_MESSAGES with FIFO eviction."""
        if len(self._history) > HISTORY_MAX_MESSAGES:
            self._history = self._history[-HISTORY_MAX_MESSAGES:]

    def _compact_completed_turns(self) -> None:
        """Strip tool results and bare planning emissions from `_history`.

        Called at the start of each turn, when every tool / `{"actions": ...}`
        entry belongs to an already-completed turn. Those are in-turn scaffolding
        — the agent needs a tool result only while composing that turn's reply,
        and the reply itself is recorded as a `{"reply": ...}` assistant entry.
        Keeping the raw dumps afterwards just bloats the context (a single note
        read or search payload can be many KB) and evicts the real conversation;
        a follow-up re-fetches what it needs (e.g. web searches re-research)
        rather than relying on a stale copy. The result is history that's just
        the user's turns and Fulloch's replies.
        """
        kept = []
        for msg in self._history:
            role = msg.get("role")
            if role == "tool":
                continue
            if role == "assistant":
                try:
                    emission = json.loads(msg.get("content") or "")
                except Exception:
                    emission = None
                # Drop a bare action-emission (planning scaffolding); keep
                # replies (what Fulloch actually said) and anything unparseable.
                if (isinstance(emission, dict)
                        and "reply" not in emission and "actions" in emission):
                    continue
            kept.append(msg)
        if len(kept) != len(self._history):
            logger.debug(
                "Compacted history: dropped %d tool/scaffolding entries",
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
        while (
            len(self._history) > CONTEXT_TRIM_KEEP_MIN
            and self._history[0].get("role") != "user"
        ):
            del self._history[0]
        logger.info(
            "Context overflow: shed %d oldest history entries, %d remain",
            drop, len(self._history),
        )
        return True

    def _generate_with_context_recovery(self, **kwargs) -> str:
        """`generate_slm` wrapper that degrades gracefully on context overflow.

        On `ContextExhaustedError`, sheds the oldest history (preserving the
        in-flight turn) and retries, so a long conversation loses its tail end
        instead of being wiped. Re-raises only when even the recent floor won't
        fit — the caller then apologises + clears as a last resort.
        """
        while True:
            try:
                return generate_slm(self.slm_model, **kwargs)
            except ContextExhaustedError:
                if not self._shed_oldest_history():
                    raise

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

    def _emit_turn_event(
        self, role: str, content: str, source: str,
        stats: Optional[TurnStats] = None,
    ) -> float:
        ts = time.time()
        event = {
            "role": role,
            "content": content,
            "ts": ts,
            "source": source,
        }
        if stats is not None:
            event["stats"] = stats.to_payload()
        self._dispatch_event(event)
        return ts

    def _emit_stats_patch(self, ref_ts: float, source: str, patch: dict) -> None:
        """Patch a stat onto an already-emitted assistant message (keyed by its
        ts). Used for TTS time, which is only known after playback starts."""
        self._dispatch_event({
            "role": "stats",
            "ts": time.time(),
            "ref_ts": ref_ts,
            "source": source,
            "patch": patch,
        })

    def _emit_agent_event(self, kind: str, payload: dict, source: str = "voice") -> None:
        """Emit a `plan` / `step` / `observation` event to listeners.

        kind:
          - "plan"        — payload = {"actions": [{...}]}  (or {"reply": "..."})
          - "step"        — payload = {"intent": str, "args": list}
          - "observation" — payload = {"intent": str, "result": str}
        """
        event = {
            "role": "agent",
            "kind": kind,
            "ts": time.time(),
            "source": source,
            "payload": payload,
        }
        self._dispatch_event(event)

    def _dispatch_event(self, event: dict) -> None:
        with self._turn_listeners_lock:
            listeners = list(self._turn_listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Turn listener failed: {e}")

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
            prompt = prompt[leading.end():].lstrip(" ,.:;-")
            if not prompt:
                return ""

        self._emit_turn_event("user", prompt, "text")
        try:
            self._maybe_reset_session()
            stats = TurnStats()  # text turns have no STT/TTS
            answer = self._handle_wakeword(prompt, source="text", stats=stats)
            cleaned = clean_for_tts(answer)
            self._emit_turn_event("assistant", cleaned, "text", stats=stats)
            return cleaned
        except Exception as e:
            logger.error(f"Text turn error: {e}")
            err = "Sorry, something went wrong handling that."
            self._emit_turn_event("assistant", err, "text")
            return err

    def _record_spoken(self, spoken: str) -> None:
        """Replace the most recent assistant entry with one whose `reply` is
        the spoken text the user actually heard. Called after a successful
        dispatch turn where the spoken answer is joined tool outputs (not
        the agent's raw JSON). Keeps `_history` representative of what the
        user experienced for downstream SLM calls.
        """
        if not spoken:
            return
        self._history.append({
            "role": "assistant",
            "content": json.dumps({"reply": spoken}),
        })
        self._trim_history()

    def _play_random_ack(self, session: Optional[TtsSession] = None, cache: Optional[list] = None) -> None:
        """Play one pre-rendered ack chunk. Safe no-op if the cache is empty
        (e.g. an early failure before `_warm_and_announce` populated it).
        Pass `cache` to use an alternate phrase pool (e.g. replan_stall_cache).
        """
        c = cache if cache is not None else self.ack_cache
        if not c:
            return
        chunks, sr = random.choice(c)
        try:
            self.play_chunks(chunks, sr, session=session or self.tts_session)
        except Exception:
            logger.exception("ack playback failed")

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

        Serialised under `_turn_lock` — llama-cpp-python isn't thread-safe
        and dashboard text turns share this method with voice turns.
        """
        with self._turn_lock:
            return AgentLoop(
                self,
                session=session,
                source=source,
                stats=stats,
                on_slm_start=on_slm_start,
            ).run(user_prompt)

    def _run_half_duplex(self, user_prompt: str, stt_seconds=None) -> None:
        """Synchronous handle + speak, with mic muted during TTS playback."""
        # A real turn is starting — close the prior follow-up window; it's
        # re-armed at the end of this method via `_mark_turn_end`.
        self.audio_capture.clear_follow_up()
        self._maybe_reset_session()
        self._emit_turn_event("user", user_prompt, "voice")
        stats = TurnStats(stt_seconds=stt_seconds)

        ack_thread: list = []

        def _start_ack() -> None:
            # Mute the recorder before the ack plays so it doesn't re-enter
            # the ASR queue. Half-duplex relies on this flag, not on AEC.
            self.audio_capture.transcribing = False
            t = threading.Thread(
                target=self._play_random_ack,
                args=(self.tts_session,),
                daemon=True,
            )
            t.start()
            ack_thread.append(t)

        answer = self._handle_wakeword(
            user_prompt, source="voice", on_slm_start=_start_ack, stats=stats
        )
        cleaned = clean_for_tts(answer)
        self.audio_capture.transcribing = False
        try:
            if ack_thread:
                ack_thread[0].join()
            # Emit before speaking so the dashboard can reveal the text while
            # TTS plays, not only after playback finishes.
            ts = self._emit_turn_event("assistant", cleaned, "voice", stats=stats)
            # Emit the TTS stat as soon as the first audio chunk is ready, so the
            # panel doesn't wait for the whole reply to finish playing.
            self.speak_stream(
                cleaned, self.voice_clone_prompt, session=self.tts_session, stats=stats,
                on_first_audio=lambda: self._emit_stats_patch(
                    ts, "voice",
                    {"tts": stats.tts_payload(), "total": stats.total_with_tts()},
                ),
            )
        finally:
            self.audio_capture.transcribing = True
        self._mark_turn_end()

    def _start_turn(self, user_prompt: str, stt_seconds=None) -> None:
        """Kick off a barge-in turn (handle + speak) on a worker thread."""
        # A real turn is starting — close the prior follow-up window; the new
        # turn re-arms it at its end via `_mark_turn_end`.
        self.audio_capture.clear_follow_up()
        # Wait briefly for any prior turn to wind down so its session can't
        # leak forward and abort the new turn.
        if self._turn_thread is not None and self._turn_thread.is_alive():
            self._turn_thread.join(timeout=2.0)
        session = TtsSession()
        self._turn_session = session
        self._turn_active = True
        self._turn_thread = threading.Thread(
            target=self._run_turn,
            args=(user_prompt, session, stt_seconds),
            daemon=True,
        )
        self._turn_thread.start()

    def _run_turn(self, user_prompt: str, session: TtsSession, stt_seconds=None) -> None:
        """Worker-thread body: handle the prompt, then speak the answer."""
        try:
            self._maybe_reset_session()
            self._emit_turn_event("user", user_prompt, "voice")
            stats = TurnStats(stt_seconds=stt_seconds)

            ack_thread: list = []

            def _start_ack() -> None:
                # Barge-in mode: recorder stays live; ack shares the turn's
                # session so a user interruption stops it mid-utterance.
                t = threading.Thread(
                    target=self._play_random_ack,
                    args=(session,),
                    daemon=True,
                )
                t.start()
                ack_thread.append(t)

            answer = self._handle_wakeword(
                user_prompt, session=session, source="voice",
                on_slm_start=_start_ack, stats=stats,
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
            # Recorded for self-echo suppression in `_check_barge_in`.
            self._last_spoken_text = cleaned.lower()
            self.tts_start_time = time.monotonic()
            # Emit before speaking so the dashboard reveals the text while TTS
            # plays. Barge-in may cut speech short; the full text still shows,
            # while _history is reconciled to the spoken text via _record_spoken.
            ts = self._emit_turn_event("assistant", cleaned, "voice", stats=stats)
            # Emit the TTS stat as soon as the first audio chunk is ready, so the
            # panel doesn't wait for the whole reply to finish playing.
            self.speak_stream(
                cleaned, self.voice_clone_prompt, session=session, stats=stats,
                on_first_audio=lambda: self._emit_stats_patch(
                    ts, "voice",
                    {"tts": stats.tts_payload(), "total": stats.total_with_tts()},
                ),
            )
            if not session.cancelled:
                # Mic stays live during barge-in TTS, so the queue contains
                # audio of our own voice captured while we were speaking.
                # Flush it before opening the follow-up window — same treatment
                # as a barge-in cancel — otherwise the tail chunk arrives with
                # onset ≈ _last_turn_end (speech_onset_t was never set because
                # AEC kept the residue below TTS_SILENCE_THRESHOLD) and falls
                # inside the follow-up window as a spurious new turn.
                self.audio_capture.flush()
                self._drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                self._mark_turn_end()
        except Exception as e:
            logger.error(f"Turn error: {e}")
        finally:
            self._turn_active = False

    def _cancel_turn(self) -> None:
        """Signal the active turn to abort and wait briefly for it to wind down."""
        if self._turn_session is not None:
            self._turn_session.stop()
        if self._turn_thread is not None:
            self._turn_thread.join(timeout=2.0)

    def _mark_turn_end(self) -> None:
        """Record the turn-end time and open the wakeword-free follow-up window.

        The recorder reads the armed window to accept short replies (which the
        full min-utterance length would otherwise drop); the transcriber gauges
        the window itself from `_last_turn_end` against each utterance's onset.
        """
        self._last_turn_end = time.monotonic()
        if self.follow_up_seconds > 0:
            self.audio_capture.arm_follow_up(self.follow_up_seconds)

    def _is_self_echo(self, text_lower: str, short_only: bool = False) -> bool:
        """True if `text_lower` looks like AEC residue of the assistant's voice.

        AEC suppresses the speaker-into-mic energy but isn't a phoneme filter,
        so ASR can still re-transcribe parts of the assistant's own answer —
        and if the answer happened to contain the wakeword (or a homophone),
        the assistant would interrupt itself. We sidestep that by checking
        whether the incoming utterance is mostly drawn from the words we just
        spoke. Real user speech rarely overlaps the assistant's prior answer
        by more than half.
        """
        spoken = self._last_spoken_text
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
            return (
                words[0] in spoken_norm
                or self._wakeword_re.search(spoken) is not None
            )
        # AEC residue always starts from a fragment the assistant actually spoke.
        # If the first word isn't in the spoken text at all, this is user speech
        # anchored by their own word (e.g. "yes, save a recipe..." after the
        # assistant offered to save one) — not an echo.
        if words[0] not in spoken_norm:
            return False
        overlap = sum(1 for w in words if w in spoken_norm) / len(words)
        return overlap >= 0.6

    def _check_barge_in(self, text_lower: str) -> bool:
        """Return True if `text_lower` should interrupt the active turn."""
        if not self._turn_active or self.barge_in == "off":
            return False
        if self._is_self_echo(text_lower):
            return False
        return (
            self._barge_re.search(text_lower) is not None
            or _BARGE_STOP_RE.search(text_lower) is not None
        )

    def _transcriber_thread(self):
        """ASR results → wakeword/intent/TTS. Owns model loading on first call."""
        self._load_models()
        logger.info("Transcriber started")

        for result in self.asr_pipe(
            self.asr_stream_generator(
                self.audio_capture.audio_queue, self._asr_onset
            ),
            batch_size=1,
            generate_kwargs={"max_new_tokens": 256}
        ):
            try:
                text = result.get("text", "").strip()
                if not text:
                    continue

                if time.monotonic() < self._drop_results_until:
                    logger.debug(f"Dropping post-cancel ASR result: {text}")
                    continue

                # Drop known ASR noise hallucinations before any routing.
                # Enforcing English causes taps and ambient sounds to surface as
                # short filler tokens ("Yeah.", "Okay.") that would otherwise
                # trigger the follow-up window. Strip punctuation before matching
                # so "Yeah." and "yeah" both hit.
                _text_bare = re.sub(r"[^\w\s]", "", text.lower()).strip()
                if _text_bare in _ASR_NOISE_TOKENS:
                    logger.debug(f"Dropped noise token: {text!r}")
                    continue

                logger.debug(f"Transcribed: {text}")

                text_lower = text.lower()
                wakeword_match = self._wakeword_re.search(text_lower)
                wakeword_present = wakeword_match is not None
                just_barged_in = False

                if self._turn_active:
                    # Mid-turn transcriptions are only acted on as potential
                    # barge-ins; otherwise they're echo of our own voice (or
                    # cross-talk during SLM thinking) and must be ignored.
                    if not self._check_barge_in(text_lower):
                        logger.debug(f"Suppressed (mid-turn, no barge-in match): {text!r}")
                        continue

                    logger.info(f"BARGE-IN: {text}")
                    self._cancel_turn()
                    # Buffer contains TTS-bleed accumulated during the
                    # cancelled turn; flush so the next user utterance is
                    # captured cleanly. Then open a short drop window so
                    # any ASR inference still finishing on contaminated
                    # audio gets discarded — without swallowing the user's
                    # real reply, which can't arrive for ≥1.5s.
                    self.audio_capture.flush()
                    self._drop_results_until = (
                        time.monotonic() + DROP_AFTER_CANCEL_S
                    )
                    just_barged_in = True
                elif not wakeword_present:
                    # Follow-up window: a brief grace period after TTS ends
                    # during which wakeword-free input is treated as a reply
                    # to the assistant's last answer. Measured from when this
                    # utterance's *speech began* (`_asr_onset`), not now — so
                    # the window means "started replying within N seconds",
                    # and a long answer isn't disqualified by its own length
                    # plus the silence-detection tail and ASR latency.
                    onset_gap = self._asr_onset['t'] - self._last_turn_end
                    in_follow_up = (
                        self.follow_up_seconds > 0
                        and self._last_turn_end > 0
                        and 0 <= onset_gap < self.follow_up_seconds
                    )
                    if not in_follow_up:
                        logger.debug(
                            f"Suppressed (no wakeword, outside follow-up window "
                            f"[gap={onset_gap:.1f}s, window={self.follow_up_seconds:.0f}s]): {text!r}"
                        )
                        continue
                    if self._skip_followup_self_echo:
                        # One-shot bypass after a wakeword-alone barge-in:
                        # this utterance is the user's deliberate question,
                        # even if every word also appears in the cancelled
                        # response. Consume the flag here.
                        self._skip_followup_self_echo = False
                    elif self._is_self_echo(text_lower, short_only=True):
                        # Suppress brief AEC reverb after TTS ends (typically
                        # 1–3 word fragments). Longer utterances are user
                        # speech — even if the words overlap the assistant's
                        # reply (e.g. user picks an option Atticus just offered).
                        logger.debug(f"Suppressed (echo in follow-up window): {text!r}")
                        continue
                    logger.info(f"FOLLOW-UP: {text}")

                if just_barged_in:
                    # Barge-in: TTS has stopped. Open the follow-up window so
                    # the user can speak their full query without re-saying the
                    # wakeword. Discard the partial chunk that triggered the
                    # barge-in — silence detection may have cut the sentence
                    # short, so the content here is unreliable.
                    self._mark_turn_end()
                    self._skip_followup_self_echo = True
                    logger.info("Barge-in — follow-up window open")
                    continue

                if wakeword_present:
                    logger.debug(f"WAKEWORD: {text}")
                    user_prompt = text_lower[wakeword_match.end():].strip(_PROMPT_STRIP_CHARS).replace('"', '')
                else:
                    # Follow-up: no wakeword required (within follow_up window).
                    # Strip a leading wakeword if the user habitually said it anyway.
                    stripped = text_lower.strip(_PROMPT_STRIP_CHARS)
                    leading_ww = self._wakeword_re.match(stripped)
                    if leading_ww:
                        stripped = stripped[leading_ww.end():].lstrip(" ,.:;-")
                    user_prompt = stripped.replace('"', '')

                if not user_prompt:
                    if wakeword_present:
                        # Wakeword said alone (often to barge in and then
                        # pause before asking) — open the follow-up window
                        # from this moment so the next utterance is heard
                        # without re-saying the wakeword. Also arm the
                        # one-shot bypass so the user's question isn't
                        # suppressed as self-echo when it shares words
                        # with the response we just cancelled.
                        self._mark_turn_end()
                        self._skip_followup_self_echo = True
                        logger.info("Wakeword alone — awaiting follow-up")
                    else:
                        logger.debug("Nothing in utterance")
                    continue

                # Snapshot the STT latency of the utterance that triggered this
                # turn before later (echo/cross-talk) transcriptions overwrite it.
                stt_seconds = getattr(self.asr_pipe, "last_transcribe_seconds", None)
                if self.barge_in == "off":
                    self._run_half_duplex(user_prompt, stt_seconds=stt_seconds)
                else:
                    self._start_turn(user_prompt, stt_seconds=stt_seconds)

            except Exception as e:
                logger.error(f"Transcription error: {e}")

    def get_state(self) -> str:
        """Return the current assistant state: idle | thinking | speaking."""
        if self.audio_capture.tts_active.is_set():
            return "speaking"
        if self._turn_active:
            return "thinking"
        return "idle"

    def speak_proactive(self, text: str) -> None:
        """Speak `text` through the cloned voice without a user turn.

        Waits for any in-progress voice turn (SLM + TTS) to fully complete
        before speaking. `_turn_lock` covers only the SLM phase; TTS runs
        without it, so we also spin on `_turn_active` (barge-in worker) and
        `tts_active` (half-duplex TTS) to avoid two audio streams overlapping.
        Mutes the mic for the duration. Blocks until playback finishes.
        """
        if not self.models_ready.wait(timeout=30):
            logger.warning("speak_proactive: models not ready")
            return
        deadline = time.monotonic() + 60.0
        while self._turn_active or self.audio_capture.tts_active.is_set():
            if time.monotonic() > deadline:
                logger.warning("speak_proactive: timed out waiting for active turn")
                break
            time.sleep(0.05)
        with self._turn_lock:
            self.audio_capture.transcribing = False
            try:
                session = TtsSession()
                cleaned = clean_for_tts(text)
                if not cleaned:
                    return
                self._emit_turn_event("assistant", cleaned, "proactive")
                self.speak_stream(cleaned, self.voice_clone_prompt, session=session)
            finally:
                self.audio_capture.transcribing = True
            self._mark_turn_end()

    def run(self):
        """Start the recorder + transcriber threads and block until Ctrl+C."""
        rec_thread = threading.Thread(
            target=self.audio_capture.recorder_thread,
            daemon=True
        )
        trans_thread = threading.Thread(
            target=self._transcriber_thread,
            daemon=True
        )

        rec_thread.start()
        trans_thread.start()

        logger.info("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.audio_capture.stop()
            rec_thread.join(timeout=2)
            trans_thread.join(timeout=2)
