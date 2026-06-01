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

from .audio import AudioCapture
from .slm import load_slm, generate_slm
from .text_utils import clean_for_tts, split_sentences
from .thinking_watchdog import ThinkingWatchdog
from .tts_session import TtsSession, parse_barge_time
from .turn_stats import TurnStats

from utils.prompts import (
    CACHE_PRIMING_USER_PROMPT,
    get_agent_system_prompt,
    get_greeting_system_prompt,
    get_greeting_user_prompt,
    get_partial_thinking_summary_prompt,
    get_thinking_system_prompt,
    get_web_summary_system_prompt,
)
from utils.intent_catch import catchAll
import utils.intents as intents
from utils.intents import MAX_AGENT_CALLS_PER_TURN

# Sentinels returned by the thinking tools — kept as module-level
# constants so renames only happen in one place.
from tools.thinking import THINKING_PREFIX, SUMMARY_PREFIX
import tools.notes as notes

logger = logging.getLogger(__name__)

# Tool intents that count as "context retrieval" for the stats panel. The
# chunk count is surfaced by the semantic paths via notes.last_retrieval.
NOTE_SEARCH_INTENTS = frozenset(
    {"search_notes", "search_notes_semantic", "read_note"}
)

# Strips every non-word character. The self-echo check uses this so ASR's
# "1254" / "am" still match the assistant's spoken "12 54" / "a m" — the
# get_time tool spells digits out with spaces and ASR re-concatenates them.
_NON_WORD_RE = re.compile(r"\W+")

# Leading/trailing punctuation peeled off the user prompt after the wakeword
# is stripped. Includes "!"/"?" so ASR's "Hey Frasier! Stop." yields the bare
# "stop" without trailing punctuation confusing intent matching.
_PROMPT_STRIP_CHARS = " ,.!?;:"

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
    "oh",
})


def _build_wakeword_pattern(wakeword: str) -> str:
    """Compile a tolerant wakeword matcher pattern.

    Tolerates two ASR foibles:
      - Punctuation / whitespace runs between tokens ("Hey, Fraser") via
        a `\\W+` join between words.
      - The s↔z swap that names like "Fraser" / "Frazer" trigger — any
        `s` or `z` in the wakeword is matched by `[sz]`, so the same
        configured wakeword catches either pronunciation.
    Wraps the whole thing in word boundaries so "fraser" doesn't fire
    inside "frasers" or "morganic".
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
from utils.phrases import (
    ACK_PHRASES,
    GREETING_TOPICS,
    NOTE_WRITE_STALL_PHRASES,
    PRE_THINKING_STALL_PHRASES,
    STALL_PHRASES,
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
# Silence timeout — past this gap with no new turn, history is cleared so a
# stale "you said earlier..." context can't bleed into a fresh conversation.
CHAT_SESSION_TIMEOUT_S = 90.0

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
        # Built in `_load_models()` once the TTS model is in memory.
        self.voice_clone_prompt = None
        self.output_device = output_device
        self.audio_capture = AudioCapture(input_device=input_device)
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
        self.asr_stream_generator = stream_generator

        from .tts import (
            speak_stream,
            warmup_model,
            synthesize,
            play_chunks,
            set_output_device,
            set_tts_active_event,
            set_voice,
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

        self.grammar, self.slm_model = load_slm()
        self.greeting_prompt = get_greeting_system_prompt()
        self.web_summary_prompt = get_web_summary_system_prompt()

        self.models_ready.set()
        self._warm_and_announce()

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
        # the tolerant regex so "Hey, Fraser ..." strips the same as
        # "Hey Fraser ...".
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

    def _play_random_ack(self, session: Optional[TtsSession] = None) -> None:
        """Play one pre-rendered ack chunk. Safe no-op if the cache is empty
        (e.g. an early failure before `_warm_and_announce` populated it).
        """
        if not self.ack_cache:
            return
        chunks, sr = random.choice(self.ack_cache)
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
            cancel_check = (lambda: session.cancelled) if session is not None else None
            logger.info(f"Handling turn: {user_prompt}")

            self._history.append({"role": "user", "content": user_prompt})
            self._trim_history()

            # Regex fast-path: if it matches, use it as the first agent emission.
            caught = catchAll(user_prompt)
            first_emission = caught if isinstance(caught, dict) else None
            if first_emission is not None:
                logger.debug(f"Regex caught: {first_emission}")

            slm_started = False
            # Holds the most recent web-search summary produced this turn.
            # Persists across replan iterations so a follow-up side-effect
            # action (e.g. write_note) doesn't bury the findings — see the
            # terminal "speak joined outputs" step.
            web_summary_text: Optional[str] = None
            # Side-effect actions (e.g. write_note) that were skipped when a
            # web-search step forced a replan before they could execute.
            # Flushed in the reply branch so they still run even when the
            # replanned agent emits a reply instead of re-emitting them.
            pending_side_effects: list = []
            # The query deep_think tagged this turn (if any). Captured at
            # dispatch, consumed once by the out-of-loop thinking call.
            thinking_query: Optional[str] = None
            for iteration in range(MAX_AGENT_CALLS_PER_TURN):
                if session is not None and session.cancelled:
                    return ""

                if iteration == 0 and first_emission is not None:
                    emission = first_emission
                    emission_text = json.dumps(emission)
                else:
                    if not slm_started:
                        slm_started = True
                        if on_slm_start is not None:
                            try:
                                on_slm_start()
                            except Exception:
                                logger.exception("on_slm_start hook raised")
                    logger.debug(f"Agent call (iter {iteration})")
                    emission_text = generate_slm(
                        self.slm_model,
                        user_prompt=None,
                        grammar=self.grammar,
                        system_prompt=get_agent_system_prompt(),
                        cancel_check=cancel_check,
                        history=self._history,
                        stats=stats,
                    )
                    logger.debug(f"Agent emission: {emission_text}")

                    if session is not None and session.cancelled:
                        return ""

                    try:
                        emission = json.loads(emission_text)
                    except Exception as e:
                        logger.error(f"Failed to parse agent emission: {emission_text!r} ({e})")
                        return random.choice([
                            "Sorry, can you repeat that",
                            "I don't understand",
                        ])

                self._history.append({"role": "assistant", "content": emission_text})
                self._trim_history()

                # Emit a `plan` event so dashboards can show what the agent decided.
                self._emit_agent_event("plan", emission, source=source)

                # Reply branch — agent's final spoken answer.
                if "reply" in emission:
                    # Flush any side-effect actions that were skipped when a
                    # web-search replan fired before they could execute.  For
                    # write_note, substitute the actual search summary so the
                    # saved note contains real findings, not the pre-search stub.
                    for deferred in pending_side_effects:
                        d_intent = deferred.get("intent", "?")
                        if d_intent == "write_note" and web_summary_text:
                            d_args = list(deferred.get("args", []))
                            if len(d_args) >= 2:
                                d_args[1] = web_summary_text
                                deferred = dict(deferred, args=d_args)
                        logger.debug(f"Executing deferred action: {deferred}")
                        d_result = intents.handle_action(deferred)
                        self._history.append({
                            "role": "tool",
                            "name": d_intent,
                            "content": d_result if isinstance(d_result, str) else str(d_result or "<error>"),
                        })
                        self._emit_agent_event("observation", {
                            "intent": d_intent,
                            "result": d_result if isinstance(d_result, str) else str(d_result or "<error>"),
                        }, source=source)
                    pending_side_effects = []
                    reply = (emission.get("reply") or "").strip()
                    if not reply:
                        return random.choice(STALL_PHRASES)
                    return reply

                actions = emission.get("actions") or []
                if not actions:
                    logger.warning("Agent emitted empty actions; stalling")
                    return random.choice(STALL_PHRASES)

                # Dispatch each action in order. Stop on the first replan trigger.
                result_strs: list = []
                replan = False
                saw_summary = False
                saw_thinking = False
                for _action_idx, action in enumerate(actions[:3]):
                    if session is not None and session.cancelled:
                        return ""
                    intent_name = action.get("intent", "?")
                    logger.debug(f"Dispatching action: {action}")
                    # A web search blocks on a SearXNG round-trip that can run
                    # many seconds (engine timeouts / rate-limits). Play the
                    # context stall BEFORE dispatch so the user hears
                    # "searching the web" during the lookup itself, not after
                    # it lands (the summarise step that follows is only ~1s).
                    if intents.is_web_search(intent_name) and self.web_search_stall_cache:
                        chunks, sr = random.choice(self.web_search_stall_cache)
                        self.play_chunks(
                            chunks, sr, session=session or self.tts_session
                        )
                        if session is not None and session.cancelled:
                            return ""
                    elif intents.is_note_write(intent_name) and self.note_write_stall_cache:
                        chunks, sr = random.choice(self.note_write_stall_cache)
                        self.play_chunks(
                            chunks, sr, session=session or self.tts_session
                        )
                        if session is not None and session.cancelled:
                            return ""
                    self._emit_agent_event("step", {
                        "intent": intent_name,
                        "args": action.get("args", []),
                    }, source=source)
                    _t_dispatch = time.monotonic()
                    result = intents.handle_action(action)
                    if stats is not None:
                        stats.tool_dispatches += 1
                        if action.get("intent") in NOTE_SEARCH_INTENTS:
                            stats.retrieval_seconds = (
                                stats.retrieval_seconds or 0.0
                            ) + (time.monotonic() - _t_dispatch)
                            chunks = notes.last_retrieval.pop("chunks", None)
                            if chunks is not None:
                                stats.retrieval_chunks = chunks

                    # Inline summariser: web search returns kilobytes of raw
                    # HTML snippets. Compress them into a short spoken answer
                    # with a focused SLM call BEFORE the agent's next view of
                    # history. (The "searching the web" stall already played
                    # before dispatch above, covering the slower lookup.)
                    web_summarised = False
                    if (
                        isinstance(result, str)
                        and result.startswith("User question:")
                    ):
                        logger.debug("Summarising web search payload")
                        if session is not None and session.cancelled:
                            return ""
                        result = self._summarise_search_result(result, cancel_check, stats=stats)
                        if session is not None and session.cancelled:
                            return ""
                        web_summarised = True
                        web_summary_text = result

                    result_str = result if isinstance(result, str) else (
                        "<error>" if result is None else str(result)
                    )
                    self._history.append({
                        "role": "tool",
                        "name": action.get("intent", "?"),
                        "content": result_str,
                    })
                    self._emit_agent_event("observation", {
                        "intent": action.get("intent", "?"),
                        "result": result_str,
                    }, source=source)
                    if isinstance(result, str):
                        if result.startswith(SUMMARY_PREFIX):
                            saw_summary = True
                        elif result.startswith(THINKING_PREFIX):
                            saw_thinking = True
                            # deep_think returns "Thinking question:\n<query>";
                            # keep the query for the out-of-loop thinking call.
                            _parts = result.split("\n", 1)
                            thinking_query = (
                                _parts[1].strip() if len(_parts) > 1 else user_prompt
                            )
                        result_strs.append(result)
                    # A summarised web search normally gets spoken directly
                    # (fast path). But when the agent planned more than the
                    # lookup alone (e.g. search + write_note), route the summary
                    # back through the agent so it writes the note from the
                    # *actual* findings now in history instead of a stub
                    # composed before the search ran.
                    if (web_summarised and len(actions) > 1) or intents.should_replan(result):
                        replan = True
                        # Save actions that haven't run yet; they'll be flushed
                        # in the reply branch if the agent doesn't re-emit them.
                        pending_side_effects = list(actions[_action_idx + 1 : 3])
                        break
                self._trim_history()

                # Special sentinel handling.
                if saw_summary:
                    # summarize_thinking — surface the captured partial directly.
                    summary = self._summarise_partial_thinking(cancel_check, stats=stats)
                    self._record_spoken(summary)
                    return summary

                if saw_thinking:
                    # deep_think flagged this query. Run ONE free-text reasoning
                    # call (NO agent grammar — the grammar permits only a JSON
                    # object, so it would forbid Qwen3's <think> block) and speak
                    # the result. Handling it here, out of the grammar loop, also
                    # stops the agent from simply re-emitting deep_think forever:
                    # with the sentinel in history and no other obvious move, it
                    # looped until MAX_AGENT_CALLS and never answered.
                    query = thinking_query or user_prompt
                    # Stall before the (slow) reasoning call so the user hears
                    # acknowledgement up front.
                    if self.pre_thinking_stall_cache:
                        chunks, sr = random.choice(self.pre_thinking_stall_cache)
                        self.play_chunks(chunks, sr, session=session or self.tts_session)
                    if session is not None and session.cancelled:
                        return ""
                    # Watchdog plays periodic "still thinking" stalls during the
                    # long /think run.
                    watchdog_session = session or self.tts_session
                    with ThinkingWatchdog(
                        self.thinking_stall_cache, self.play_chunks, watchdog_session,
                    ):
                        answer = generate_slm(
                            self.slm_model,
                            user_prompt=query,
                            system_prompt=get_thinking_system_prompt(),
                            cancel_check=cancel_check,
                            history=self._history,
                            thinking_mode=True,
                            stats=stats,
                        )
                    if session is not None and session.cancelled:
                        # Stash the partial reasoning for a follow-up
                        # summarize_thinking ("what have you got so far?").
                        if answer:
                            self._last_thinking_partial = answer
                            self._last_thinking_question = query
                            self._last_thinking_cancelled_at = time.monotonic()
                            logger.debug(
                                f"Captured {len(answer)} chars of partial thinking"
                            )
                        return ""
                    cleaned = clean_for_tts(answer)
                    if not cleaned:
                        cleaned = random.choice(STALL_PHRASES)
                    self._record_spoken(cleaned)
                    return cleaned

                if replan:
                    # Reactive question: (HA 400/404, multi-event calendar) or
                    # error — re-call the agent with observations in history.
                    # No stall here; the agent's next call is usually <1s
                    # because the input is small. `User question:` payloads
                    # are intercepted earlier by the inline summariser.
                    continue

                # All actions succeeded without replan — speak joined outputs.
                parts = [s.strip() for s in result_strs if s and s.strip()]
                # If the turn researched something and then took a side-effect
                # action (e.g. write_note), the web summary lives in history
                # but isn't in this iteration's result_strs. Surface it first
                # so the user hears the findings, not just "saved a note".
                if web_summary_text:
                    summary = web_summary_text.strip().rstrip(".")
                    already = any(p.rstrip(".") == summary for p in parts)
                    if summary and not already:
                        parts.insert(0, summary)
                spoken = ". ".join(parts)
                if not spoken:
                    spoken = "Done."
                self._record_spoken(spoken)
                return spoken

            # Cap exhausted
            logger.warning(
                f"Hit MAX_AGENT_CALLS_PER_TURN={MAX_AGENT_CALLS_PER_TURN}"
            )
            return "Sorry, I couldn't finish that."

    def _run_half_duplex(self, user_prompt: str, stt_seconds=None) -> None:
        """Synchronous handle + speak, with mic muted during TTS playback."""
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
        self._last_turn_end = time.monotonic()

    def _start_turn(self, user_prompt: str, stt_seconds=None) -> None:
        """Kick off a barge-in turn (handle + speak) on a worker thread."""
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
                self._last_turn_end = time.monotonic()
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

    def _is_self_echo(self, text_lower: str) -> bool:
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
        if len(words) == 1:
            # Single-word transcript — too short to score by overlap.
            # Treat as echo if the word (or the wakeword) appears in what
            # we just said. Catches single-letter residue like the "A"
            # tail of a spoken time ("...12:28 A M."). The wakeword check
            # runs against the un-stripped `spoken` so it still resolves
            # multi-word wakewords like "hey fraser" (spoken_norm has no
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
        return self._barge_re.search(text_lower) is not None

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
                        continue
                    if self._skip_followup_self_echo:
                        # One-shot bypass after a wakeword-alone barge-in:
                        # this utterance is the user's deliberate question,
                        # even if every word also appears in the cancelled
                        # response. Consume the flag here.
                        self._skip_followup_self_echo = False
                    elif self._is_self_echo(text_lower):
                        # Trailing AEC residue after our own TTS regularly
                        # looks like a brief utterance ("A.", "the.")
                        # landing inside the follow-up window. Suppress it
                        # the same way the barge-in path does.
                        continue
                    logger.info(f"FOLLOW-UP: {text}")

                if just_barged_in:
                    # Barge-in: TTS has stopped. Open the follow-up window so
                    # the user can speak their full query without re-saying the
                    # wakeword. Discard the partial chunk that triggered the
                    # barge-in — silence detection may have cut the sentence
                    # short, so the content here is unreliable.
                    self._last_turn_end = time.monotonic()
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
                        self._last_turn_end = time.monotonic()
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
            self._last_turn_end = time.monotonic()

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
