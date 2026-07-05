"""
Main assistant orchestration module.

Handles wakeword detection, intent processing, and response generation.
"""

import json
import logging
import queue
import random
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

# Sentinels returned by the thinking tools — kept as module-level
# constants so renames only happen in one place.
from utils.completeness import should_commit_provisional
from utils.intent_catch import catchAll
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
from .backends import ASR, LLM, TTS, get_module, resolve_models
from .noise_baseline import BackgroundNoiseBaseline
from .slm import ContextExhaustedError, RemoteUnreachable, generate_slm, load_slm
from .text_utils import clean_for_tts, split_sentences
from .tts_session import TtsSession, parse_barge_time
from .turn_stats import TurnStats, set_model_labels

logger = logging.getLogger(__name__)

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

# Strips every non-word character. The self-echo check uses this so ASR's
# "1254" / "am" still match the assistant's spoken "12 54" / "a m" — the
# get_time tool spells digits out with spaces and ASR re-concatenates them.
_NON_WORD_RE = re.compile(r"\W+")

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
    ACK_PHRASES,
    GREETING_TOPICS,
    LLM_ERROR_PHRASES,
    NO_AI_PHRASES,
    NOTE_WRITE_STALL_PHRASES,
    PRE_THINKING_STALL_PHRASES,
    REMINDER_PREFIX_PHRASES,
    REPLAN_STALL_PHRASES,
    THINKING_STALL_PHRASES,
    TOOL_UNAVAILABLE_PHRASES,
    WEB_SEARCH_STALL_PHRASES,
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
CHAT_SESSION_TIMEOUT_S = 600.0
# Cap on a compacted tool-result entry's length (chars). Raw tool payloads
# (a web summary, a note dump, a state list) can be many KB; once a turn is
# complete, only a short trace survives compaction — see
# _compact_completed_turns.
COMPACTED_TOOL_TRACE_MAX_CHARS = 160

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

# A bare wakeword (no command following) is the shape the ASR decoder most
# readily fabricates from voiced non-speech: the `asr_context_hint` prompt primes
# the wakeword spelling with nothing to anchor it, so a cough/sigh/TV burst
# decodes to "hey atticus". `_verify_bare_wakeword` gates opening the follow-up
# window on one. The loudness gate rejects a bare wakeword that sits at or below
# the background-speech baseline by this margin (dBFS) — a genuine wakeword is
# the user, closer/louder than ambient media. 0.0 = must be at least as loud as
# the baseline; raise it to demand a clearer margin over background.
BARE_WAKEWORD_MIN_OVER_BASELINE_DB = 0.0


class Assistant:
    """Owns the audio capture and runs the wakeword → intent → TTS loop."""

    def __init__(
        self,
        wakeword: str,
        wakeword_pattern: Optional[str] = None,
        voice_clone: Optional[str] = None,
        tts_speed: Optional[float] = None,
        barge_in: str = "off",
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
        models: Optional[dict] = None,
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
            barge_in: "off" (half-duplex) or "wakeword" (interrupt with wakeword)
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
        # Built in `_load_models()` once the TTS model is in memory.
        self.voice_clone_prompt = None
        self.audio_capture = AudioCapture(
            barge_in_threshold_dbfs=barge_in_threshold_dbfs,
            use_vad=use_vad,
            vad_threshold=vad_threshold,
            vad_endpoint_silence_ms=vad_endpoint_silence_ms,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_soft_endpoint_silence_ms=vad_soft_endpoint_silence_ms,
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
        self._noise_baseline = BackgroundNoiseBaseline()

        # Turn state (only meaningful in barge-in mode). A "turn" runs the SLM
        # intent/chat path + stall + final TTS in a worker thread; the
        # transcriber loop can cancel it via _cancel_turn().
        self._turn_active = False
        self._turn_session: Optional[TtsSession] = None
        self._turn_thread: Optional[threading.Thread] = None
        # Cancel handle for whichever turn is currently in flight — voice
        # barge-in, voice half-duplex, or a dashboard text turn. `request_stop`
        # signals it so the SLM stream and any TTS abort; set/cleared by each
        # turn path, None when idle. Also drives `get_state` so the dashboard
        # knows the agent is working even outside barge-in mode.
        self._active_session: Optional[TtsSession] = None
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

        # Backend selection. `models` is the parsed `models:` config block
        # (or None — defaults to the Qwen stack). Resolved once through the
        # registry so `_load_models` and the no-LLM gate share one source of
        # truth. `llm_enabled` is False for `llm.backend: none`, which runs a
        # regex-only assistant that never touches the SLM.
        self._models = resolve_models(models)
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
        # Pre-rendered "no AI model" fallback clips (see NO_AI_PHRASES).
        # Populated in `_warm_and_announce`; played on a no-LLM miss.
        self.no_ai_cache: list = []
        # Pre-rendered "AI server unreachable" clips (see LLM_ERROR_PHRASES).
        # Played when the configured remote LLM times out or errors.
        self.llm_error_cache: list = []
        # Pre-rendered "tool not available" clips (see TOOL_UNAVAILABLE_PHRASES).
        # Played by the hallucinated-tool guard instead of a fabricated answer.
        self.tool_unavailable_cache: list = []
        # Pre-rendered opening-greeting clips. `_warm_and_announce` synthesises
        # and attempts to play these during startup — before any browser could
        # possibly be connected (the WebSocket satellite only exists once the
        # dashboard is already serving, which is after warmup) — so that first
        # attempt silently drops the audio (no sink yet). Cached here and
        # replayed once by `replay_greeting()` on the first satellite
        # connection instead. `_greeting_delivered` guards against replaying
        # on every reconnect (tab refresh, follow-up browser tab, etc).
        self.greeting_cache: list = []
        self._greeting_delivered = False

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

        # Satellite (browser WebSocket) audio session.
        # _satellite_chunk_q: fed by the WebSocket handler; consumed by satellite_recorder_thread.
        # _satellite_always_listen: when True, the wakeword check is bypassed entirely.
        self._satellite_chunk_q: Optional[queue.Queue] = None
        self._satellite_always_listen: bool = False
        # Mirrors the sink handed to the TTS module in set_satellite_sink, so
        # _play_alarm_tone can push straight to it without reaching into the
        # active TTS backend module's private state.
        self._satellite_sink: Optional[queue.Queue] = None

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
        self.asr_stream_generator = asr_mod.stream_generator
        # Pay the ONNX cold-start (ORT kernel/arena init) now, not on the user's
        # first command. No-op on backends that don't implement it (e.g. GPU Qwen).
        if hasattr(self.asr_pipe, "warmup"):
            self.asr_pipe.warmup()

        # --- TTS -----------------------------------------------------------
        self._loading_backend = tts_cfg["spec"]
        tts_mod = get_module(TTS, tts_cfg["backend"])
        # Kept for the settings-console hot-apply (set_voice/set_speed live).
        # The backend gates which of those can apply without a restart: Kokoro
        # swaps voice/speed instantly; Qwen's clone needs a warmup + phrase-cache
        # re-render, so it stays restart-only (see apply_hot_config).
        self._tts_module = tts_mod
        self._tts_backend = tts_cfg["backend"]
        # The satellite recorder uses this to switch to a stricter silence
        # threshold while we're talking, so AEC residue doesn't keep
        # utterances alive.
        tts_mod.set_tts_active_event(self.audio_capture.tts_active)
        logger.info(f"Using {tts_cfg['spec'].display_name} with voice clone: {self.voice_clone}")
        tts_mod.load_tts(model_id=tts_cfg["model"], **tts_cfg["opts"])
        if self.tts_speed is not None and hasattr(tts_mod, "set_speed"):
            tts_mod.set_speed(self.tts_speed)
        self.voice_clone_prompt = tts_mod.set_voice(self.voice_clone)
        tts_mod.warmup_model(self.voice_clone_prompt)

        self.speak_stream = tts_mod.speak_stream
        self.synthesize = tts_mod.synthesize
        self.play_chunks = tts_mod.play_chunks
        self.web_search_stall_cache: list = []
        self.note_write_stall_cache: list = []
        self.pre_thinking_stall_cache: list = []
        self.thinking_stall_cache: list = []
        self.ack_cache: list = []
        self.replan_stall_cache: list = []

        # --- LLM -----------------------------------------------------------
        if self.llm_enabled:
            self._loading_backend = llm_cfg["spec"]
            logger.info(f"Using {llm_cfg['spec'].display_name}")
            backend = llm_cfg["backend"]
            if backend == "openai":
                from .llm_openai import load_openai

                self.grammar, self.slm_model = load_openai(
                    model=llm_cfg["model"], **llm_cfg["opts"]
                )
            else:  # local llama.cpp (default)
                load_kwargs = {
                    "model_path": llm_cfg["model"],
                    # Reasoning-directive family (qwen | "" for Gemma/other).
                    "think_style": llm_cfg["spec"].think_style,
                }
                if llm_cfg["n_context"]:
                    load_kwargs["n_ctx"] = llm_cfg["n_context"]
                self.grammar, self.slm_model = load_slm(**load_kwargs, **llm_cfg["opts"])
            self.greeting_prompt = get_greeting_system_prompt(self.wakeword_name)
            self.web_summary_prompt = get_web_summary_system_prompt()
        else:
            logger.info("No LLM backend — running regex-only (basic commands)")

        self.models_ready.set()
        # Probe the remote LLM (if any) in the background so a down/misconfigured
        # endpoint surfaces the dashboard banner without waiting for a first turn.
        if self.llm_backend == "openai":
            threading.Thread(target=self._probe_remote_llm, daemon=True).start()
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

    def _note_llm_remote_status(self, ok: bool, error: str = "") -> None:
        """Record the last-known reachability of the remote LLM endpoint.

        No-op unless the LLM is the remote OpenAI backend. Called from the agent
        loop (failure on a RemoteUnreachable catch, success after a generation)
        and the startup probe, so the dashboard reflects live reachability.
        """
        if self.llm_backend != "openai":
            return
        self._llm_remote_ok = ok
        self._llm_remote_error = "" if ok else (error or "unreachable")

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
            # Pre-render the "no AI model" fallback + ack clips. The fallback
            # is the no-LLM tier's spoken miss-handler (and the remote-down
            # path in Step 6); both play regardless of whether an SLM loaded.
            logger.info("Caching no-AI fallback phrases...")
            self.no_ai_cache = [
                self.synthesize(phrase, self.voice_clone_prompt) for phrase in NO_AI_PHRASES
            ]
            logger.info("Caching LLM-error fallback phrases...")
            self.llm_error_cache = [
                self.synthesize(phrase, self.voice_clone_prompt) for phrase in LLM_ERROR_PHRASES
            ]
            logger.info("Caching ack phrases...")
            self.ack_cache = [
                self.synthesize(phrase, self.voice_clone_prompt) for phrase in ACK_PHRASES
            ]
            logger.info("Caching tool-unavailable phrases...")
            self.tool_unavailable_cache = [
                self.synthesize(phrase, self.voice_clone_prompt)
                for phrase in TOOL_UNAVAILABLE_PHRASES
            ]

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
                        system_prompt=get_agent_system_prompt(self.wakeword_name),
                    )
                except RemoteUnreachable:
                    logger.warning("Remote LLM unreachable at startup — skipping cache prime")

                # Pre-render the context-specific stall phrases so each plays
                # instantly and in the cloned voice (synthesizing them live each
                # turn made them too short for the clone to lock in). These are
                # only reached on SLM-driven turns, so skip them with no LLM.
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
                logger.info("Caching replan stall phrases...")
                self.replan_stall_cache = [
                    self.synthesize(phrase, self.voice_clone_prompt)
                    for phrase in REPLAN_STALL_PHRASES
                ]

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
                logger.info(f"Generating opening greeting (topic: {topic})...")
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

            # Splitting the greeting into individual sentences and
            # synthesising each as its own worker job populates the
            # reduce-overhead CUDA-graph cache with extra prefill shapes.
            sentences = split_sentences(cleaned) or [cleaned]
            logger.info(f"Synthesising greeting ({len(sentences)} sentence(s))...")
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
                    "use the browser satellite or dashboard for voice.", e
                )

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
        sometimes invent a new number instead of dispatching a fresh tool call
        (docs/TODO.md #7). Keeping a short trace of which tool ran gives it
        that signal back without re-bloating history with full payloads.
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
        self,
        role: str,
        content: str,
        source: str,
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
        """
        event = {
            "role": "agent",
            "kind": kind,
            "ts": time.time(),
            "source": source,
            "payload": payload,
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

    def set_llm_model(self, model: str) -> dict:
        """Hot-swap the remote LLM model on the live handle — no restart.

        Only valid when the running LLM backend is the remote OpenAI client: the
        model is a per-request string there (`core/llm_openai.py`), so switching
        it needs no reload, just a guarded attribute swap. A local llama / none
        backend loads its weights at startup and can't switch live, so we refuse
        and the caller tells the user a restart is required.

        Serialised under `_turn_lock` (llama-cpp-python isn't thread-safe and the
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

    def connect_satellite(self, wakeword_bypass: bool = False) -> "queue.Queue":
        """Start a browser satellite session.  Returns the audio chunk queue.

        The WebSocket handler feeds float32 16 kHz mono chunks into the queue;
        the satellite_recorder_thread drains it and pushes utterances to the
        ASR pipeline. `wakeword_bypass=True` enables always-listen mode (no
        wakeword required); False keeps normal wakeword-after-follow-up behaviour
        but opens a 60 s grace window so the first utterance needs no wakeword.
        """
        self._satellite_always_listen = wakeword_bypass
        chunk_q: "queue.Queue" = queue.Queue(maxsize=100)
        self._satellite_chunk_q = chunk_q
        threading.Thread(
            target=self.audio_capture.satellite_recorder_thread,
            args=(chunk_q,),
            daemon=True,
            name="satellite-recorder",
        ).start()
        # Open a generous initial window so the user can speak immediately
        # without having to say the wakeword after clicking the button.
        if not wakeword_bypass:
            self._last_turn_end = time.monotonic()
            self.audio_capture.arm_follow_up(60)
        logger.info("Satellite connected (always_listen=%s)", wakeword_bypass)
        return chunk_q

    def disconnect_satellite(self) -> None:
        """Tear down the satellite session; stop the satellite recorder thread."""
        q = self._satellite_chunk_q
        if q is not None:
            q.put(None)  # sentinel → satellite_recorder_thread exits
        self._satellite_chunk_q = None
        self._satellite_always_listen = False
        self.audio_capture.clear_follow_up()
        logger.info("Satellite disconnected")

    def set_satellite_wakeword(self, bypass: bool) -> None:
        """Toggle always-listen mode on a live satellite session."""
        self._satellite_always_listen = bypass
        if bypass:
            self.audio_capture.arm_follow_up(3600)
        logger.info("Satellite always_listen=%s", bypass)

    def set_satellite_sink(self, q: Optional["queue.Queue"]) -> None:
        """Route TTS output to the WebSocket satellite queue (or None to restore speakers)."""
        self._satellite_sink = q
        tts_mod = getattr(self, "_tts_module", None)
        if tts_mod is not None and hasattr(tts_mod, "set_satellite_sink"):
            tts_mod.set_satellite_sink(q)

    def replay_greeting(self) -> None:
        """Play the cached opening greeting once, on the first satellite connect.

        See greeting_cache's docstring in __init__ for why _warm_and_announce's
        own playback attempt during startup almost never reaches a listener.
        No-op on any later reconnect, or if there's no cached greeting (e.g.
        startup hit an error before _warm_and_announce populated it).
        """
        if self._greeting_delivered or not self.greeting_cache:
            return
        self._greeting_delivered = True
        for chunks, sr in self.greeting_cache:
            if chunks:
                self.play_chunks(chunks, sr, session=self.tts_session)

    def _play_alarm_tone(self, session: Optional[TtsSession] = None) -> float:
        """Push the pre-rendered alarm tone straight to the satellite sink.

        Uses the same ("start", sr) / (chunk, None) / ("end",) sink protocol
        as speak_stream, so it queues seamlessly ahead of a spoken reminder on
        the browser (satPlayAt in index.js chains consecutive tts_start/end
        pairs back-to-back rather than resetting). Returns the tone's
        duration in seconds (0.0 if it couldn't be played) so the caller can
        add it to the reminder's own playback-end estimate.
        """
        sink = self._satellite_sink
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

    # Config paths the running assistant can apply without a restart. Two are
    # conditional on the TTS backend (Kokoro only) — see apply_hot_config.
    _HOT_CONFIG_PATHS = frozenset(
        {
            "general.log_level",
            "general.barge_in",
            "general.follow_up_time",
            "general.tts_speed",
            "general.voice_clone",
            "general.use_vad",
            "general.barge_in_threshold_dbfs",
            "general.vad_threshold",
            "general.vad_endpoint_silence_ms",
            "general.vad_min_speech_ms",
            "general.vad_soft_endpoint_silence_ms",
        }
    )

    # (cache attribute, phrase pool, llm_only) — the phrase clips pre-rendered at
    # startup. Re-rendered on a live voice change so stalls/acks don't keep
    # playing in the old voice. LLM-only pools are skipped without an SLM.
    _PHRASE_CACHE_SPECS = (
        ("no_ai_cache", NO_AI_PHRASES, False),
        ("llm_error_cache", LLM_ERROR_PHRASES, False),
        ("tool_unavailable_cache", TOOL_UNAVAILABLE_PHRASES, False),
        ("ack_cache", ACK_PHRASES, False),
        ("web_search_stall_cache", WEB_SEARCH_STALL_PHRASES, True),
        ("note_write_stall_cache", NOTE_WRITE_STALL_PHRASES, True),
        ("pre_thinking_stall_cache", PRE_THINKING_STALL_PHRASES, True),
        ("thinking_stall_cache", THINKING_STALL_PHRASES, True),
        ("replan_stall_cache", REPLAN_STALL_PHRASES, True),
    )

    def _rerender_phrase_caches(self) -> None:
        """Re-synthesise the pre-rendered phrase clips in the current voice.

        Each cache is rebuilt fully then rebound (atomic), so the transcriber
        thread reading a cache mid-rebuild sees the old or new list, never a
        partial one — same discipline as the HA deny-list swap.
        """
        for attr, phrases, llm_only in self._PHRASE_CACHE_SPECS:
            if llm_only and not self.llm_enabled:
                continue
            rendered = [self.synthesize(p, self.voice_clone_prompt) for p in phrases]
            setattr(self, attr, rendered)
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
        kokoro = self._tts_backend == "kokoro-onnx"
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
                elif path == "general.follow_up_time":
                    self.follow_up_seconds = parse_barge_time(value)
                elif path == "general.tts_speed":
                    # Qwen has no speed knob — the change is a no-op there, so a
                    # restart wouldn't help either: report applied (don't nag).
                    if kokoro and hasattr(self._tts_module, "set_speed"):
                        self._tts_module.set_speed(value)
                        self.tts_speed = value
                elif path == "general.voice_clone":
                    # Kokoro swaps the voice instantly; Qwen needs a clone warmup
                    # + phrase-cache re-render, so leave it restart-only.
                    if not kokoro:
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

        self._emit_turn_event("user", prompt, "text")
        # Per-turn cancel handle so the dashboard Stop button can abort the SLM
        # stream mid-turn (request_stop signals `_active_session`).
        session = TtsSession()
        self._active_session = session
        try:
            self._maybe_reset_session()
            stats = TurnStats()  # text turns have no STT/TTS
            answer = self._handle_wakeword(prompt, session=session, source="text", stats=stats)
            if session.cancelled:
                # Stopped from the dashboard — stand down silently, no bubble.
                return ""
            cleaned = clean_for_tts(answer)
            self._emit_turn_event("assistant", cleaned, "text", stats=stats)
            return cleaned
        except Exception as e:
            logger.error(f"Text turn error: {e}")
            self._note_runtime_error(e)
            err = "Sorry, something went wrong handling that."
            self._emit_turn_event("assistant", err, "text")
            return err
        finally:
            self._active_session = None

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
        self, session: Optional[TtsSession] = None, cache: Optional[list] = None
    ) -> None:
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

    def _speak_no_ai_fallback(self, session: Optional[TtsSession], source: str) -> str:
        """No-LLM miss handler: surface a 'basic commands only' phrase.

        Records the phrase in `_history`. For a voice turn with a populated
        cache it plays the matching pre-rendered clip and emits the assistant
        bubble itself, returning "" so the voice caller doesn't speak it again
        (see the `if cleaned:` guards in `_run_turn` / `_run_half_duplex`).
        Otherwise it returns the phrase for the caller to speak/show.
        """
        i = random.randrange(len(NO_AI_PHRASES))
        phrase = NO_AI_PHRASES[i]
        self._record_spoken(phrase)
        if source == "voice" and self.no_ai_cache:
            self._emit_turn_event("assistant", phrase, "voice")
            chunks, sr = self.no_ai_cache[i]
            try:
                self.play_chunks(chunks, sr, session=session or self.tts_session)
            except Exception:
                logger.exception("no-AI fallback playback failed")
            return ""
        return phrase

    def _speak_llm_error_fallback(self, session: Optional[TtsSession], source: str) -> str:
        """Remote LLM failure handler: surface an 'AI server unreachable' phrase.

        Used when `RemoteUnreachable` fires during an agent turn and the regex
        fast-path didn't catch the request. Same contract as
        `_speak_no_ai_fallback`: records phrase in `_history`; voice path plays
        the pre-rendered clip and returns "" so the caller doesn't double-speak.
        """
        i = random.randrange(len(LLM_ERROR_PHRASES))
        phrase = LLM_ERROR_PHRASES[i]
        self._record_spoken(phrase)
        if source == "voice" and self.llm_error_cache:
            self._emit_turn_event("assistant", phrase, "voice")
            chunks, sr = self.llm_error_cache[i]
            try:
                self.play_chunks(chunks, sr, session=session or self.tts_session)
            except Exception:
                logger.exception("LLM-error fallback playback failed")
            return ""
        return phrase

    def _speak_tool_unavailable_fallback(self, session: Optional[TtsSession], source: str) -> str:
        """Hallucinated-tool guard handler: the agent named a tool that isn't
        loaded. Speak a canned 'can't do that' instead of letting the model
        fabricate an answer from priors.

        Same contract as `_speak_no_ai_fallback`: records the phrase in
        `_history`; for a voice turn with a populated cache it plays the matching
        pre-rendered clip, emits the assistant bubble, and returns "" so the
        caller doesn't speak it again. Otherwise returns the phrase to speak/show.
        """
        i = random.randrange(len(TOOL_UNAVAILABLE_PHRASES))
        phrase = TOOL_UNAVAILABLE_PHRASES[i]
        self._record_spoken(phrase)
        if source == "voice" and self.tool_unavailable_cache:
            self._emit_turn_event("assistant", phrase, "voice")
            chunks, sr = self.tool_unavailable_cache[i]
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

        # Per-turn cancel handle so the dashboard Stop button can abort the SLM
        # stream and TTS (request_stop signals `_active_session`).
        session = TtsSession()
        self._active_session = session

        ack_thread: list = []

        def _start_ack() -> None:
            # Mute the recorder before the ack plays so it doesn't re-enter
            # the ASR queue. Half-duplex relies on this flag, not on AEC.
            self.audio_capture.transcribing = False
            t = threading.Thread(
                target=self._play_random_ack,
                args=(session,),
                daemon=True,
            )
            t.start()
            ack_thread.append(t)

        try:
            answer = self._handle_wakeword(
                user_prompt,
                session=session,
                source="voice",
                on_slm_start=_start_ack,
                stats=stats,
            )
            cleaned = clean_for_tts(answer)
            self.audio_capture.transcribing = False
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
                    ts = self._emit_turn_event("assistant", cleaned, "voice", stats=stats)
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
            finally:
                self.audio_capture.transcribing = True
        finally:
            self._active_session = None
        # A stop stands down without a follow-up window.
        if not session.cancelled:
            self._mark_turn_end(playback_end)

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
        self._active_session = session
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
                user_prompt,
                session=session,
                source="voice",
                on_slm_start=_start_ack,
                stats=stats,
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
            # Empty answer = the no-LLM bypass already played a pre-rendered
            # fallback clip (and emitted its bubble), so don't speak again.
            playback_end = None
            if cleaned:
                # Recorded for self-echo suppression in `_check_barge_in`.
                self._last_spoken_text = cleaned.lower()
                self.tts_start_time = time.monotonic()
                # Emit before speaking so the dashboard reveals the text while TTS
                # plays. Barge-in may cut speech short; the full text still shows,
                # while _history is reconciled to the spoken text via _record_spoken.
                ts = self._emit_turn_event("assistant", cleaned, "voice", stats=stats)
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
            if not session.cancelled:
                # Mic stays live during barge-in TTS, so the queue contains
                # audio of our own voice captured while we were speaking.
                # Flush it before opening the follow-up window — same treatment
                # as a barge-in cancel — otherwise the tail chunk arrives with
                # onset ≈ _last_turn_end (speech_onset_t was never set because
                # the residue stayed below the barge-in floor) and falls
                # inside the follow-up window as a spurious new turn.
                self.audio_capture.flush()
                self._drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                self._mark_turn_end(playback_end)
        except Exception as e:
            logger.error(f"Turn error: {e}")
            self._note_runtime_error(e)
        finally:
            self._turn_active = False
            self._active_session = None

    def _cancel_turn(self) -> None:
        """Signal the active turn to abort and wait briefly for it to wind down."""
        if self._turn_session is not None:
            self._turn_session.stop()
        if self._turn_thread is not None:
            self._turn_thread.join(timeout=2.0)

    def request_stop(self) -> None:
        """Stop whatever the agent is doing right now — SLM generation, TTS
        playback, an in-flight turn — silently, and stand down with no
        follow-up window. Safe no-op when idle. This is the dashboard Stop
        button's effect; the voice equivalent is a pure-stop barge-in. An
        in-flight tool dispatch runs to completion (see Known Gaps); the turn
        cancels at the next checkpoint.
        """
        session = self._active_session
        if session is not None:
            session.stop()
        # Belt-and-suspenders for any out-of-band playback on the shared session.
        self.tts_session.stop()
        if self._turn_active:
            # Barge-in worker: flush the mic and drop in-flight ASR so our own
            # speech captured mid-turn doesn't replay, mirroring a voice barge-in.
            self.audio_capture.flush()
            self._drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
        # A stop means stand down — no wakeword-free follow-up window.
        self._last_turn_end = 0.0
        self.audio_capture.clear_follow_up()
        self._dispatch_event({"role": "stopped", "ts": time.time()})
        logger.info("Stop requested — standing down")

    def _mark_turn_end(self, playback_end: Optional[float] = None) -> None:
        """Record the turn-end time and open the wakeword-free follow-up window.

        `playback_end` is the estimated monotonic-clock time the TTS audio
        actually finishes playing on the browser (`speak_stream`'s return
        value) — chunks are handed to the satellite sink as fast as they're
        generated with no flow control, so generation can finish well before
        playback does. Without this, the follow-up window was armed (and
        could silently expire) while the reply was still audibly playing, so
        a user who replied the instant the voice stopped could still land
        outside the window. Falls back to now() when no estimate is available
        (e.g. no satellite connected).

        The recorder reads the armed window to accept short replies (which the
        full min-utterance length would otherwise drop); the transcriber gauges
        the window itself from `_last_turn_end` against each utterance's onset.
        In satellite always-listen mode the recorder window is kept open
        indefinitely so short utterances are never dropped.
        """
        now = time.monotonic()
        self._last_turn_end = max(now, playback_end) if playback_end is not None else now
        window = 3600.0 if self._satellite_always_listen else self.follow_up_seconds
        if window > 0:
            self.audio_capture.arm_follow_up(window)

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
            return words[0] in spoken_norm or self._wakeword_re.search(spoken) is not None
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
        for result in self.asr_pipe(
            self.asr_stream_generator(
                self.audio_capture.audio_queue,
                self._asr_onset,
                self._asr_loudness,
                self._asr_provisional,
                self._asr_audio,
            ),
            batch_size=1,
            generate_kwargs={"max_new_tokens": 256},
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

                # Context-prompt echo: the decoder occasionally regurgitates the
                # asr_context_hint scaffolding verbatim off non-speech
                # ("Technical terms: hey atticus, <garbage>"). That carries the
                # wakeword plus a non-term tail, so neither _is_context_echo nor
                # the bare-wakeword guard catches it — but the leading marker is
                # an unambiguous tell a user never utters. Drop the whole result.
                if self._context_prompt_marker and self._context_prompt_marker in text.lower():
                    logger.debug(f"Dropped context-prompt echo: {text!r}")
                    continue

                # Provisional (soft-endpoint) snapshot of an unfinished
                # utterance. It may only *commit* a turn at the dispatch gate
                # below; the side-effecting special cases (stand-down, follow-up
                # open, baseline feed) all guard on `provisional` and drop it so
                # the real hard endpoint stays authoritative. Mid-turn a partial
                # is ignored outright — barge-in waits for the hard endpoint.
                provisional = self._asr_provisional.get("flag", False)
                if provisional and self._turn_active:
                    logger.debug(f"Ignoring provisional during active turn: {text!r}")
                    continue

                text_lower = text.lower()
                wakeword_match = self._wakeword_re.search(text_lower)
                wakeword_present = wakeword_match is not None
                just_barged_in = False

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
                baseline_db = self._noise_baseline.value()
                _vol = f"{loudness_db:.1f}" if loudness_db is not None else "n/a"
                logger.debug(
                    f"Transcribed: {text} [volume={_vol} dBFS, baseline={baseline_db:.1f} dBFS]"
                )

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
                    self._drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                    just_barged_in = True
                elif not wakeword_present:
                    # Follow-up window: a brief grace period after TTS ends
                    # during which wakeword-free input is treated as a reply
                    # to the assistant's last answer. Measured from when this
                    # utterance's *speech began* (`_asr_onset`), not now — so
                    # the window means "started replying within N seconds",
                    # and a long answer isn't disqualified by its own length
                    # plus the silence-detection tail and ASR latency.
                    onset_gap = self._asr_onset["t"] - self._last_turn_end
                    in_follow_up = (
                        self.follow_up_seconds > 0
                        and self._last_turn_end > 0
                        and 0 <= onset_gap < self.follow_up_seconds
                    ) or self._satellite_always_listen
                    if not in_follow_up:
                        # Genuine background: non-wakeword, not an accepted
                        # follow-up, and not mid-turn (that path is handled by
                        # the `_turn_active` branch above). This is the only
                        # site that feeds the noise baseline — the room/TV
                        # talking to itself, which is exactly what we estimate.
                        # A provisional is not a final utterance — never feed it
                        # into the background-noise baseline estimate.
                        if loudness_db is not None and not provisional:
                            self._noise_baseline.add(loudness_db)
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
                    if self._is_pure_stop(text_lower):
                        # Pure "stop"/"cancel": the turn is already cancelled.
                        # Stand down to idle — no follow-up window, no spoken
                        # acknowledgement. A bare stop is an instruction to
                        # cease entirely, not a preface to a new request, so the
                        # wakeword is required again before the next turn.
                        logger.info("Barge-in (pure stop) — standing down")
                        continue
                    # Redirect barge-in ("stop — actually do X"): TTS has
                    # stopped. Open the follow-up window so the user can speak
                    # their full query without re-saying the wakeword. Discard
                    # the partial chunk that triggered the barge-in — silence
                    # detection may have cut the sentence short, so the content
                    # here is unreliable.
                    self._mark_turn_end()
                    self._skip_followup_self_echo = True
                    logger.info("Barge-in — follow-up window open")
                    continue

                # Pure stop in the follow-up window: cease, don't issue a new
                # command. Stand down silently and close the window (wakeword
                # required again). Active turns use the barge-in path above; a
                # fresh wakeword turn while idle still routes "stop" to the agent.
                if not wakeword_present and self._is_pure_stop(text_lower):
                    # Don't stand down on a partial — wait for the hard endpoint
                    # to confirm the user really only said "stop".
                    if provisional:
                        continue
                    self._last_turn_end = 0.0
                    self.audio_capture.clear_follow_up()
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
                    logger.debug(f"Suppressed (context-bias echo, likely non-speech): {text!r}")
                    continue

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
                            continue
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
                    self.audio_capture.flush()
                    self._drop_results_until = time.monotonic() + DROP_AFTER_CANCEL_S
                    logger.info(f"Committing early on soft endpoint: {user_prompt!r}")

                # Snapshot the STT latency of the utterance that triggered this
                # turn before later (echo/cross-talk) transcriptions overwrite it.
                stt_seconds = getattr(self.asr_pipe, "last_transcribe_seconds", None)
                if self.barge_in == "off":
                    self._run_half_duplex(user_prompt, stt_seconds=stt_seconds)
                else:
                    self._start_turn(user_prompt, stt_seconds=stt_seconds)

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                self._note_runtime_error(e)

    def get_state(self) -> str:
        """Return the current assistant state: idle | thinking | speaking."""
        if self.audio_capture.tts_active.is_set():
            return "speaking"
        if self._turn_active or self._active_session is not None:
            return "thinking"
        return "idle"

    def speak_proactive(self, text: str, alarm: bool = False) -> None:
        """Speak `text` through the cloned voice without a user turn.

        Waits for any in-progress voice turn (SLM + TTS) to fully complete
        before speaking. `_turn_lock` covers only the SLM phase; TTS runs
        without it, so we also spin on `_turn_active` (barge-in worker) and
        `tts_active` (half-duplex TTS) to avoid two audio streams overlapping.
        Mutes the mic for the duration. Blocks until playback finishes.

        `alarm=True` (timer completions) plays the pre-rendered alert tone
        immediately before the spoken text.
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
                alarm_seconds = self._play_alarm_tone(session) if alarm else 0.0
                self._emit_turn_event("assistant", cleaned, "proactive")
                playback_end = self.speak_stream(cleaned, self.voice_clone_prompt, session=session)
                playback_end += alarm_seconds
            finally:
                self.audio_capture.transcribing = True
            self._mark_turn_end(playback_end)

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
            self.audio_capture.stop()
            trans_thread.join(timeout=2)
