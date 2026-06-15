"""The unified agent loop.

Extracted from `core.assistant.Assistant._handle_wakeword` so the most
bug-prone region of the turn lives behind a small, clearly-scoped interface.

`AgentLoop` runs one user turn: it drives the regex fast-path, the
grammar-constrained SLM agent calls, in-order action dispatch, the inline
web-search summariser, the deep-think / summarise-thinking sentinel branches,
and the terminal "speak joined outputs" step. It holds a reference to the
owning `Assistant` (`self.host`) for the shared services a turn needs —
history, stall-phrase caches, the SLM handle, event emission, TTS playback,
and the out-of-band summarisers — and keeps the per-turn context (session,
source, stats, on_slm_start hook) as its own fields.

The module is intentionally self-contained: it imports only from `utils`,
`tools`, and the leaf `core` modules (`slm`, `text_utils`, `thinking_watchdog`),
none of which import `core.assistant`, so there's no import cycle. `assistant`
imports `AgentLoop` to run turns and re-imports `_PROMPT_STRIP_CHARS` (still used
by its wakeword stripping); `NOTE_SEARCH_INTENTS` / `_normalise_search_query`
now live here and are referenced directly.
"""

import json
import logging
import random
import re
import threading
import time
from typing import Optional

from tools import notes
from utils import intents
from utils.intent_catch import catchAll
from utils.intents import MAX_AGENT_CALLS_PER_TURN, StepKind, StepResult
from utils.phrases import STALL_PHRASES
from utils.prompts import get_agent_system_prompt, get_thinking_system_prompt

from .slm import ContextExhaustedError
from .text_utils import clean_for_tts
from .thinking_watchdog import ThinkingWatchdog

logger = logging.getLogger(__name__)

# Tool intents that count as "context retrieval" for the stats panel. The
# chunk count is surfaced by the semantic paths via notes.last_retrieval.
NOTE_SEARCH_INTENTS = frozenset(
    {"search_notes", "search_notes_semantic", "read_note"}
)

# Leading/trailing punctuation peeled off a user prompt / search query. Includes
# "!"/"?" so ASR's "Hey Atticus! Stop." yields the bare "stop" without trailing
# punctuation confusing intent matching.
_PROMPT_STRIP_CHARS = " ,.!?;:"


def _normalise_search_query(args) -> Optional[str]:
    """Stable cache key for a web-search action's query.

    Lowercases, collapses whitespace, and strips surrounding punctuation so
    trivial variants of the same query dedupe. A no-arg call (the search tool
    falls back to its default query) maps to a fixed sentinel so repeated
    default-news lookups also dedupe. Returns None for a non-string first arg.
    """
    if not args:
        return "__default__"
    q = args[0]
    if not isinstance(q, str):
        return None
    q = re.sub(r"\s+", " ", q).strip().lower().strip(_PROMPT_STRIP_CHARS)
    return q or "__default__"


class AgentLoop:
    """Runs one user turn through regex catch → agent SLM → tool dispatch.

    Construct per turn with the owning Assistant plus the turn context, then
    call `run(user_prompt)`. Must be invoked under the host's `_turn_lock`
    (llama-cpp-python isn't thread-safe); the caller in `Assistant` holds it.
    """

    def __init__(
        self,
        host,
        *,
        session=None,
        source: str = "voice",
        stats=None,
        on_slm_start: Optional[callable] = None,
    ):
        self.host = host
        self.session = session
        self.source = source
        self.stats = stats
        self.on_slm_start = on_slm_start
        self.cancel_check = (
            (lambda: session.cancelled) if session is not None else None
        )

    def run(self, user_prompt: str) -> str:
        """Drive the agent loop for `user_prompt` and return the spoken text.

        Returns an empty or partial string if barge-in cancels mid-turn.
        """
        host = self.host
        session = self.session
        source = self.source
        stats = self.stats
        on_slm_start = self.on_slm_start
        cancel_check = self.cancel_check

        logger.info(f"Handling turn: {user_prompt}")

        # Every tool result / planning emission in history now belongs to an
        # already-finished turn, so drop them: the conversation is carried by
        # the user messages and Fulloch's recorded replies, and anything a
        # follow-up needs (notes, web findings) is re-fetched rather than
        # recalled from a stale tool dump. Keeps history lean so long
        # conversations don't blow N_CONTEXT.
        host._compact_completed_turns()

        host._history.append({"role": "user", "content": user_prompt})
        host._trim_history()

        # Regex fast-path: if it matches, use it as the first agent emission.
        caught = catchAll(user_prompt)
        first_emission = caught if isinstance(caught, dict) else None
        if first_emission is not None:
            logger.debug(f"Regex caught: {first_emission}")

        slm_started = False
        # Holds the most recent web-search summary produced this turn.
        # Persists across replan iterations so a follow-up action (e.g. a
        # note save the agent composes after seeing the findings) doesn't
        # bury the result — see the terminal "speak joined outputs" step.
        web_summary_text: Optional[str] = None
        # The query deep_think tagged this turn (if any). Captured at
        # dispatch, consumed once by the out-of-loop thinking call.
        thinking_query: Optional[str] = None
        # Per-turn web-search cache {normalised_query: summary}. A web search
        # always hands control back to the agent; if the agent re-issues the
        # *same* query, reusing the summary avoids a second SearXNG round-trip
        # + summarise. A genuinely different follow-up query is a cache miss
        # and still searches, so chained "drill into a result" research works.
        search_cache: dict = {}
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
                else:
                    # Replan iterations — play a "mid-process" phrase so
                    # the SLM thinking time isn't silent. Uses the replan
                    # cache ("Working through it.", "Almost there.", etc.)
                    # rather than ACK_PHRASES so the user hears progress
                    # rather than repeated acknowledgements.
                    if session is None or not session.cancelled:
                        threading.Thread(
                            target=host._play_random_ack,
                            args=(session or host.tts_session,),
                            kwargs={"cache": host.replan_stall_cache},
                            daemon=True,
                        ).start()
                logger.debug(f"Agent call (iter {iteration})")
                try:
                    # Recovery wrapper sheds oldest history and retries on
                    # overflow; only re-raises if the recent floor won't fit.
                    emission_text = host._generate_with_context_recovery(
                        user_prompt=None,
                        grammar=host.grammar,
                        system_prompt=get_agent_system_prompt(),
                        cancel_check=cancel_check,
                        history=host._history,
                        stats=stats,
                    )
                except ContextExhaustedError:
                    return host._context_exhausted_reply()
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

            host._history.append({"role": "assistant", "content": emission_text})
            host._trim_history()

            # Emit a `plan` event so dashboards can show what the agent decided.
            host._emit_agent_event("plan", emission, source=source)

            # Reply branch — agent's final spoken answer.
            if "reply" in emission:
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
                host._emit_agent_event("step", {
                    "intent": intent_name,
                    "args": action.get("args", []),
                }, source=source)

                # Per-turn idempotent web search: if this exact query already
                # ran this turn, reuse the cached summary instead of paying
                # for another SearXNG round-trip + summarise.
                search_query = None
                cached_summary = None
                if intents.is_web_search(intent_name):
                    search_query = _normalise_search_query(
                        action.get("args") or []
                    )
                    if search_query is not None:
                        cached_summary = search_cache.get(search_query)

                web_summarised = False
                if cached_summary is not None:
                    logger.debug(
                        "Reusing cached web summary for repeated query"
                    )
                    step = StepResult(StepKind.NORMAL, cached_summary, in_output=True)
                    web_summarised = True
                    web_summary_text = cached_summary
                else:
                    # A web search blocks on a SearXNG round-trip that can run
                    # many seconds (engine timeouts / rate-limits). Play the
                    # context stall BEFORE dispatch so the user hears
                    # "searching the web" during the lookup itself, not after
                    # it lands (the summarise step that follows is only ~1s).
                    if intents.is_web_search(intent_name) and host.web_search_stall_cache:
                        chunks, sr = random.choice(host.web_search_stall_cache)
                        host.play_chunks(
                            chunks, sr, session=session or host.tts_session
                        )
                        if session is not None and session.cancelled:
                            return ""
                    elif intents.is_note_write(intent_name) and host.note_write_stall_cache:
                        chunks, sr = random.choice(host.note_write_stall_cache)
                        host.play_chunks(
                            chunks, sr, session=session or host.tts_session
                        )
                        if session is not None and session.cancelled:
                            return ""
                    _t_dispatch = time.monotonic()
                    # Single typed boundary: handle_action runs the tool,
                    # classify_step maps any leading sentinel to a StepKind so
                    # the rest of the loop routes on the kind, not the raw text.
                    step = intents.classify_step(intents.handle_action(action))
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
                    if step.kind is StepKind.WEB_SEARCH:
                        logger.debug("Summarising web search payload")
                        if session is not None and session.cancelled:
                            return ""
                        # Summarise is a full SLM call (~2-3s). The web-search
                        # stall covered the SearXNG round-trip; play a parallel
                        # ack now so this gap isn't silent too.
                        if session is None or not session.cancelled:
                            threading.Thread(
                                target=host._play_random_ack,
                                args=(session or host.tts_session,),
                                daemon=True,
                            ).start()
                        summary = host._summarise_search_result(
                            step.text, cancel_check, stats=stats
                        )
                        if session is not None and session.cancelled:
                            return ""
                        # Replace the raw payload with the summary; the loop
                        # still forces a replan via web_summarised below.
                        step = StepResult(StepKind.NORMAL, summary, in_output=True)
                        web_summarised = True
                        web_summary_text = summary
                        if search_query is not None:
                            search_cache[search_query] = summary

                host._history.append({
                    "role": "tool",
                    "name": action.get("intent", "?"),
                    "content": step.text,
                })
                host._emit_agent_event("observation", {
                    "intent": action.get("intent", "?"),
                    "result": step.text,
                }, source=source)
                if step.kind is StepKind.SUMMARY:
                    saw_summary = True
                elif step.kind is StepKind.THINKING:
                    # deep_think returns "Thinking question:\n<query>";
                    # keep the query for the out-of-loop thinking call.
                    _parts = step.text.split("\n", 1)
                    thinking_query = (
                        _parts[1].strip() if len(_parts) > 1 else user_prompt
                    )
                    saw_thinking = True
                if step.in_output:
                    result_strs.append(step.text)
                # A web search always hands control back to the agent: its
                # summary is now in history, so the agent decides the next
                # move (another search, a follow-up tool, or a reply) from
                # the actual findings. Any later actions the agent bundled
                # with the search are dropped here and re-decided on replan,
                # so a save is composed from real findings, never a stub.
                if web_summarised or step.should_replan:
                    replan = True
                    break
            host._trim_history()

            # Special sentinel handling.
            if saw_summary:
                # summarize_thinking — surface the captured partial directly.
                summary = host._summarise_partial_thinking(cancel_check, stats=stats)
                host._record_spoken(summary)
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
                if host.pre_thinking_stall_cache:
                    chunks, sr = random.choice(host.pre_thinking_stall_cache)
                    host.play_chunks(chunks, sr, session=session or host.tts_session)
                if session is not None and session.cancelled:
                    return ""
                # Watchdog plays periodic "still thinking" stalls during the
                # long /think run.
                watchdog_session = session or host.tts_session
                try:
                    with ThinkingWatchdog(
                        host.thinking_stall_cache, host.play_chunks, watchdog_session,
                    ):
                        answer = host._generate_with_context_recovery(
                            user_prompt=query,
                            system_prompt=get_thinking_system_prompt(),
                            cancel_check=cancel_check,
                            history=host._history,
                            thinking_mode=True,
                            stats=stats,
                        )
                except ContextExhaustedError:
                    return host._context_exhausted_reply()
                if session is not None and session.cancelled:
                    # Stash the partial reasoning for a follow-up
                    # summarize_thinking ("what have you got so far?").
                    if answer:
                        host._last_thinking_partial = answer
                        host._last_thinking_question = query
                        host._last_thinking_cancelled_at = time.monotonic()
                        logger.debug(
                            f"Captured {len(answer)} chars of partial thinking"
                        )
                    return ""
                cleaned = clean_for_tts(answer)
                if not cleaned:
                    cleaned = random.choice(STALL_PHRASES)
                host._record_spoken(cleaned)
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
            # If the turn researched something and the agent then took a
            # follow-up action (e.g. saving a note), the web summary lives
            # in history but not in this iteration's result_strs. Surface it
            # first so the user hears the findings, not just "saved a note".
            if web_summary_text:
                summary = web_summary_text.strip().rstrip(".")
                already = any(p.rstrip(".") == summary for p in parts)
                if summary and not already:
                    parts.insert(0, summary)
            spoken = ". ".join(parts)
            if not spoken:
                spoken = "Done."
            host._record_spoken(spoken)
            return spoken

        # Cap exhausted.
        logger.warning(
            f"Hit MAX_AGENT_CALLS_PER_TURN={MAX_AGENT_CALLS_PER_TURN}"
        )
        # If we researched something, speak the findings instead of a flat
        # apology — the lookup succeeded even though the agent never settled.
        if web_summary_text:
            spoken = web_summary_text.strip()
            if spoken:
                host._record_spoken(spoken)
                return spoken
        return "Sorry, I couldn't finish that."
