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

from .satellite_context import current_satellite_id as _current_satellite_id
from .slm import ContextExhaustedError, RemoteUnreachable
from .text_utils import clean_for_tts
from .thinking_watchdog import ThinkingWatchdog

logger = logging.getLogger(__name__)

# Tool intents that count as "context retrieval" for the stats panel. The
# chunk count is surfaced by the semantic paths via notes.last_retrieval.
NOTE_SEARCH_INTENTS = frozenset({"search_notes", "search_notes_semantic", "read_note"})

# Leading/trailing punctuation peeled off a user prompt / search query. Includes
# "!"/"?" so ASR's "Hey Atticus! Stop." yields the bare "stop" without trailing
# punctuation confusing intent matching.
_PROMPT_STRIP_CHARS = " ,.!?;:"

# The calling satellite's id for the duration of the current `AgentLoop.run`
# call (None outside a turn, or for a turn with no satellite — e.g. a fresh
# text turn's "dashboard-text" pseudo-id still sets this). This is the one
# hook tools need to read "which satellite/room is this turn for" without
# every `@tool` signature growing a satellite_id parameter — only the HA
# per-satellite area default (#14) reads it so far. Lives in
# `core/satellite_context.py`, not here, so a tool module can import it
# without pulling in this module's much heavier dependency chain (`core.slm`
# alone imports `torch`).


def _normalise_search_query(args) -> Optional[str]:
    """Stable cache key for a web-search action's query.

    Lowercases, collapses whitespace, and strips surrounding punctuation so
    trivial variants of the same query dedupe. A no-arg call (the search tool
    falls back to its default query) maps to a fixed sentinel so repeated
    default-news lookups also dedupe. Returns None for a non-string first arg.

    Shape-robust: a grammar-less remote model may emit `args` as a kwargs object
    (`{"query": "x"}`) or a bare string rather than the list the grammar forces,
    so indexing `args[0]` blindly would raise `KeyError(0)` on a dict.
    """
    if isinstance(args, dict):
        q = next(iter(args.values()), None)  # kwargs-style: first value
    elif isinstance(args, (list, tuple)):
        q = args[0] if args else None  # positional: first arg
    elif isinstance(args, str):
        q = args
    else:
        q = None
    if q is None:
        return "__default__"
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
        satellite_id: Optional[str] = None,
        satellite=None,
    ):
        self.host = host
        self.session = session
        self.source = source
        self.stats = stats
        self.on_slm_start = on_slm_start
        self.cancel_check = (lambda: session.cancelled) if session is not None else None
        # satellite_id/satellite identify the calling satellite (or
        # "dashboard-text" for a typed turn); cheap to pass since AgentLoop is
        # constructed fresh per turn anyway. Read by `run` to set
        # `_current_satellite_id` for the duration of the call — the only
        # hook tools need to read the calling satellite (#14).
        self.satellite_id = satellite_id
        self.satellite = satellite

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
        token = _current_satellite_id.set(self.satellite_id)
        try:
            return self._run(host, session, source, stats, on_slm_start, cancel_check, user_prompt)
        finally:
            _current_satellite_id.reset(token)

    def _run(
        self, host, session, source, stats, on_slm_start, cancel_check, user_prompt: str
    ) -> str:
        logger.info(f"Handling turn: {user_prompt}")

        # Every tool result / planning emission in history now belongs to an
        # already-finished turn, so drop them: the conversation is carried by
        # the user messages and Fulloch's recorded replies, and anything a
        # follow-up needs (notes, web findings) is re-fetched rather than
        # recalled from a stale tool dump. Keeps history lean so long
        # conversations don't blow N_CONTEXT.
        host._compact_completed_turns()

        host._history_for(self.satellite).append({"role": "user", "content": user_prompt})
        host._trim_history()

        # Regex fast-path: if it matches, use it as the first agent emission.
        caught = catchAll(user_prompt)
        first_emission = caught if isinstance(caught, dict) else None
        if first_emission is not None:
            logger.debug(f"Regex caught: {first_emission}")
            # A2 route stat: overwritten to "agent" below if an SLM call ever
            # actually runs this turn (e.g. a replan after the regex catch).
            if stats is not None:
                stats.route = "regex"

        # No-LLM tier (llm.backend: none): the regex catch is the only path.
        # Dispatch a match, or speak the 'basic commands only' fallback —
        # never touch the SLM.
        if not host.llm_enabled:
            if stats is not None:
                stats.route = "no_llm"
            return self._run_without_llm(user_prompt, first_emission)

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
        # Set true once a note-write tool actually dispatches this turn, so a
        # confabulated "I saved this to your notes" can be scrubbed from the
        # spoken reply when no write happened (see strip_unfounded_save_claim).
        note_written = False
        # Set true once a data-lookup result (intents.LOOKUP_TOOLS) has been
        # handed back for a composing replan this turn. Bounds the cost to a
        # single extra agent call: a second lookup result (or the same one
        # re-fetched) falls through to the verbatim path rather than looping
        # replan→re-fetch up to the call cap. Mirrors the web-search contract.
        lookup_composed = False
        for iteration in range(MAX_AGENT_CALLS_PER_TURN):
            if session is not None and session.cancelled:
                return ""

            if iteration == 0 and first_emission is not None:
                emission = first_emission
                emission_text = json.dumps(emission)
            else:
                if not slm_started:
                    slm_started = True
                    if stats is not None:
                        stats.route = "agent"
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
                            kwargs={
                                "cache": host.replan_stall_cache,
                                "sink": getattr(host._turn_local, "sink", None),
                                "tts_active_event": getattr(host._turn_local, "tts_active_event", None),
                            },
                            daemon=True,
                        ).start()
                logger.debug(f"Agent call (iter {iteration})")
                try:
                    # Periodic progress stalls so a slow generation isn't silent
                    # — esp. a remote LLM, where a long reply can be many seconds
                    # of dead air (the one-shot replan/ack stall above plays only
                    # at the start). The first stall fires after one interval, so
                    # fast local calls (the GBNF grammar stops them quickly) never
                    # trigger it. Recovery wrapper sheds oldest history and retries
                    # on overflow; only re-raises if the recent floor won't fit.
                    with ThinkingWatchdog(
                        host.replan_stall_cache,
                        host.play_chunks,
                        session or host.tts_session,
                        sink=getattr(host._turn_local, "sink", None),
                        tts_active_event=getattr(host._turn_local, "tts_active_event", None),
                    ):
                        emission_text = host._generate_with_context_recovery(
                            user_prompt=None,
                            grammar=host.grammar,
                            system_prompt=get_agent_system_prompt(
                                host.wakeword_name,
                                vault_context=getattr(host, "_vault_current_file", None),
                                satellite_area=(
                                    self.satellite.ha_area if self.satellite is not None else None
                                ),
                            ),
                            cancel_check=cancel_check,
                            history=host._history_for(self.satellite),
                            stats=stats,
                        )
                except ContextExhaustedError:
                    return host._context_exhausted_reply()
                except RemoteUnreachable as e:
                    logger.warning("Remote LLM unreachable; regex-only this turn")
                    host._note_llm_remote_status(False, str(e))
                    if first_emission is None:
                        return host._speak_llm_error_fallback(
                            self.session, self.source, satellite_id=self.satellite_id
                        )
                    return self._run_without_llm(user_prompt, first_emission)
                host._note_llm_remote_status(True)
                logger.debug(f"Agent emission: {emission_text}")

                if session is not None and session.cancelled:
                    return ""

                try:
                    # Tolerant parse: strips reasoning artefacts (<think> blocks,
                    # stray </think>, repeated objects) a grammar-less remote LLM
                    # can wrap around the JSON, and takes the first balanced object.
                    emission = intents.parse_agent_emission(emission_text)
                except Exception as e:
                    # A grammar-less remote model often answers in plain prose
                    # instead of the JSON envelope (response_format is only a hint;
                    # some servers ignore it, and even the one repair retry can come
                    # back as prose). That prose is almost always a valid spoken
                    # reply, so don't drop the turn — treat it as `{"reply": ...}`
                    # and fall through to the reply branch (which also prefers a
                    # grounded web summary after a search). Only genuinely empty
                    # output, or a malformed JSON *fragment* (starts with { or [ —
                    # speaking that would read brace-junk aloud), gives up.
                    prose = (emission_text or "").strip()
                    if not prose or prose[:1] in ("{", "["):
                        logger.error(f"Failed to parse agent emission: {emission_text!r} ({e})")
                        return random.choice(
                            [
                                "Sorry, can you repeat that",
                                "I don't understand",
                            ]
                        )
                    logger.warning(f"Agent emission was prose, not JSON; treating as a reply ({e})")
                    emission = {"reply": prose}
                # Cap actions to 3 — GBNF enforces this for local llama; do
                # it structurally here so the remote path (no grammar) can't
                # emit a 20-room scan. Applies before history, plan event, and
                # dispatch all see the emission, so they stay consistent.
                if isinstance(emission.get("actions"), list) and len(emission["actions"]) > 3:
                    logger.warning(
                        "Agent emitted %d actions; capping to 3",
                        len(emission["actions"]),
                    )
                    emission = {**emission, "actions": emission["actions"][:3]}
                # Canonicalise what goes into history so reasoning junk doesn't
                # pollute the context the model sees on the next call.
                emission_text = json.dumps(emission)

            # A grammar-less model sometimes bundles its spoken answer as a
            # `reply` pseudo-action *inside* `actions` (e.g. [append_to_today(...),
            # reply("Done")]) instead of using the {"reply": ...} envelope. Split
            # it out: the real tools still dispatch, the text becomes the spoken
            # reply, and the hallucinated-tool guard doesn't block the whole turn
            # on the non-tool `reply`.
            bundled_reply = None
            _acts = emission.get("actions")
            if isinstance(_acts, list):
                kept = []
                for a in _acts:
                    _intent = a.get("intent") if isinstance(a, dict) else None
                    if (
                        isinstance(_intent, str)
                        and _intent.lower() in intents.REPLY_PSEUDO_INTENTS
                        and not intents.is_registered_tool(_intent)
                    ):
                        text = intents.coerce_reply_text(a.get("args"))
                        if text:
                            bundled_reply = text
                    else:
                        kept.append(a)
                if len(kept) != len(_acts):  # a pseudo-reply was split out
                    # Only a pseudo-reply (no real tools) → it's just a reply.
                    emission = {"actions": kept} if kept else {"reply": bundled_reply or ""}
                    if not kept:
                        bundled_reply = None
                    emission_text = json.dumps(emission)

            host._history_for(self.satellite).append({"role": "assistant", "content": emission_text})
            host._trim_history()

            # Emit a `plan` event so dashboards can show what the agent decided.
            # iteration > 0 means the agent was re-called: this plan supersedes
            # the previous one (e.g. a web search dropped the actions bundled
            # after it and re-decided from the findings). Flag it as a replan so
            # the trace shows the prior plan was scrapped, not silently failed.
            host._emit_agent_event(
                "plan",
                emission,
                source=source,
                replan=(iteration > 0),
            )

            # Reply branch — agent's final spoken answer.
            if "reply" in emission:
                reply = (emission.get("reply") or "").strip()
                # Grounding guard: if a web search ran this turn, the inline
                # summariser already built a grounded answer from the actual
                # SearXNG snippets (`web_summary_text`). A replan `reply` here is
                # the model re-answering from a compressed summary in history —
                # a fabrication opening a weak model takes (Gemma invented news
                # items — inflation figures, sports results — in testing). Speak
                # the grounded summary instead, and reconcile the just-appended
                # history entry to it so the fabricated reply doesn't linger for
                # the next turn. (A search followed by a real follow-up action
                # goes through the actions branch + terminal path, not here.)
                if web_summary_text and web_summary_text.strip():
                    grounded = web_summary_text.strip()
                    host._history_for(self.satellite)[-1] = {
                        "role": "assistant",
                        "content": json.dumps({"reply": grounded}),
                    }
                    return grounded
                if not reply:
                    return random.choice(STALL_PHRASES)
                return intents.strip_unfounded_save_claim(reply, note_written)

            actions = emission.get("actions") or []
            if not actions:
                logger.warning("Agent emitted empty actions; stalling")
                return random.choice(STALL_PHRASES)

            # Hallucinated-tool guard (direct registry match, not a heuristic
            # read of the observation): a weaker model — especially a remote one
            # without the GBNF grammar — sometimes invents a tool name that was
            # never in its prompt, then fabricates an answer from priors when
            # told it "failed". If ANY action this emission names a tool that
            # isn't loaded, block the turn up front (before dispatching the valid
            # ones, so no partial side effects) and speak a canned "can't do
            # that" instead of letting it replan into a fabrication.
            unknown = [
                a.get("intent", "?")
                for a in actions[:3]
                if not intents.is_registered_tool(a.get("intent"))
            ]
            if unknown:
                logger.warning(
                    "Agent called unregistered tool(s) %s; blocking fabrication",
                    unknown,
                )
                host._emit_agent_event(
                    "observation",
                    {
                        "intent": unknown[0],
                        "result": f"Blocked: tool {unknown[0]!r} is not available.",
                    },
                    source=source,
                )
                return host._speak_tool_unavailable_fallback(
                    session, source, satellite_id=self.satellite_id
                )

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
                host._emit_agent_event(
                    "step",
                    {
                        "intent": intent_name,
                        "args": action.get("args", []),
                    },
                    source=source,
                )

                # Per-turn idempotent web search: if this exact query already
                # ran this turn, reuse the cached summary instead of paying
                # for another SearXNG round-trip + summarise.
                search_query = None
                cached_summary = None
                if intents.is_web_search(intent_name):
                    search_query = _normalise_search_query(action.get("args") or [])
                    if search_query is not None:
                        cached_summary = search_cache.get(search_query)

                web_summarised = False
                if cached_summary is not None:
                    logger.debug("Reusing cached web summary for repeated query")
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
                        host.play_chunks(chunks, sr, session=session or host.tts_session)
                        if session is not None and session.cancelled:
                            return ""
                    elif intents.is_note_write(intent_name) and host.note_write_stall_cache:
                        chunks, sr = random.choice(host.note_write_stall_cache)
                        host.play_chunks(chunks, sr, session=session or host.tts_session)
                        if session is not None and session.cancelled:
                            return ""
                    _t_dispatch = time.monotonic()
                    # Single typed boundary: handle_action runs the tool,
                    # classify_step maps any leading sentinel to a StepKind so
                    # the rest of the loop routes on the kind, not the raw text.
                    step = intents.classify_step(intents.handle_action(action))
                    # Record a genuine note write (dispatched and didn't bounce
                    # to a reactive question / error) so the reply guard knows a
                    # save really happened.
                    if intents.is_note_write(intent_name) and step.kind is StepKind.NORMAL:
                        note_written = True
                    if stats is not None:
                        stats.tool_dispatches += 1
                        if action.get("intent") in NOTE_SEARCH_INTENTS:
                            stats.retrieval_seconds = (stats.retrieval_seconds or 0.0) + (
                                time.monotonic() - _t_dispatch
                            )
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
                                kwargs={
                                    "sink": getattr(host._turn_local, "sink", None),
                                    "tts_active_event": getattr(host._turn_local, "tts_active_event", None),
                                },
                                daemon=True,
                            ).start()
                        try:
                            summary = host._summarise_search_result(
                                step.text, cancel_check, stats=stats
                            )
                        except RemoteUnreachable as e:
                            logger.warning("Remote LLM unreachable mid-turn; regex-only")
                            host._note_llm_remote_status(False, str(e))
                            if first_emission is None:
                                return host._speak_llm_error_fallback(
                            self.session, self.source, satellite_id=self.satellite_id
                        )
                            return self._run_without_llm(user_prompt, first_emission)
                        if session is not None and session.cancelled:
                            return ""
                        # Replace the raw payload with the summary; the loop
                        # still forces a replan via web_summarised below.
                        step = StepResult(StepKind.NORMAL, summary, in_output=True)
                        web_summarised = True
                        web_summary_text = summary
                        if search_query is not None:
                            search_cache[search_query] = summary

                host._history_for(self.satellite).append(
                    {
                        "role": "tool",
                        "name": action.get("intent", "?"),
                        "content": step.text,
                    }
                )
                host._emit_agent_event(
                    "observation",
                    {
                        "intent": action.get("intent", "?"),
                        "result": step.text,
                    },
                    source=source,
                )
                if step.kind is StepKind.SUMMARY:
                    saw_summary = True
                elif step.kind is StepKind.THINKING:
                    # deep_think returns "Thinking question:\n<query>";
                    # keep the query for the out-of-loop thinking call.
                    _parts = step.text.split("\n", 1)
                    thinking_query = _parts[1].strip() if len(_parts) > 1 else user_prompt
                    saw_thinking = True
                # A data-lookup tool returns raw records (a state-change dump, a
                # conversation transcript, note chunks), not a spoken answer.
                # Hand the result — now in history above — back for one composing
                # replan so the agent answers the actual question from it, rather
                # than the verbatim path reading the whole dump aloud. Guarded by
                # `lookup_composed` so it fires at most once per turn.
                lookup_replan = (
                    not lookup_composed
                    and step.kind is StepKind.NORMAL
                    and intents.is_lookup(intent_name)
                )
                # A result destined for a replan must not also be joined into the
                # spoken output (it would be both read raw AND recomposed).
                if step.in_output and not lookup_replan:
                    result_strs.append(step.text)
                # A web search always hands control back to the agent: its
                # summary is now in history, so the agent decides the next
                # move (another search, a follow-up tool, or a reply) from
                # the actual findings. Any later actions the agent bundled
                # with the search are dropped here and re-decided on replan,
                # so a save is composed from real findings, never a stub.
                if web_summarised or step.should_replan or lookup_replan:
                    if lookup_replan:
                        lookup_composed = True
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
                        host.thinking_stall_cache,
                        host.play_chunks,
                        watchdog_session,
                        sink=getattr(host._turn_local, "sink", None),
                        tts_active_event=getattr(host._turn_local, "tts_active_event", None),
                    ):
                        answer = host._generate_with_context_recovery(
                            user_prompt=query,
                            system_prompt=get_thinking_system_prompt(host.wakeword_name),
                            cancel_check=cancel_check,
                            history=host._history_for(self.satellite),
                            thinking_mode=True,
                            stats=stats,
                        )
                except ContextExhaustedError:
                    return host._context_exhausted_reply()
                except RemoteUnreachable as e:
                    logger.warning("Remote LLM unreachable mid-think; regex-only")
                    host._note_llm_remote_status(False, str(e))
                    if first_emission is None:
                        return host._speak_llm_error_fallback(
                            self.session, self.source, satellite_id=self.satellite_id
                        )
                    return self._run_without_llm(user_prompt, first_emission)
                if session is not None and session.cancelled:
                    # Stash the partial reasoning for a follow-up
                    # summarize_thinking ("what have you got so far?").
                    if answer:
                        host._last_thinking_partial = answer
                        host._last_thinking_question = query
                        host._last_thinking_cancelled_at = time.monotonic()
                        logger.debug(f"Captured {len(answer)} chars of partial thinking")
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

            # All actions succeeded without replan — speak the answer.
            # If the model bundled its spoken reply with the tool actions (split
            # out above), the tools have now run, so speak that reply (e.g.
            # "Done, saved to your notes") rather than raw joined tool outputs —
            # unless a web search ran, where the grounded summary wins (below).
            if bundled_reply and not web_summary_text:
                spoken = intents.strip_unfounded_save_claim(bundled_reply, note_written)
                host._record_spoken(spoken)
                return spoken
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
            spoken = intents.strip_unfounded_save_claim(spoken, note_written)
            host._record_spoken(spoken)
            return spoken

        # Cap exhausted.
        logger.warning(f"Hit MAX_AGENT_CALLS_PER_TURN={MAX_AGENT_CALLS_PER_TURN}")
        # If we researched something, speak the findings instead of a flat
        # apology — the lookup succeeded even though the agent never settled.
        if web_summary_text:
            spoken = web_summary_text.strip()
            if spoken:
                host._record_spoken(spoken)
                return spoken
        return "Sorry, I couldn't finish that."

    def _run_without_llm(self, user_prompt: str, first_emission) -> str:
        """Regex-only turn for `llm.backend: none` — never calls the SLM.

        A `catchAll` match dispatches its action(s) and speaks the joined
        tool output (a regex `{"reply": ...}` like the notes-refusal is spoken
        directly). Anything that would normally replan into the SLM — an
        unresolved-entity / error `Reactive question:` sentinel, a web search,
        or `deep_think` — can't proceed without a model, so it (and an outright
        miss) falls back to a spoken "basic commands only" phrase.
        """
        host = self.host
        session = self.session
        source = self.source

        if first_emission is None:
            return host._speak_no_ai_fallback(session, source, satellite_id=self.satellite_id)

        host._history_for(self.satellite).append({"role": "assistant", "content": json.dumps(first_emission)})
        host._trim_history()
        host._emit_agent_event("plan", first_emission, source=source)

        if "reply" in first_emission:
            reply = (first_emission.get("reply") or "").strip()
            return reply or host._speak_no_ai_fallback(session, source, satellite_id=self.satellite_id)

        actions = first_emission.get("actions") or []
        if not actions:
            return host._speak_no_ai_fallback(session, source, satellite_id=self.satellite_id)

        result_strs: list = []
        for action in actions[:3]:
            if session is not None and session.cancelled:
                return ""
            intent_name = action.get("intent", "?")
            host._emit_agent_event(
                "step",
                {
                    "intent": intent_name,
                    "args": action.get("args", []),
                },
                source=source,
            )
            step = intents.classify_step(intents.handle_action(action))
            host._history_for(self.satellite).append(
                {
                    "role": "tool",
                    "name": intent_name,
                    "content": step.text,
                }
            )
            host._emit_agent_event(
                "observation",
                {
                    "intent": intent_name,
                    "result": step.text,
                },
                source=source,
            )
            # No SLM to replan with. A REACTIVE step still ran the tool and
            # produced a real observation (e.g. HA "couldn't find that entity")
            # — speak that directly rather than the generic "no AI" phrase, since
            # the tool didn't need AI. Other replan kinds (web search, deep_think,
            # summary, dispatch error) genuinely need the model → fall back.
            if step.should_replan:
                if step.kind is intents.StepKind.REACTIVE:
                    spoken = intents.reactive_to_speech(step.text)
                    host._record_spoken(spoken)
                    return spoken
                return host._speak_no_ai_fallback(session, source, satellite_id=self.satellite_id)
            if step.in_output:
                result_strs.append(step.text)
        host._trim_history()

        parts = [s.strip() for s in result_strs if s and s.strip()]
        spoken = ". ".join(parts) or "Done."
        host._record_spoken(spoken)
        return spoken
