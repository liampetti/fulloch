"""Per-turn intent matching, model calls, tool dispatch, and reply selection.

The owning Assistant supplies history, model access, playback, and event sinks.
This module does not import Assistant, keeping the dependency one-way.
"""

import json
import logging
import random
import threading
import time
from typing import Optional

from tools import notes
from tools.capabilities import native_access_class, native_requires_deep_think
from tools.tool_registry import tool_registry
from utils import intents
from utils.intent_catch import catchAll, is_contextual_web_search_request
from utils.intents import MAX_AGENT_CALLS_PER_TURN, StepKind
from utils.phrases import ACK_PHRASES
from utils.prompts import (
    assemble_foreground_history,
    get_agent_system_prompt,
)

from .agent_emission import normalize_emission, parse_model_emission, should_style_satellite_message
from .agent_follow_up import route_report_follow_up, startup_greeting_follow_up
from .agent_search import PROMPT_STRIP_CHARS as _PROMPT_STRIP_CHARS  # noqa: F401 — Assistant export
from .agent_search import TurnSearch, last_user_question, normalise_search_query
from .satellite_context import current_satellite_id as _current_satellite_id
from .slm import ContextExhaustedError, RemoteUnreachable
from .telemetry import event as telemetry_event
from .thinking_watchdog import ThinkingWatchdog

logger = logging.getLogger(__name__)

# Tool intents that count as "context retrieval" for the stats panel. The
# chunk count is surfaced by the semantic paths via notes.last_retrieval.
NOTE_SEARCH_INTENTS = frozenset({"search_notes", "search_notes_semantic", "read_note"})


def _personality(host) -> Optional[str]:
    """Get the configured personality without coupling test hosts to Assistant."""
    getter = getattr(host, "_personality_for_prompt", None)
    if getter is not None:
        return getter()
    if getattr(host, "personality", None) == "custom":
        return getattr(host, "personality_custom", "").strip() or None
    return getattr(host, "personality", None)


def _llm_unavailable_label(host) -> str:
    """Name the configured backend rather than its shared transport contract."""
    return (
        "Local llama-server unavailable"
        if getattr(host, "llm_backend", None) == "local"
        else "Remote LLM unreachable"
    )


def _play_music_search_ack(host, session) -> None:
    """Cover Spotify's remote search and playback-dispatch latency."""
    cache = getattr(host, "music_search_stall_cache", None)
    if cache:
        host._play_random_ack(session or getattr(host, "tts_session", None), cache=cache)


class AgentLoop:
    """Runs one user turn through regex catch → agent SLM → tool dispatch.

    Construct per turn with the owning Assistant plus the turn context, then
    call `run(user_prompt)`. Must be invoked under the host's `_turn_lock`
    (the local model server and history are shared); the caller in `Assistant` holds it.
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
        # run() exposes this satellite to tools through core.satellite_context.
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
        greeting_response = getattr(host, "_startup_greeting_response", None)
        if greeting_response:
            host._startup_greeting_response = None
        startup_greeting = startup_greeting_follow_up(greeting_response, user_prompt)
        report_route = route_report_follow_up(
            user_prompt, satellite_id=self.satellite_id, cancel_check=cancel_check, stats=stats,
            consume_report=getattr(host, "consume_completed_thinking_report", None),
            answer_report=getattr(host, "answer_completed_thinking_report", None),
            active_task=getattr(host, "active_thinking_task", None), catch_intent=catchAll,
        )
        if report_route.reply:
            return report_route.reply
        caught = report_route.caught
        # Compact finished turns while retaining short tool traces for follow-ups.
        host._compact_completed_turns()

        history = host._history_for(self.satellite)
        prior_question = last_user_question(history)
        if startup_greeting:
            history.append({"role": "assistant", "content": startup_greeting})
        history.append({"role": "user", "content": user_prompt})
        host._trim_history()

        # Regex fast-path: if it matches, use it as the first agent emission.
        regex_emission = caught if isinstance(caught, dict) else None
        first_emission = regex_emission
        if should_style_satellite_message(
            regex_emission, llm_enabled=getattr(host, "llm_enabled", False),
            personality=_personality(host),
        ):
            # Other fast commands stay deterministic. A named outbound message
            # is delivery copy, so let a non-balanced personality phrase it.
            first_emission = None
        if (
            first_emission is None
            and prior_question
            and is_contextual_web_search_request(user_prompt)
        ):
            first_emission = {
                "actions": [{"intent": "external_information", "args": [prior_question]}]
            }
            logger.debug("Contextual web-search follow-up uses prior question: %r", prior_question)
        if first_emission is not None:
            logger.debug(f"Regex caught: {first_emission}")
            # A later SLM call changes the route to "agent".
            if stats is not None:
                stats.route = "regex"

        # No-LLM tier (llm.backend: none): the regex catch is the only path.
        # Dispatch a match, or speak the 'basic commands only' fallback —
        # never touch the SLM.
        if not host.llm_enabled:
            if stats is not None:
                stats.route = "no_llm"
            return self._run_without_llm(user_prompt, first_emission or regex_emission)

        def remote_fallback() -> str:
            return self._remote_llm_unavailable_fallback(host)

        if getattr(host, "_remote_llm_retry_blocked", lambda: False)():
            logger.info("Remote LLM retry cooldown active; using regex-only path")
            return self._run_without_llm(
                user_prompt, first_emission or regex_emission, unavailable_fallback=remote_fallback
            )

        slm_started = False
        # A replan acknowledgement may play while its next LLM call runs. It
        # must finish before that call's result can start another TTS stream:
        # browser TTS controls are not stream-id scoped, so a late ack `end`
        # would otherwise deactivate the final reply and discard its PCM.
        replan_ack_thread: Optional[threading.Thread] = None
        # Fresh even when callers reuse this AgentLoop instance for another run.
        search = TurnSearch()
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
                    # Replan iterations get the same short acknowledgement as
                    # the initial call; do not narrate internal model work.
                    if session is None or not session.cancelled:
                        replan_ack_thread = threading.Thread(
                            target=host._play_random_ack,
                            args=(session or host.tts_session,),
                            kwargs={
                                "cache": host.replan_stall_cache,
                                "sink": getattr(host._turn_local, "sink", None),
                                "tts_active_event": getattr(
                                    host._turn_local, "tts_active_event", None
                                ),
                            },
                            daemon=True,
                        )
                        replan_ack_thread.start()
                logger.debug(f"Agent call (iter {iteration})")
                telemetry_event(
                    "llm_start", iteration=iteration, source=source, history_entries=len(history)
                )
                llm_started_at = time.monotonic()
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
                        # A replan already plays one immediate acknowledgement.
                        # Do not repeat it while a slow remote server is stalled.
                        max_stalls=1 if iteration == 0 else 0,
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
                                personality=_personality(host),
                                higgs_tts=getattr(host, "_tts_backend", None) == "higgs-gguf",
                                conversation_mode=bool(
                                    source == "voice"
                                    and self.satellite is not None
                                    and self.satellite.conversation_mode
                                ),
                                wakeword_barge_in=bool(
                                    source == "voice"
                                    and getattr(host, "barge_in", None) == "wakeword"
                                ),
                                obsidian_edit_enabled=notes._obsidian_edit_allowed(),
                            ),
                            cancel_check=cancel_check,
                            history=assemble_foreground_history(host._history_for(self.satellite)),
                            stats=stats,
                        )
                    telemetry_event(
                        "llm_complete",
                        iteration=iteration,
                        source=source,
                        seconds=round(time.monotonic() - llm_started_at, 3),
                        response_chars=len(emission_text or ""),
                    )
                except ContextExhaustedError:
                    telemetry_event(
                        "llm_error", iteration=iteration, source=source, error="context_exhausted"
                    )
                    return host._context_exhausted_reply()
                except RemoteUnreachable as e:
                    telemetry_event(
                        "llm_error", iteration=iteration, source=source, error="unreachable"
                    )
                    logger.warning("%s; regex-only this turn: %s", _llm_unavailable_label(host), e)
                    host._note_llm_remote_status(False, str(e))
                    if first_emission is None and regex_emission is None:
                        return remote_fallback()
                    # A replan failure occurs after the regex action has already
                    # run; never dispatch it a second time just to degrade.
                    if iteration > 0:
                        return remote_fallback()
                    return self._run_without_llm(
                        user_prompt,
                        first_emission or regex_emission,
                        unavailable_fallback=remote_fallback,
                    )
                finally:
                    if replan_ack_thread is not None:
                        replan_ack_thread.join()
                        replan_ack_thread = None
                host._note_llm_remote_status(True)
                logger.debug(f"Agent emission: {emission_text}")

                if session is not None and session.cancelled:
                    return ""

                try:
                    emission = parse_model_emission(
                        emission_text, parse=intents.parse_agent_emission,
                    )
                except ValueError:
                    return random.choice(["Sorry, can you repeat that", "I don't understand"])
                # Canonicalise what goes into history so reasoning junk doesn't
                # pollute the context the model sees on the next call.
                emission_text = json.dumps(emission)

            normalized = normalize_emission(
                emission, emission_text, regex_emission=regex_emission,
                user_prompt=user_prompt, personality=_personality(host), registry=tool_registry,
                intent_services=intents, requires_deep_think=native_requires_deep_think,
                access_class=native_access_class,
            )
            emission = normalized.emission
            emission_text = normalized.history_text
            delivery = normalized.delivery
            bundled_reply = normalized.bundled_reply

            host._history_for(self.satellite).append(
                {"role": "assistant", "content": emission_text}
            )
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
                # Prefer the source-grounded summary over a replan's answer;
                # keep history consistent with what the user hears.
                if grounded := search.grounded_reply():
                    host._history_for(self.satellite)[-1] = {
                        "role": "assistant",
                        "content": json.dumps({"reply": grounded}),
                    }
                    return grounded
                if not reply:
                    return random.choice(ACK_PHRASES)
                return intents.strip_unfounded_save_claim(reply, note_written)

            actions = emission.get("actions") or []
            if not actions:
                logger.warning("Agent emitted empty actions; stalling")
                return random.choice(ACK_PHRASES)

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
            for action in actions[:3]:
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
                cached_step = None
                if intents.is_web_search(intent_name):
                    search_query = normalise_search_query(action.get("args") or [])
                    cached_step = search.cached(search_query)

                web_summarised = False
                if cached_step is not None:
                    logger.debug("Reusing cached web summary for repeated query")
                    step = cached_step
                    web_summarised = True
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
                    elif intent_name == "play_song":
                        _play_music_search_ack(host, session)
                    if intent_name == "deep_think":
                        # The foreground agent often compresses the topic. The
                        # original turn carries the complete constraints needed
                        # by the deliberate worker and its matching playbooks.
                        action = {**action, "args": [user_prompt]}
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
                        # Searching has already announced itself before the
                        # SearXNG request. If the summary model stalls, give
                        # bounded search-specific progress rather than piling
                        # generic "Aha" acknowledgements onto the response.
                        try:
                            summary = search.summarise(
                                step.text, summariser=host._summarise_search_result,
                                watchdog=ThinkingWatchdog, clips=host.web_search_stall_cache,
                                play_chunks=host.play_chunks, session=session or host.tts_session,
                                sink=getattr(host._turn_local, "sink", None),
                                tts_active_event=getattr(
                                    host._turn_local, "tts_active_event", None
                                ),
                                cancel_check=cancel_check, stats=stats,
                            )
                        except RemoteUnreachable as e:
                            logger.warning(
                                "%s mid-turn; regex-only: %s", _llm_unavailable_label(host), e
                            )
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
                        step = search.accept(search_query, summary, step)
                        web_summarised = True

                host._history_for(self.satellite).append(
                    {
                        "role": "tool",
                        "name": action.get("intent", "?"),
                        "content": step.text,
                    }
                )
                observation = {"intent": action.get("intent", "?"), "result": step.text}
                if step.artifact is not None:
                    observation["artifact"] = step.artifact
                host._emit_agent_event("observation", observation, source=source)
                # Deliberate work owns the research plan. Do not let a bundled
                # foreground search or paper lookup race it and speak an
                # unrelated result before the background report is ready.
                if intent_name == "deep_think":
                    spoken = (
                        step.text.strip() or "I'll look into that and let you know when I'm done."
                    )
                    host._record_spoken(spoken)
                    return spoken
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
            if bundled_reply and not search.latest:
                spoken = intents.strip_unfounded_save_claim(bundled_reply, note_written)
                host._record_spoken(spoken)
                return spoken
            parts = search.output_parts(result_strs)
            # `delivery` is the one concise, post-success confirmation. Do not
            # append it after raw tool results, which would restate the same
            # completed actions to the user.
            spoken = delivery if delivery else ". ".join(parts)
            if not spoken:
                spoken = "Done."
            spoken = intents.strip_unfounded_save_claim(spoken, note_written)
            host._record_spoken(spoken)
            return spoken

        # Cap exhausted.
        logger.warning(f"Hit MAX_AGENT_CALLS_PER_TURN={MAX_AGENT_CALLS_PER_TURN}")
        # If we researched something, speak the findings instead of a flat
        # apology — the lookup succeeded even though the agent never settled.
        if spoken := search.grounded_reply():
            host._record_spoken(spoken)
            return spoken
        return "Sorry, I couldn't finish that."

    def _remote_llm_unavailable_fallback(self, host) -> str:
        fallback = getattr(host, "_remote_llm_unavailable_fallback", None)
        if fallback is not None:
            return fallback(self.session, self.source, satellite_id=self.satellite_id)
        return host._speak_llm_error_fallback(
            self.session, self.source, satellite_id=self.satellite_id
        )

    def _run_without_llm(self, user_prompt: str, first_emission, unavailable_fallback=None) -> str:
        """Regex-only turn for `llm.backend: none` — never calls the SLM.

        A `catchAll` match dispatches its action(s) and speaks the joined
        tool output (a regex `{"reply": ...}` like the notes-refusal is spoken
        directly). Anything that would normally replan into the SLM — an
        unresolved-entity / error `Reactive question:` sentinel, a web search,
        or `deep_think` — can't proceed without a model, so it (and an outright
        miss) uses the caller's configured unavailable-model fallback.
        """
        host = self.host
        session = self.session
        source = self.source

        def fallback() -> str:
            if unavailable_fallback is not None:
                return unavailable_fallback()
            return host._speak_no_ai_fallback(session, source, satellite_id=self.satellite_id)

        if first_emission is None:
            return fallback()

        host._history_for(self.satellite).append(
            {"role": "assistant", "content": json.dumps(first_emission)}
        )
        host._trim_history()
        host._emit_agent_event("plan", first_emission, source=source)

        if "reply" in first_emission:
            reply = (first_emission.get("reply") or "").strip()
            return reply or fallback()

        actions = first_emission.get("actions") or []
        if not actions:
            return fallback()

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
            if intent_name == "play_song":
                _play_music_search_ack(host, session)
            step = intents.classify_step(intents.handle_action(action))
            host._history_for(self.satellite).append(
                {
                    "role": "tool",
                    "name": intent_name,
                    "content": step.text,
                }
            )
            observation = {"intent": intent_name, "result": step.text}
            if step.artifact is not None:
                observation["artifact"] = step.artifact
            host._emit_agent_event("observation", observation, source=source)
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
                return fallback()
            if step.in_output:
                result_strs.append(step.text)
        host._trim_history()

        parts = [s.strip() for s in result_strs if s and s.strip()]
        spoken = ". ".join(parts) or "Done."
        host._record_spoken(spoken)
        return spoken
