"""Per-turn intent matching, model calls, tool dispatch, and reply selection.

The owning Assistant supplies history, model access, playback, and event sinks.
This module does not import Assistant, keeping the dependency one-way.
"""

import json
import logging
import random
import re
import threading
import time
import uuid
from typing import Optional

from tools import notes
from tools.capabilities import native_access_class, native_requires_deep_think
from tools.thinking_context import reset_handoff, set_handoff
from tools.tool_registry import tool_registry
from utils import intents
from utils.intent_catch import catchAll, is_contextual_web_search_request
from utils.intents import MAX_AGENT_CALLS_PER_TURN, NOTE_WRITE_TOOLS, StepKind
from utils.phrases import ACK_PHRASES
from utils.prompts import (
    assemble_foreground_history,
    get_agent_system_prompt,
)

from .action_evidence import (
    FALLBACK,
    REPAIR,
    ActionEvidence,
    claims_execution,
    claims_research_execution,
    without_execution_claims,
)
from .agent_emission import normalize_emission, parse_model_emission, should_style_satellite_message
from .agent_follow_up import (
    disputed_answer_note,
    route_report_follow_up,
    startup_greeting_follow_up,
)
from .agent_plan import TurnPlan
from .agent_search import PROMPT_STRIP_CHARS as _PROMPT_STRIP_CHARS  # noqa: F401 — Assistant export
from .agent_search import TurnSearch, last_user_question, normalise_search_query
from .foreground_context import is_standalone_research_request, select_prompt_profile
from .satellite_context import (
    current_satellite_ha_area as _current_satellite_ha_area,
)
from .satellite_context import (
    current_satellite_id as _current_satellite_id,
)
from .slm import ContextExhaustedError, RemoteUnreachable
from .telemetry import event as telemetry_event
from .thinking_watchdog import ThinkingWatchdog

logger = logging.getLogger(__name__)

# Tool intents that count as "context retrieval" for the stats panel. The
# chunk count is surfaced by the semantic paths via notes.last_retrieval.
NOTE_SEARCH_INTENTS = frozenset({"search_notes", "search_notes_semantic", "read_note"})


def _contextual_turn_onoff(
    user_prompt: str, history: list[dict], *, ha_area: Optional[str] = None,
) -> Optional[dict]:
    """Resolve a short on/off pronoun only from the immediately prior success."""
    match = re.fullmatch(
        r"\s*(?:please\s+)?(?:turn|switch)\s+"
        r"(?:them|those|these|it|that)(?:\s+(?:lights?|ones?))?\s+"
        r"(?:back\s+)?(?P<state>on|off)(?:\s+again)?[.!?]*\s*",
        user_prompt,
        re.IGNORECASE,
    )
    if not match:
        return None

    desired = match.group("state").lower()
    expected_intent = "turn_off" if desired == "on" else "turn_on"
    prior = history[:-1] if history and history[-1].get("role") == "user" else history
    latest_user = max(
        (i for i, message in enumerate(prior) if message.get("role") == "user"),
        default=-1,
    )
    if latest_user < 0:
        return None

    pending: list[tuple[str, str]] = []
    last_success: Optional[tuple[str, str]] = None
    for message in prior[latest_user + 1 :]:
        if message.get("role") == "assistant":
            try:
                emission = json.loads(message.get("content") or "")
            except (TypeError, ValueError):
                continue
            for action in emission.get("actions", []) if isinstance(emission, dict) else []:
                if not isinstance(action, dict):
                    continue
                intent = action.get("intent")
                args, kwargs = intents.coerce_args(action.get("args"))
                target = args[0] if args else kwargs.get("entity")
                if intent in {"turn_on", "turn_off"} and target:
                    if not isinstance(target, str):
                        return None
                    pending.append((intent, target))
        elif message.get("role") == "tool":
            if "ha_area" in message and message["ha_area"] != ha_area:
                # A browser may have moved/re-pinned rooms. Never reinterpret
                # the prior bare "lights" using the new room's default.
                return None
            # More than one observation or another tool makes the referent
            # ambiguous; leave compound commands to the normal agent route.
            if last_success is not None or message.get("name") not in {"turn_on", "turn_off"}:
                return None
            tool_name = message.get("name")
            result = str(message.get("content") or "")
            for index, (intent, target) in enumerate(pending):
                if intent != tool_name:
                    continue
                pending.pop(index)
                lowered = result.casefold()
                if not result.strip() or any(marker in lowered for marker in (
                    "reactive question:", "couldn't", "could not", "error:",
                    "not available", "no command was sent", "isn't set up", "sorry",
                )):
                    return None
                else:
                    last_success = (intent, target)
                break

    if pending or last_success is None or last_success[0] != expected_intent:
        return None
    return {"actions": [{"intent": "turn_on" if desired == "on" else "turn_off",
                         "args": [last_success[1]]}]}


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
        ha_area: Optional[str] = None,
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
        self.ha_area = ha_area if ha_area is not None else getattr(satellite, "ha_area", None)
        self.turn_id = getattr(stats, "turn_id", None) or uuid.uuid4().hex

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
        if session is not None and session.cancelled:
            return ""
        token = _current_satellite_id.set(self.satellite_id)
        area_token = _current_satellite_ha_area.set(self.ha_area)
        try:
            spoken = self._run(host, session, source, stats, on_slm_start, cancel_check, user_prompt)
            return "" if session is not None and session.cancelled else spoken
        finally:
            _current_satellite_ha_area.reset(area_token)
            _current_satellite_id.reset(token)

    def _run(
        self, host, session, source, stats, on_slm_start, cancel_check, user_prompt: str
    ) -> str:
        logger.info(f"Handling turn: {user_prompt}")
        greeting_response = getattr(host, "_startup_greeting_response", None)
        if greeting_response:
            host._startup_greeting_response = None
        startup_greeting = startup_greeting_follow_up(greeting_response, user_prompt)
        resume_pending = getattr(host, "resume_pending_thinking_task", None)
        if resume_pending is not None and resume_pending(self.satellite_id, user_prompt):
            return "Thanks, I'll continue that investigation with the proposed correction."
        task_follow_up = getattr(host, "research_task_follow_up", None)
        if task_follow_up is not None:
            task_reply = task_follow_up(self.satellite_id, user_prompt)
            if task_reply:
                return task_reply
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
        if disputed := disputed_answer_note(history, user_prompt):
            history.append(disputed)
        history.append({"role": "user", "content": user_prompt})
        host._trim_history()
        research_handoff_required = is_standalone_research_request(user_prompt)

        # Resolve only a narrowly phrased on/off pronoun follow-up from the
        # immediately preceding successful device action. Small local models
        # commonly echo "turn them back on" instead of resolving "them".
        contextual_action = _contextual_turn_onoff(user_prompt, history, ha_area=self.ha_area)

        # Regex fast-path: if it matches, use it as the first agent emission.
        regex_emission = caught if isinstance(caught, dict) else None
        if regex_emission is None and contextual_action is not None:
            regex_emission = contextual_action
            logger.debug("Resolved contextual device follow-up: %s", contextual_action)
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

        # Non-generative CPU backends dispatch an existing regex match first.
        # Laya gets one bounded semantic-routing attempt only after a miss;
        # neither path can start the conversational SLM.
        if not host.llm_enabled:
            laya_routed = False
            if first_emission is None and getattr(host, "laya_enabled", False):
                router = getattr(host, "laya_router", None)
                if router is not None:
                    laya_started = time.monotonic()
                    try:
                        context = getattr(host, "_laya_context", None)
                        first_emission = (
                            router.route(user_prompt, context=context)
                            if context is not None
                            else router.route(user_prompt)
                        )
                        laya_routed = first_emission is not None
                    except Exception:
                        logger.exception("Laya routing failed; using CPU fallback")
                    finally:
                        if stats is not None:
                            stats.laya_seconds = time.monotonic() - laya_started
            if stats is not None:
                stats.route = "laya" if laya_routed else ("regex" if first_emission is not None else "no_llm")
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
        plan = TurnPlan(user_prompt, tool_registry, native_access_class)
        # Set true once a note-write tool actually dispatches this turn, so a
        # confabulated "I saved this to your notes" can be scrubbed from the
        # spoken reply when no write happened (see strip_unfounded_save_claim).
        note_written = False
        evidence = ActionEvidence()
        # Each distinct raw lookup gets a composing replan. A repeated lookup
        # reuses its observation and cannot start another composing loop.
        lookup_composed = set()
        for iteration in range(MAX_AGENT_CALLS_PER_TURN):
            if session is not None and session.cancelled:
                return ""

            ready = plan.ready() if iteration else []
            if ready:
                # Explicitly independent work and a research handoff survive a
                # tool's replan. Dependent arguments still require model repair.
                emission = {"actions": [{key: value for key, value in step.action.items()
                                         if key != "depends_on"} for step in ready]}
                emission_text = json.dumps(emission)
            elif iteration == 0 and first_emission is not None:
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
                    "llm_start", iteration=iteration, source=source, history_entries=len(history),
                    turn_id=self.turn_id,
                    process_generation=getattr(
                        getattr(host, "slm_model", None), "_fulloch_process_generation", None
                    ),
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
                        turn_history = host._history_for(self.satellite)
                        prompt_profile = select_prompt_profile(
                            user_prompt, turn_history, plan_active=bool(plan.steps) or iteration > 0,
                        )
                        model_history = (
                            [{"role": "user", "content": user_prompt}]
                            if prompt_profile == "conversation"
                            else assemble_foreground_history(turn_history)
                        )
                        if stats is not None:
                            stats.prompt_profile = prompt_profile
                        if plan.steps:
                            model_history.append({"role": "tool", "name": "plan_progress",
                                                  "content": json.dumps(plan.progress())})
                        emission_text = host._generate_with_context_recovery(
                            user_prompt=None,
                            grammar=host.grammar,
                            system_prompt=get_agent_system_prompt(
                                host.wakeword_name,
                                vault_context=getattr(host, "_vault_current_file", None),
                                satellite_area=self.ha_area,
                                personality=_personality(host),
                                higgs_tts=getattr(host, "_tts_backend", None) == "higgs-gguf",
                                breeze_tts=getattr(host, "_tts_backend", None) == "breeze-tts-2-gguf",
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
                                context_tokens=getattr(getattr(host, "slm_model", None), "context_tokens", None),
                                profile=prompt_profile,
                                user_prompt=user_prompt,
                            ),
                            cancel_check=cancel_check,
                            history=model_history,
                            stats=stats,
                        )
                    telemetry_event(
                        "llm_complete",
                        iteration=iteration,
                        source=source,
                        seconds=round(time.monotonic() - llm_started_at, 3),
                        response_chars=len(emission_text or ""),
                        turn_id=self.turn_id,
                        process_generation=getattr(getattr(host, "slm_model", None), "_fulloch_process_generation", None),
                    )
                except ContextExhaustedError as exc:
                    logger.warning("Foreground request exceeded model context: %s", exc)
                    if session is not None and session.cancelled:
                        return ""
                    telemetry_event(
                        "llm_error", iteration=iteration, source=source, error="context_exhausted",
                        turn_id=self.turn_id,
                        process_generation=getattr(getattr(host, "slm_model", None), "_fulloch_process_generation", None),
                    )
                    return plan.finish(host._context_exhausted_reply(), include_results=True)
                except RemoteUnreachable as e:
                    if session is not None and session.cancelled:
                        return ""
                    telemetry_event(
                        "llm_error", iteration=iteration, source=source, error="unreachable",
                        turn_id=self.turn_id,
                        process_generation=getattr(getattr(host, "slm_model", None), "_fulloch_process_generation", None),
                    )
                    logger.warning("%s; regex-only this turn: %s", _llm_unavailable_label(host), e)
                    host._note_llm_remote_status(False, str(e))
                    # A replan failure occurs after the regex action has already
                    # run, or after model-authored work. Preserve either plan;
                    # never redispatch its first actions just to degrade.
                    if plan.steps:
                        return plan.finish(remote_fallback(), include_results=True)
                    if first_emission is None and regex_emission is None:
                        return remote_fallback()
                    return self._run_without_llm(
                        user_prompt,
                        first_emission or regex_emission,
                        unavailable_fallback=remote_fallback,
                    )
                except Exception:
                    if session is not None and session.cancelled:
                        return ""
                    logger.exception("Agent generation failed while continuing a plan")
                    return plan.finish("I couldn't continue because the language model failed.", include_results=True)
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
                    if not plan.steps:
                        return random.choice(["Sorry, can you repeat that", "I don't understand"])
                    return plan.finish("Sorry, I couldn't continue the plan.", include_results=True)
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

            if (iteration == 0 and research_handoff_required
                    and "reply" in emission and not emission.get("actions")):
                # Acknowledgement prose cannot queue a job. Enforce the narrow
                # standalone research contract before history/UI sees the plan,
                # then use normal dispatch so only the tool's outcome is spoken.
                logger.warning("Replacing reply-only research emission with an actual handoff")
                emission = (
                    {"actions": [{"intent": "deep_think", "args": [user_prompt]}]}
                    if tool_registry.is_available("deep_think")
                    else {"reply": "Background research is unavailable right now; I haven't queued an investigation."}
                )
                emission_text = json.dumps(emission)
                delivery = bundled_reply = None

            if emission.get("actions") and plan.repeated_without_progress(emission["actions"]):
                blocker = "I couldn't make progress after one correction; the current plan repeated without a new result."
                findings = search.output_parts(plan.output(include_lookups=True))
                spoken = " ".join([*findings, blocker])
                logger.warning(
                    "Stopping unchanged agent plan turn_id=%s observations=%d",
                    self.turn_id, len(plan.observations),
                )
                telemetry_event(
                    "agent_no_progress", turn_id=self.turn_id,
                    observations=len(plan.observations),
                    pending_goals=[plan.name(step.action) for step in plan.pending()],
                )
                host._emit_agent_event(
                    "observation", {"intent": "plan_guard", "result": blocker}, source=source,
                )
                host._history_for(self.satellite).append(
                    {"role": "assistant", "content": json.dumps({"reply": spoken})},
                )
                host._record_spoken(spoken)
                return spoken

            if "reply" in emission and not emission.get("actions") and plan.pending():
                # A model-authored final reply must not erase an unexecuted
                # suffix. Show progress again and ask it to repair the arguments.
                logger.debug("Withholding premature reply: %d pending plan steps", len(plan.pending()))
                if "?" in str(emission.get("reply") or "") and any(
                    observation["status"] == "needs_input" for observation in plan.observations
                ):
                    spoken = plan.finish(without_execution_claims(emission["reply"]), include_results=True)
                    host._record_spoken(spoken)
                    return spoken
                if iteration < MAX_AGENT_CALLS_PER_TURN - 1:
                    continue
                return plan.finish("I couldn't finish the remaining steps.", include_results=True)

            if emission.get("actions"):
                bundled_reply = bundled_reply or emission.get("reply")
                emission = {key: value for key, value in emission.items() if key != "reply"}
                # Store the plan, not a model-authored outcome written before
                # execution. Only the actual tool observations establish results.
                emission_text = json.dumps({"actions": emission["actions"]})

            host._history_for(self.satellite).append(
                {"role": "assistant", "content": emission_text}
            )
            host._trim_history()

            # Emit a `plan` event so dashboards can show what the agent decided.
            # Later emissions revise arguments or resume independent work;
            # pending goals in the previous plan remain tracked separately.
            host._emit_agent_event(
                "plan",
                emission,
                source=source,
                replan=(iteration > 0),
            )

            # Reply branch — agent's final spoken answer.
            if "reply" in emission:
                reply = (emission.get("reply") or "").strip()
                # A single quick search may use its focused summary verbatim.
                # Mixed plans must compose from all tool findings instead.
                non_search_results = any(
                    step.result is not None and not intents.is_web_search(step.action.get("intent"))
                    for step in plan.steps
                )
                if not non_search_results and len(search.summaries) == 1 and (grounded := search.grounded_reply()):
                    if evidence.records:
                        grounded = ". ".join(search.output_parts([evidence.response()]))
                    reply = plan.finish(grounded)
                    turn_history = host._history_for(self.satellite)
                    if turn_history:
                        turn_history[-1] = {"role": "assistant", "content": json.dumps({"reply": reply})}
                    return reply
                if claims_execution(reply):
                    turn_history = host._history_for(self.satellite)
                    # Remove the unspoken fabrication before any recovery call.
                    if turn_history:
                        turn_history.pop()
                    logger.warning("Withheld model execution claim (operations=%d)", len(evidence.records))
                    if claims_research_execution(reply) and evidence.research_records:
                        reply = evidence.research_response()
                    elif evidence.records:
                        remainder = without_execution_claims(reply)
                        reply = " ".join(part for part in (evidence.response(), remainder) if part)
                    elif evidence.repairs == 0 and iteration < MAX_AGENT_CALLS_PER_TURN - 1:
                        evidence.repairs += 1
                        turn_history.append({"role": "tool", "name": "reply_guard", "content": REPAIR})
                        host._emit_agent_event("observation", {"intent": "reply_guard", "result": REPAIR}, source=source)
                        continue
                    else:
                        reply = FALLBACK
                    reply = plan.finish(reply, include_results=bool(search.latest))
                    turn_history.append({"role": "assistant", "content": json.dumps({"reply": reply})})
                    return reply
                if not reply:
                    if plan.steps:
                        return plan.finish("I couldn't compose the final answer.", include_results=True)
                    return random.choice(ACK_PHRASES)
                reply = plan.finish(intents.strip_unfounded_save_claim(reply, note_written))
                turn_history = host._history_for(self.satellite)
                if turn_history:
                    turn_history[-1] = {"role": "assistant", "content": json.dumps({"reply": reply})}
                return reply

            actions = emission.get("actions") or []
            if not actions:
                logger.warning("Agent emitted empty actions; stalling")
                if plan.pending():
                    continue
                if plan.steps:
                    return plan.finish("I couldn't continue the plan.", include_results=True)
                return random.choice(ACK_PHRASES)
            batch = plan.stage(actions[:3])

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
                return plan.finish(host._speak_tool_unavailable_fallback(
                    session, source, satellite_id=self.satellite_id
                ), include_results=True)

            # Dispatch each action in order. Stop on the first replan trigger.
            replan = False
            for planned_step in batch:
                if session is not None and session.cancelled:
                    return ""
                action = planned_step.action
                if planned_step.needs_revision:
                    replan = True
                    break
                intent_name = tool_registry.canonical_name(action.get("intent", "")) or action.get("intent", "?")
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
                reused = False
                prior_result = plan.cached(planned_step)
                if prior_result is not None and not intents.is_web_search(intent_name):
                    step = prior_result
                    reused = True
                    logger.debug("Reusing prior outcome; not repeating a completed or uncertain action")
                elif cached_step is not None:
                    logger.debug("Reusing cached web summary for repeated query")
                    step = cached_step
                    web_summarised = True
                    reused = True
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
                        args, kwargs = intents.coerce_args(action.get("args"))
                        research_task = kwargs.get("query") or (args[0] if args else user_prompt)
                        action = {**action, "args": [user_prompt]}
                    _t_dispatch = time.monotonic()
                    # Single typed boundary: handle_action runs the tool,
                    # classify_step maps any leading sentinel to a StepKind so
                    # the rest of the loop routes on the kind, not the raw text.
                    handoff_token = set_handoff(plan.handoff(research_task)) if intent_name == "deep_think" else None
                    try:
                        step = intents.classify_step(intents.handle_action(action))
                    finally:
                        if handoff_token is not None:
                            reset_handoff(handoff_token)
                    # Record a genuine note write (dispatched and didn't bounce
                    # to a reactive question / error) so the reply guard knows a
                    # save really happened.
                    if intents.is_note_write(intent_name) and step.kind is StepKind.NORMAL and not step.should_replan:
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
                            # The tools already ran. Never redispatch the regex
                            # plan (especially mutations) to degrade gracefully.
                            summary = "The search results could not be summarised; original source material has been retained."
                        except Exception:
                            logger.exception("Search summarisation failed; retaining sources and pending work")
                            summary = "The search results could not be summarised; original source material has been retained."
                        if session is not None and session.cancelled:
                            return ""
                        # Replace the raw payload with the summary; the loop
                        # still forces a replan via web_summarised below.
                        step = search.accept(search_query, summary, step)
                        web_summarised = True

                plan.record(planned_step, step, reused=reused)
                if not reused:
                    evidence.record({**action, "intent": intent_name}, step)
                host._history_for(self.satellite).append(
                    {
                        "role": "tool",
                        "name": action.get("intent", "?"),
                        "content": step.text,
                        **({"ha_area": self.ha_area} if intent_name in {"turn_on", "turn_off"} else {}),
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
                    if step.should_replan:
                        plan.defer(batch, planned_step)
                        replan = True
                        break
                    remaining = batch[batch.index(planned_step) + 1:]
                    independent = [later for later in remaining if later.action.get("depends_on") == []]
                    for later in remaining:
                        if later not in independent:
                            later.status = "blocked"
                            later.needs_revision = True
                            later.reason = "This step awaits the background research report."
                    if independent:
                        continue
                    spoken = (
                        step.text.strip() or "I'll look into that and let you know when I'm done."
                    )
                    spoken = plan.finish("", include_results=True) or spoken
                    host._record_spoken(spoken)
                    return spoken
                # A data-lookup tool returns raw records (a state-change dump, a
                # conversation transcript, note chunks), not a spoken answer.
                # Hand the result — now in history above — back for one composing
                # replan so the agent answers the actual question from it, rather
                # than the verbatim path reading the whole dump aloud. Each
                # distinct lookup is composed once, even in a multi-lookup plan.
                lookup_replan = (
                    planned_step.key not in lookup_composed
                    and step.kind is StepKind.NORMAL
                    and intents.is_lookup(intent_name)
                )
                # A web search always hands control back to the agent: its
                # summary is now in history, so the agent decides the next
                # move (another search, a follow-up tool, or a reply) from
                # the actual findings. The suffix remains pending: dependent
                # arguments must be re-decided, but its goals are not dropped.
                dependent_replan = any(
                    planned_step.id in later.dependencies
                    and plan.name(later.action) != "deep_think"
                    and (later.action.get("depends_on") or plan.name(later.action) in NOTE_WRITE_TOOLS)
                    for later in batch[batch.index(planned_step) + 1:]
                )
                if web_summarised or step.should_replan or lookup_replan or dependent_replan:
                    if lookup_replan:
                        lookup_composed.add(planned_step.key)
                    replan = True
                    plan.defer(batch, planned_step)
                    break
            host._trim_history()

            if replan:
                # Reactive question: (HA 400/404, multi-event calendar) or
                # error — re-call the agent with observations in history.
                # No stall here; the agent's next call is usually <1s
                # because the input is small. `User question:` payloads
                # are intercepted earlier by the inline summariser.
                continue

            if plan.pending():
                continue

            # No further replan requested. NORMAL includes plain-string failures;
            # operation confirmations must therefore preserve actual tool output,
            # never replace it with model-authored delivery/bundled prose.
            if evidence.records:
                spoken = plan.finish("", include_results=True) or evidence.response()
                host._record_spoken(spoken)
                return spoken
            # A lookup is not evidence that an operation occurred. Guard bundled
            # prose too; never repeat tools merely to repair their confirmation.
            if claims_execution(bundled_reply or "") or claims_execution(delivery or ""):
                bundled_reply = None
                delivery = None
            if bundled_reply and not search.latest:
                spoken = intents.strip_unfounded_save_claim(bundled_reply, note_written)
                host._record_spoken(spoken)
                return spoken
            parts = plan.output()
            # `delivery` is the one concise, post-success confirmation. Do not
            # append it after raw tool results, which would restate the same
            # completed actions to the user.
            spoken = delivery if delivery else ". ".join(parts)
            if not spoken:
                spoken = FALLBACK
            spoken = plan.finish(intents.strip_unfounded_save_claim(spoken, note_written))
            host._record_spoken(spoken)
            return spoken

        # Cap exhausted.
        logger.warning(f"Hit MAX_AGENT_CALLS_PER_TURN={MAX_AGENT_CALLS_PER_TURN}")
        # If we researched something, speak the findings instead of a flat
        # apology — the lookup succeeded even though the agent never settled.
        if spoken := plan.finish("", include_results=True):
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

        plan = TurnPlan(user_prompt, tool_registry, native_access_class)
        batch = plan.stage(actions[:3])
        needs_model = False
        reactive_parts = []
        for planned_step in batch:
            if session is not None and session.cancelled:
                return ""
            if planned_step.needs_revision:
                continue
            action = planned_step.action
            intent_name = tool_registry.canonical_name(action.get("intent", "")) or action.get("intent", "?")
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
            if intent_name == "deep_think":
                step = intents.StepResult(StepKind.ERROR, "Research requires an available language model.", False)
                needs_model = True
            else:
                step = plan.cached(planned_step) or intents.classify_step(intents.handle_action(action))
            observed_text = step.text
            if step.kind is StepKind.WEB_SEARCH:
                # Raw source blobs cannot be spoken without a summariser.
                step = intents.StepResult(StepKind.ERROR, "Web findings require an available summariser.", False,
                                          artifact=step.artifact, outcome=step.outcome)
                needs_model = True
            if step.kind is StepKind.ERROR:
                needs_model = True
            plan.record(planned_step, step)
            host._history_for(self.satellite).append(
                {
                    "role": "tool",
                    "name": intent_name,
                    "content": observed_text,
                    **({"ha_area": self.ha_area} if intent_name in {"turn_on", "turn_off"} else {}),
                }
            )
            observation = {"intent": intent_name, "result": observed_text}
            if step.artifact is not None:
                observation["artifact"] = step.artifact
            host._emit_agent_event("observation", observation, source=source)
            if step.kind is intents.StepKind.NORMAL and not step.should_replan:
                record_context = getattr(host, "_record_laya_action", None)
                if record_context is not None:
                    record_context(user_prompt, action, step.text)
            # No SLM to replan with. A REACTIVE step still ran the tool and
            # produced a real observation (e.g. HA "couldn't find that entity")
            # — speak that directly rather than the generic "no AI" phrase, since
            # the tool didn't need AI. Other replan kinds (web search, deep_think,
            # summary, dispatch error) genuinely need the model → fall back.
            dependent_replan = any(
                planned_step.id in later.dependencies
                and (later.action.get("depends_on") or plan.name(later.action) in NOTE_WRITE_TOOLS)
                for later in batch[batch.index(planned_step) + 1:]
            )
            if step.should_replan or dependent_replan:
                if step.kind is intents.StepKind.REACTIVE:
                    reactive_parts.append(intents.reactive_to_speech(step.text))
                plan.defer(batch, planned_step)
        host._trim_history()

        parts = plan.output(include_lookups=True)
        parts.extend(part for part in reactive_parts if part not in parts)
        if needs_model and not parts:
            return plan.finish(fallback())
        spoken = plan.finish(" ".join(parts)) or fallback()
        host._record_spoken(spoken)
        return spoken
