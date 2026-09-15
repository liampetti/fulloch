"""Normalize foreground model emissions before plan events and dispatch."""

import json
import logging
import re
from dataclasses import dataclass

from .agent_search import normalise_search_query

logger = logging.getLogger(__name__)

_FINANCE_ADVICE_RE = re.compile(
    r"\b(buy|sell|hold|invest(?:ment)?|allocat(?:e|ion)|margin|options?|short(?:ing)?|tax|legal)\b",
    re.I,
)
_ANNOUNCEMENT_SUFFIXES = {
    "playful": "The plates are ready for their moment.",
    "calm": "Come through when you're ready.",
    "wry": "The kitchen's patience has been noted.",
}
_LITERAL_ANNOUNCEMENT_RE = re.compile(
    r"\b(?:verbatim|exact(?:ly)?|quote|alarm|emergency|evacuat|fire|smoke|carbon monoxide|"
    r"medic(?:al|ine|ation)|ambulance|call 000|call 911)\b", re.I,
)
_SENSITIVE_DELIVERY_INTENTS = frozenset({"ha_lock", "ha_unlock"})


def should_style_satellite_message(emission, *, llm_enabled, personality) -> bool:
    if not llm_enabled or personality in (None, "balanced"):
        return False
    actions = emission.get("actions") if isinstance(emission, dict) else None
    return (
        isinstance(actions, list) and len(actions) == 1
        and actions[0].get("intent") == "send_satellite_message"
    )


def route_deep_think_only_tools(emission, user_prompt, *, registry, requires_deep_think):
    actions = emission.get("actions")
    finance_advice = _FINANCE_ADVICE_RE.search(user_prompt) and any(
        isinstance(action, dict)
        and registry.canonical_name(str(action.get("intent") or ""))
        in {"get_finance_quote", "get_exchange_rate", "get_watchlist_brief", "get_market_brief"}
        for action in actions or []
    )
    if (
        not isinstance(actions, list)
        or not (finance_advice or any(
            isinstance(action, dict)
            and requires_deep_think(str(action.get("intent") or "")) for action in actions
        ))
        or not registry.is_available("deep_think")
    ):
        return emission
    return {"actions": [{"intent": "deep_think", "args": [user_prompt]}]}


def _satellite_message_args(emission):
    actions = emission.get("actions") if isinstance(emission, dict) else None
    if not isinstance(actions, list) or len(actions) != 1:
        return None
    action = actions[0]
    if not isinstance(action, dict):
        return None
    args = action.get("args")
    if (
        action.get("intent") != "send_satellite_message"
        or not isinstance(args, list) or len(args) < 2
    ):
        return None
    return args


def apply_announcement_fallback(personality, user_prompt, raw_emission, emission):
    suffix = _ANNOUNCEMENT_SUFFIXES.get(personality)
    raw_args = _satellite_message_args(raw_emission)
    action_args = _satellite_message_args(emission)
    if not suffix or not raw_args or not action_args or _LITERAL_ANNOUNCEMENT_RE.search(user_prompt):
        return
    raw_text, proposed = raw_args[1], action_args[1]
    if not isinstance(raw_text, str) or not isinstance(proposed, str):
        return
    if normalise_search_query([raw_text]) == normalise_search_query([proposed]):
        action_args[1] = f"{proposed.rstrip('. ')}. {suffix}"


def can_speak_delivery(personality, actions, *, access_class) -> bool:
    return personality not in (None, "balanced") and isinstance(actions, list) and all(
        not isinstance(action, dict) or (
            action.get("intent") not in _SENSITIVE_DELIVERY_INTENTS
            and access_class(action.get("intent", "")) == "execute"
        ) for action in actions
    )


def parse_model_emission(text, *, parse):
    """Tolerate reasoning/prose, reject empty or malformed JSON, cap remote plans."""
    try:
        emission = parse(text)
    except Exception as exc:
        prose = (text or "").strip()
        if not prose or prose[:1] in ("{", "["):
            logger.error("Failed to parse agent emission: %r (%s)", text, exc)
            raise ValueError("Unusable agent emission") from exc
        logger.warning("Agent emission was prose, not JSON; treating as a reply (%s)", exc)
        emission = {"reply": prose}
    if isinstance(emission.get("actions"), list) and len(emission["actions"]) > 3:
        logger.warning("Agent emitted %d actions; capping to 3", len(emission["actions"]))
        emission = {**emission, "actions": emission["actions"][:3]}
    return emission


@dataclass(frozen=True)
class NormalizedEmission:
    emission: dict
    history_text: str
    delivery: str | None
    bundled_reply: str | None


def normalize_emission(
    emission, history_text, *, regex_emission, user_prompt, personality,
    registry, intent_services, requires_deep_think, access_class,
) -> NormalizedEmission:
    """Apply announcement/research policy and separate pseudo-replies from tools.

    Keep the original serialized emission unless splitting pseudo-actions, as
    before: plan routing and history representation have distinct contracts.
    """
    if emission is not regex_emission:
        apply_announcement_fallback(personality, user_prompt, regex_emission, emission)
    emission = route_deep_think_only_tools(
        emission, user_prompt, registry=registry, requires_deep_think=requires_deep_think,
    )
    actions = emission.get("actions")
    can_deliver = can_speak_delivery(personality, actions, access_class=access_class)
    delivery = emission.get("delivery")
    delivery = delivery.strip() or None if isinstance(delivery, str) and can_deliver else None
    bundled_reply = None
    if isinstance(actions, list):
        kept = []
        for action in actions:
            name = action.get("intent") if isinstance(action, dict) else None
            if (
                isinstance(name, str) and name.lower() in intent_services.REPLY_PSEUDO_INTENTS
                and not intent_services.is_registered_tool(name)
            ):
                if text := intent_services.coerce_reply_text(action.get("args")):
                    bundled_reply = text
            else:
                kept.append(action)
        if len(kept) != len(actions):
            emission = (
                {"actions": kept, **({"delivery": delivery} if delivery else {})}
                if kept else {"reply": bundled_reply or ""}
            )
            if not kept:
                bundled_reply = None
            history_text = json.dumps(emission)
    return NormalizedEmission(emission, history_text, delivery, bundled_reply)
