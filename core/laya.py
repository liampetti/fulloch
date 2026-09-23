"""Bounded semantic command routing for the CPU-only Laya backend.

Laya selects only from schemas supplied by Fulloch. It never generates tool
arguments: every required value is either a static default or a candidate
validated by local parsing or Home Assistant's loaded alias map.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.tool_registry import tool_registry
from utils.duration import duration_components, normalise_duration
from utils.value_parsing import parse_percentage

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./data/models/laya"
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
MAX_VALUE_CANDIDATES = 9
MAX_ACTION_CHOICES = 9

_WORD_RE = re.compile(r"[a-z0-9]+")
_TARGET_WORDS = frozenset(
    {
        "a", "an", "and", "at", "can", "could", "for", "hey", "i", "in", "is", "it", "me",
        "my", "now", "on", "please", "the", "this", "to", "turn", "what", "with", "you",
    }
)
_NUMBER_WORD = (
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand)\b"
)
_TIMER_REMINDER_RE = re.compile(
    rf"\b(?:remind|reminder|so that)\b|\bto\s+(?!\d|{_NUMBER_WORD})[a-z]",
    re.I,
)
_WEATHER_ARGUMENT_RE = re.compile(
    r"\b(?:today|tomorrow|yesterday|tonight|this\s+(?:week|weekend)|next\s+\w+|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|in|near|at)\b",
    re.I,
)
_MEDIA_TARGET_RE = re.compile(r"\b(?:in|on|at)\s+(?:the\s+)?[a-z]", re.I)
_MEDIA_GENERIC_WORDS = frozenset(
    {"a", "all", "and", "audio", "back", "continue", "music", "next", "now", "pause", "playback",
     "please", "previous", "resume", "skip", "song", "stop", "the", "this", "whatever", "is", "playing"}
)
_ACTION_HINTS = {
    "turn_on": re.compile(r"\b(?:turn|switch|power)\s+on\b", re.I),
    "turn_off": re.compile(r"\b(?:turn|switch|power|shut)\s+(?:off|down)\b", re.I),
    "toggle": re.compile(r"\b(?:toggle|switch)\b", re.I),
    "get_entity_state": re.compile(r"\b(?:state|status|on or off)\b", re.I),
    "ha_set_brightness": re.compile(r"\b(?:dim|brighten|brighton|brighter|brightness|percent)\b", re.I),
    "ha_set_color": re.compile(r"\b(?:color|colour|red|blue|green|yellow|white|purple|pink)\b", re.I),
    "ha_volume_set": re.compile(r"\b(?:volume|louder|quieter|percent)\b", re.I),
    "ha_volume_up": re.compile(r"\b(?:louder|volume up|turn up)\b", re.I),
    "ha_volume_down": re.compile(r"\b(?:quieter|volume down|turn down)\b", re.I),
    "ha_lock": re.compile(r"\b(?:lock)\b", re.I),
    "ha_unlock": re.compile(r"\b(?:unlock)\b", re.I),
    "ha_open_cover": re.compile(r"\b(?:open|raise)\b", re.I),
    "ha_close_cover": re.compile(r"\b(?:close|lower)\b", re.I),
    "get_temperature": re.compile(r"\b(?:temperature|temp|how warm|how cold)\b", re.I),
    "get_home_overview": re.compile(r"\b(?:home|house)\s+(?:status|overview|check)\b", re.I),
    "get_energy_overview": re.compile(r"\b(?:energy|power usage|solar status)\b", re.I),
    "get_security_overview": re.compile(r"\b(?:security|secure|security check)\b", re.I),
    "ha_stop_cover": re.compile(r"\b(?:stop|halt)\b.*\b(?:blind|curtain|shade|shutter|garage|cover|valve)\b", re.I),
    "ha_set_cover_position": re.compile(
        r"\b(?:blind|curtain|shade|shutter|garage|cover|valve)\b.*\b(?:percent|%|halfway)\b", re.I
    ),
    "ha_set_fan_speed": re.compile(r"\b(?:fan)\b.*\b(?:speed|percent|%)\b", re.I),
    "ha_run_script": re.compile(r"\b(?:run|start)\b.*\b(?:script|automation)\b", re.I),
    "ha_activate_scene": re.compile(r"\b(?:activate|set|turn on)\b.*\bscene\b", re.I),
}

_CONTEXT_PRONOUN_RE = r"(?:it|them|that|those|these)"
_CONTEXT_LIGHT_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?:turn\s+)?{_CONTEXT_PRONOUN_RE}\s+"
    r"(?:back\s+)?(?P<state>on|off)\s*(?:again)?[.!?]*\s*$",
    re.I,
)
_CONTEXT_BRIGHTNESS_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?P<verb>dim|brighten|brighton)\s+{_CONTEXT_PRONOUN_RE}"
    r"(?:\s+again)?[.!?]*\s*$",
    re.I,
)
_CONTEXT_VOLUME_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?:turn\s+)?{_CONTEXT_PRONOUN_RE}\s+"
    r"(?P<direction>up|down)\s*[.!?]*\s*$",
    re.I,
)
_CONTEXT_TRANSPORT_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?P<action>pause|resume|skip)\s+{_CONTEXT_PRONOUN_RE}"
    r"\s*[.!?]*\s*$",
    re.I,
)
_CONTEXT_COVER_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?P<action>open|close|stop|halt)\s+{_CONTEXT_PRONOUN_RE}"
    r"\s*[.!?]*\s*$",
    re.I,
)
_CONTEXT_LOCK_RE = re.compile(
    rf"^\s*(?:now\s+)?(?:please\s+)?(?P<action>lock|unlock)\s+{_CONTEXT_PRONOUN_RE}"
    r"\s*[.!?]*\s*$",
    re.I,
)


@dataclass(frozen=True)
class _Action:
    intent: str
    family: str
    argument_resolver: Optional[Callable[[str], Optional[list[Any]]]] = None
    optional_entity_domain: Optional[str] = None
    entity_domain: Optional[str] = None
    value_candidates: Optional[Callable[[str], dict[str, Any]]] = None


def _entity_candidates(
    command: str, domain: Optional[str] = None, preferred: Optional[str] = None,
) -> dict[str, str]:
    """Return a small lexical shortlist from HA's live, configured alias map."""
    try:
        from tools import ha_client

        ha_client._ensure_loaded()
        aliases = ha_client._ENTITY_ALIASES
    except Exception:
        logger.exception("Could not prepare Home Assistant entity candidates")
        return {preferred: preferred} if preferred else {}

    tokens = set(_WORD_RE.findall(command.lower())) - _TARGET_WORDS
    if not tokens:
        return {preferred: preferred} if preferred else {}
    scored = []
    for alias, entity_id in aliases.items():
        if domain and not entity_id.startswith(f"{domain}."):
            continue
        overlap = tokens & set(_WORD_RE.findall(alias.lower()))
        if overlap:
            scored.append((len(overlap), -len(alias), alias, entity_id))
    scored.sort(reverse=True)
    candidates = {alias: entity_id for _, _, alias, entity_id in scored[:MAX_VALUE_CANDIDATES]}
    if preferred:
        candidates = {preferred: preferred, **candidates}
    return candidates


def _timer_duration_candidates(command: str) -> dict[str, str]:
    """Return one complete, normalized duration without a reminder."""
    if _TIMER_REMINDER_RE.search(command):
        return {}
    components = duration_components(command)
    if not components:
        return {}
    return {" ".join(component[2] for component in components): normalise_duration(components)}


def _entity_argument(command: str) -> Optional[list[Any]]:
    candidates = _entity_candidates(command)
    return [candidates] if candidates else None


def _temperature_argument(command: str) -> Optional[list[Any]]:
    """Return climate and temperature-sensor candidates for a temperature ask."""
    # Match the temperature tool's own resolver order: a climate zone wins over
    # a sensor with the same friendly alias.
    candidates = _entity_candidates(command, "sensor")
    candidates.update(_entity_candidates(command, "climate"))
    return [candidates] if candidates else None


def _timer_argument(command: str) -> Optional[list[Any]]:
    candidates = _timer_duration_candidates(command)
    return [candidates] if candidates else None


def _percentage_candidates(command: str) -> dict[str, str]:
    """Return raw percentage phrases; tools own their final coercion."""
    candidates = {}
    for match in re.finditer(r"\b\d{1,3}\s*(?:percent|%)\b", command, re.I):
        candidates[match.group(0)] = match.group(0)
    for match in re.finditer(_NUMBER_WORD, command, re.I):
        phrase = match.group(0)
        if re.match(r"\s*(?:percent\b|%)", command[match.end() :], re.I):
            phrase += " percent"
        try:
            parse_percentage(phrase)
        except ValueError:
            continue
        candidates[phrase] = phrase
    lowered = command.lower()
    if "dim" in lowered:
        candidates.setdefault("30 percent", "30 percent")
    if "brighten" in lowered:
        candidates.setdefault("100 percent", "100 percent")
    if "brighton" in lowered or "brighter" in lowered:
        candidates.setdefault("100 percent", "100 percent")
    # Laya can distinguish qualitative requests such as "a bit darker" even
    # when the ASR text contains no numeric phrase. The tool still validates
    # the selected raw percentage before Home Assistant receives it.
    if not candidates:
        candidates = {"30 percent": "30 percent", "100 percent": "100 percent"}
    return dict(list(candidates.items())[:MAX_VALUE_CANDIDATES])


def _numeric_percentage_candidates(command: str) -> dict[str, int]:
    """Return normalised percentage values for tools that require an integer."""
    lowered = command.lower()
    if "halfway" in lowered:
        return {"halfway": 50}
    # Unlike light brightness, fan and cover controls must never infer a
    # default level: an omitted value is ambiguous and should fall through.
    if not re.search(r"(?:\bpercent\b|%|\bhalf\b|\bfull\b|\bmaximum\b|\bmax\b)", lowered):
        return {}
    candidates = {}
    for phrase in _percentage_candidates(command):
        try:
            candidates[phrase] = parse_percentage(phrase)
        except ValueError:
            continue
    return candidates


def _colour_candidates(command: str) -> dict[str, str]:
    colours = (
        "warm white", "cool white", "red", "green", "blue", "yellow", "orange", "purple", "pink", "white",
    )
    return {colour: colour for colour in colours if re.search(rf"\b{re.escape(colour)}\b", command, re.I)}


def _action_priority(action: _Action, command: str) -> int:
    hint = _ACTION_HINTS.get(action.intent)
    return 0 if hint is not None and hint.search(command) else 1


def _has_explicit_media_target(command: str) -> bool:
    """Whether omitting a media entity would change the user's command."""
    tokens = set(_WORD_RE.findall(command.lower()))
    return bool(_MEDIA_TARGET_RE.search(command) or tokens - _MEDIA_GENERIC_WORDS)


# Actions remain an explicit product contract, while their descriptions and
# availability come from the live tool registry. New tools need an argument
# resolver before becoming selectable in the non-generative CPU backend.
_ACTION_TEMPLATES = (
    _Action("get_current_time", "information"),
    _Action("get_weather_forecast", "information"),
    _Action("get_temperature", "information", _temperature_argument),
    _Action("get_home_overview", "information"),
    _Action("get_energy_overview", "information"),
    _Action("get_security_overview", "information"),
    _Action("get_timer_status", "timers"),
    _Action("start_countdown", "timers", _timer_argument),
    _Action("cancel_timer", "timers"),
    _Action("pause", "media", optional_entity_domain="media_player"),
    _Action("resume", "media", optional_entity_domain="media_player"),
    _Action("skip", "media", optional_entity_domain="media_player"),
    _Action("previous", "media", optional_entity_domain="media_player"),
    _Action("turn_on", "home_assistant", _entity_argument),
    _Action("turn_off", "home_assistant", _entity_argument),
    _Action("toggle", "home_assistant", _entity_argument),
    _Action("get_entity_state", "home_assistant", _entity_argument),
    _Action("ha_set_brightness", "home_assistant", entity_domain="light", value_candidates=_percentage_candidates),
    _Action("ha_set_color", "home_assistant", entity_domain="light", value_candidates=_colour_candidates),
    _Action("ha_volume_set", "media", entity_domain="media_player", value_candidates=_percentage_candidates),
    _Action("ha_volume_up", "media", optional_entity_domain="media_player"),
    _Action("ha_volume_down", "media", optional_entity_domain="media_player"),
    _Action("ha_lock", "home_assistant", _entity_argument),
    _Action("ha_unlock", "home_assistant", _entity_argument),
    _Action("ha_open_cover", "home_assistant", _entity_argument),
    _Action("ha_close_cover", "home_assistant", _entity_argument),
    _Action("ha_stop_cover", "home_assistant", entity_domain="cover"),
    _Action("ha_set_cover_position", "home_assistant", entity_domain="cover", value_candidates=_numeric_percentage_candidates),
    _Action("ha_set_fan_speed", "home_assistant", entity_domain="fan", value_candidates=_numeric_percentage_candidates),
    _Action("ha_run_script", "home_assistant", entity_domain="script"),
    _Action("ha_activate_scene", "home_assistant", entity_domain="scene"),
)


class LayaRouter:
    """Run bounded semantic classification and return a complete agent emission."""

    def __init__(self, agent: Any, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._agent = agent
        self._confidence_threshold = confidence_threshold

    def route(self, command: str, context: Optional[dict] = None) -> Optional[dict]:
        """Return a complete, safe action or ``None`` for the CPU fallback."""
        if contextual := self._contextual_route(command, context):
            return contextual
        actions = self._available_actions(command, context)[:MAX_ACTION_CHOICES]
        if not actions:
            return None
        # Keep the initial choice space flat and below the checkpoint's 11+
        # option calibration bucket. A separate family decision was less
        # confident than direct action selection in a real CPU smoke test.
        action_by_intent = {action.intent: action for action in actions}
        intent = self._choose(
            self._model_command(command, context),
            "Which supported Fulloch action should run?",
            {name: self._tool_description(name) for name in action_by_intent},
        )
        action = action_by_intent.get(intent)
        if action is None:
            return None
        args = self._resolve_args(action, command, context)
        if args is None:
            return None
        return {"actions": [{"intent": action.intent, "args": args}]}

    @staticmethod
    def _contextual_route(command: str, context: Optional[dict]) -> Optional[dict]:
        """Resolve unambiguous pronoun follow-ups without another model decision.

        A target enters this context only after a successful local tool call.
        These routes therefore retain normal tool-side resolution and voice
        deny-list checks while avoiding a confidence-sensitive model hop for
        commands such as ``now dim them`` or ``turn it back on``.
        """
        if not isinstance(context, dict):
            return None
        target = context.get("target")
        prior = context.get("intent")
        if not isinstance(target, str) or not target:
            return None
        light_intents = {"turn_on", "turn_off", "ha_set_brightness", "ha_set_color"}
        media_intents = {"pause", "resume", "skip", "previous", "ha_volume_set", "ha_volume_up", "ha_volume_down"}
        cover_intents = {"ha_open_cover", "ha_close_cover", "ha_stop_cover", "ha_set_cover_position"}
        lock_intents = {"ha_lock", "ha_unlock"}
        def emission(intent: str, args: list[Any]) -> Optional[dict]:
            return {"actions": [{"intent": intent, "args": args}]} if tool_registry.is_available(intent) else None

        if prior in light_intents:
            if match := _CONTEXT_LIGHT_RE.match(command):
                return emission(f"turn_{match.group('state').lower()}", [target])
            if match := _CONTEXT_BRIGHTNESS_RE.match(command):
                level = 30 if match.group("verb").lower() == "dim" else 100
                return emission("ha_set_brightness", [target, level])
        if prior in media_intents:
            if match := _CONTEXT_VOLUME_RE.match(command):
                intent = "ha_volume_up" if match.group("direction").lower() == "up" else "ha_volume_down"
                return emission(intent, [target])
            if match := _CONTEXT_TRANSPORT_RE.match(command):
                return emission(match.group("action").lower(), [target])
        if prior in cover_intents and (match := _CONTEXT_COVER_RE.match(command)):
            action = match.group("action").lower()
            intent = {"open": "ha_open_cover", "close": "ha_close_cover", "stop": "ha_stop_cover", "halt": "ha_stop_cover"}[action]
            return emission(intent, [target])
        if prior in lock_intents and (match := _CONTEXT_LOCK_RE.match(command)):
            return emission(f"ha_{match.group('action').lower()}", [target])
        return None

    def _available_actions(self, command: str, context: Optional[dict] = None) -> list[_Action]:
        actions = [a for a in _ACTION_TEMPLATES if tool_registry.is_available(a.intent)]
        # This backend only supports the weather tool's default arguments.
        # Date and location requests must fall through instead of being lost.
        if _WEATHER_ARGUMENT_RE.search(command):
            actions = [a for a in actions if a.intent != "get_weather_forecast"]
        prior_intent = context.get("intent") if isinstance(context, dict) else None
        return sorted(
            actions,
            # The current request's lexical action signal takes precedence over
            # a previous command.  Otherwise "now dim them" after turning lights
            # on can put turn_on ahead of ha_set_brightness in Laya's small
            # calibrated choice budget.
            key=lambda action: (_action_priority(action, command), 0 if action.intent == prior_intent else 1),
        )

    def _resolve_args(self, action: _Action, command: str, context: Optional[dict]) -> Optional[list[Any]]:
        model_command = self._model_command(command, context)
        prior_target = self._prior_target(context, action.entity_domain or action.optional_entity_domain)
        if action.entity_domain is not None:
            candidates = (
                _entity_candidates(command, action.entity_domain, preferred=prior_target)
                if prior_target
                else _entity_candidates(command, action.entity_domain)
            )
            if not candidates:
                return None
            choice = self._choose(
                model_command,
                f"Which {action.entity_domain} should {action.intent} use?",
                {name: f"Use {name}." for name in candidates},
            )
            if choice not in candidates:
                return None
            args = [candidates[choice]]
            if action.value_candidates is not None:
                values = action.value_candidates(command)
                if not values and isinstance(context, dict) and context.get("intent") == action.intent:
                    prior_value = context.get("value")
                    if isinstance(prior_value, str):
                        values = {prior_value: prior_value}
                if not values:
                    return None
                choice = self._choose(
                    model_command,
                    f"Which value should {action.intent} use?",
                    {name: f"Use {name}." for name in values},
                )
                if choice not in values:
                    return None
                args.append(values[choice])
            return args
        if action.argument_resolver is not None:
            # Generic turn on/off actions can target any HA domain, but a
            # pronoun in an immediate follow-up still has one safe prior target.
            # The generic resolver cannot express that preference itself.
            if action.intent in {"turn_on", "turn_off"} and prior_target:
                candidates = _entity_candidates(command, preferred=prior_target)
                if candidates:
                    choice = self._choose(
                        model_command,
                        f"Which entity should {action.intent} use?",
                        {name: f"Use {name}." for name in candidates},
                    )
                    return [candidates[choice]] if choice in candidates else None
            resolved = action.argument_resolver(command)
            if not resolved:
                return None
            candidates = resolved[0]
            choice = self._choose(
                model_command,
                f"Which value should {action.intent} use?",
                {name: f"Use {name}." for name in candidates},
            )
            return [candidates[choice]] if choice in candidates else None
        if action.optional_entity_domain is not None:
            candidates = (
                _entity_candidates(command, action.optional_entity_domain, preferred=prior_target)
                if prior_target
                else _entity_candidates(command, action.optional_entity_domain)
            )
            if not candidates:
                return None if _has_explicit_media_target(command) else []
            choice = self._choose(
                model_command,
                f"Which media player should {action.intent} use?",
                {name: f"Media player named {name}." for name in candidates},
            )
            return [candidates[choice]] if choice in candidates else None
        return []

    @staticmethod
    def _model_command(command: str, context: Optional[dict]) -> str:
        """Keep one successful prior operation visible without supplying chat history."""
        if not isinstance(context, dict) or not context.get("request") or not context.get("result"):
            return command
        target = f" Target: {context['target']}." if context.get("target") else ""
        value = f" Value: {context['value']}." if context.get("value") else ""
        return (
            f"Previous request: {context['request']}. Action: {context.get('intent', 'unknown')}."
            f"{target}{value} Result: {context['result']}. Current request: {command}"
        )

    @staticmethod
    def _prior_target(context: Optional[dict], domain: Optional[str]) -> Optional[str]:
        """Reuse a target only when the prior operation was in the same domain."""
        if not isinstance(context, dict) or not isinstance(context.get("target"), str):
            return None
        intent = context.get("intent")
        domains = {
            "ha_set_brightness": "light",
            "ha_set_color": "light",
            "ha_volume_set": "media_player",
            "ha_volume_up": "media_player",
            "ha_volume_down": "media_player",
            "pause": "media_player",
            "resume": "media_player",
            "skip": "media_player",
            "previous": "media_player",
        }
        # turn_on/off accept entities from several HA domains, but their target
        # is still an unambiguous and useful light reference for a brightness
        # follow-up such as "now dim them".
        if intent in {"turn_on", "turn_off"} and domain in {None, "light"}:
            return context["target"]
        return context["target"] if domain is not None and domains.get(intent) == domain else None

    def _choose(self, command: str, instructions: str, criteria: dict[str, str]) -> Optional[str]:
        if not criteria:
            return None
        question = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
        question["criteria"]["unsupported"] = "No listed option safely matches this request."
        try:
            result = self._agent.predict({"command": command}, {"route": question})
            answer = result["answers"]["route"]
            confidence = float(answer["confidence"])
            choice = answer["choice"]
        except (KeyError, TypeError, ValueError):
            logger.warning("Laya returned an invalid decision payload")
            return None
        if not math.isfinite(confidence) or confidence < self._confidence_threshold:
            return None
        return choice if isinstance(choice, str) and choice in criteria else None

    @staticmethod
    def _tool_description(intent: str) -> str:
        schema = tool_registry.get_schema(intent)
        return schema.description if schema is not None else intent.replace("_", " ")

def load_laya(
    model_path: str = DEFAULT_MODEL_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    **_opts,
) -> LayaRouter:
    """Load the local Laya checkpoint without importing it during setup mode."""
    try:
        import laya
    except ImportError as exc:  # pragma: no cover - installation is image-specific
        raise RuntimeError("Laya is not installed; rebuild with the current requirements file.") from exc
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("models.llm.confidence_threshold must be between 0 and 1")
    logger.info("Loading Laya semantic command model from %s", model_path)
    return LayaRouter(laya.load(model_path, device="cpu"), confidence_threshold)
