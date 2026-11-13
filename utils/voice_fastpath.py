"""Language-pack voice repairs and verified light follow-ups."""

import logging
import math
import re
import time
from functools import lru_cache
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

logger = logging.getLogger(__name__)
_BUNDLED = Path(__file__).resolve().parent.parent / "data" / "intents"


def fastpath_section(section, *, language="en", directory="./data/intents"):
    """Load a named section for a selected language, respecting local edits."""
    for root in (Path(directory), _BUNDLED, Path("/app/seed/intents")):
        path = root / language / "fulloch-fastpath.yaml"
        if path.is_file():
            try:
                data = YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {}
                if section in data:
                    return data[section]
            except (OSError, AttributeError, TypeError, YAMLError, re.error):
                logger.exception("Invalid fast-path rules: %s", path)
    return None


@lru_cache(maxsize=None)
def fastpath_pattern(name, flags=re.IGNORECASE, *, language="en", directory="./data/intents"):
    """Compile a named pattern from the selected language's fast-path pack."""
    patterns = fastpath_section("patterns", language=language, directory=directory) or {}
    if name not in patterns:
        # An older customized pack may not contain keys introduced by an
        # upgrade; use the bundled default without overwriting the user's file.
        patterns = fastpath_section("patterns", language=language, directory="/app/seed/intents") or {}
    if name not in patterns:
        raise KeyError(f"Missing fast-path pattern: {name}")
    return re.compile(patterns[name], flags)


def contextual_followup(command, context):
    """Preserve the existing deterministic contextual tool routes."""
    from tools.tool_registry import tool_registry

    if not isinstance(context, dict):
        return None
    target, prior = context.get("target"), context.get("intent")
    if not isinstance(target, str) or not target:
        return None

    def emission(intent, args):
        return {"actions": [{"intent": intent, "args": args}]} if tool_registry.is_available(intent) else None

    if prior in {"turn_on", "turn_off", "ha_set_brightness", "ha_set_color"}:
        if match := fastpath_pattern("laya_context_light").match(command):
            return emission(f"turn_{match.group('state').lower()}", [target])
        if match := fastpath_pattern("laya_context_brightness").match(command):
            return emission("ha_set_brightness", [target, 30 if match.group("verb").lower() == "dim" else 100])
    if prior in {"pause", "resume", "skip", "previous", "ha_volume_set", "ha_volume_up", "ha_volume_down"}:
        if match := fastpath_pattern("laya_context_volume").match(command):
            intent = "ha_volume_up" if match.group("direction").lower() == "up" else "ha_volume_down"
            return emission(intent, [target])
        if match := fastpath_pattern("laya_context_transport").match(command):
            return emission(match.group("action").lower(), [target])
    if prior in {"ha_open_cover", "ha_close_cover", "ha_stop_cover", "ha_set_cover_position"}:
        if match := fastpath_pattern("laya_context_cover").match(command):
            action = match.group("action").lower()
            intent = {"open": "ha_open_cover", "close": "ha_close_cover", "stop": "ha_stop_cover", "halt": "ha_stop_cover"}[action]
            return emission(intent, [target])
    if prior in {"ha_lock", "ha_unlock"}:
        if match := fastpath_pattern("laya_context_lock").match(command):
            return emission(f"ha_{match.group('action').lower()}", [target])
    return None


def load_hassil_supplements(language, directory):
    """Load existing HassIL sentences from the language's fast-path file."""
    for root in (Path(directory), _BUNDLED, Path("/app/seed/intents")):
        path = root / language / "fulloch-fastpath.yaml"
        if path.is_file():
            try:
                value = YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {}
                if not isinstance(value, dict):
                    raise ValueError("fast-path file must be a mapping")
                if "hassil" in value:
                    if not isinstance(value["hassil"], dict):
                        raise ValueError("hassil section must be a mapping")
                    return value["hassil"]
            except (OSError, ValueError, TypeError, YAMLError):
                logger.exception("Invalid fast-path file: %s", path)
                return {}
    return {}


class ContextualEmission(dict):
    """A full, explicitly matched follow-up, including its ASR variants."""

    complete_followup = True


class VoiceFastPath:
    def __init__(self, languages, directory):
        self.rules = []
        self.commands = []
        for language in languages:
            path = Path(directory) / language / "fulloch-fastpath.yaml"
            if not path.is_file():
                path = _BUNDLED / language / "fulloch-fastpath.yaml"
            if not path.is_file():
                continue
            try:
                data = YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {}
                corrections = [(re.compile(r["pattern"], re.IGNORECASE), r["replacement"])
                               for r in data.get("corrections", [])]
                followups = [(re.compile(r["pattern"], re.IGNORECASE), r["operation"], r.get("value"))
                             for r in data.get("followups", [])]
                holds = [re.compile(pattern, re.IGNORECASE) for pattern in data.get("hold_partial", [])]
                self.rules.append((language, corrections, followups, holds))
                for item in data.get("commands", []):
                    if not item.get("enabled", True):
                        continue
                    pattern = item.get("pattern")
                    if pattern is None:
                        pattern = data["patterns"][item["pattern_key"]]
                    self.commands.append((language, re.compile(pattern, re.IGNORECASE),
                                          item["operation"], item.get("value")))
            except (OSError, ValueError, KeyError, TypeError, re.error, YAMLError):
                logger.exception("Invalid voice fast-path rules: %s", path)

    def command(self, text, area, matcher=None):
        """Resolve pack-defined commands through a bounded canonical operation set."""
        from tools import ha_client

        candidates = []
        for language, pattern, operation, configured_value in self.commands:
            match = pattern.fullmatch(text)
            if not match:
                continue
            groups = match.groupdict()
            simple = {
                "time": ("get_current_time", []), "date": ("get_current_time", []),
                "weather": ("get_weather_forecast", []),
                "pause": ("pause", []), "resume": ("resume", []), "skip": ("skip", []),
                "volume_up": ("ha_volume_up", []), "volume_down": ("ha_volume_down", []),
                "timer_status": ("get_timer_status", []), "timer_cancel": ("cancel_timer", []),
            }
            if operation in simple:
                intent, args = simple[operation]
            elif operation in {"play", "timer"}:
                # HassIL normalizes multilingual durations and validates media
                # target/location qualifiers; never pass raw words to English
                # duration parsing or discard an explicit player/room.
                resolved = matcher.match(text) if matcher is not None else None
                actions = resolved.get("actions", []) if isinstance(resolved, dict) else []
                expected = "play_song" if operation == "play" else "start_countdown"
                if len(actions) != 1 or actions[0].get("intent") != expected:
                    continue
                candidates.append((language, actions[0], operation, resolved))
                continue
            elif operation == "deep_think":
                topic = groups.get("topic")
                if not topic or not topic.strip():
                    continue
                intent, args = "deep_think", [topic.strip()]
            elif operation == "satellite_message":
                target, message = groups.get("target"), groups.get("message")
                if not target or not message or not target.strip() or not message.strip():
                    continue
                intent, args = "send_satellite_message", [target.strip(), message.strip()]
            elif operation == "currency":
                mapping = configured_value if isinstance(configured_value, dict) else {}
                base, quote = groups.get("base"), groups.get("quote")
                if not base or not quote:
                    continue
                currencies = {str(k).casefold(): v for k, v in mapping.items()}
                base = currencies.get(base.strip().casefold(), base.strip())
                quote = currencies.get(quote.strip().casefold(), quote.strip())
                if not isinstance(base, str) or not isinstance(quote, str):
                    continue
                base, quote = base.upper(), quote.upper()
                if not re.fullmatch(r"[A-Z]{3}", base) or not re.fullmatch(r"[A-Z]{3}", quote):
                    continue
                try:
                    amount = float(groups["amount"].replace(",", ".")) if groups.get("amount") else None
                except ValueError:
                    continue
                if amount is not None and (not math.isfinite(amount) or amount < 0):
                    continue
                intent, args = "get_exchange_rate", [base, quote, amount]
            elif operation in {"area_lights_on", "area_lights_off"}:
                location = groups.get("area")
                if not ha_client._loaded or not location:
                    continue
                areas = [name for name in ha_client._AREA_MAP.values()
                         if name and name.casefold() == location.strip().casefold()]
                if len(areas) != 1:
                    continue
                intent = "get_entities_in_area_state"
                args = [areas[0], "light", "on" if operation == "area_lights_on" else "off"]
            elif operation in {"lock", "unlock", "cover_open", "cover_close", "toggle", "light_color"}:
                if not ha_client._loaded or not groups.get("name"):
                    continue
                aliases = dict(ha_client._ENTITY_ALIASES_MULTI)
                for alias, eid in ha_client._ENTITY_ALIASES.items():
                    aliases.setdefault(alias, [eid])
                ids = {eid for alias, eids in aliases.items()
                       if alias.casefold() == groups["name"].strip().casefold() for eid in eids}
                if len(ids) != 1:
                    continue
                target = next(iter(ids))
                domain = target.split(".", 1)[0]
                allowed_domains = ({"lock"} if operation in {"lock", "unlock"} else
                                   {"cover"} if operation in {"cover_open", "cover_close"} else
                                   {"light"} if operation == "light_color" else {"light", "switch"})
                if domain not in allowed_domains or target in ha_client.get_denylist():
                    continue
                intent = {"lock": "ha_lock", "unlock": "ha_unlock", "cover_open": "ha_open_cover",
                          "cover_close": "ha_close_cover", "toggle": "toggle", "light_color": "ha_set_color"}[operation]
                args = [target]
                if operation == "light_color":
                    # Colors are canonical tool values supplied by the pack,
                    # never arbitrary language-specific text passed unchecked.
                    allowed = {"red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "warm white", "cool white", "cyan", "magenta"}
                    if not isinstance(configured_value, str) or configured_value not in allowed:
                        continue
                    args.append(configured_value)
            elif operation in {"light_on", "light_off", "light_dim", "light_brighten", "light_brightness"}:
                if not ha_client._loaded:
                    continue
                aliases = dict(ha_client._ENTITY_ALIASES_MULTI)
                for alias, eid in ha_client._ENTITY_ALIASES.items():
                    aliases.setdefault(alias, [eid])
                denied = ha_client.get_denylist()
                if any(alias.strip() and (len(set(ids)) > 1 or any(eid in denied for eid in ids))
                       and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
                       for alias, ids in aliases.items()):
                    continue
                location = groups.get("area")
                name = groups.get("name")
                if location and name:
                    continue
                level = configured_value if configured_value is not None else (30 if operation == "light_dim" else 100)
                if operation == "light_brightness":
                    try:
                        level = int(groups.get("value", ""))
                    except (TypeError, ValueError):
                        continue
                if type(level) is not int or not 0 <= level <= 100:
                    continue
                state = "on" if operation == "light_on" else "off"
                brightness = operation in {"light_dim", "light_brighten", "light_brightness"}
                if location:
                    areas = [key for key, value in ha_client._AREA_MAP.items()
                             if value and value.casefold() == location.strip().casefold()]
                    if len(areas) != 1:
                        continue
                    intent = "ha_hassil_area"
                    args = ["area", areas[0], "light", "brightness" if brightness else state,
                            level if brightness else None]
                elif name:
                    ids = {eid for alias, eids in aliases.items() if alias.casefold() == name.strip().casefold()
                           for eid in eids}
                    if len(ids) != 1:
                        continue
                    target = next(iter(ids))
                    if not target.startswith("light.") or target in denied:
                        continue
                    intent = "ha_set_brightness" if brightness else f"turn_{state}"
                    args = [target, level] if brightness else [target]
                elif area:
                    intent = "ha_set_brightness" if brightness else f"turn_{state}"
                    args = ["lights", level] if brightness else ["lights"]
                else:
                    continue
            else:
                logger.warning("Unsupported fast-path operation %r for %s", operation, language)
                continue
            candidates.append((language, {"intent": intent, "args": args}, operation, None))
        if not candidates or any(action != candidates[0][1] for _, action, _, _ in candidates[1:]):
            return None
        language, action, operation, emission = candidates[0]
        if emission is not None:
            return emission
        if operation in {"time", "date"}:
            if any(other != language or other_operation != operation
                   for other, _, other_operation, _ in candidates[1:]):
                return None
            from utils.hassil_fastpath import FastPathEmission

            return FastPathEmission(action, language,
                                    "HassGetCurrentDate" if operation == "date" else "HassGetCurrentTime", {})
        return {"actions": [action]}

    def corrected_match(self, text, matcher, *, area=None):
        """Try only unique repairs that resolve through the normal HA matcher."""
        from tools import ha_client

        candidates = []
        aliases = {**ha_client._ENTITY_ALIASES, **ha_client._ENTITY_ALIASES_MULTI}
        for _, corrections, _, _ in self.rules:
            for pattern, replacement in corrections:
                match = pattern.fullmatch(text)
                if not match:
                    continue
                word = match.groupdict().get("word")
                if word:
                    start, end = match.span("word")
                    if any(
                        found and found.start() <= start and found.end() >= end
                        for alias in aliases if alias.strip()
                        for found in [re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)]
                    ):
                        continue
                try:
                    repaired = match.expand(replacement)
                except re.error:
                    logger.warning("Invalid voice fast-path replacement for %s", pattern.pattern)
                    continue
                emission = self.command(repaired, area, matcher) or matcher.match(repaired)
                if emission is not None:
                    candidates.append((repaired, emission))
        if not candidates or any(item[1] != candidates[0][1] for item in candidates[1:]):
            return None
        logger.debug("Voice fast-path repair: %r -> %r", text, candidates[0][0])
        return candidates[0][1]

    def followup(self, text, context, area):
        if (not context or context.get("satellite_area") != area
                or time.monotonic() - context.get("time", 0) > 1800):
            return None
        candidates = []
        for _, _, rules, _ in self.rules:
            for pattern, operation, configured_value in rules:
                match = pattern.fullmatch(text)
                if not match:
                    continue
                value = configured_value
                if value == "capture":
                    try:
                        value = int(match.group("value"))
                    except (IndexError, TypeError, ValueError):
                        continue
                if operation not in {"on", "off", "brightness", "brightness_step"}:
                    continue
                if operation == "brightness" and (type(value) is not int or not 0 <= value <= 100):
                    continue
                if operation == "brightness_step" and (type(value) is not int or not -100 <= value <= 100 or not value):
                    continue
                if context.get("scope") == "area":
                    action = {"intent": "ha_hassil_area", "args": ["area", context["target"], "light", operation, value]}
                elif context.get("scope") == "entity":
                    intent = {"brightness": "ha_set_brightness", "brightness_step": "ha_adjust_light_brightness"}.get(operation, f"turn_{operation}")
                    action = {"intent": intent,
                              "args": [context["target"], value] if operation in {"brightness", "brightness_step"} else [context["target"]]}
                else:
                    continue
                candidates.append(action)
        if not candidates or any(action != candidates[0] for action in candidates[1:]):
            return None
        return ContextualEmission(actions=[candidates[0]])

    def hold_partial(self, text):
        from tools import ha_client

        for _, _, _, patterns in self.rules:
            for pattern in patterns:
                match = pattern.fullmatch(text)
                area = match.groupdict().get("area") if match else None
                if area and any(area.casefold() == name.casefold() for name in ha_client._AREA_MAP.values()):
                    return True
        return False
