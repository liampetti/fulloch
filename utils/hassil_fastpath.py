"""Conservative Home Assistant sentence matching using installed HassIL packs.

Matching only produces existing tool emissions. All service calls still go
through the HA client, including its live voice deny-list enforcement.
"""

import logging
import math
import re
import threading
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path

from hassil import Intents, recognize, recognize_all
from hassil.intents import TextSlotList, WildcardSlotList

from utils.voice_fastpath import fastpath_pattern as _pattern

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchMetadata:
    """Details belonging to one matched turn, never shared by the matcher."""

    language: str
    intent: str
    response_key: str
    slots: dict
    target: str | None


class FastPathEmission(dict):
    """An ordinary action emission with non-serialized matching details."""

    def __init__(self, action, language, intent, slots):
        super().__init__(actions=[action])
        values = {key: value for key, value in slots.items() if value != _DEFAULT_AREA}
        self.metadata = MatchMetadata(
            language=language,
            intent=intent,
            response_key=intent,
            slots=values,
            target=values.get("name") or values.get("area") or values.get("floor"),
        )


_DEFAULT_AREA = "__fulloch_default_media_target__"
_SATELLITE_AREA = "__fulloch_satellite_area__"
_LOCAL_INTENTS = frozenset({
    "HassGetCurrentTime", "HassGetCurrentDate", "HassGetWeather",
    "HassStartTimer", "HassCancelTimer", "HassTimerStatus",
    "HassMediaPause", "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious",
    "HassMediaSearchAndPlay", "HassSetVolume",
})
_MEDIA_ACTIONS = {
    "HassMediaPause": "pause",
    "HassMediaUnpause": "resume",
    "HassMediaNext": "skip",
    "HassMediaPrevious": "previous",
    "FullochVolumeUp": "ha_volume_up",
    "FullochVolumeDown": "ha_volume_down",
}
_GENERIC_MUSIC = frozenset({
    "music", "musik", "musique", "la musique", "música", "la música", "musica", "muziek",
})
_QUERY_LOCATION = _pattern("hassil_location_guard")
def _supplemental_intents(language, skip_words, supplements, names):
    return Intents.from_dict({
        "language": language,
        "skip_words": skip_words,
        "lists": {"search_query": {"wildcard": True}},
        "intents": {name: {"data": [{"sentences": supplements[name]}]} for name in names},
    })


def _transcript_variants(text):
    # Retry a leading English hesitation that multilingual ASR commonly
    # produces. Do not strip "um" within an utterance (e.g. "um acht Uhr").
    stripped = _pattern("hassil_leading_hesitation").sub("", text)
    return (text, stripped) if stripped != text else (text,)


class HassilFastPath:
    """Immutable set of language tables; entity slots are read live per match."""

    def __init__(self, languages: list[str], intents_dir: str = "./data/intents"):
        from server.downloader import HASSIL_REVISION
        from utils.voice_fastpath import VoiceFastPath, load_hassil_supplements

        self.voice_fastpath = VoiceFastPath(languages, intents_dir)

        tables = []
        for language in languages:
            path = Path(intents_dir) / language / f"{HASSIL_REVISION}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"Intent pack not installed: {language}")
            tables.append((language, Intents.from_files([path])))
        self.tables = tuple(tables)
        supplements_by_language = {
            language: load_hassil_supplements(language, intents_dir) for language, _ in tables
        }
        # Default media sentences in HA's tables expect an area in their
        # conversation context. Use a separate table so that synthetic default
        # context cannot broaden Home Assistant device-control matches.
        local_tables = []
        for language, table in tables:
            supplements = supplements_by_language[language].get("local", {})
            specific = [name for name in supplements if name != "HassMediaSearchAndPlay"]
            if specific:
                local_tables.append((language, _supplemental_intents(language, table.skip_words, supplements, specific)))
            local_tables.append((language, replace(table, intents={name: intent for name, intent in table.intents.items() if name in _LOCAL_INTENTS})))
            if "HassMediaSearchAndPlay" in supplements:
                local_tables.append((language, _supplemental_intents(language, table.skip_words, supplements, ["HassMediaSearchAndPlay"])))
        self.local_tables = tuple(local_tables)
        self.light_level_tables = tuple(
            (language, _supplemental_intents(language, table.skip_words, supplements_by_language[language]["light_level"],
                                              list(supplements_by_language[language]["light_level"])))
            for language, table in tables if supplements_by_language[language].get("light_level")
        )
        self.area_light_tables = tuple(
            (language, _supplemental_intents(language, table.skip_words, supplements_by_language[language]["area_light"],
                                             list(supplements_by_language[language]["area_light"])))
            for language, table in tables if supplements_by_language[language].get("area_light")
        )
        self._discovery_lock = threading.Lock()
        self._discovery_thread = None

    def start_discovery(self):
        """Start shared HA discovery without blocking voice or dashboard turns."""
        from tools import ha_client

        if not ha_client.HA_TOKEN or ha_client._loaded:
            return
        with self._discovery_lock:
            if self._discovery_thread and self._discovery_thread.is_alive():
                return
            self._discovery_thread = threading.Thread(
                target=ha_client._ensure_loaded, daemon=True, name="ha-intent-discovery"
            )
            self._discovery_thread.start()

    @staticmethod
    def _slots(client):
        # Duplicate aliases are ambiguous even when one of the entities is
        # blocked: never silently redirect a request to a different device.
        aliases = dict(client._ENTITY_ALIASES_MULTI)
        for name, eid in client._ENTITY_ALIASES.items():
            aliases.setdefault(name, [eid])
        denied = client.get_denylist()
        names = []
        for name, ids in aliases.items():
            ids = list(dict.fromkeys(ids))
            if len(ids) != 1 or ids[0] in denied or not name.strip():
                continue
            eid = ids[0]
            names.append((name, eid, {"domain": client._domain_of(eid)}))
        # Preserve the existing per-satellite default-room lights behavior.
        # Never replace a real (including blocked/ambiguous) alias named lights.
        from core.satellite_context import current_satellite_ha_area

        if current_satellite_ha_area.get() and not any(name.casefold() == "lights" for name in aliases):
            names.append(("lights", "@satellite-lights", {"domain": "light"}))
        def unique_locations(mapping):
            counts = {}
            for name in mapping.values():
                if name:
                    counts[name.casefold()] = counts.get(name.casefold(), 0) + 1
            return [(name, key) for key, name in mapping.items() if name and counts[name.casefold()] == 1]

        areas = unique_locations(client._AREA_MAP)
        floors = unique_locations(client._FLOOR_MAP)
        return {
            "name": TextSlotList.from_tuples(names, allow_template=False),
            "area": TextSlotList.from_tuples(areas, allow_template=False),
            "floor": TextSlotList.from_tuples(floors, allow_template=False),
        }

    def match(self, text: str):
        """Return a safe tool emission or None; never perform device actions."""
        from tools import ha_client as client

        # The shared discovery cache is populated asynchronously at startup.
        # Never stall unrelated turns on HA's retry loop.
        loaded = client._loaded
        if not loaded:
            self.start_discovery()
        slots = self._slots(client) if loaded else {
            name: TextSlotList.from_strings([]) for name in ("name", "area", "floor")
        }
        # Timer names and delayed commands cannot be mapped safely onto the
        # local timer-ID interface. Exclude their wildcard alternatives rather
        # than accidentally treating articles as names or running a command now.
        local_slots = {**slots, **{
            name: TextSlotList.from_strings([]) for name in ("timer_name", "timer_command")
        }}
        local_slots["search_query"] = WildcardSlotList(name="search_query")
        aliases = dict(client._ENTITY_ALIASES_MULTI) if loaded else {}
        if loaded:
            for alias, eid in client._ENTITY_ALIASES.items():
                aliases.setdefault(alias, [eid])
        denied = client.get_denylist() if loaded else set()
        blocked = any(
            (len(set(ids)) > 1 or any(eid in denied for eid in ids)) and alias.strip()
            and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
            for alias, ids in aliases.items()
        )
        local = self._match_local(text, local_slots, blocked=blocked)
        if local is not None:
            return local
        if not loaded:
            return None
        # Do not reinterpret an explicitly named blocked or ambiguous entity as
        # a broader area command through another sentence pattern.
        if blocked:
            return None
        for language, table in self.area_light_tables:
            for variant in _transcript_variants(text):
                try:
                    result = recognize(variant, table, slot_lists=slots)
                except Exception:  # noqa: BLE001
                    logger.exception("HassIL area light matching failed for %s", language)
                    continue
                if result:
                    area = result.entities["area"].value
                    operation, value, intent = {
                        "FullochAreaLightsOn": ("on", None, "HassTurnOn"),
                        "FullochAreaLightsOff": ("off", None, "HassTurnOff"),
                        "FullochAreaLightsDim": ("brightness", 30, "HassLightSet"),
                        "FullochAreaLightsBrighten": ("brightness", 100, "HassLightSet"),
                    }[result.intent.name]
                    values = {"area": area, "domain": "light"}
                    if value is not None:
                        values["brightness"] = value
                    return FastPathEmission(
                        {"intent": "ha_hassil_area", "args": ["area", area, "light", operation, value]},
                        language, intent, values,
                    )
        for language, table in self.light_level_tables:
            for variant in _transcript_variants(text):
                try:
                    result = recognize(variant, table, slot_lists=slots)
                except Exception:  # noqa: BLE001 — a bad pack must not break turns
                    logger.exception("HassIL light level matching failed for %s", language)
                    continue
                if result:
                    values = {key: entity.value for key, entity in result.entities.items()}
                    target = values.get("name", "lights")
                    if target != "lights" and (not isinstance(target, str) or not target.startswith("light.")):
                        return None
                    level = 30 if result.intent.name == "FullochDimLights" else 100
                    return FastPathEmission(
                        {"intent": "ha_set_brightness", "args": [target, level]},
                        language, "HassLightSet", {"name": target, "brightness": level},
                    )
        # HA's language packs contain room-local light sentences that require
        # an area in conversation context. Supply a sentinel, not a real area:
        # only accept matches which explicitly copy it into the area slot, and
        # let the existing light tool resolve the current room safely.
        from core.satellite_context import current_satellite_ha_area

        if current_satellite_ha_area.get():
            # A real entity alias named "light" must not capture a bare
            # room-local command before HA's generic light sentence can match.
            # Keep area/floor slots, but omit entity names only for this probe;
            # explicit names still use the normal matcher below.
            room_slots = {**slots, "name": TextSlotList.from_strings([])}
            for language, table in self.tables:
                for variant in _transcript_variants(text):
                    try:
                        result = recognize(variant, table, slot_lists=room_slots, intent_context={"area": _SATELLITE_AREA})
                        if result:
                            values = {key: entity.value for key, entity in result.entities.items()}
                            if (values.get("area") == _SATELLITE_AREA
                                    and values.get("domain") == "light"
                                    and set(values) == {"area", "domain"}
                                    and result.intent.name in {"HassTurnOn", "HassTurnOff"}):
                                action = {"intent": "turn_on" if result.intent.name == "HassTurnOn" else "turn_off", "args": ["lights"]}
                                return FastPathEmission(action, language, result.intent.name, {"name": "@satellite-lights"})
                            if (values.get("area") == _SATELLITE_AREA
                                    and values.get("domain") == "light"
                                    and set(values) == {"area", "domain", "brightness"}
                                    and result.intent.name == "HassLightSet"):
                                action = {"intent": "ha_set_brightness", "args": ["lights", values["brightness"]]}
                                return FastPathEmission(action, language, result.intent.name,
                                                        {"name": "@satellite-lights", "brightness": values["brightness"]})
                    except Exception:  # noqa: BLE001
                        logger.exception("HassIL room matching failed for %s", language)
        for language, table in self.tables:
            for variant in _transcript_variants(text):
                try:
                    result = recognize(variant, table, slot_lists=slots)
                    if result:
                        values = {key: entity.value for key, entity in result.entities.items()}
                        action = self._action(result.intent.name, values)
                        if action:
                            return FastPathEmission(action, language, result.intent.name, values)
                        return None  # Never broaden an unsupported or ambiguous match.
                except Exception:  # noqa: BLE001 — a bad pack must not break turns
                    logger.exception("HassIL matching failed for %s", language)
        return None

    def _match_local(self, text, slots, *, blocked=False):
        """Match local commands without requiring HA discovery or a room."""
        for language, table in self.local_tables:
            for variant in _transcript_variants(text):
                try:
                    first = None
                    for result in recognize_all(variant, table, slot_lists=slots, intent_context={"area": _DEFAULT_AREA}):
                        values = {key: entity.value for key, entity in result.entities.items()}
                        action = self._local_action(result.intent.name, values)
                        if not action:
                            continue
                        if blocked and (result.intent.name in _MEDIA_ACTIONS or result.intent.name in {"HassMediaSearchAndPlay", "HassSetVolume"}):
                            return None
                        # Prefer a specified player/area over a wildcard search
                        # that swallowed the same location as part of the query.
                        if result.intent.name == "HassMediaSearchAndPlay":
                            if values.get("name") or values.get("area") not in (None, _DEFAULT_AREA):
                                return FastPathEmission(action, language, result.intent.name, values)
                            # If a location-like phrase was swallowed by the
                            # wildcard, don't play it on the wrong default player.
                            if _QUERY_LOCATION.search(values.get("search_query", "")):
                                continue
                            first = first or FastPathEmission(action, language, result.intent.name, values)
                            continue
                        return FastPathEmission(action, language, result.intent.name, values)
                    if first:
                        return first
                except Exception:  # noqa: BLE001 — a bad language pack must not break turns
                    logger.exception("HassIL local matching failed for %s", language)
        return None

    def response_language_confirmed(self, text: str, emission) -> bool:
        """Conservatively detect competing language matches for supported replies.

        This does not select or execute actions and never changes routing order.
        Rechecking the chosen match binds the decision to this utterance and
        current entity cache. A failed check or incomplete HA discovery in a
        multilingual installation leaves the response language unconfirmed.
        """
        from tools import ha_client as client

        metadata = getattr(emission, "metadata", None)
        if not isinstance(metadata, MatchMetadata) or metadata.intent not in {
            "HassGetCurrentTime", "HassGetCurrentDate", "HassStartTimer", "HassCancelTimer",
            "HassTurnOn", "HassTurnOff", "HassTimerStatus", "HassLightSet",
            "HassClimateSetTemperature", "HassSetPosition", "HassSetVolume",
        }:
            return False
        try:
            current = self.match(text)
            if current != emission or getattr(current, "metadata", None) != metadata:
                from core.satellite_context import current_satellite_ha_area

                current = self.voice_fastpath.command(text, current_satellite_ha_area.get(), self)
            if current != emission or getattr(current, "metadata", None) != metadata:
                return False
            languages = {language for language, _ in self.tables}
            if metadata.language not in languages:
                return False
            if len(languages) == 1:
                return True
            if not client._loaded:
                return False
            slots = self._slots(client)
            local_slots = {**slots, **{
                name: TextSlotList.from_strings([]) for name in ("timer_name", "timer_command")
            }}
            local_slots["search_query"] = WildcardSlotList(name="search_query")
            for language, table in self.local_tables:
                if language == metadata.language:
                    continue
                for variant in _transcript_variants(text):
                    if next(recognize_all(variant, table, slot_lists=local_slots,
                                          intent_context={"area": _DEFAULT_AREA}), None):
                        return False
            for language, table in self.tables:
                if language == metadata.language:
                    continue
                for variant in _transcript_variants(text):
                    if recognize(variant, table, slot_lists=slots):
                        return False
        except Exception:  # noqa: BLE001 — uncertainty must never confirm a language
            logger.exception("Could not determine fast-path response language")
            return False
        return True

    @staticmethod
    def _local_action(intent: str, slots: dict):
        values = dict(slots)
        if values.get("area") == _DEFAULT_AREA:
            values.pop("area")
        if intent in {"HassGetCurrentTime", "HassGetCurrentDate"} and not values:
            return {"intent": "get_current_time", "args": []}
        if intent == "HassGetWeather" and not values:
            return {"intent": "get_weather_forecast", "args": []}
        if intent == "HassStartTimer" and values and set(values) <= {"hours", "minutes", "seconds"}:
            total = 0
            for unit, factor in (("hours", 3600), ("minutes", 60), ("seconds", 1)):
                value = values.get(unit, 0)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 or int(value) != value:
                    return None
                total += int(value) * factor
            if 0 < total <= 7 * 24 * 60 * 60:
                return {"intent": "start_countdown", "args": [f"{total} seconds"]}
        if intent in {"HassCancelTimer", "HassTimerStatus"} and not values:
            return {"intent": "cancel_timer" if intent == "HassCancelTimer" else "get_timer_status", "args": []}
        if intent in _MEDIA_ACTIONS or intent in {"HassMediaSearchAndPlay", "HassSetVolume"}:
            name, area = values.get("name"), values.get("area")
            if (name and area) or (name and (not isinstance(name, str) or not name.startswith("media_player."))):
                return None
            target = name or area
            if intent in _MEDIA_ACTIONS and set(values) <= {"name", "area"}:
                return {"intent": _MEDIA_ACTIONS[intent], "args": [target] if target else []}
            if intent == "HassSetVolume" and set(values) <= {"name", "area", "volume_level"} and "volume_level" in values:
                return {"intent": "ha_volume_set", "args": [target, values["volume_level"]]}
            if intent == "HassMediaSearchAndPlay" and set(values) <= {"name", "area", "search_query", "media_class"}:
                query, media_class = values.get("search_query"), values.get("media_class")
                if not isinstance(query, str) or not query.strip() or media_class not in (None, "artist", "album", "track", "playlist"):
                    return None
                query = query.strip()
                if query.casefold() in _GENERIC_MUSIC:
                    query = "music"
                return {"intent": "play_song", "args": [query, None, target] if target else [query]}
        return None

    @staticmethod
    def _action(intent: str, slots: dict):
        name, area, floor = slots.get("name"), slots.get("area"), slots.get("floor")
        permitted = {
            "HassTurnOn": {"name", "area", "floor", "domain"},
            "HassTurnOff": {"name", "area", "floor", "domain"},
            "HassLightSet": {"name", "area", "floor", "domain", "brightness", "color"},
            "HassClimateSetTemperature": {"name", "area", "floor", "domain", "temperature"},
            "HassSetPosition": {"name", "domain", "position"},
        }
        if intent not in permitted or set(slots) - permitted[intent]:
            return None
        # Disambiguating a named entity by its room needs registry membership;
        # do not guess or silently ignore any stated location.
        if (name and (area or floor)) or (area and floor):
            return None
        if name:
            if name == "@satellite-lights":
                if intent in {"HassTurnOn", "HassTurnOff"}:
                    return {"intent": "turn_on" if intent == "HassTurnOn" else "turn_off", "args": ["lights"]}
                if intent == "HassLightSet" and "brightness" in slots and "color" not in slots:
                    return {"intent": "ha_set_brightness", "args": ["lights", slots["brightness"]]}
                return None
            if not isinstance(name, str) or "." not in name:
                return None
            domain = name.split(".", 1)[0]
            if slots.get("domain") and slots["domain"] != domain:
                return None
            if intent in {"HassTurnOn", "HassTurnOff"}:
                on = intent == "HassTurnOn"
                if domain == "lock":
                    tool = "ha_lock" if on else "ha_unlock"
                elif domain in {"cover", "valve"}:
                    tool = "ha_open_cover" if on else "ha_close_cover"
                elif domain in {"light", "switch", "fan", "input_boolean", "media_player"} or (on and domain in {"scene", "script"}):
                    tool = "turn_on" if on else "turn_off"
                else:
                    return None
                return {"intent": tool, "args": [name]}
            if intent == "HassLightSet" and domain == "light":
                if "brightness" in slots and "color" not in slots:
                    return {"intent": "ha_set_brightness", "args": [name, slots["brightness"]]}
                if "color" in slots and "brightness" not in slots:
                    return {"intent": "ha_set_color", "args": [name, slots["color"]]}
            if intent == "HassClimateSetTemperature" and domain == "climate" and "temperature" in slots:
                return {"intent": "ha_set_climate", "args": [name, slots["temperature"]]}
            if intent == "HassSetPosition" and domain in {"cover", "valve"} and "position" in slots:
                return {"intent": "ha_set_cover_position", "args": [name, slots["position"]]}
            return None
        if area or floor:
            domain = slots.get("domain")
            if intent in {"HassTurnOn", "HassTurnOff"} and domain in {"light", "switch", "fan"}:
                return {"intent": "ha_hassil_area", "args": ["area" if area else "floor", area or floor, domain, "on" if intent == "HassTurnOn" else "off", None]}
            if intent == "HassLightSet" and "brightness" in slots and "color" not in slots:
                if domain not in (None, "light"):
                    return None
                return {"intent": "ha_hassil_area", "args": ["area" if area else "floor", area or floor, "light", "brightness", slots["brightness"]]}
            if intent == "HassClimateSetTemperature" and "temperature" in slots:
                if domain not in (None, "climate"):
                    return None
                return {"intent": "ha_hassil_area", "args": ["area" if area else "floor", area or floor, "climate", "temperature", slots["temperature"]]}
        return None
