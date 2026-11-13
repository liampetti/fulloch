"""Conservative, per-satellite repair for incomplete light requests."""

import re
import time

from utils.voice_fastpath import fastpath_section


def _room_context(text, packs, satellite_area):
    from tools import ha_client

    if not ha_client._loaded:
        return None
    names = {match.group("room").strip().casefold() for pack in packs
             if pack.get("room") for match in [re.fullmatch(pack["room"], text, re.I)] if match}
    areas = {key for key, name in ha_client._AREA_MAP.items() if name and name.casefold() in names}
    if len(areas) != 1:
        return None
    return {"scope": "area", "target": areas.pop(), "satellite_area": satellite_area, "time": time.monotonic()}


def clarify_light(satellite, text, area, languages, directory):
    """Return (reply, emission), or (None, None) for normal routing.

    Only exact, incomplete requests and exact answers are interpreted. Never
    reinterpret a negation or a guessed phonetic verb as a device command.
    """
    if satellite is None:
        return None, None
    packs = [fastpath_section("light_clarification", language=language, directory=directory) or {}
             for language in languages]
    pending = getattr(satellite, "pending_light_clarification", None)
    satellite.pending_light_clarification = None
    if pending and time.monotonic() - pending["time"] < 30 and pending["area"] == area:
        states = {state for pack in packs for state in ("on", "off")
                  if pack.get(state) and re.fullmatch(pack[state], text, re.I)}
        context = pending.get("context")
        state = states.pop() if len(states) == 1 else None
        room = _room_context(text, packs, area) if not states and state is None else None
        if room:
            context = room
            state = pending.get("state")
            if not state:
                pending.update(context=context, time=time.monotonic())
                satellite.pending_light_clarification = pending
                return pending["question"], None
        if state:
            if context and context.get("scope") == "area":
                action = {"intent": "ha_hassil_area", "args": ["area", context["target"], "light", state, None]}
            elif context and context.get("scope") == "entity":
                action = {"intent": f"turn_{state}", "args": [context["target"]]}
            elif area:
                action = {"intent": f"turn_{state}", "args": ["lights"]}
            else:
                pending.update(state=state, time=time.monotonic())
                satellite.pending_light_clarification = pending
                return pending["missing_room"], None
            return None, {"actions": [action]}
    matching = [pack for pack in packs if pack.get("bare") and re.fullmatch(pack["bare"], text, re.I)]
    if not matching:
        return None, None
    context = getattr(satellite, "fastpath_light_context", None)
    if not context or context.get("satellite_area") != area or time.monotonic() - context.get("time", 0) > 1800:
        context = None
    pack = matching[0]
    satellite.pending_light_clarification = {"time": time.monotonic(), "area": area, "context": context,
                                             "missing_room": pack.get("missing_room", ""),
                                             "question": pack.get("question", "")}
    return pack.get("question", ""), None
