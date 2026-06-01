"""Fulloch Home Assistant integration.

Connects to a running Fulloch dashboard (the optional FastAPI server) and
exposes:
  - sensor.fulloch_status / sensor.fulloch_last_utterance / sensor.fulloch_last_response
  - switch.fulloch_mic
  - fulloch.speak / fulloch.chat services
  - notify.fulloch notification target
  - fulloch_wakeword_detected / fulloch_turn_ended HA events (via SSE)
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import aiohttp_client

from .const import (
    CONF_HOST,
    CONF_PORT,
    DOMAIN,
    EVENT_TURN_ENDED,
    EVENT_WAKEWORD_DETECTED,
    PLATFORMS,
)
from .coordinator import FullochCoordinator

logger = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    host = entry.data[CONF_HOST]
    port = entry.data[CONF_PORT]
    url = f"http://{host}:{port}"

    session = aiohttp_client.async_get_clientsession(hass)
    coordinator = FullochCoordinator(hass, session, url)

    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as exc:
        raise ConfigEntryNotReady(f"Cannot reach Fulloch at {url}") from exc

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
        "session": session,
        "url": url,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Services ---------------------------------------------------------------

    async def _post(path: str, payload: dict) -> None:
        try:
            async with session.post(
                f"{url}{path}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ):
                pass
        except aiohttp.ClientError as exc:
            logger.warning("Fulloch service call failed: %s", exc)

    async def handle_speak(call: ServiceCall) -> None:
        await _post("/speak", {"text": call.data["text"]})

    async def handle_chat(call: ServiceCall) -> None:
        # Run the full agent loop, then speak the result — this is the
        # intended path for HA automations (morning briefing, reminders, etc.)
        try:
            async with session.post(
                f"{url}/chat",
                json={"text": call.data["text"]},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                answer = data.get("answer", "")
        except aiohttp.ClientError as exc:
            logger.warning("Fulloch chat call failed: %s", exc)
            return
        if answer:
            await _post("/speak", {"text": answer})

    async def handle_mic(call: ServiceCall) -> None:
        await _post("/mic", {"enabled": call.data["enabled"]})
        await coordinator.async_refresh()

    hass.services.async_register(
        DOMAIN,
        "speak",
        handle_speak,
        schema=vol.Schema({vol.Required("text"): str}),
    )
    hass.services.async_register(
        DOMAIN,
        "chat",
        handle_chat,
        schema=vol.Schema({vol.Required("text"): str}),
    )
    hass.services.async_register(
        DOMAIN,
        "mic",
        handle_mic,
        schema=vol.Schema({vol.Required("enabled"): bool}),
    )

    # SSE listener — fires HA events and refreshes coordinator on turns ------
    sse_task = entry.async_create_background_task(
        hass,
        _sse_listener(hass, coordinator, session, url),
        "fulloch_sse_listener",
    )
    hass.data[DOMAIN][entry.entry_id]["sse_task"] = sse_task

    return True


async def _sse_listener(
    hass: HomeAssistant,
    coordinator: FullochCoordinator,
    session: aiohttp.ClientSession,
    url: str,
) -> None:
    """Stream /stream SSE events and fire HA bus events."""
    while True:
        try:
            async with session.get(
                f"{url}/stream",
                timeout=aiohttp.ClientTimeout(total=None, connect=10),
            ) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except ValueError:
                        continue

                    role = data.get("role")
                    source = data.get("source", "")

                    if role == "user":
                        hass.bus.async_fire(
                            EVENT_WAKEWORD_DETECTED,
                            {"utterance": data.get("content", ""), "source": source},
                        )
                        # Update coordinator data in-place so last_utterance is
                        # current without waiting for the next 30s poll.
                        if coordinator.data:
                            coordinator.async_set_updated_data(
                                {**coordinator.data, "last_utterance": data.get("content", "")}
                            )

                    elif role == "assistant":
                        hass.bus.async_fire(
                            EVENT_TURN_ENDED,
                            {"response": data.get("content", ""), "source": source},
                        )
                        if coordinator.data:
                            coordinator.async_set_updated_data(
                                {**coordinator.data, "last_response": data.get("content", "")}
                            )

        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("SSE listener disconnected (%s), retrying in 10s", exc)
            await asyncio.sleep(10)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    data = hass.data[DOMAIN].get(entry.entry_id, {})

    sse_task = data.get("sse_task")
    if sse_task is not None:
        sse_task.cancel()
        with suppress(asyncio.CancelledError):
            await sse_task

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    hass.services.async_remove(DOMAIN, "speak")
    hass.services.async_remove(DOMAIN, "chat")
    hass.services.async_remove(DOMAIN, "mic")

    hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
