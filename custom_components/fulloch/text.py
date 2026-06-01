import aiohttp
from homeassistant.components.text import TextEntity, TextMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        FullochSpeakText(data["session"], data["url"], entry),
        FullochChatText(data["session"], data["url"], entry),
    ])


class _FullochTextBase(TextEntity):
    _attr_native_min = 0
    _attr_native_max = 500
    _attr_mode = TextMode.TEXT
    _attr_native_value = ""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        entry: ConfigEntry,
    ) -> None:
        self._session = session
        self._url = url
        self._entry = entry

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Fulloch",
            "manufacturer": "Fulloch",
        }


class FullochSpeakText(_FullochTextBase):
    _attr_name = "Speak"
    _attr_icon = "mdi:text-to-speech"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_speak_text"

    async def async_set_value(self, value: str) -> None:
        try:
            async with self._session.post(
                f"{self._url}/speak",
                json={"text": value},
                timeout=aiohttp.ClientTimeout(total=10),
            ):
                pass
        except aiohttp.ClientError:
            pass
        self._attr_native_value = value
        self.async_write_ha_state()


class FullochChatText(_FullochTextBase):
    _attr_name = "Chat"
    _attr_icon = "mdi:chat-processing"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_chat_text"

    async def async_set_value(self, value: str) -> None:
        try:
            async with self._session.post(
                f"{self._url}/chat",
                json={"text": value},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                answer = data.get("answer", "")
            if answer:
                async with self._session.post(
                    f"{self._url}/speak",
                    json={"text": answer},
                    timeout=aiohttp.ClientTimeout(total=10),
                ):
                    pass
        except aiohttp.ClientError:
            pass
        self._attr_native_value = value
        self.async_write_ha_state()
