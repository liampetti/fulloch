import aiohttp
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import FullochCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([FullochMicSwitch(data["coordinator"], data["session"], data["url"], entry)])


class FullochMicSwitch(CoordinatorEntity, SwitchEntity):
    _attr_name = "Fulloch Mic"
    _attr_icon = "mdi:microphone"

    def __init__(
        self,
        coordinator: FullochCoordinator,
        session: aiohttp.ClientSession,
        url: str,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator)
        self._session = session
        self._url = url
        self._entry = entry

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_mic"

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Fulloch",
            "manufacturer": "Fulloch",
        }

    @property
    def is_on(self) -> bool:
        if self.coordinator.data:
            return bool(self.coordinator.data.get("mic_enabled", True))
        return True

    async def async_turn_on(self, **kwargs) -> None:
        await self._set_mic(True)

    async def async_turn_off(self, **kwargs) -> None:
        await self._set_mic(False)

    async def _set_mic(self, enabled: bool) -> None:
        try:
            async with self._session.post(
                f"{self._url}/mic",
                json={"enabled": enabled},
                timeout=aiohttp.ClientTimeout(total=5),
            ):
                pass
        except aiohttp.ClientError:
            pass
        await self.coordinator.async_refresh()
