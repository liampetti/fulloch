from homeassistant.components.sensor import SensorEntity
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
    coordinator: FullochCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([
        FullochStatusSensor(coordinator, entry),
        FullochLastUtteranceSensor(coordinator, entry),
        FullochLastResponseSensor(coordinator, entry),
    ])


class _FullochSensorBase(CoordinatorEntity, SensorEntity):
    def __init__(self, coordinator: FullochCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Fulloch",
            "manufacturer": "Fulloch",
        }


class FullochStatusSensor(_FullochSensorBase):
    _attr_name = "Fulloch Status"
    _attr_icon = "mdi:microphone"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_status"

    @property
    def native_value(self) -> str:
        if self.coordinator.data:
            return self.coordinator.data.get("state", "unknown")
        return "unknown"


def _truncate(text: str, limit: int = 255) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class FullochLastUtteranceSensor(_FullochSensorBase):
    _attr_name = "Fulloch Last Utterance"
    _attr_icon = "mdi:account-voice"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_last_utterance"

    @property
    def native_value(self) -> str:
        text = (self.coordinator.data or {}).get("last_utterance", "")
        return _truncate(text)

    @property
    def extra_state_attributes(self) -> dict:
        return {"full_text": (self.coordinator.data or {}).get("last_utterance", "")}


class FullochLastResponseSensor(_FullochSensorBase):
    _attr_name = "Fulloch Last Response"
    _attr_icon = "mdi:message-text"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_last_response"

    @property
    def native_value(self) -> str:
        text = (self.coordinator.data or {}).get("last_response", "")
        return _truncate(text)

    @property
    def extra_state_attributes(self) -> dict:
        return {"full_text": (self.coordinator.data or {}).get("last_response", "")}
