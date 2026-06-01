import logging
from datetime import timedelta

import aiohttp
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN

logger = logging.getLogger(__name__)


class FullochCoordinator(DataUpdateCoordinator):
    def __init__(self, hass, session: aiohttp.ClientSession, url: str) -> None:
        super().__init__(
            hass,
            logger,
            name=DOMAIN,
            update_interval=timedelta(seconds=30),
        )
        self.session = session
        self.url = url

    async def _async_update_data(self) -> dict:
        try:
            async with self.session.get(
                f"{self.url}/status",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    raise UpdateFailed(f"HTTP {resp.status}")
                return await resp.json()
        except aiohttp.ClientError as exc:
            raise UpdateFailed(f"Connection error: {exc}") from exc
