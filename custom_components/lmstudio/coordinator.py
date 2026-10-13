"""DataUpdateCoordinator for LM Studio."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import (
    LMStudioAuthError,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioData,
    LMStudioError,
)
from .const import CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL, DOMAIN

_LOGGER = logging.getLogger(__name__)

type LMStudioConfigEntry = ConfigEntry[LMStudioCoordinator]


class LMStudioCoordinator(DataUpdateCoordinator[LMStudioData]):
    """Polls the LM Studio models endpoint."""

    config_entry: LMStudioConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        entry: LMStudioConfigEntry,
        client: LMStudioClient,
    ) -> None:
        """Initialise the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(
                seconds=entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
            ),
        )
        self.client = client
        # Serialises load/unload so repeated calls can't stack reloads.
        self.mutation_lock = asyncio.Lock()

    async def _async_update_data(self) -> LMStudioData:
        try:
            return await self.client.async_get_data()
        except LMStudioAuthError as err:
            raise ConfigEntryAuthFailed(f"Authentication failed: {err}") from err
        except LMStudioConnectionError as err:
            raise UpdateFailed(f"Cannot reach LM Studio: {err}") from err
        except LMStudioError as err:
            # Invalid payload etc: keep the previous snapshot, flag unhealthy.
            raise UpdateFailed(f"Unexpected response from LM Studio: {err}") from err
