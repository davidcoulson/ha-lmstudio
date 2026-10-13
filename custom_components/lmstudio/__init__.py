"""The LM Studio integration."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import LMStudioClient, LMStudioError
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_CONTEXT_LENGTH,
    ATTR_FLASH_ATTENTION,
    ATTR_INSTANCE_ID,
    ATTR_MODEL,
    ATTR_PARALLEL,
    CONF_API_TOKEN,
    DOMAIN,
    SERVICE_LOAD_MODEL,
    SERVICE_UNLOAD_MODEL,
)
from .coordinator import LMStudioConfigEntry, LMStudioCoordinator

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)

LOAD_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Required(ATTR_MODEL): cv.string,
        vol.Optional(ATTR_CONTEXT_LENGTH): vol.All(vol.Coerce(int), vol.Range(min=1)),
        vol.Optional(ATTR_PARALLEL): vol.All(vol.Coerce(int), vol.Range(min=1)),
        vol.Optional(ATTR_FLASH_ATTENTION): cv.boolean,
    }
)
UNLOAD_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Required(ATTR_INSTANCE_ID): cv.string,
    }
)


def _get_coordinator(hass: HomeAssistant, call: ServiceCall) -> LMStudioCoordinator:
    entries = [
        e
        for e in hass.config_entries.async_loaded_entries(DOMAIN)
        if call.data.get(ATTR_CONFIG_ENTRY_ID, e.entry_id) == e.entry_id
    ]
    if len(entries) != 1:
        raise ServiceValidationError(
            "Specify config_entry_id: expected exactly one matching LM Studio server"
            f", found {len(entries)}"
        )
    return entries[0].runtime_data


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the integration's services."""

    async def load_model(call: ServiceCall) -> None:
        coordinator = _get_coordinator(hass, call)
        try:
            await coordinator.client.async_load_model(
                call.data[ATTR_MODEL],
                call.data.get(ATTR_CONTEXT_LENGTH),
                call.data.get(ATTR_PARALLEL),
                call.data.get(ATTR_FLASH_ATTENTION),
            )
        except LMStudioError as err:
            raise HomeAssistantError(f"Loading model failed: {err}") from err
        await coordinator.async_request_refresh()

    async def unload_model(call: ServiceCall) -> None:
        coordinator = _get_coordinator(hass, call)
        try:
            await coordinator.client.async_unload_model(call.data[ATTR_INSTANCE_ID])
        except LMStudioError as err:
            raise HomeAssistantError(f"Unloading model failed: {err}") from err
        await coordinator.async_request_refresh()

    hass.services.async_register(DOMAIN, SERVICE_LOAD_MODEL, load_model, LOAD_SCHEMA)
    hass.services.async_register(
        DOMAIN, SERVICE_UNLOAD_MODEL, unload_model, UNLOAD_SCHEMA
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: LMStudioConfigEntry) -> bool:
    """Set up an LM Studio server from a config entry."""
    client = LMStudioClient(
        async_get_clientsession(hass),
        entry.data[CONF_HOST],
        entry.data[CONF_PORT],
        entry.data.get(CONF_API_TOKEN),
    )
    coordinator = LMStudioCoordinator(hass, entry, client)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    return True


async def _async_options_updated(
    hass: HomeAssistant, entry: LMStudioConfigEntry
) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LMStudioConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
