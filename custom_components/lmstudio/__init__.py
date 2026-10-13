"""The LM Studio integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import LMStudioClient, LMStudioError, LoadedInstance
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_CONTEXT_LENGTH,
    ATTR_FLASH_ATTENTION,
    ATTR_INSTANCE_ID,
    ATTR_MODEL,
    ATTR_PARALLEL,
    CONF_API_TOKEN,
    DOMAIN,
    MAX_CONTEXT_LENGTH,
    MAX_PARALLEL,
    MIN_CONTEXT_LENGTH,
    MIN_PARALLEL,
    SERVICE_LOAD_MODEL,
    SERVICE_UNLOAD_MODEL,
)
from .coordinator import LMStudioConfigEntry, LMStudioCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)


def _bounded_int(low: int, high: int):
    """Validate a strict integer: no bools, no fractions, within bounds."""

    def validate(value: Any) -> int:
        if isinstance(value, bool):
            raise vol.Invalid("expected an integer, not a boolean")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise vol.Invalid("expected a whole number")
        if not low <= value <= high:
            raise vol.Invalid(f"must be between {low} and {high}")
        return value

    return validate


_ID = vol.All(cv.string, vol.Strip, vol.Length(min=1))

LOAD_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CONFIG_ENTRY_ID): _ID,
        vol.Required(ATTR_MODEL): _ID,
        vol.Optional(ATTR_CONTEXT_LENGTH): _bounded_int(
            MIN_CONTEXT_LENGTH, MAX_CONTEXT_LENGTH
        ),
        vol.Optional(ATTR_PARALLEL): _bounded_int(MIN_PARALLEL, MAX_PARALLEL),
        vol.Optional(ATTR_FLASH_ATTENTION): cv.boolean,
    }
)
UNLOAD_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CONFIG_ENTRY_ID): _ID,
        vol.Required(ATTR_INSTANCE_ID): _ID,
    }
)


def _get_coordinator(hass: HomeAssistant, call: ServiceCall) -> LMStudioCoordinator:
    """Resolve the target server without ever guessing between several."""
    entries = hass.config_entries.async_entries(DOMAIN)
    target_id = call.data.get(ATTR_CONFIG_ENTRY_ID)
    if target_id is not None:
        matches = [e for e in entries if e.entry_id == target_id]
        if not matches:
            raise ServiceValidationError(f"Unknown config entry {target_id}")
        entry = matches[0]
    elif len(entries) == 1:
        entry = entries[0]
    elif not entries:
        raise ServiceValidationError("No LM Studio server is configured")
    else:
        raise ServiceValidationError(
            "Several LM Studio servers are configured; set config_entry_id"
        )
    if entry.state is not ConfigEntryState.LOADED:
        raise ServiceValidationError(
            f"LM Studio server '{entry.title}' is not loaded (state: {entry.state})"
        )
    return entry.runtime_data


def _matches(instance: LoadedInstance, call_data: dict[str, Any]) -> bool:
    return (
        call_data.get(ATTR_CONTEXT_LENGTH) in (None, instance.context_length)
        and call_data.get(ATTR_PARALLEL) in (None, instance.parallel)
        and call_data.get(ATTR_FLASH_ATTENTION) in (None, instance.flash_attention)
    )


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the integration's services."""

    async def load_model(call: ServiceCall) -> None:
        coordinator = _get_coordinator(hass, call)
        model_key = call.data[ATTR_MODEL]
        async with coordinator.mutation_lock:
            # Reconcile with the server first: loading again would stack a
            # second instance (and may reload), so skip an identical one.
            await coordinator.async_refresh()
            model = (coordinator.data.models if coordinator.data else {}).get(model_key)
            if model and any(_matches(i, call.data) for i in model.instances):
                _LOGGER.info("%s is already loaded with these settings", model_key)
                return
            try:
                await coordinator.client.async_load_model(
                    model_key,
                    call.data.get(ATTR_CONTEXT_LENGTH),
                    call.data.get(ATTR_PARALLEL),
                    call.data.get(ATTR_FLASH_ATTENTION),
                )
            except LMStudioError as err:
                # Outcome may be ambiguous (e.g. timeout); never retry blindly.
                await coordinator.async_refresh()
                raise HomeAssistantError(f"Loading model failed: {err}") from err
            await coordinator.async_refresh()

    async def unload_model(call: ServiceCall) -> None:
        coordinator = _get_coordinator(hass, call)
        async with coordinator.mutation_lock:
            await coordinator.async_refresh()
            try:
                await coordinator.client.async_unload_model(call.data[ATTR_INSTANCE_ID])
            except LMStudioError as err:
                await coordinator.async_refresh()
                raise HomeAssistantError(f"Unloading model failed: {err}") from err
            await coordinator.async_refresh()

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
