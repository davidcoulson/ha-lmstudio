"""Binary sensors for LM Studio."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import LMStudioConfigEntry, LMStudioCoordinator
from .entity import LMStudioEntity, LMStudioModelEntity, sync_model_entities


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LMStudioConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensors."""
    coordinator = entry.runtime_data
    async_add_entities([LMStudioServerSensor(coordinator)])
    sync_model_entities(
        hass,
        entry,
        coordinator,
        async_add_entities,
        "binary_sensor",
        lambda key: [LMStudioFlashAttentionSensor(coordinator, key)],
        ("flash_attention",),
    )


class LMStudioServerSensor(LMStudioEntity, BinarySensorEntity):
    """On when the LM Studio API answers."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "server"

    def __init__(self, coordinator: LMStudioCoordinator) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.config_entry.unique_id}_server"

    @property
    def available(self) -> bool:
        """Stay available while the server is down; that is the point."""
        return True

    @property
    def is_on(self) -> bool:
        """Return True if the last poll succeeded."""
        return self.coordinator.last_update_success

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()


class LMStudioFlashAttentionSensor(LMStudioModelEntity, BinarySensorEntity):
    """Flash attention enabled for a loaded model."""

    _attr_translation_key = "flash_attention"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: LMStudioCoordinator, model_key: str) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, model_key, "flash_attention")

    @property
    def is_on(self) -> bool | None:
        """Return flash attention state."""
        inst = self.instance
        return None if inst is None else inst.flash_attention
