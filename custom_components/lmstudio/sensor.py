"""Sensors for LM Studio."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, UnitOfInformation
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import LMStudioConfigEntry, LMStudioCoordinator
from .entity import LMStudioEntity, LMStudioModelEntity, sync_model_entities


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LMStudioConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors."""
    coordinator = entry.runtime_data
    async_add_entities(
        [LMStudioLoadedModels(coordinator), LMStudioAvailableModels(coordinator)]
    )
    sync_model_entities(
        hass,
        entry,
        coordinator,
        async_add_entities,
        "sensor",
        lambda key: [
            LMStudioContextLength(coordinator, key),
            LMStudioParallel(coordinator, key),
            LMStudioModelSize(coordinator, key),
        ],
        ("context_length", "parallel", "model_size"),
    )


class LMStudioLoadedModels(LMStudioEntity, SensorEntity):
    """Number of loaded model instances."""

    _attr_translation_key = "loaded_models"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: LMStudioCoordinator) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.config_entry.unique_id}_loaded_models"

    @property
    def native_value(self) -> int:
        """Return the count of loaded instances."""
        return len(self.coordinator.data.instance_ids)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the loaded instance ids."""
        return {"instances": self.coordinator.data.instance_ids}


class LMStudioAvailableModels(LMStudioEntity, SensorEntity):
    """Number of downloaded models."""

    _attr_translation_key = "available_models"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:database"

    def __init__(self, coordinator: LMStudioCoordinator) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.config_entry.unique_id}_available_models"

    @property
    def native_value(self) -> int:
        """Return the number of downloaded models."""
        return len(self.coordinator.data.models)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the downloaded model keys."""
        return {"models": list(self.coordinator.data.models)}


class LMStudioContextLength(LMStudioModelEntity, SensorEntity):
    """Context length a model is loaded with."""

    _attr_translation_key = "context_length"
    _attr_icon = "mdi:text-long"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: LMStudioCoordinator, model_key: str) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, model_key, "context_length")

    @property
    def native_value(self) -> int | None:
        """Return the loaded context length."""
        inst = self.instance
        return None if inst is None else inst.context_length

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return model details."""
        model, inst = self.model, self.instance
        if model is None or inst is None:
            return {}
        return {
            "model": model.key,
            "instance_id": inst.id,
            "max_context_length": model.max_context_length,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "params": model.params_string,
            "eval_batch_size": inst.eval_batch_size,
            "offload_kv_cache_to_gpu": inst.offload_kv_cache_to_gpu,
            "instances": len(model.instances),
        }


class LMStudioParallel(LMStudioModelEntity, SensorEntity):
    """Parallel slots a model is loaded with."""

    _attr_translation_key = "parallel"
    _attr_icon = "mdi:source-branch"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: LMStudioCoordinator, model_key: str) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, model_key, "parallel")

    @property
    def native_value(self) -> int | None:
        """Return the parallel slot count."""
        inst = self.instance
        return None if inst is None else inst.parallel


class LMStudioModelSize(LMStudioModelEntity, SensorEntity):
    """On-disk size of a loaded model."""

    _attr_translation_key = "model_size"
    _attr_device_class = SensorDeviceClass.DATA_SIZE
    _attr_native_unit_of_measurement = UnitOfInformation.GIGABYTES
    _attr_suggested_display_precision = 1
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: LMStudioCoordinator, model_key: str) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, model_key, "model_size")

    @property
    def native_value(self) -> float | None:
        """Return the size in GB (decimal)."""
        model = self.model
        if model is None or model.size_bytes is None:
            return None
        return round(model.size_bytes / 1e9, 3)
