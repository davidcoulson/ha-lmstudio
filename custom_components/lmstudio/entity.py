"""Shared entity helpers for LM Studio."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .api import LoadedInstance, Model
from .const import DOMAIN
from .coordinator import LMStudioConfigEntry, LMStudioCoordinator


class LMStudioEntity(CoordinatorEntity[LMStudioCoordinator]):
    """Base entity attached to the per-server device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: LMStudioCoordinator) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.unique_id or entry.entry_id)},
            name=entry.title,
            manufacturer="LM Studio",
            configuration_url=(f"http://{entry.data['host']}:{entry.data['port']}"),
        )


class LMStudioModelEntity(LMStudioEntity):
    """Entity describing one loaded model; keyed by model key."""

    def __init__(
        self, coordinator: LMStudioCoordinator, model_key: str, suffix: str
    ) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        self.model_key = model_key
        self._attr_unique_id = (
            f"{coordinator.config_entry.unique_id}_{model_key}_{suffix}"
        )
        model = coordinator.data.models.get(model_key)
        self._attr_translation_placeholders = {
            "model": model.display_name if model else model_key
        }

    @property
    def model(self) -> Model | None:
        """Return the model, if still known."""
        return self.coordinator.data.models.get(self.model_key)

    @property
    def instance(self) -> LoadedInstance | None:
        """Return the lowest-id loaded instance of the model."""
        model = self.model
        return model.primary_instance if model else None

    @property
    def available(self) -> bool:
        """Unavailable while the server is down or the model is unloaded."""
        return super().available and self.instance is not None


def sync_model_entities(
    entry: LMStudioConfigEntry,
    coordinator: LMStudioCoordinator,
    async_add_entities: AddConfigEntryEntitiesCallback,
    factory: Callable[[str], list[Entity]],
) -> None:
    """Add entities for LLMs the first time they are seen loaded.

    Entities are keyed by model key (``{host:port}_{model key}_{suffix}``).
    They are never removed: once a model unloads they stay registered but
    unavailable, so entity ids stay stable and the same entities return when
    the model loads again.
    """
    known: set[str] = set()

    @callback
    def _sync() -> None:
        new = set(coordinator.data.loaded_llms) - known
        if new:
            known.update(new)
            async_add_entities([e for key in sorted(new) for e in factory(key)])

    _sync()
    entry.async_on_unload(coordinator.async_add_listener(_sync))
