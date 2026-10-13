"""Shared entity helpers for LM Studio."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
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
        """Return the first loaded instance of the model."""
        model = self.model
        return model.instances[0] if model and model.instances else None

    @property
    def available(self) -> bool:
        """Unavailable while the server is down or the model is unloaded."""
        return super().available and self.instance is not None


def sync_model_entities(
    hass: HomeAssistant,
    entry: LMStudioConfigEntry,
    coordinator: LMStudioCoordinator,
    async_add_entities: AddConfigEntryEntitiesCallback,
    platform: str,
    factory: Callable[[str], list[Entity]],
    suffixes: tuple[str, ...],
) -> None:
    """Add entities for newly loaded LLMs and remove those that unloaded.

    Entities are keyed by model key (``{host:port}_{model key}_{suffix}``) so a
    reload of the same model brings back the same entities.
    """
    registry = er.async_get(hass)
    prefix = f"{entry.unique_id}_"
    known: set[str] = set()

    @callback
    def _sync() -> None:
        loaded = set(coordinator.data.loaded_llms)
        new = loaded - known
        if new:
            known.update(new)
            async_add_entities([e for key in sorted(new) for e in factory(key)])

        for gone in known - loaded:
            known.discard(gone)
            for suffix in suffixes:
                entity_id = registry.async_get_entity_id(
                    platform, DOMAIN, f"{prefix}{gone}_{suffix}"
                )
                if entity_id:
                    registry.async_remove(entity_id)

    # Drop registry leftovers for models that were unloaded while HA was down.
    loaded_now = set(coordinator.data.loaded_llms)
    for reg_entry in er.async_entries_for_config_entry(registry, entry.entry_id):
        if reg_entry.domain != platform or not reg_entry.unique_id.startswith(prefix):
            continue
        rest = reg_entry.unique_id[len(prefix) :]
        for suffix in suffixes:
            if rest.endswith(f"_{suffix}"):
                key = rest[: -len(suffix) - 1]
                if key not in loaded_now:
                    registry.async_remove(reg_entry.entity_id)
                break

    _sync()
    entry.async_on_unload(coordinator.async_add_listener(_sync))
