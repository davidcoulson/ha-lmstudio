"""Tests for setup, entities and services."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.lmstudio.api import LMStudioConnectionError
from custom_components.lmstudio.const import DOMAIN

from .conftest import LLM_LOADED

PREFIX = "gemma_4_26b_a4b_qat"


async def _setup(hass, entry) -> None:
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _poll(hass, seconds: int = 31) -> None:
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=seconds))
    await hass.async_block_till_done()


async def test_entities(hass, mock_entry, mock_api) -> None:
    """Entities report the loaded model's configuration."""
    await _setup(hass, mock_entry)

    assert hass.states.get("binary_sensor.lm_studio_10_2_3_62_server").state == "on"
    assert hass.states.get("sensor.lm_studio_10_2_3_62_loaded_models").state == "1"
    assert hass.states.get("sensor.lm_studio_10_2_3_62_loaded_models").attributes[
        "instances"
    ] == ["google/gemma-4-26b-a4b-qat"]
    assert hass.states.get("sensor.lm_studio_10_2_3_62_available_models").state == "3"

    ctx = hass.states.get(
        "sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_context_length"
    )
    assert ctx.state == "262144"
    assert (
        hass.states.get(
            "sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_parallel_slots"
        ).state
        == "2"
    )
    size = hass.states.get("sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_size")
    assert float(size.state) == pytest.approx(15.6, abs=0.1)
    assert (
        hass.states.get(
            "binary_sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_flash_attention"
        ).state
        == "on"
    )


async def test_dynamic_entities(hass, mock_entry, mock_api, payload) -> None:
    """Entities appear when a model loads and vanish when it unloads."""
    await _setup(hass, mock_entry)
    reg = er.async_get(hass)
    assert len(er.async_entries_for_config_entry(reg, mock_entry.entry_id)) == 7

    # Unload the model.
    payload["models"][0]["loaded_instances"] = []
    await _poll(hass)
    assert hass.states.get("sensor.lm_studio_10_2_3_62_loaded_models").state == "0"
    assert len(er.async_entries_for_config_entry(reg, mock_entry.entry_id)) == 3
    assert (
        hass.states.get("sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_context_length")
        is None
    )

    # Load another with a new config; entities are created, config change shows.
    other = payload["models"][2]
    other["loaded_instances"] = [
        {
            "id": "openai/gpt-oss-20b",
            "config": {
                "context_length": 65536,
                "parallel": 4,
                "flash_attention": False,
            },
        }
    ]
    await _poll(hass, 62)
    assert (
        hass.states.get("sensor.lm_studio_10_2_3_62_gpt_oss_20b_context_length").state
        == "65536"
    )
    assert (
        hass.states.get(
            "binary_sensor.lm_studio_10_2_3_62_gpt_oss_20b_flash_attention"
        ).state
        == "off"
    )
    assert len(er.async_entries_for_config_entry(reg, mock_entry.entry_id)) == 7


async def test_context_change_updates_state(
    hass, mock_entry, mock_api, payload
) -> None:
    """A context length change after restart is visible in the sensor state."""
    await _setup(hass, mock_entry)
    cfg = payload["models"][0]["loaded_instances"][0]["config"]
    cfg["context_length"], cfg["parallel"] = 65536, 4
    await _poll(hass)
    entity = "sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_context_length"
    assert hass.states.get(entity).state == "65536"
    assert (
        hass.states.get(
            "sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_parallel_slots"
        ).state
        == "4"
    )


async def test_server_offline(hass, mock_entry, mock_api) -> None:
    """Server sensor goes off; other entities go unavailable."""
    await _setup(hass, mock_entry)
    mock_api.get.side_effect = LMStudioConnectionError("down")
    await _poll(hass)
    assert hass.states.get("binary_sensor.lm_studio_10_2_3_62_server").state == "off"
    assert (
        hass.states.get("sensor.lm_studio_10_2_3_62_loaded_models").state
        == "unavailable"
    )


async def test_services(hass, mock_entry, mock_api) -> None:
    """Load and unload services call the client (mocked)."""
    await _setup(hass, mock_entry)
    await hass.services.async_call(
        DOMAIN,
        "load_model",
        {
            "model": LLM_LOADED["key"],
            "context_length": 262144,
            "parallel": 2,
            "flash_attention": True,
        },
        blocking=True,
    )
    mock_api.load.assert_awaited_once_with(LLM_LOADED["key"], 262144, 2, True)

    await hass.services.async_call(
        DOMAIN, "unload_model", {"instance_id": "abc"}, blocking=True
    )
    mock_api.unload.assert_awaited_once_with("abc")


async def test_service_errors(hass, mock_entry, mock_api) -> None:
    """Client errors become HomeAssistantError; no entry is a validation error."""
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "unload_model", {"instance_id": "x"}, blocking=True
        )
    await _setup(hass, mock_entry)
    mock_api.unload.side_effect = LMStudioConnectionError("nope")
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN, "unload_model", {"instance_id": "x"}, blocking=True
        )


async def test_unload_entry(hass, mock_entry, mock_api) -> None:
    """The entry unloads cleanly."""
    await _setup(hass, mock_entry)
    assert await hass.config_entries.async_unload(mock_entry.entry_id)
