"""Tests for setup, entities and services."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import patch

import pytest
import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.lmstudio.const import DOMAIN, LOAD_TIMEOUT

from .conftest import BASE, LLM_IDLE, LLM_LOADED, posts, serve

CTX = "sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_context_length"
LOADED = "sensor.lm_studio_10_2_3_62_loaded_models"
SERVER = "binary_sensor.lm_studio_10_2_3_62_server"
KEY = LLM_LOADED["key"]


async def _setup(hass, entry) -> None:
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _poll(hass, seconds: int = 31) -> None:
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=seconds))
    await hass.async_block_till_done()


def _entry_count(hass, entry) -> int:
    return len(er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id))


async def test_entities(hass, mock_entry, mock_api) -> None:
    """Entities report the loaded model's configuration."""
    await _setup(hass, mock_entry)

    assert hass.states.get(SERVER).state == "on"
    assert hass.states.get(LOADED).state == "1"
    assert hass.states.get(LOADED).attributes["instances"] == [KEY]
    assert hass.states.get("sensor.lm_studio_10_2_3_62_available_models").state == "3"
    assert hass.states.get(CTX).state == "262144"
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


async def test_unload_keeps_entities_unavailable(
    hass, mock_entry, mock_api, payload
) -> None:
    """Unloaded models stay registered (stable ids) but go unavailable."""
    await _setup(hass, mock_entry)
    assert _entry_count(hass, mock_entry) == 7

    payload["models"][0]["loaded_instances"] = []
    serve(mock_api, payload)
    await _poll(hass)
    assert hass.states.get(LOADED).state == "0"
    assert _entry_count(hass, mock_entry) == 7
    assert hass.states.get(CTX).state == "unavailable"

    # Another model loads: new entities appear alongside the old ones.
    payload["models"][2]["loaded_instances"] = [
        {
            "id": "openai/gpt-oss-20b",
            "config": {
                "context_length": 65536,
                "parallel": 4,
                "flash_attention": False,
            },
        }
    ]
    serve(mock_api, payload)
    await _poll(hass, 62)
    assert (
        hass.states.get("sensor.lm_studio_10_2_3_62_gpt_oss_20b_context_length").state
        == "65536"
    )
    assert _entry_count(hass, mock_entry) == 11

    # The first model returns on the very same entity id.
    payload["models"][0]["loaded_instances"] = [LLM_LOADED["loaded_instances"][0]]
    serve(mock_api, payload)
    await _poll(hass, 93)
    assert hass.states.get(CTX).state == "262144"
    assert _entry_count(hass, mock_entry) == 11


async def test_context_change_updates_state(
    hass, mock_entry, mock_api, payload
) -> None:
    """A context length change after restart is visible in the sensor state."""
    await _setup(hass, mock_entry)
    cfg = payload["models"][0]["loaded_instances"][0]["config"]
    cfg["context_length"], cfg["parallel"] = 65536, 4
    serve(mock_api, payload)
    await _poll(hass)
    assert hass.states.get(CTX).state == "65536"


async def test_instance_selection_is_deterministic(
    hass, mock_entry, mock_api, payload
) -> None:
    """With several instances the lowest id is reported, whatever the order."""
    a = {"id": "m:1", "config": {"context_length": 4096, "parallel": 1}}
    b = {"id": "m:2", "config": {"context_length": 65536, "parallel": 1}}
    payload["models"][0]["loaded_instances"] = [b, a]
    serve(mock_api, payload)
    await _setup(hass, mock_entry)
    assert hass.states.get(CTX).state == "4096"
    assert hass.states.get(CTX).attributes["instances"] == 2

    payload["models"][0]["loaded_instances"] = [a, b]
    serve(mock_api, payload)
    await _poll(hass)
    assert hass.states.get(CTX).state == "4096"


@pytest.mark.parametrize(
    "bad",
    [{}, {"error": "server unavailable"}, {"models": None}, {"models": ["x"]}, []],
)
async def test_malformed_response_keeps_snapshot(
    hass, mock_entry, mock_api, bad
) -> None:
    """A 200 with a wrong shape is a failure, not 'zero models'."""
    await _setup(hass, mock_entry)
    serve(mock_api, bad)
    await _poll(hass)
    assert hass.states.get(SERVER).state == "off"
    assert hass.states.get(LOADED).state == "unavailable"
    assert _entry_count(hass, mock_entry) == 7
    assert mock_entry.runtime_data.data.instance_ids == [KEY]  # snapshot kept

    serve(mock_api, {"models": [LLM_LOADED]})
    await _poll(hass, 62)
    assert hass.states.get(SERVER).state == "on"
    assert hass.states.get(LOADED).state == "1"


async def test_invalid_json_is_failure(hass, mock_entry, mock_api) -> None:
    """Non-JSON body does not escape as ValueError."""
    await _setup(hass, mock_entry)
    mock_api.clear_requests()
    mock_api.get(f"{BASE}/api/v1/models", text="<html>nope</html>")
    await _poll(hass)
    assert hass.states.get(SERVER).state == "off"


async def test_server_offline_and_recovery(hass, mock_entry, mock_api, payload) -> None:
    """Server sensor goes off; other entities go unavailable, then recover."""
    await _setup(hass, mock_entry)
    mock_api.clear_requests()
    mock_api.get(f"{BASE}/api/v1/models", exc=TimeoutError)
    await _poll(hass)
    assert hass.states.get(SERVER).state == "off"
    assert hass.states.get(LOADED).state == "unavailable"

    serve(mock_api, payload)
    await _poll(hass, 62)
    assert hass.states.get(SERVER).state == "on"
    assert hass.states.get(LOADED).state == "1"


async def test_auth_failure_starts_reauth(hass, mock_entry, mock_api) -> None:
    """A 401 on poll raises a reauth flow instead of silent retry."""
    await _setup(hass, mock_entry)
    mock_api.clear_requests()
    mock_api.get(f"{BASE}/api/v1/models", status=401)
    await _poll(hass)
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [f["context"]["source"] for f in flows] == ["reauth"]


async def test_startup_auth_failure_starts_reauth(hass, mock_entry, aioclient_mock):
    """A 401 at startup puts the entry in setup_error and begins reauth."""
    aioclient_mock.get(f"{BASE}/api/v1/models", status=401)
    mock_entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(mock_entry.entry_id)
    assert mock_entry.state is ConfigEntryState.SETUP_ERROR
    assert hass.config_entries.flow.async_progress_by_handler(DOMAIN)


async def test_startup_offline_retries(hass, mock_entry, aioclient_mock) -> None:
    """Offline at startup is a retry, not a failure."""
    aioclient_mock.get(f"{BASE}/api/v1/models", exc=TimeoutError)
    mock_entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(mock_entry.entry_id)
    assert mock_entry.state is ConfigEntryState.SETUP_RETRY


async def test_load_and_unload_use_real_client(hass, mock_entry, mock_api) -> None:
    """Services POST the right bodies through the real client."""
    await _setup(hass, mock_entry)
    await hass.services.async_call(
        DOMAIN,
        "load_model",
        {"model": LLM_IDLE["key"], "context_length": 8192, "parallel": 2},
        blocking=True,
    )
    assert posts(mock_api, "/models/load") == [
        {"model": LLM_IDLE["key"], "context_length": 8192, "parallel": 2}
    ]
    await hass.services.async_call(
        DOMAIN, "unload_model", {"instance_id": "abc"}, blocking=True
    )
    assert posts(mock_api, "/models/unload") == [{"instance_id": "abc"}]


async def test_load_identical_is_skipped(hass, mock_entry, mock_api) -> None:
    """Re-issuing a load for an identical loaded config sends nothing."""
    await _setup(hass, mock_entry)
    await hass.services.async_call(
        DOMAIN,
        "load_model",
        {"model": KEY, "context_length": 262144, "parallel": 2},
        blocking=True,
    )
    assert posts(mock_api, "/models/load") == []


async def test_load_uses_long_timeout(hass, mock_entry, mock_api) -> None:
    """Load POSTs with the long timeout, polling with the short one."""
    await _setup(hass, mock_entry)
    client = mock_entry.runtime_data.client
    with patch.object(client, "_request", wraps=client._request) as spy:
        await client.async_load_model("m")
        await client.async_unload_model("i")
    assert spy.call_args_list[0].args[3] == LOAD_TIMEOUT
    assert spy.call_args_list[1].args[3] < LOAD_TIMEOUT


async def test_load_failure_not_retried(hass, mock_entry, mock_api) -> None:
    """A failed load raises once, with state refreshed, and is not retried."""
    await _setup(hass, mock_entry)
    mock_api.clear_requests()
    mock_api.get(f"{BASE}/api/v1/models", json={"models": [LLM_LOADED, LLM_IDLE]})
    mock_api.post(f"{BASE}/api/v1/models/load", exc=TimeoutError)
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN, "load_model", {"model": LLM_IDLE["key"]}, blocking=True
        )
    assert len(posts(mock_api, "/models/load")) == 1


async def test_v0_only_server_reports_unsupported(
    hass, mock_entry, aioclient_mock
) -> None:
    """Services say model management needs API v1 on a v0-only server."""
    aioclient_mock.get(f"{BASE}/api/v1/models", status=404)
    aioclient_mock.get(
        f"{BASE}/api/v0/models",
        json={"data": [{"id": "m", "type": "llm", "state": "not-loaded"}]},
    )
    aioclient_mock.post(f"{BASE}/api/v1/models/unload", status=404)
    await _setup(hass, mock_entry)
    with pytest.raises(HomeAssistantError, match="API v1"):
        await hass.services.async_call(
            DOMAIN, "unload_model", {"instance_id": "m"}, blocking=True
        )


async def test_service_serialised(hass, mock_entry, mock_api) -> None:
    """Concurrent service calls run one at a time."""
    await _setup(hass, mock_entry)
    client = mock_entry.runtime_data.client
    active = peak = 0

    async def slow(*args, **kwargs) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    with patch.object(client, "async_unload_model", slow):
        await asyncio.gather(
            *(
                hass.services.async_call(
                    DOMAIN, "unload_model", {"instance_id": str(n)}, blocking=True
                )
                for n in range(3)
            )
        )
    assert peak == 1


async def test_multiple_entries_require_target(hass, mock_entry, mock_api) -> None:
    """With two servers (one not loaded) an omitted target is rejected."""
    await _setup(hass, mock_entry)
    second = MockConfigEntry(
        domain=DOMAIN,
        title="B",
        unique_id="10.2.3.99:1234",
        data={"host": "10.2.3.99", "port": 1234},
    )
    second.add_to_hass(hass)  # never set up: not loaded
    assert second.state is ConfigEntryState.NOT_LOADED

    with pytest.raises(ServiceValidationError, match="config_entry_id"):
        await hass.services.async_call(
            DOMAIN, "unload_model", {"instance_id": "x"}, blocking=True
        )
    with pytest.raises(ServiceValidationError, match="not loaded"):
        await hass.services.async_call(
            DOMAIN,
            "unload_model",
            {"instance_id": "x", "config_entry_id": second.entry_id},
            blocking=True,
        )
    with pytest.raises(ServiceValidationError, match="Unknown"):
        await hass.services.async_call(
            DOMAIN,
            "unload_model",
            {"instance_id": "x", "config_entry_id": "nope"},
            blocking=True,
        )
    assert posts(mock_api, "/models/unload") == []

    await hass.services.async_call(
        DOMAIN,
        "unload_model",
        {"instance_id": "x", "config_entry_id": mock_entry.entry_id},
        blocking=True,
    )
    assert posts(mock_api, "/models/unload") == [{"instance_id": "x"}]


@pytest.mark.parametrize(
    ("service", "data"),
    [
        ("load_model", {"model": ""}),
        ("load_model", {"model": "   "}),
        ("load_model", {"model": "m", "context_length": 1.9}),
        ("load_model", {"model": "m", "context_length": True}),
        ("load_model", {"model": "m", "context_length": 511}),
        ("load_model", {"model": "m", "context_length": 1048577}),
        ("load_model", {"model": "m", "parallel": 0}),
        ("load_model", {"model": "m", "parallel": 17}),
        ("load_model", {"model": "m", "parallel": False}),
        ("unload_model", {"instance_id": ""}),
    ],
)
async def test_service_validation(hass, mock_entry, mock_api, service, data) -> None:
    """Bad input is rejected before any request is made."""
    await _setup(hass, mock_entry)
    with pytest.raises(vol.Invalid):
        await hass.services.async_call(DOMAIN, service, data, blocking=True)
    assert posts(mock_api, "/models/load") == []
    assert posts(mock_api, "/models/unload") == []


async def test_service_validation_accepts_whole_floats(
    hass, mock_entry, mock_api
) -> None:
    """UI number selectors send 4.0; that is accepted as 4."""
    await _setup(hass, mock_entry)
    await hass.services.async_call(
        DOMAIN,
        "load_model",
        {"model": "new", "context_length": 4096.0, "parallel": 4.0},
        blocking=True,
    )
    assert posts(mock_api, "/models/load") == [
        {"model": "new", "context_length": 4096, "parallel": 4}
    ]


async def test_unload_entry(hass, mock_entry, mock_api) -> None:
    """The entry unloads cleanly."""
    await _setup(hass, mock_entry)
    assert await hass.config_entries.async_unload(mock_entry.entry_id)
    assert mock_entry.state is ConfigEntryState.NOT_LOADED
