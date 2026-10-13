"""Tests for the API client and parsers."""

from __future__ import annotations

import pytest
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.lmstudio.api import (
    LMStudioAuthError,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioInvalidResponseError,
    parse_v0,
    parse_v1,
)


def test_parse_v1(payload) -> None:
    """Parse the sample payload."""
    data = parse_v1(payload)
    assert len(data.models) == 3
    assert list(data.loaded_llms) == ["google/gemma-4-26b-a4b-qat"]
    inst = data.loaded_llms["google/gemma-4-26b-a4b-qat"].instances[0]
    assert (inst.context_length, inst.parallel, inst.flash_attention) == (
        262144,
        2,
        True,
    )
    assert data.instance_ids == ["google/gemma-4-26b-a4b-qat"]


def test_parse_v0() -> None:
    """Parse the legacy payload."""
    data = parse_v0(
        {
            "data": [
                {
                    "id": "m",
                    "type": "llm",
                    "state": "loaded",
                    "loaded_context_length": 4096,
                    "max_context_length": 8192,
                },
                {"id": "e", "type": "embeddings", "state": "not-loaded"},
            ]
        }
    )
    assert data.models["m"].instances[0].context_length == 4096
    assert not data.models["e"].loaded


async def test_client_falls_back_to_v0(hass, aioclient_mock) -> None:
    """A 404 on v1 triggers the v0 endpoint; the token is sent as Bearer."""
    aioclient_mock.get("http://h:1/api/v1/models", status=404)
    aioclient_mock.get(
        "http://h:1/api/v0/models",
        json={"data": [{"id": "m", "type": "llm", "state": "not-loaded"}]},
    )
    client = LMStudioClient(async_get_clientsession(hass), "h", 1, "tok")
    data = await client.async_get_data()
    assert "m" in data.models
    assert aioclient_mock.mock_calls[-1][3]["Authorization"] == "Bearer tok"


async def test_client_errors(hass, aioclient_mock) -> None:
    """401 maps to auth error, connection failure to connection error."""
    aioclient_mock.get("http://h:1/api/v1/models", status=401)
    client = LMStudioClient(async_get_clientsession(hass), "h", 1)
    with pytest.raises(LMStudioAuthError):
        await client.async_get_data()

    aioclient_mock.clear_requests()
    aioclient_mock.get("http://h:1/api/v1/models", exc=TimeoutError)
    with pytest.raises(LMStudioConnectionError):
        await client.async_get_data()


@pytest.mark.parametrize("bad", [{}, [], {"models": None}, {"models": [1]}])
def test_parse_v1_rejects_bad_envelope(bad) -> None:
    """Wrong shapes raise instead of parsing as zero models."""
    with pytest.raises(LMStudioInvalidResponseError):
        parse_v1(bad)


def test_primary_instance_is_lowest_id() -> None:
    """Instance choice does not depend on list order."""
    from custom_components.lmstudio.api import LoadedInstance, Model

    model = Model("k", "K", "llm", instances=[LoadedInstance("b"), LoadedInstance("a")])
    assert model.primary_instance.id == "a"
