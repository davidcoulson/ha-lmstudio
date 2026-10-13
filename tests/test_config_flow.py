"""Tests for the config and options flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.lmstudio.api import (
    LMStudioAuthError,
    LMStudioConnectionError,
    parse_v1,
)
from custom_components.lmstudio.const import CONF_API_TOKEN, CONF_SCAN_INTERVAL, DOMAIN

GET = "custom_components.lmstudio.api.LMStudioClient.async_get_data"


async def test_user_flow(hass, payload) -> None:
    """Happy path creates an entry with unique id host:port."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    with (
        patch(GET, AsyncMock(return_value=parse_v1(payload))),
        patch("custom_components.lmstudio.async_setup_entry", return_value=True),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"host": "10.2.3.62", "port": 1234, CONF_API_TOKEN: "secret"},
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        "host": "10.2.3.62",
        "port": 1234,
        CONF_API_TOKEN: "secret",
    }
    assert result["result"].unique_id == "10.2.3.62:1234"


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (LMStudioConnectionError("x"), "cannot_connect"),
        (LMStudioAuthError("x"), "invalid_auth"),
        (RuntimeError("x"), "unknown"),
    ],
)
async def test_user_flow_errors(hass, exc, error) -> None:
    """Errors surface on the form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(GET, AsyncMock(side_effect=exc)):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"host": "h", "port": 1234}
        )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}


async def test_duplicate_aborts(hass, mock_entry, mock_api) -> None:
    """The same host:port cannot be added twice."""
    mock_entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"host": "10.2.3.62", "port": 1234}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_flow(hass, mock_entry, mock_api, aioclient_mock) -> None:
    """Reauth validates and stores the new token, then reloads."""
    mock_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_entry.entry_id)
    result = await mock_entry.start_reauth_flow(hass)
    assert result["step_id"] == "reauth_confirm"

    with patch(GET, AsyncMock(side_effect=LMStudioAuthError("x"))):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_TOKEN: "bad"}
        )
    assert result["errors"] == {"base": "invalid_auth"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_TOKEN: "new"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_entry.data[CONF_API_TOKEN] == "new"


async def test_options_flow(hass, mock_entry, mock_api) -> None:
    """The scan interval is configurable."""
    mock_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_entry.entry_id)
    result = await hass.config_entries.options.async_init(mock_entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_SCAN_INTERVAL: 60}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert mock_entry.options[CONF_SCAN_INTERVAL] == 60
    await hass.async_block_till_done()
    assert mock_entry.runtime_data.update_interval.total_seconds() == 60
