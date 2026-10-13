"""Fixtures for LM Studio tests."""

from __future__ import annotations

import copy

import pytest
from homeassistant.const import CONF_HOST, CONF_PORT
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.lmstudio.const import DOMAIN

pytest_plugins = "pytest_homeassistant_custom_component"

LLM_LOADED = {
    "type": "llm",
    "publisher": "google",
    "key": "google/gemma-4-26b-a4b-qat",
    "display_name": "Gemma 4 26B A4B QAT",
    "architecture": "gemma4",
    "quantization": {"name": "Q4_0", "bits_per_weight": 4},
    "size_bytes": 15634264392,
    "params_string": "26B-A4B",
    "loaded_instances": [
        {
            "id": "google/gemma-4-26b-a4b-qat",
            "config": {
                "context_length": 262144,
                "eval_batch_size": 2048,
                "parallel": 2,
                "flash_attention": True,
                "offload_kv_cache_to_gpu": True,
            },
        }
    ],
    "max_context_length": 262144,
}
EMBEDDING = {
    "type": "embedding",
    "key": "text-embedding-nomic-embed-text-v1.5",
    "display_name": "Nomic Embed Text v1.5",
    "quantization": {"name": "Q4_K_M", "bits_per_weight": 4},
    "size_bytes": 84106624,
    "params_string": None,
    "loaded_instances": [],
    "max_context_length": 2048,
}
LLM_IDLE = {
    "type": "llm",
    "key": "openai/gpt-oss-20b",
    "display_name": "GPT-OSS 20B",
    "architecture": "gpt-oss",
    "quantization": {"name": "MXFP4", "bits_per_weight": 4},
    "size_bytes": 12109664217,
    "params_string": "20B",
    "loaded_instances": [],
    "max_context_length": 131072,
}

PAYLOAD = {"models": [LLM_LOADED, EMBEDDING, LLM_IDLE]}


@pytest.fixture
def payload() -> dict:
    """Return a fresh copy of the sample /api/v1/models payload."""
    return copy.deepcopy(PAYLOAD)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Allow loading custom_components in every test."""


@pytest.fixture
def mock_entry() -> MockConfigEntry:
    """Return a config entry for the default server."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="LM Studio (10.2.3.62)",
        unique_id="10.2.3.62:1234",
        data={CONF_HOST: "10.2.3.62", CONF_PORT: 1234},
    )


BASE = "http://10.2.3.62:1234"


def serve(aioclient_mock, payload, *, status=200):
    """(Re)register the mocked server's model list."""
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/api/v1/models", json=payload, status=status)
    aioclient_mock.post(f"{BASE}/api/v1/models/load", json={"status": "loaded"})
    aioclient_mock.post(f"{BASE}/api/v1/models/unload", json={"instance_id": "x"})


@pytest.fixture
def mock_api(aioclient_mock, payload):
    """Serve the sample payload from a mocked aiohttp session (real client)."""
    serve(aioclient_mock, payload)
    return aioclient_mock


def posts(aioclient_mock, suffix):
    """Return JSON bodies of POSTs to /api/v1/models/<suffix>."""
    return [
        call[2]
        for call in aioclient_mock.mock_calls
        if call[0] == "POST" and str(call[1]).endswith(suffix)
    ]
