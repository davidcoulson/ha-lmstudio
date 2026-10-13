"""Minimal async client and parser for the LM Studio REST API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .const import REQUEST_TIMEOUT


class LMStudioError(Exception):
    """Base error for the LM Studio client."""


class LMStudioConnectionError(LMStudioError):
    """The server could not be reached."""


class LMStudioAuthError(LMStudioError):
    """The server rejected the API token."""


@dataclass(slots=True)
class LoadedInstance:
    """One loaded instance of a model."""

    id: str
    context_length: int | None = None
    parallel: int | None = None
    flash_attention: bool | None = None
    eval_batch_size: int | None = None
    offload_kv_cache_to_gpu: bool | None = None


@dataclass(slots=True)
class Model:
    """A downloaded model and its loaded instances."""

    key: str
    display_name: str
    type: str
    architecture: str | None = None
    quantization: str | None = None
    bits_per_weight: float | None = None
    size_bytes: int | None = None
    params_string: str | None = None
    max_context_length: int | None = None
    instances: list[LoadedInstance] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        """Return True if at least one instance is loaded."""
        return bool(self.instances)


@dataclass(slots=True)
class LMStudioData:
    """Parsed snapshot of the server."""

    models: dict[str, Model]

    @property
    def loaded_models(self) -> list[Model]:
        """Models with at least one loaded instance."""
        return [m for m in self.models.values() if m.loaded]

    @property
    def loaded_llms(self) -> dict[str, Model]:
        """Loaded LLMs keyed by model key."""
        return {k: m for k, m in self.models.items() if m.loaded and m.type == "llm"}

    @property
    def instance_ids(self) -> list[str]:
        """Ids of every loaded instance (all model types)."""
        return [i.id for m in self.models.values() for i in m.instances]


def _parse_v1_instance(raw: dict[str, Any]) -> LoadedInstance:
    cfg = raw.get("config") or {}
    return LoadedInstance(
        id=str(raw.get("id")),
        context_length=cfg.get("context_length"),
        parallel=cfg.get("parallel"),
        flash_attention=cfg.get("flash_attention"),
        eval_batch_size=cfg.get("eval_batch_size"),
        offload_kv_cache_to_gpu=cfg.get("offload_kv_cache_to_gpu"),
    )


def parse_v1(payload: dict[str, Any]) -> LMStudioData:
    """Parse a GET /api/v1/models response."""
    models: dict[str, Model] = {}
    for raw in payload.get("models") or []:
        key = raw.get("key")
        if not key:
            continue
        quant = raw.get("quantization") or {}
        models[key] = Model(
            key=key,
            display_name=raw.get("display_name") or key,
            type=raw.get("type") or "llm",
            architecture=raw.get("architecture"),
            quantization=quant.get("name"),
            bits_per_weight=quant.get("bits_per_weight"),
            size_bytes=raw.get("size_bytes"),
            params_string=raw.get("params_string"),
            max_context_length=raw.get("max_context_length"),
            instances=[
                _parse_v1_instance(i)
                for i in raw.get("loaded_instances") or []
                if i.get("id")
            ],
        )
    return LMStudioData(models=models)


def parse_v0(payload: dict[str, Any]) -> LMStudioData:
    """Parse the older GET /api/v0/models response (fewer fields)."""
    models: dict[str, Model] = {}
    for raw in payload.get("data") or []:
        key = raw.get("id")
        if not key:
            continue
        instances = []
        if raw.get("state") == "loaded":
            instances.append(
                LoadedInstance(id=key, context_length=raw.get("loaded_context_length"))
            )
        models[key] = Model(
            key=key,
            display_name=key,
            type="llm" if raw.get("type") in ("llm", "vlm") else raw.get("type", "llm"),
            architecture=raw.get("arch"),
            quantization=raw.get("quantization"),
            max_context_length=raw.get("max_context_length"),
            instances=instances,
        )
    return LMStudioData(models=models)


class LMStudioClient:
    """Talks to one LM Studio server."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        token: str | None = None,
    ) -> None:
        """Initialise the client."""
        self._session = session
        self._base = f"http://{host}:{port}"
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> Any:
        try:
            async with self._session.request(
                method,
                f"{self._base}{path}",
                json=json,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status in (401, 403):
                    raise LMStudioAuthError(f"HTTP {resp.status}")
                if resp.status == 404:
                    raise FileNotFoundError(path)
                resp.raise_for_status()
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError) as err:
            raise LMStudioConnectionError(str(err)) from err

    async def async_get_data(self) -> LMStudioData:
        """Fetch and parse the model list, falling back to /api/v0."""
        try:
            return parse_v1(await self._request("GET", "/api/v1/models"))
        except FileNotFoundError:
            pass
        try:
            return parse_v0(await self._request("GET", "/api/v0/models"))
        except FileNotFoundError as err:
            raise LMStudioConnectionError("No models endpoint found") from err

    async def async_load_model(
        self,
        model: str,
        context_length: int | None = None,
        parallel: int | None = None,
        flash_attention: bool | None = None,
    ) -> None:
        """Load a model with the given settings."""
        body: dict[str, Any] = {"model": model}
        for name, value in (
            ("context_length", context_length),
            ("parallel", parallel),
            ("flash_attention", flash_attention),
        ):
            if value is not None:
                body[name] = value
        await self._request("POST", "/api/v1/models/load", body)

    async def async_unload_model(self, instance_id: str) -> None:
        """Unload a loaded instance."""
        await self._request(
            "POST", "/api/v1/models/unload", {"instance_id": instance_id}
        )
