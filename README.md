# LM Studio for Home Assistant

Custom integration that polls an [LM Studio](https://lmstudio.ai) server's REST API
(`GET /api/v1/models`, falling back to `/api/v0/models`) and shows what is **actually
loaded**: context length, parallel slots, flash attention and model size. Use it to alert
when LM Studio reloads a model with different settings after a restart.

GPU load and VRAM are deliberately not included; use LibreHardwareMonitor or similar.
The LM Studio API exposes no generation activity, so none is faked.

## Install

**HACS:** HACS → ⋮ → Custom repositories → add this repo URL, category *Integration* →
install *LM Studio* → restart Home Assistant.

**Manual:** copy `custom_components/lmstudio` to `/config/custom_components/lmstudio`
and restart.

Then Settings → Devices & services → Add integration → *LM Studio*. Enter host, port
(defaults `10.2.3.62:1234`) and an optional API token (sent as `Authorization: Bearer`).
Polling interval (default 30 s) is in the integration's *Configure* options.

## Entities (one device per server)

| Entity | Notes |
| --- | --- |
| `binary_sensor.<server>_server` | connectivity; on when the API answers |
| `sensor.<server>_loaded_models` | count; attribute `instances` lists loaded instance ids |
| `sensor.<server>_available_models` | downloaded models; attribute `models` |
| `sensor.<server>_<model>_context_length` | per loaded LLM; attributes include max context, quantization |
| `sensor.<server>_<model>_parallel_slots` | per loaded LLM |
| `sensor.<server>_<model>_size` | GB (decimal) |
| `binary_sensor.<server>_<model>_flash_attention` | per loaded LLM |

Per-model entities are created the first time an LLM is seen loaded. When it unloads they
stay registered but become *unavailable*, so entity ids stay stable and the same entities
come back when it loads again. Unique ids are keyed by model key.

If a model has several loaded instances, the entities report the instance with the
lowest id (deterministic); the context-length sensor's `instances` attribute gives the count.

An HTTP 200 response that is not a valid model list is treated as a failure: the server
sensor goes off, entities go unavailable and the last good snapshot is kept.

### API v1 vs v0

Monitoring works against `/api/v1/models` and falls back to `/api/v0/models`, which has
no parallel / flash attention / size (those report unknown). The services below need
API v1 and fail with a clear error on a v0-only server.

## Alert on a config change

```yaml
trigger:
  - platform: numeric_state
    entity_id: sensor.lm_studio_10_2_3_62_gemma_4_26b_a4b_qat_context_length
    below: 262144
```

## Services

- `lmstudio.load_model` (`model`, `context_length`, `parallel`, `flash_attention`) → `POST /api/v1/models/load`
- `lmstudio.unload_model` (`instance_id`) → `POST /api/v1/models/unload`

Both can interrupt running requests. `config_entry_id` is optional only when exactly one
server is configured; with several it is required, and the target must be loaded.
Calls are serialised per server, refresh state before and after, never retry on their own,
and use a 300 s timeout for loads. `load_model` adds an instance and is skipped if an
identical one is already loaded. Limits: `context_length` 512–1048576, `parallel` 1–16.

## Authentication and security

Authentication failures start Home Assistant's re-authentication flow so the token can be
replaced. The API token is sent as a Bearer header over **plain HTTP**: use it on a trusted
LAN only, not across untrusted networks.

## Development

```bash
uv venv .venv && uv pip install -r requirements_test.txt --python .venv/bin/python
.venv/bin/ruff check . && .venv/bin/pytest
.venv/bin/python scripts/smoke.py [host] [port]   # read-only parse of the live server
```

Tests mock load/unload; nothing in them touches a real server.
