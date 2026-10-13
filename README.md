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

Per-model entities are created when an LLM loads and removed when it unloads. Their
unique ids are keyed by model key, so reloading the same model restores the same entities.
The v0 fallback has no parallel / flash attention / size, so those report unknown.

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

Both interrupt running requests. `config_entry_id` is optional with a single server.

## Development

```bash
uv venv .venv && uv pip install -r requirements_test.txt --python .venv/bin/python
.venv/bin/ruff check . && .venv/bin/pytest
.venv/bin/python scripts/smoke.py [host] [port]   # read-only parse of the live server
```

Tests mock load/unload; nothing in them touches a real server.
