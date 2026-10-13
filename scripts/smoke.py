#!/usr/bin/env python3
"""Parse the live LM Studio /api/v1/models response with the integration's parser.

Read-only (a single GET). Usage: python scripts/smoke.py [host] [port]
Needs only aiohttp (api.py is loaded directly, not via the HA package).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "custom_components" / "lmstudio"


def load_api():
    """Load const.py and api.py without importing the HA-dependent __init__."""
    pkg = types.ModuleType("lmstudio_smoke")
    pkg.__path__ = [str(PKG)]
    sys.modules["lmstudio_smoke"] = pkg
    for name in ("const", "api"):
        spec = importlib.util.spec_from_file_location(
            f"lmstudio_smoke.{name}", PKG / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["lmstudio_smoke.api"]


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "10.2.3.62"
    port = sys.argv[2] if len(sys.argv) > 2 else "1234"
    url = f"http://{host}:{port}/api/v1/models"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
        payload = json.load(resp)

    api = load_api()
    data = api.parse_v1(payload)
    unique = {m["key"] for m in payload["models"]}
    assert set(data.models) == unique, "model keys mismatch"

    print(f"GET {url}")
    print(f"available models : {len(data.models)}")
    print(f"loaded instances : {data.instance_ids}")
    for key, model in data.models.items():
        size = f"{model.size_bytes / 1e9:.1f} GB" if model.size_bytes else "?"
        print(f"- {key} [{model.type}] {size}, max ctx {model.max_context_length}")
        for inst in model.instances:
            print(
                f"    loaded {inst.id}: ctx={inst.context_length} "
                f"parallel={inst.parallel} flash={inst.flash_attention}"
            )
    print("OK: parsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
