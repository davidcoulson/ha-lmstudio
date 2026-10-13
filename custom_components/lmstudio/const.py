"""Constants for the LM Studio integration."""

from __future__ import annotations

DOMAIN = "lmstudio"

CONF_API_TOKEN = "api_token"
CONF_SCAN_INTERVAL = "scan_interval"

DEFAULT_HOST = "10.2.3.62"
DEFAULT_PORT = 1234
DEFAULT_SCAN_INTERVAL = 30
MIN_SCAN_INTERVAL = 5
MAX_SCAN_INTERVAL = 3600

REQUEST_TIMEOUT = 10

ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_MODEL = "model"
ATTR_CONTEXT_LENGTH = "context_length"
ATTR_PARALLEL = "parallel"
ATTR_FLASH_ATTENTION = "flash_attention"
ATTR_INSTANCE_ID = "instance_id"

SERVICE_LOAD_MODEL = "load_model"
SERVICE_UNLOAD_MODEL = "unload_model"
