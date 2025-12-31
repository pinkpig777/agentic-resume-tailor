from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_USER_CONFIG_PATH = "config/user_settings.json"


def get_user_config_path() -> str:
    return os.environ.get("USER_SETTINGS_FILE", DEFAULT_USER_CONFIG_PATH)


def load_user_config(path: str | None = None) -> Dict[str, Any]:
    config_path = Path(path or get_user_config_path())
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_user_config(path: str | None, config: Dict[str, Any]) -> Dict[str, Any]:
    config_path = Path(path or get_user_config_path())
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config
