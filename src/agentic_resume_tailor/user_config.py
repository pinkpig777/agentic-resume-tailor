from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ALLOWED_KEYS = {
    "auto_reingest_on_save",
    "export_file",
}


def load_user_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in ALLOWED_KEYS if key in data}


def save_user_config(path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = {key: config[key] for key in ALLOWED_KEYS if key in config}
    config_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return filtered
