from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_USER_CONFIG_PATH = "config/user_settings.json"
LOCAL_USER_CONFIG_PATH = "config/user_settings.local.json"


def _load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON config file.

    Args:
        path: Filesystem path.

    Returns:
        Dictionary result.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def get_user_config_path() -> str:
    """Get user config path.

    Returns:
        String result.
    """
    env_path = os.environ.get("USER_SETTINGS_FILE")
    if env_path:
        return env_path
    local_path = Path(LOCAL_USER_CONFIG_PATH)
    if local_path.exists():
        return str(local_path)
    return DEFAULT_USER_CONFIG_PATH


def load_user_config(path: str | None = None) -> Dict[str, Any]:
    """Load user config.

    Args:
        path: Filesystem path (optional).

    Returns:
        Dictionary result.
    """
    if path:
        return _load_config_file(path)

    env_path = os.environ.get("USER_SETTINGS_FILE")
    if env_path:
        return _load_config_file(env_path)

    base = _load_config_file(DEFAULT_USER_CONFIG_PATH)
    local = _load_config_file(LOCAL_USER_CONFIG_PATH)
    if local:
        return {**base, **local}
    return base


def save_user_config(path: str | None, config: Dict[str, Any]) -> Dict[str, Any]:
    """Save user config.

    Args:
        path: Filesystem path.
        config: Configuration mapping.

    Returns:
        Dictionary result.
    """
    if path:
        config_path = Path(path)
    else:
        env_path = os.environ.get("USER_SETTINGS_FILE")
        config_path = Path(env_path or LOCAL_USER_CONFIG_PATH)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config
