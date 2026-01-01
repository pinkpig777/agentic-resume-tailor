from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_USER_CONFIG_PATH = "config/user_settings.json"
LOCAL_USER_CONFIG_PATH = "config/user_settings.local.json"
DOCKER_USER_CONFIG_PATH = "config/user_settings.docker.json"


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


def _in_docker() -> bool:
    """Check if running inside a Docker container.

    Returns:
        True if running in Docker, otherwise False.
    """
    if os.path.exists("/.dockerenv"):
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text(encoding="utf-8"):
            return True
    except Exception:
        return False
    return False


def get_user_config_path() -> str:
    """Get user config path.

    Returns:
        String result.
    """
    env_path = os.environ.get("USER_SETTINGS_FILE")
    if env_path:
        return env_path
    if _in_docker():
        return DOCKER_USER_CONFIG_PATH
    return LOCAL_USER_CONFIG_PATH


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
    if _in_docker():
        docker = _load_config_file(DOCKER_USER_CONFIG_PATH)
        return {**base, **docker} if docker else base

    local = _load_config_file(LOCAL_USER_CONFIG_PATH)
    return {**base, **local} if local else base


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
        if env_path:
            config_path = Path(env_path)
        elif _in_docker():
            config_path = Path(DOCKER_USER_CONFIG_PATH)
        else:
            config_path = Path(LOCAL_USER_CONFIG_PATH)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config
