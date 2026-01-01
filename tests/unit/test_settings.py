import json

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.user_config import load_user_config, save_user_config


def test_save_and_load_user_config(tmp_path, monkeypatch) -> None:
    """Test saving and loading user config."""
    config_path = tmp_path / "user_settings.json"
    monkeypatch.setenv("USER_SETTINGS_FILE", str(config_path))
    get_settings.cache_clear()
    config = {"export_file": "data/custom.json", "max_bullets": 12}
    save_user_config(None, config)
    loaded = load_user_config(None)
    assert loaded == config


def test_settings_override_from_user_config(tmp_path, monkeypatch) -> None:
    """Test settings load values from user config."""
    config_path = tmp_path / "user_settings.json"
    monkeypatch.setenv("USER_SETTINGS_FILE", str(config_path))
    config_path.write_text(json.dumps({"max_bullets": 21}), encoding="utf-8")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.max_bullets == 21
