import json

from agentic_resume_tailor.settings import (
    get_settings,
    live_settings_fields,
    restart_required_fields,
)
from agentic_resume_tailor.user_config import load_user_config, save_user_config


def test_save_and_load_user_config(tmp_path, monkeypatch) -> None:
    """Test saving and loading user config."""
    config_path = tmp_path / "user_settings.json"
    monkeypatch.setenv("USER_SETTINGS_FILE", str(config_path))
    get_settings.cache_clear()
    config = {"export_file": "data/custom.json", "max_bullets": 12}
    save_user_config(None, config)
    loaded = load_user_config(None)
    assert loaded["export_file"] == "data/custom.json"
    assert loaded["max_bullets"] == 12


def test_settings_override_from_user_config(tmp_path, monkeypatch) -> None:
    """Test settings load values from user config."""
    config_path = tmp_path / "user_settings.json"
    monkeypatch.setenv("USER_SETTINGS_FILE", str(config_path))
    config_path.write_text(json.dumps({"max_bullets": 21}), encoding="utf-8")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.max_bullets == 21


def test_settings_runtime_metadata_partition() -> None:
    restart_fields = set(restart_required_fields())
    live_fields = set(live_settings_fields())

    assert "db_path" in restart_fields
    assert "sql_db_url" in restart_fields
    assert "rewrite_style" in live_fields
    assert "jd_model" in live_fields
    assert "openai_api_key" not in live_fields
    assert restart_fields.isdisjoint(live_fields)
