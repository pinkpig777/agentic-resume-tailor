import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.user_config import load_user_config, save_user_config


class TestSettings(unittest.TestCase):
    def setUp(self) -> None:
        """Create a temporary user settings file."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmpdir.name) / "user_settings.json"
        os.environ["USER_SETTINGS_FILE"] = str(self.config_path)
        get_settings.cache_clear()

    def tearDown(self) -> None:
        """Clean up temporary settings."""
        get_settings.cache_clear()
        if "USER_SETTINGS_FILE" in os.environ:
            del os.environ["USER_SETTINGS_FILE"]
        self.tmpdir.cleanup()

    def test_save_and_load_user_config(self) -> None:
        """Test saving and loading user config."""
        config = {"export_file": "data/custom.json", "max_bullets": 12}
        save_user_config(None, config)
        loaded = load_user_config(None)
        self.assertEqual(loaded, config)

    def test_settings_override_from_user_config(self) -> None:
        """Test settings load values from user config."""
        self.config_path.write_text(json.dumps({"max_bullets": 21}), encoding="utf-8")
        settings = get_settings()
        self.assertEqual(settings.max_bullets, 21)


if __name__ == "__main__":
    unittest.main()
