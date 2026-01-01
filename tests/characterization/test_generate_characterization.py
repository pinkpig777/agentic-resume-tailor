import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.characterization
def test_generate_characterization() -> None:
    """Run the characterization script when explicitly enabled."""
    if not os.environ.get("RUN_CHARACTERIZATION"):
        pytest.skip("Set RUN_CHARACTERIZATION=1 to run characterization test.")
    script_path = Path(__file__).with_name("run_generate_characterization.py")
    result = subprocess.run([sys.executable, str(script_path)], check=False, text=True)
    assert result.returncode == 0
