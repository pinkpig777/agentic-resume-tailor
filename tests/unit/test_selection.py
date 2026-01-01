import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.core.selection import select_topk


class TestSelection(unittest.TestCase):
    def test_select_topk_dedupes_and_limits(self) -> None:
        """Test select topk dedupes and limits.
        """
        candidates = [
            SimpleNamespace(bullet_id="a"),
            SimpleNamespace(bullet_id="a"),
            SimpleNamespace(bullet_id="b"),
            SimpleNamespace(bullet_id="c"),
        ]

        selected, decisions = select_topk(candidates, max_bullets=2)

        self.assertEqual(selected, ["a", "b"])
        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[1].action, "skipped")
        self.assertEqual(decisions[1].reason, "duplicate_bullet_id")


if __name__ == "__main__":
    unittest.main()
