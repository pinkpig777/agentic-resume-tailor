import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.core.retrieval import _compute_quant_bonus


class TestRetrievalQuantBonus(unittest.TestCase):
    def test_quant_bonus_for_numbers(self) -> None:
        """Test quant bonus for numbers.
        """
        bonus = _compute_quant_bonus("Improved latency by 45% and cut costs by 2x")
        self.assertGreaterEqual(bonus, 0.05)

    def test_quant_bonus_absent(self) -> None:
        """Test quant bonus absent.
        """
        bonus = _compute_quant_bonus("Built a scalable service for users")
        self.assertEqual(bonus, 0.0)

    def test_quant_bonus_caps(self) -> None:
        """Test quant bonus caps at configured max.
        """
        text = (
            "Improved latency by 45% from 200 ms to 50 ms, cut costs by 2x for 10 users, "
            "processed 1gb data, auc 0.91"
        )
        bonus = _compute_quant_bonus(text)
        self.assertLessEqual(bonus, 0.20)
        self.assertEqual(bonus, 0.20)


if __name__ == "__main__":
    unittest.main()
