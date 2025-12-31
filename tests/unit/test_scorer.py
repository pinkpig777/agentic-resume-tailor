import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.core.scorer import compute_retrieval_norm, score


class TestScorer(unittest.TestCase):
    def test_compute_retrieval_norm_caps_at_one(self) -> None:
        selected = [SimpleNamespace(total_weighted=2.0), SimpleNamespace(total_weighted=1.0)]
        all_candidates = [
            SimpleNamespace(total_weighted=2.0),
            SimpleNamespace(total_weighted=1.0),
            SimpleNamespace(total_weighted=0.5),
        ]

        self.assertAlmostEqual(compute_retrieval_norm(selected, all_candidates), 1.0)

    def test_compute_retrieval_norm_uses_effective_weight(self) -> None:
        selected = [
            SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
            SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
        ]
        all_candidates = [
            SimpleNamespace(total_weighted=1.0, effective_total_weighted=1.0),
            SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
        ]

        self.assertAlmostEqual(compute_retrieval_norm(selected, all_candidates), 0.8235294)

    def test_score_returns_full_match(self) -> None:
        selected = [SimpleNamespace(total_weighted=2.0), SimpleNamespace(total_weighted=1.0)]
        all_candidates = [
            SimpleNamespace(total_weighted=2.0),
            SimpleNamespace(total_weighted=1.0),
            SimpleNamespace(total_weighted=0.5),
        ]
        profile_keywords = {
            "must_have": [{"raw": "Python", "canonical": "python"}],
            "nice_to_have": [],
        }
        evidence = [SimpleNamespace(keyword="python", tier="exact")]

        result = score(
            selected_candidates=selected,
            all_candidates=all_candidates,
            profile_keywords=profile_keywords,
            must_evs_all=evidence,
            nice_evs_all=[],
            must_evs_bullets_only=evidence,
            nice_evs_bullets_only=[],
            alpha=0.5,
            must_weight=1.0,
        )

        self.assertEqual(result.final_score, 100)
        self.assertEqual(result.must_missing_bullets_only, [])


if __name__ == "__main__":
    unittest.main()
