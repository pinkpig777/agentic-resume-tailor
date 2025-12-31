import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.core.keyword_matcher import (
    latex_to_plain_for_matching,
    match_keywords_against_bullets,
)


class TestKeywordMatcher(unittest.TestCase):
    def test_latex_to_plain_for_matching(self) -> None:
        latex = r"Built \\textbf{API} with C\_plus\_plus in \LaTeX{}"
        plain = latex_to_plain_for_matching(latex)
        self.assertIn("api", plain.lower())
        self.assertIn("plus", plain.lower())

    def test_match_keywords_exact(self) -> None:
        keywords = [{"raw": "Python", "canonical": "python"}]
        bullets = [{"bullet_id": "b1", "text_latex": "Built with Python", "meta": {}}]

        evidences = match_keywords_against_bullets(keywords, bullets)

        self.assertEqual(len(evidences), 1)
        self.assertEqual(evidences[0].tier, "exact")
        self.assertEqual(evidences[0].bullet_ids, ["b1"])


if __name__ == "__main__":
    unittest.main()
