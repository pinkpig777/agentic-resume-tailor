from agentic_resume_tailor.core.keyword_matcher import (
    latex_to_plain_for_matching,
    match_keywords_against_bullets,
)


def test_latex_to_plain_for_matching() -> None:
    """Test latex to plain for matching."""
    latex = r"Built \\textbf{API} with C\_plus\_plus in \LaTeX{}"
    plain = latex_to_plain_for_matching(latex)
    assert "api" in plain.lower()
    assert "plus" in plain.lower()


def test_match_keywords_exact() -> None:
    """Test match keywords exact."""
    keywords = [{"raw": "Python", "canonical": "python"}]
    bullets = [{"bullet_id": "b1", "text_latex": "Built with Python", "meta": {}}]

    evidences = match_keywords_against_bullets(keywords, bullets)

    assert len(evidences) == 1
    assert evidences[0].tier == "exact"
    assert evidences[0].bullet_ids == ["b1"]
