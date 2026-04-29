from agentic_resume_tailor.core.agents.scoring_agent import score_resume
from agentic_resume_tailor.settings import get_settings


class DummyCandidate:
    def __init__(self, bullet_id: str, weight: float) -> None:
        self.bullet_id = bullet_id
        self.effective_total_weighted = weight
        self.total_weighted = weight


def test_length_score_penalizes_short_bullets() -> None:
    settings = get_settings().model_copy(
        update={
            "rewrite_min_chars": 10,
            "rewrite_max_chars": 20,
            "length_weight": 0.0,
            "redundancy_weight": 0.0,
            "quality_weight": 0.0,
            "quant_bonus_per_hit": 0.0,
            "quant_bonus_cap": 0.0,
            "alpha": 1.0,
        }
    )

    candidates = [DummyCandidate("exp:job:b01", 1.0)]
    bullets = [{"bullet_id": "exp:job:b01", "text_latex": "Short"}]
    rewrites = {"exp:job:b01": "Short"}

    result = score_resume(
        "JD",
        target_profile=None,
        selected_candidates=candidates,
        all_candidates=candidates,
        selected_bullets_original=bullets,
        rewritten_bullets=rewrites,
        skills_text="",
        settings=settings,
    )

    assert result.length_score == 0.5


def test_length_score_rewards_in_range_bullets() -> None:
    settings = get_settings().model_copy(
        update={
            "rewrite_min_chars": 10,
            "rewrite_max_chars": 20,
            "length_weight": 0.0,
            "redundancy_weight": 0.0,
            "quality_weight": 0.0,
            "quant_bonus_per_hit": 0.0,
            "quant_bonus_cap": 0.0,
            "alpha": 1.0,
        }
    )

    candidates = [DummyCandidate("exp:job:b01", 1.0)]
    bullets = [{"bullet_id": "exp:job:b01", "text_latex": "Lengthy text here"}]
    rewrites = {"exp:job:b01": "Lengthy text here"}

    result = score_resume(
        "JD",
        target_profile=None,
        selected_candidates=candidates,
        all_candidates=candidates,
        selected_bullets_original=bullets,
        rewritten_bullets=rewrites,
        skills_text="",
        settings=settings,
    )

    assert result.length_score == 1.0
