from agentic_resume_tailor.core.agents import scoring_agent
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


def test_scoring_keeps_numeric_deterministic_with_semantic_feedback(monkeypatch) -> None:
    settings = get_settings().model_copy(
        update={
            "rewrite_min_chars": 10,
            "rewrite_max_chars": 120,
            "alpha": 0.8,
            "must_weight": 0.8,
            "length_weight": 0.1,
            "redundancy_weight": 0.1,
            "quality_weight": 0.05,
        }
    )

    candidates = [DummyCandidate("exp:job:b01", 1.0)]
    bullets = [{"bullet_id": "exp:job:b01", "text_latex": "Built backend APIs."}]
    rewrites = {"exp:job:b01": "Built backend APIs."}
    target_profile = {
        "must_have": [{"raw": "Kafka", "canonical": "kafka"}],
        "nice_to_have": [],
        "responsibilities": [],
        "domain_terms": [],
    }

    semantic_output = scoring_agent.ScoringAgentOutput(
        must_missing_bullets_only=["kafka"],
        nice_missing_bullets_only=[],
        must_missing_all=["kafka"],
        nice_missing_all=[],
        candidate_boost_terms=["kafka"],
        summary="Strong base alignment but missing kafka evidence.",
        notes=["Consider a bullet with kafka ownership."],
    )
    monkeypatch.setattr(scoring_agent, "call_llm_json", lambda *_args, **_kwargs: semantic_output)

    with_semantic = score_resume(
        "JD",
        target_profile=target_profile,
        selected_candidates=candidates,
        all_candidates=candidates,
        selected_bullets_original=bullets,
        rewritten_bullets=rewrites,
        skills_text="",
        settings=settings,
    )

    def failing_call(*_args, **_kwargs):
        raise ValueError("llm unavailable")

    monkeypatch.setattr(scoring_agent, "call_llm_json", failing_call)
    deterministic = score_resume(
        "JD",
        target_profile=target_profile,
        selected_candidates=candidates,
        all_candidates=candidates,
        selected_bullets_original=bullets,
        rewritten_bullets=rewrites,
        skills_text="",
        settings=settings,
    )

    assert with_semantic.final_score == deterministic.final_score
    assert with_semantic.retrieval_score == deterministic.retrieval_score
    assert with_semantic.semantic_summary == semantic_output.summary
    assert with_semantic.semantic_notes == semantic_output.notes
    assert with_semantic.boost_terms == ["kafka"]
