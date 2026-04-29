from agentic_resume_tailor.core.agents.rewrite_agent import RewriteContext
from agentic_resume_tailor.core.prompts.query import build_query_prompt
from agentic_resume_tailor.core.prompts.rewrite import build_rewrite_prompt
from agentic_resume_tailor.core.prompts.scoring import build_scoring_prompt


def test_query_prompt_contains_dense_retrieval_constraints() -> None:
    system_prompt, user_prompt = build_query_prompt("Backend engineer JD")

    assert "no boolean operators" in system_prompt.lower()
    assert "semantically dense" in system_prompt.lower()
    assert "return strict json only" in system_prompt.lower()
    assert "Job description" in user_prompt


def test_rewrite_prompt_varies_by_style_and_preserves_constraints() -> None:
    context = RewriteContext(
        target_profile_summary={"must_have": ["python"]},
        query_plan_summary=[{"query": "backend api python", "purpose": "core_stack"}],
        jd_excerpt="Build backend APIs",
    )

    system_prompt, conservative_prompt = build_rewrite_prompt(
        rewrite_context=context,
        bullets_payload="[]",
        min_chars=50,
        max_chars=140,
        rewrite_style="conservative",
    )
    _, creative_prompt = build_rewrite_prompt(
        rewrite_context=context,
        bullets_payload="[]",
        min_chars=50,
        max_chars=140,
        rewrite_style="creative",
    )

    assert "do not add new numbers" in system_prompt.lower()
    assert "light clarity edits" in conservative_prompt.lower()
    assert "stronger verbs" in creative_prompt.lower()
    assert "never introduce new facts" in creative_prompt.lower()


def test_scoring_prompt_limits_llm_to_semantic_feedback() -> None:
    system_prompt, user_prompt = build_scoring_prompt(
        jd_text="JD",
        target_profile_json="{}",
        skills_text="python",
        selected_bullets_json="[]",
        rewritten_bullets_json="{}",
        signals_json="{}",
        min_chars=10,
        max_chars=120,
    )

    assert "do not calculate numeric scores" in system_prompt.lower()
    assert "candidate_boost_terms" in user_prompt
    assert "summary" in user_prompt
