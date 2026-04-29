from agentic_resume_tailor.core.agents.query_agent import QueryPlanItem
from agentic_resume_tailor.core.agents.rewrite_agent import (
    RewriteConstraints,
    RewriteContext,
    RewriteAgentOutput,
    RewriteBulletInfoOut,
    RewriteBulletOut,
    build_rewrite_context,
    rewrite_bullets,
)


def _fake_output(bullet_id: str, text: str) -> RewriteAgentOutput:
    return RewriteAgentOutput(
        rewritten_bullets=[RewriteBulletOut(bullet_id=bullet_id, rewritten_text=text)],
        bullet_info=[RewriteBulletInfoOut(bullet_id=bullet_id, changed=True, notes="")],
    )


def test_rewrite_prompt_includes_context(monkeypatch) -> None:
    captured = {}

    def fake_call(prompt, *_args, **_kwargs):
        captured["prompt"] = prompt
        return _fake_output("exp:acme:b01", "Built APIs in Python.")

    monkeypatch.setattr(
        "agentic_resume_tailor.core.agents.rewrite_agent.call_llm_json", fake_call
    )

    target_profile = {
        "role_title": "Backend Engineer",
        "role_summary": "Owns API platform work.",
        "must_have": [
            {
                "raw": "Python",
                "canonical": "python",
                "evidence": [{"snippet": "Proficient in Python"}],
            }
        ],
        "nice_to_have": [
            {
                "raw": "AWS",
                "canonical": "aws",
                "evidence": [{"snippet": "AWS experience"}],
            }
        ],
    }
    query_plan_items = [
        QueryPlanItem(
            text="backend api performance",
            purpose="core_stack",
            weight=1.2,
            boost_keywords=["fastapi"],
        )
    ]
    context = build_rewrite_context(target_profile, query_plan_items, "Requirements: APIs")

    bullets = [{"bullet_id": "exp:acme:b01", "text_latex": "Built APIs in Python."}]
    constraints = RewriteConstraints(enabled=True, min_chars=10, max_chars=200)

    rewrite_bullets(
        rewrite_context=context,
        bullets_original=bullets,
        allowlist={"exp:acme:b01": {"built", "apis", "python"}},
        constraints=constraints,
    )

    prompt = captured.get("prompt", "")
    assert "backend api performance" in prompt
    assert "fastapi" in prompt
    assert "python" in prompt
    assert "Proficient in Python" in prompt
    assert "Requirements: APIs" in prompt


def test_rewrite_prompt_conservative_without_profile(monkeypatch) -> None:
    captured = {}

    def fake_call(prompt, *_args, **_kwargs):
        captured["prompt"] = prompt
        return _fake_output("exp:acme:b01", "Built APIs in Python.")

    monkeypatch.setattr(
        "agentic_resume_tailor.core.agents.rewrite_agent.call_llm_json", fake_call
    )

    context = RewriteContext(
        target_profile_summary=None,
        query_plan_summary=[{"query": "backend api"}],
        jd_excerpt="Responsibilities: Build APIs",
    )

    bullets = [{"bullet_id": "exp:acme:b01", "text_latex": "Built APIs in Python."}]
    constraints = RewriteConstraints(enabled=True, min_chars=10, max_chars=200)

    rewrite_bullets(
        rewrite_context=context,
        bullets_original=bullets,
        allowlist={"exp:acme:b01": {"built", "apis", "python"}},
        constraints=constraints,
    )

    prompt = captured.get("prompt", "")
    assert "light clarity edits" in prompt.lower()


def test_rewrite_falls_back_on_new_numbers(monkeypatch) -> None:
    def fake_call(*_args, **_kwargs):
        return _fake_output("exp:acme:b01", "Improved latency by 20%.")

    monkeypatch.setattr(
        "agentic_resume_tailor.core.agents.rewrite_agent.call_llm_json", fake_call
    )

    bullets = [{"bullet_id": "exp:acme:b01", "text_latex": "Improved latency."}]
    constraints = RewriteConstraints(enabled=True, min_chars=10, max_chars=200)

    result = rewrite_bullets(
        rewrite_context=None,
        bullets_original=bullets,
        allowlist={"exp:acme:b01": {"improved", "latency"}},
        constraints=constraints,
    )

    info = result.bullet_info["exp:acme:b01"]
    assert info.rewritten_text == "Improved latency."
    assert "new_numbers" in info.validation.violations


def test_rewrite_falls_back_on_length_violation(monkeypatch) -> None:
    def fake_call(*_args, **_kwargs):
        return _fake_output("exp:acme:b01", "This sentence is far too long.")

    monkeypatch.setattr(
        "agentic_resume_tailor.core.agents.rewrite_agent.call_llm_json", fake_call
    )

    bullets = [{"bullet_id": "exp:acme:b01", "text_latex": "Short."}]
    constraints = RewriteConstraints(enabled=True, min_chars=5, max_chars=10)

    result = rewrite_bullets(
        rewrite_context=None,
        bullets_original=bullets,
        allowlist={"exp:acme:b01": {"short"}},
        constraints=constraints,
    )

    info = result.bullet_info["exp:acme:b01"]
    assert info.rewritten_text == "Short."
    assert "too_long" in info.validation.violations
