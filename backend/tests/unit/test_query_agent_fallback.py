from agentic_resume_tailor.core.agents import query_agent
from agentic_resume_tailor.settings import get_settings


def test_query_agent_fallback_on_llm_error(monkeypatch) -> None:
    def fake_call_llm_json(*_args, **_kwargs):
        raise ValueError("bad output")

    monkeypatch.setattr(query_agent, "call_llm_json", fake_call_llm_json)

    settings = get_settings().model_copy(update={"use_jd_parser": True})
    plan = query_agent.build_query_plan("Backend engineer with Python and SQL.", settings)

    assert plan.items
    assert plan.agent_fallback is True


def test_query_agent_repairs_invalid_queries(monkeypatch) -> None:
    output = query_agent.QueryAgentOutput(
        target_profile=query_agent.TargetProfile(
            role_title="Backend Engineer",
            role_summary="Build API services",
            must_have=[
                query_agent.QueryKeywordItem(
                    raw="Python",
                    canonical="python",
                    type="hard_skill",
                    priority=1,
                ),
                query_agent.QueryKeywordItem(
                    raw="FastAPI",
                    canonical="fastapi",
                    type="framework",
                    priority=1,
                ),
            ],
            responsibilities=[
                query_agent.QueryKeywordItem(
                    raw="Design APIs",
                    canonical="design apis",
                    type="responsibility",
                    priority=2,
                )
            ],
            retrieval_plan=query_agent.RetrievalPlan(
                experience_queries=[
                    query_agent.QueryItem(
                        query="python backend",
                        purpose="core_stack",
                        boost_keywords=["python"],
                        weight=1.0,
                    ),
                    query_agent.QueryItem(
                        query="backend api and cloud reliability operations",
                        purpose="general",
                        boost_keywords=[],
                        weight=1.0,
                    ),
                ]
            ),
        ),
        retrieval_plan=query_agent.RetrievalPlan(),
    )

    monkeypatch.setattr(query_agent, "call_llm_json", lambda *_args, **_kwargs: output)

    settings = get_settings().model_copy(update={"use_jd_parser": True})
    plan = query_agent.build_query_plan("Backend engineer with Python and FastAPI", settings)

    assert plan.items
    assert len(plan.items) >= 3
    assert all(" and " not in item.text.lower() for item in plan.items)
    assert all(6 <= len(item.text.split()) <= 14 for item in plan.items)
    assert any("python" in item.text for item in plan.items)
