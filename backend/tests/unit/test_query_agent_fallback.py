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
