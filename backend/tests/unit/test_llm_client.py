from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agentic_resume_tailor.core.agents import llm_client
from agentic_resume_tailor.settings import get_settings


class DummySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class FakeMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class FakeCompletions:
    def __init__(self):
        self.calls = 0
        self.requests = []

    def parse(self, **_kwargs):
        self.requests.append(_kwargs)
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad json")
        return FakeResponse(DummySchema(value="ok"))


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.responses = FakeCompletions()


def test_call_llm_json_retries_with_repair(monkeypatch) -> None:
    fake = FakeOpenAI()
    monkeypatch.setattr(llm_client, "OpenAI", lambda **_kwargs: fake)

    settings = get_settings().model_copy(
        update={"openai_api_key": "test", "agent_model": "gpt-test"}
    )
    result = llm_client.call_llm_json("prompt", DummySchema, settings=settings)

    assert result.value == "ok"
    assert fake.responses.calls == 2


def test_call_llm_json_repair_prompt_includes_schema_hint(monkeypatch) -> None:
    fake = FakeOpenAI()
    monkeypatch.setattr(llm_client, "OpenAI", lambda **_kwargs: fake)

    settings = get_settings().model_copy(
        update={"openai_api_key": "test", "agent_model": "gpt-test"}
    )
    result = llm_client.call_llm_json("prompt", DummySchema, settings=settings)

    assert result.value == "ok"
    assert len(fake.responses.requests) == 2
    repair_messages = fake.responses.requests[1]["input"]
    repair_content = str(repair_messages[-1].get("content") or "")
    assert "Schema:" in repair_content
    assert '"value"' in repair_content
