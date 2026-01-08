from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from agentic_resume_tailor.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _build_messages(system_prompt: Optional[str], user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def _schema_hint(schema_model: Type[BaseModel]) -> str:
    try:
        schema = schema_model.model_json_schema()
    except Exception:
        return ""
    try:
        return json.dumps(schema, indent=2, ensure_ascii=False)
    except Exception:
        return ""


def call_llm_json(
    prompt: str,
    schema_model: Type[T],
    *,
    system_prompt: Optional[str] = None,
    settings: Any | None = None,
    model: Optional[str] = None,
) -> T:
    """Call an LLM with strict JSON schema parsing and one repair retry."""
    settings = settings or get_settings()
    api_key = getattr(settings, "openai_api_key", None)
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing; cannot call LLM.")

    model = model or getattr(settings, "agent_model", None) or getattr(settings, "jd_model", None)
    if not model:
        raise ValueError("No model configured for LLM client.")

    temperature = float(getattr(settings, "agent_temperature", 0.2))
    timeout_s = float(getattr(settings, "agent_timeout_s", 60.0))

    client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
    messages = _build_messages(system_prompt, prompt)

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema_model,
            temperature=temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned empty parsed response.")
        return parsed
    except Exception as exc:
        logger.warning("LLM JSON parse failed; attempting repair: %s", exc)

    schema_hint = _schema_hint(schema_model)
    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous output did not match the required JSON schema. "
        "Return ONLY valid JSON that satisfies the schema exactly. "
        "Do not include extra keys or commentary."
    )
    if schema_hint:
        repair_prompt = f"{repair_prompt}\n\nSchema:\n{schema_hint}"

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=_build_messages(system_prompt, repair_prompt),
        response_format=schema_model,
        temperature=temperature,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM repair failed to return parsed response.")
    return parsed
