from __future__ import annotations

QUERY_PROMPT_VERSION = "query_v2"

QUERY_SYSTEM_PROMPT = """You are the Query Agent for Agentic Resume Tailor.

Goal: parse the job description into a target profile and a retrieval plan.

Rules:
- Use ONLY information present in the JD; do not guess.
- Canonical terms must be lowercase, concise, and tool/skill specific.
- Prefer atomic skills over combined phrases.
- Include must-have vs nice-to-have skill lists and responsibilities.
- retrieval_plan.experience_queries: 3-7 queries, 6-14 words each, no boolean operators.
- Queries must be semantically dense: combine action + domain + technical specifics.
- Avoid generic standalone queries like "leadership", "deployment", or "python".
- Return STRICT JSON only; no extra keys or commentary.
"""


def build_query_prompt(jd_text: str) -> tuple[str, str]:
    user_prompt = """Job description:
{jd_text}

Return JSON with this shape:
{{
  "target_profile": {{
    "role_title": "...",
    "role_summary": "...",
    "must_have": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 1}}],
    "nice_to_have": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 2}}],
    "responsibilities": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 2}}],
    "domain_terms": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 3}}],
    "retrieval_plan": {{"experience_queries": [{{"query": "...", "purpose": "...", "boost_keywords": [], "weight": 1.0}}]}}
  }},
  "retrieval_plan": {{"experience_queries": [{{"query": "...", "purpose": "...", "boost_keywords": [], "weight": 1.0}}]}}
}}
""".format(jd_text=jd_text)
    return QUERY_SYSTEM_PROMPT, user_prompt
