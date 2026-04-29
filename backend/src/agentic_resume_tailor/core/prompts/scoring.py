from __future__ import annotations

SCORING_PROMPT_VERSION = "scoring_v2"

SCORING_SYSTEM_PROMPT = """You are the Scoring Agent for Agentic Resume Tailor.

Evaluate semantic alignment between the resume draft and the JD/target profile.

Rules:
- Use only the provided JD, target profile, skills text, and bullets.
- Do not calculate numeric scores.
- Missing keyword lists must be drawn from target_profile must-have/nice-to-have terms.
- candidate_boost_terms must be a subset of must_missing_bullets_only.
- Return STRICT JSON only; no extra keys or commentary.
"""


def build_scoring_prompt(
    *,
    jd_text: str,
    target_profile_json: str,
    skills_text: str,
    selected_bullets_json: str,
    rewritten_bullets_json: str,
    signals_json: str,
    min_chars: int,
    max_chars: int,
) -> tuple[str, str]:
    user_prompt = """Job description:
{jd_text}

Target profile:
{target_profile}

Skills text:
{skills_text}

Selected bullets (original):
{selected_bullets}

Rewritten bullets (final):
{rewritten_bullets}

Deterministic signals:
{signals}

Length constraints:
{{"min_chars": {min_chars}, "max_chars": {max_chars}}}

Return JSON with this shape:
{{
  "must_missing_bullets_only": ["..."],
  "nice_missing_bullets_only": ["..."],
  "must_missing_all": ["..."],
  "nice_missing_all": ["..."],
  "candidate_boost_terms": ["..."],
  "summary": "...",
  "notes": ["..."]
}}
""".format(
        jd_text=jd_text,
        target_profile=target_profile_json,
        skills_text=skills_text or "",
        selected_bullets=selected_bullets_json,
        rewritten_bullets=rewritten_bullets_json,
        signals=signals_json,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    return SCORING_SYSTEM_PROMPT, user_prompt
