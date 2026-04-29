from __future__ import annotations

from typing import Any

REWRITE_PROMPT_VERSION = "rewrite_v2"

REWRITE_SYSTEM_PROMPT = """You are the Bullet Rewrite Agent for Agentic Resume Tailor.

Task: rephrase bullets to be clearer, tighter and more JD-focused while preserving facts.

Hard constraints:
- Do NOT add new numbers, metrics, tools, companies, or claims.
- Only rephrase; keep meaning and facts identical.
- Output must use the original LaTeX format, including all commands and special characters.
- Each rewritten bullet must respect the provided min/max character limits.
- Return STRICT JSON only; no extra keys or commentary.
- No period at the end of bullets.
"""


def build_rewrite_prompt(
    *,
    rewrite_context: Any,
    bullets_payload: str,
    min_chars: int,
    max_chars: int,
    rewrite_style: str,
) -> tuple[str, str]:
    mode_guidance = {
        "creative": "\n".join(
            [
                "- Use stronger verbs when supported by the original bullet.",
                "- Prefer accomplishment-first framing and clause reordering when it improves readability.",
                "- Be more expressive, but never introduce new facts, scope, or technical details.",
                "- If a creative phrasing risks semantic drift, stay conservative instead.",
            ]
        ),
        "conservative": "\n".join(
            [
                "- Prefer light clarity edits over aggressive rewriting.",
                "- Keep sentence structure close to the original unless a small rewrite clearly improves it.",
            ]
        ),
    }["creative" if rewrite_style == "creative" else "conservative"]

    user_prompt = """Rewrite conditioning context (primary sources):
Target profile summary (keywords + evidence snippets):
{target_profile_summary}

Query plan summary (purpose + weight + boost_keywords):
{query_plan_summary}

Tone reference (do not invent facts):
{jd_excerpt}

Guidance:
- Prefer target profile + query plan over the JD excerpt for wording.
- Reuse only verbs/phrases from evidence snippets; do not add new facts.
- If target profile summary is empty, make only light clarity edits and avoid keyword injection.
- Use allowed_terms for any tools/technologies/numbers that appear in a rewrite.
{mode_guidance}

Length constraints:
- min_chars: {min_chars}
- max_chars: {max_chars}

Bullets to rewrite (LaTeX-ready). Each bullet includes allowed_terms from the original text.
{bullets_payload}

Return JSON with this shape:
{{
  "rewritten_bullets": [{{"bullet_id": "...", "rewritten_text": "..."}}],
  "bullet_info": [{{"bullet_id": "...", "changed": true, "notes": ""}}]
}}
""".format(
        target_profile_summary=rewrite_context.target_profile_summary_json,
        query_plan_summary=rewrite_context.query_plan_summary_json,
        jd_excerpt=rewrite_context.jd_excerpt_text,
        mode_guidance=mode_guidance,
        min_chars=min_chars,
        max_chars=max_chars,
        bullets_payload=bullets_payload,
    )
    return REWRITE_SYSTEM_PROMPT, user_prompt
