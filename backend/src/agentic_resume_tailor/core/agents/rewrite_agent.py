from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set

from pydantic import BaseModel, ConfigDict, Field

from agentic_resume_tailor.core.agents.llm_client import call_llm_json
from agentic_resume_tailor.core.agents.rewrite_validation import (
    ValidationResult,
    validate_rewrite,
)
from agentic_resume_tailor.core.keyword_matcher import latex_to_plain_for_matching
from agentic_resume_tailor.settings import get_settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./#-]*")
logger = logging.getLogger(__name__)


class RewriteBulletOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bullet_id: str = Field(min_length=1)
    rewritten_text: str = Field(min_length=1)


class RewriteBulletInfoOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bullet_id: str = Field(min_length=1)
    changed: bool
    notes: str = ""


class RewriteAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rewritten_bullets: List[RewriteBulletOut]
    bullet_info: List[RewriteBulletInfoOut] = Field(default_factory=list)


@dataclass(frozen=True)
class RewriteConstraints:
    enabled: bool
    min_chars: int
    max_chars: int


@dataclass
class RewriteBulletInfo:
    bullet_id: str
    original_text: str
    rewritten_text: str
    changed: bool
    fallback_used: bool
    validation: ValidationResult


@dataclass
class RewriteResult:
    rewritten_bullets: Dict[str, str]
    bullet_info: Dict[str, RewriteBulletInfo]
    agent_used: bool = False
    agent_fallback: bool = False
    agent_model: str | None = None


SYSTEM_PROMPT = """You are the Bullet Rewrite Agent for Agentic Resume Tailor.

Task: rephrase bullets to be clearer and tighter while preserving facts.

Hard constraints:
- Do NOT add new numbers, metrics, tools, companies, or claims.
- Only rephrase; keep meaning and facts identical.
- Output must remain LaTeX-ready.
- Each rewritten bullet must respect the provided min/max character limits.
- Return STRICT JSON only; no extra keys or commentary.
"""


USER_TEMPLATE = """Target profile summary (may be empty):
{profile_summary}

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
"""


def _tokenize_terms(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _load_json(path: str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _expand_canon_terms(text: str, canon_cfg: Dict[str, Any]) -> Set[str]:
    tokens = set(_tokenize_terms(text))
    lowered = (text or "").lower()
    for group in canon_cfg.get("canon_groups", []) or []:
        canonical = group.get("canonical")
        variants = group.get("variants") or []
        phrases = [p for p in [canonical, *variants] if isinstance(p, str)]
        if not phrases:
            continue
        if any(p.lower() in lowered for p in phrases if p):
            for phrase in phrases:
                tokens.update(_tokenize_terms(phrase))
    return tokens


def build_rewrite_allowlist_by_bullet(
    bullets_original: Iterable[Dict[str, Any]],
    settings: Any | None = None,
) -> Dict[str, Set[str]]:
    """Build per-bullet allowlists from original text + canonicalization variants."""
    settings = settings or get_settings()
    canon_cfg = _load_json(settings.canon_config)

    allowlists: Dict[str, Set[str]] = {}
    for bullet in bullets_original:
        bullet_id = str(bullet.get("bullet_id") or "")
        text = str(bullet.get("text_latex") or "")
        if not bullet_id:
            continue
        tokens = _expand_canon_terms(text, canon_cfg)
        allowlists[bullet_id] = {t for t in tokens if t}
    return allowlists


def _profile_summary(target_profile: Any | None) -> str:
    if target_profile is None:
        return ""
    if hasattr(target_profile, "model_dump"):
        target_profile = target_profile.model_dump()
    must_have = ", ".join(
        [
            str(it.get("canonical") or it.get("raw") or "")
            for it in target_profile.get("must_have", [])
        ]
    )
    nice_to_have = ", ".join(
        [
            str(it.get("canonical") or it.get("raw") or "")
            for it in target_profile.get("nice_to_have", [])
        ]
    )
    role_title = str(target_profile.get("role_title") or "")
    role_summary = str(target_profile.get("role_summary") or "")
    parts = []
    if role_title:
        parts.append(f"role_title: {role_title}")
    if role_summary:
        parts.append(f"role_summary: {role_summary}")
    if must_have:
        parts.append(f"must_have: {must_have}")
    if nice_to_have:
        parts.append(f"nice_to_have: {nice_to_have}")
    return " | ".join(parts).strip()


def _similarity_ratio(a: str, b: str) -> float:
    a_plain = latex_to_plain_for_matching(a or "").lower()
    b_plain = latex_to_plain_for_matching(b or "").lower()
    if not a_plain or not b_plain:
        return 0.0
    return SequenceMatcher(None, a_plain, b_plain).ratio()


def _allowlist_for_bullet(
    allowlist: Iterable[str] | Mapping[str, Iterable[str]], bullet_id: str
) -> Set[str]:
    if isinstance(allowlist, Mapping):
        return {t.lower() for t in allowlist.get(bullet_id, []) if t}
    return {t.lower() for t in allowlist if t}


def rewrite_bullets(
    target_profile: Any | None,
    bullets_original: List[Dict[str, Any]],
    allowlist: Iterable[str] | Mapping[str, Iterable[str]],
    constraints: RewriteConstraints,
) -> RewriteResult:
    """Rewrite bullets safely using an LLM with validation and fallback."""
    rewritten: Dict[str, str] = {}
    info: Dict[str, RewriteBulletInfo] = {}

    settings = get_settings()
    model = getattr(settings, "agent_model", None) or getattr(settings, "jd_model", None)
    agent_used = False
    agent_fallback = False

    if not bullets_original:
        return RewriteResult(rewritten_bullets=rewritten, bullet_info=info)

    rewrites_from_llm: Dict[str, str] = {}
    if constraints.enabled:
        agent_used = True
        bullets_payload = [
            {
                "bullet_id": str(b.get("bullet_id") or ""),
                "text_latex": str(b.get("text_latex") or ""),
                "allowed_terms": sorted(
                    _allowlist_for_bullet(allowlist, str(b.get("bullet_id") or ""))
                ),
            }
            for b in bullets_original
            if b.get("bullet_id")
        ]
        prompt = USER_TEMPLATE.format(
            profile_summary=_profile_summary(target_profile),
            min_chars=constraints.min_chars,
            max_chars=constraints.max_chars,
            bullets_payload=json.dumps(bullets_payload, ensure_ascii=False),
        )
        try:
            output = call_llm_json(
                prompt,
                RewriteAgentOutput,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                model=model,
            )
            for item in output.rewritten_bullets:
                rewrites_from_llm[str(item.bullet_id)] = str(item.rewritten_text)
        except Exception:
            logger.exception("Rewrite agent failed; falling back to original bullets.")
            agent_fallback = True
    else:
        agent_used = False

    for bullet in bullets_original:
        bullet_id = str(bullet.get("bullet_id") or "")
        original = str(bullet.get("text_latex") or "")
        if not bullet_id:
            continue

        candidate = rewrites_from_llm.get(bullet_id, original).strip()
        fallback_used = False
        if not candidate:
            candidate = original
            fallback_used = True

        allow = _allowlist_for_bullet(allowlist, bullet_id)
        validation = validate_rewrite(original, candidate, allow)
        violations = list(validation.violations)

        length = len(candidate.strip())
        if constraints.enabled:
            if length < constraints.min_chars:
                violations.append("too_short")
            elif length > constraints.max_chars:
                violations.append("too_long")

        similarity = _similarity_ratio(original, candidate)
        drift_threshold = float(getattr(settings, "rewrite_similarity_threshold", 0.55))
        if candidate and similarity < drift_threshold:
            violations.append("semantic_drift")

        if violations:
            validation = ValidationResult(
                ok=False,
                violations=violations,
                new_numbers=validation.new_numbers,
                new_tools=validation.new_tools,
            )
            candidate = original
            fallback_used = True

        rewritten[bullet_id] = candidate
        info[bullet_id] = RewriteBulletInfo(
            bullet_id=bullet_id,
            original_text=original,
            rewritten_text=candidate,
            changed=candidate != original,
            fallback_used=fallback_used or agent_fallback,
            validation=validation,
        )

    return RewriteResult(
        rewritten_bullets=rewritten,
        bullet_info=info,
        agent_used=agent_used,
        agent_fallback=agent_fallback,
        agent_model=model if agent_used else None,
    )
