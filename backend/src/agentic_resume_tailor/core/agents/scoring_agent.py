from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from agentic_resume_tailor.core.agents.llm_client import call_llm_json
from agentic_resume_tailor.core.keyword_matcher import (
    extract_profile_keywords,
    latex_to_plain_for_matching,
    match_keywords_against_bullets,
)
from agentic_resume_tailor.core.retrieval import _compute_quant_bonus
from agentic_resume_tailor.core.scorer import clamp01, compute_coverage_norm, compute_retrieval_norm

_WORD_RE = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)


class ScoringAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    final_score: int = Field(ge=0, le=100)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    coverage_bullets_only: float = Field(ge=0.0, le=1.0)
    coverage_all: float = Field(ge=0.0, le=1.0)
    length_score: float = Field(ge=0.0, le=1.0)
    redundancy_penalty: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    must_missing_bullets_only: List[str] = Field(default_factory=list)
    nice_missing_bullets_only: List[str] = Field(default_factory=list)
    must_missing_all: List[str] = Field(default_factory=list)
    nice_missing_all: List[str] = Field(default_factory=list)
    boost_terms: List[str] = Field(default_factory=list)


@dataclass
class ScoreResultV3:
    final_score: int
    retrieval_score: float
    coverage_bullets_only: float
    coverage_all: float
    length_score: float
    redundancy_penalty: float
    quality_score: float
    must_missing_bullets_only: List[str]
    nice_missing_bullets_only: List[str]
    must_missing_all: List[str]
    nice_missing_all: List[str]
    length_by_bullet: Dict[str, int]
    redundancy_pairs: List[Tuple[str, str, float]]
    boost_terms: List[str]
    agent_used: bool = False
    agent_fallback: bool = False
    agent_model: str | None = None


SYSTEM_PROMPT = """You are the Scoring Agent for Agentic Resume Tailor.

Evaluate the resume draft against the JD and target profile. Provide a strict JSON score breakdown.

Rules:
- Use only the provided JD, target profile, skills text, and bullets.
- Missing keyword lists must be drawn from target_profile must-have/nice-to-have terms.
- boost_terms must be a subset of must_missing_bullets_only.
- Respect the provided retrieval_score/length_score/redundancy_penalty as fixed signals.
- Output STRICT JSON only; no extra keys or commentary.
"""


USER_TEMPLATE = """Job description:
{jd_text}

Target profile:
{target_profile}

Skills text:
{skills_text}

Selected bullets (original):
{selected_bullets}

Rewritten bullets (final):
{rewritten_bullets}

Signals:
{signals}

Length constraints:
{{"min_chars": {min_chars}, "max_chars": {max_chars}}}

Return JSON with this shape:
{{
  "final_score": 0,
  "retrieval_score": 0.0,
  "coverage_bullets_only": 0.0,
  "coverage_all": 0.0,
  "length_score": 0.0,
  "redundancy_penalty": 0.0,
  "quality_score": 0.0,
  "must_missing_bullets_only": ["..."],
  "nice_missing_bullets_only": ["..."],
  "must_missing_all": ["..."],
  "nice_missing_all": ["..."],
  "boost_terms": ["..."]
}}
"""


def _mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _token_set(text: str) -> set[str]:
    plain = latex_to_plain_for_matching(text or "").lower()
    return set(_WORD_RE.findall(plain))


def _length_score(lengths: Dict[str, int], min_chars: int, max_chars: int) -> float:
    scores: List[float] = []
    for _, count in lengths.items():
        if count <= 0:
            scores.append(0.0)
            continue
        if min_chars <= count <= max_chars:
            scores.append(1.0)
        elif count < min_chars:
            scores.append(count / float(min_chars))
        else:
            scores.append(max_chars / float(count))
    return clamp01(_mean(scores))


def _redundancy_pairs(texts: Dict[str, str], threshold: float) -> List[Tuple[str, str, float]]:
    items = list(texts.items())
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(items)):
        a_id, a_text = items[i]
        for j in range(i + 1, len(items)):
            b_id, b_text = items[j]
            a_tokens = _token_set(a_text)
            b_tokens = _token_set(b_text)
            if not a_tokens or not b_tokens:
                continue
            overlap = len(a_tokens & b_tokens) / float(len(a_tokens | b_tokens))
            if overlap >= threshold:
                pairs.append((a_id, b_id, overlap))
            else:
                ratio = SequenceMatcher(None, a_text, b_text).ratio()
                if ratio >= threshold:
                    pairs.append((a_id, b_id, ratio))
    return pairs


def _normalize_terms(terms: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for term in terms or []:
        norm = (term or "").strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _filter_terms(terms: List[str], allowed: List[str]) -> List[str]:
    allowed_map = {t.lower(): t for t in allowed if t}
    out: List[str] = []
    seen = set()
    for term in terms or []:
        norm = (term or "").strip().lower()
        if not norm or norm not in allowed_map or norm in seen:
            continue
        seen.add(norm)
        out.append(allowed_map[norm])
    return out


def _deterministic_score(
    target_profile: Any | None,
    selected_candidates: List[Any],
    all_candidates: List[Any],
    selected_bullets_original: List[Dict[str, Any]],
    rewritten_bullets: Dict[str, str],
    skills_text: str,
    settings: Any,
    *,
    agent_used: bool = False,
    agent_model: str | None = None,
    agent_fallback: bool | None = None,
) -> ScoreResultV3:
    """Deterministic fallback scoring using existing hybrid logic."""
    fallback_flag = agent_fallback if agent_fallback is not None else bool(agent_used)
    retrieval_score = compute_retrieval_norm(selected_candidates, all_candidates)

    length_by_bullet: Dict[str, int] = {}
    rewritten_texts: Dict[str, str] = {}
    for bullet in selected_bullets_original:
        bullet_id = str(bullet.get("bullet_id") or "")
        original_text = str(bullet.get("text_latex") or "")
        text = rewritten_bullets.get(bullet_id, original_text)
        rewritten_texts[bullet_id] = text
        length_by_bullet[bullet_id] = len(text.strip())

    length_score = _length_score(
        length_by_bullet, settings.rewrite_min_chars, settings.rewrite_max_chars
    )

    redundancy_pairs = _redundancy_pairs(rewritten_texts, settings.redundancy_threshold)
    total_pairs = max(len(rewritten_texts) * (len(rewritten_texts) - 1) / 2, 1)
    redundancy_penalty = clamp01(len(redundancy_pairs) / float(total_pairs))

    quality_scores: List[float] = []
    for text in rewritten_texts.values():
        bonus = _compute_quant_bonus(
            text,
            per_hit=settings.quant_bonus_per_hit,
            cap=settings.quant_bonus_cap,
        )
        if settings.quant_bonus_cap > 0:
            quality_scores.append(min(bonus, settings.quant_bonus_cap) / settings.quant_bonus_cap)
    quality_score = clamp01(_mean(quality_scores))

    if target_profile is None:
        base = retrieval_score
        return ScoreResultV3(
            final_score=int(round(100 * clamp01(base))),
            retrieval_score=retrieval_score,
            coverage_bullets_only=base,
            coverage_all=base,
            length_score=length_score,
            redundancy_penalty=redundancy_penalty,
            quality_score=quality_score,
            must_missing_bullets_only=[],
            nice_missing_bullets_only=[],
            must_missing_all=[],
            nice_missing_all=[],
            length_by_bullet=length_by_bullet,
            redundancy_pairs=redundancy_pairs,
            boost_terms=[],
            agent_used=agent_used,
            agent_fallback=fallback_flag,
            agent_model=agent_model,
        )

    profile_keywords = extract_profile_keywords(target_profile)

    selected_bullets_for_match = [
        {"bullet_id": bid, "text_latex": txt, "meta": {"section": "selected"}}
        for bid, txt in rewritten_texts.items()
    ]
    must_evs = match_keywords_against_bullets(
        profile_keywords["must_have"], selected_bullets_for_match
    )
    nice_evs = match_keywords_against_bullets(
        profile_keywords["nice_to_have"], selected_bullets_for_match
    )
    coverage_bullets_only, must_missing_bullets, nice_missing_bullets = compute_coverage_norm(
        profile_keywords, must_evs, nice_evs, must_weight=settings.must_weight
    )

    all_plus_skills = list(selected_bullets_for_match)
    if skills_text:
        all_plus_skills.append(
            {"bullet_id": "__skills__", "text_latex": skills_text, "meta": {"section": "skills"}}
        )

    must_all = match_keywords_against_bullets(profile_keywords["must_have"], all_plus_skills)
    nice_all = match_keywords_against_bullets(profile_keywords["nice_to_have"], all_plus_skills)
    coverage_all, must_missing_all, nice_missing_all = compute_coverage_norm(
        profile_keywords, must_all, nice_all, must_weight=settings.must_weight
    )

    base = clamp01(
        settings.alpha * retrieval_score + (1.0 - settings.alpha) * coverage_bullets_only
    )
    final = clamp01(
        base
        + settings.length_weight * length_score
        + settings.quality_weight * quality_score
        - settings.redundancy_weight * redundancy_penalty
    )

    boost_terms = _normalize_terms(must_missing_bullets)[: settings.boost_top_n_missing]

    return ScoreResultV3(
        final_score=int(round(100 * final)),
        retrieval_score=retrieval_score,
        coverage_bullets_only=coverage_bullets_only,
        coverage_all=coverage_all,
        length_score=length_score,
        redundancy_penalty=redundancy_penalty,
        quality_score=quality_score,
        must_missing_bullets_only=must_missing_bullets,
        nice_missing_bullets_only=nice_missing_bullets,
        must_missing_all=must_missing_all,
        nice_missing_all=nice_missing_all,
        length_by_bullet=length_by_bullet,
        redundancy_pairs=redundancy_pairs,
        boost_terms=boost_terms,
        agent_used=agent_used,
        agent_fallback=fallback_flag,
        agent_model=agent_model,
    )


def score_resume(
    jd_text: str,
    target_profile: Any | None,
    selected_candidates: List[Any],
    all_candidates: List[Any],
    selected_bullets_original: List[Dict[str, Any]],
    rewritten_bullets: Dict[str, str],
    skills_text: str,
    settings: Any,
) -> ScoreResultV3:
    """Score a resume draft using retrieval, coverage, length, redundancy, and quality."""
    retrieval_score = compute_retrieval_norm(selected_candidates, all_candidates)

    length_by_bullet: Dict[str, int] = {}
    rewritten_texts: Dict[str, str] = {}
    for bullet in selected_bullets_original:
        bullet_id = str(bullet.get("bullet_id") or "")
        original_text = str(bullet.get("text_latex") or "")
        text = rewritten_bullets.get(bullet_id, original_text)
        rewritten_texts[bullet_id] = text
        length_by_bullet[bullet_id] = len(text.strip())

    length_score = _length_score(
        length_by_bullet, settings.rewrite_min_chars, settings.rewrite_max_chars
    )

    redundancy_pairs = _redundancy_pairs(rewritten_texts, settings.redundancy_threshold)
    total_pairs = max(len(rewritten_texts) * (len(rewritten_texts) - 1) / 2, 1)
    redundancy_penalty = clamp01(len(redundancy_pairs) / float(total_pairs))

    quality_scores: List[float] = []
    for text in rewritten_texts.values():
        bonus = _compute_quant_bonus(
            text,
            per_hit=settings.quant_bonus_per_hit,
            cap=settings.quant_bonus_cap,
        )
        if settings.quant_bonus_cap > 0:
            quality_scores.append(min(bonus, settings.quant_bonus_cap) / settings.quant_bonus_cap)
    quality_score_hint = clamp01(_mean(quality_scores))

    if target_profile is None:
        return _deterministic_score(
            target_profile,
            selected_candidates,
            all_candidates,
            selected_bullets_original,
            rewritten_bullets,
            skills_text,
            settings,
        )

    model = getattr(settings, "agent_model", None) or getattr(settings, "jd_model", None)
    signals = {
        "retrieval_score": retrieval_score,
        "length_score": length_score,
        "redundancy_penalty": redundancy_penalty,
        "quant_bonus_hint": quality_score_hint,
    }

    try:
        prompt = USER_TEMPLATE.format(
            jd_text=jd_text,
            target_profile=json.dumps(
                target_profile.model_dump()
                if hasattr(target_profile, "model_dump")
                else target_profile,
                ensure_ascii=False,
            ),
            skills_text=skills_text or "",
            selected_bullets=json.dumps(selected_bullets_original, ensure_ascii=False),
            rewritten_bullets=json.dumps(rewritten_texts, ensure_ascii=False),
            signals=json.dumps(signals, ensure_ascii=False),
            min_chars=settings.rewrite_min_chars,
            max_chars=settings.rewrite_max_chars,
        )
        output = call_llm_json(
            prompt,
            ScoringAgentOutput,
            system_prompt=SYSTEM_PROMPT,
            settings=settings,
            model=model,
        )

        profile_keywords = extract_profile_keywords(target_profile)
        must_allowed = _normalize_terms(
            [k.get("canonical") or k.get("raw") for k in profile_keywords.get("must_have", [])]
        )
        nice_allowed = _normalize_terms(
            [k.get("canonical") or k.get("raw") for k in profile_keywords.get("nice_to_have", [])]
        )

        must_missing_bullets_only = _filter_terms(output.must_missing_bullets_only, must_allowed)
        nice_missing_bullets_only = _filter_terms(output.nice_missing_bullets_only, nice_allowed)
        must_missing_all = _filter_terms(output.must_missing_all, must_allowed)
        nice_missing_all = _filter_terms(output.nice_missing_all, nice_allowed)

        boost_terms = _filter_terms(output.boost_terms, must_missing_bullets_only)

        return ScoreResultV3(
            final_score=int(output.final_score),
            retrieval_score=retrieval_score,
            coverage_bullets_only=output.coverage_bullets_only,
            coverage_all=output.coverage_all,
            length_score=length_score,
            redundancy_penalty=redundancy_penalty,
            quality_score=output.quality_score,
            must_missing_bullets_only=must_missing_bullets_only,
            nice_missing_bullets_only=nice_missing_bullets_only,
            must_missing_all=must_missing_all,
            nice_missing_all=nice_missing_all,
            length_by_bullet=length_by_bullet,
            redundancy_pairs=redundancy_pairs,
            boost_terms=boost_terms,
            agent_used=True,
            agent_fallback=False,
            agent_model=model,
        )
    except Exception:
        logger.exception("Scoring agent failed; falling back to deterministic scoring.")
        return _deterministic_score(
            target_profile,
            selected_candidates,
            all_candidates,
            selected_bullets_original,
            rewritten_bullets,
            skills_text,
            settings,
            agent_used=True,
            agent_model=model,
        )
