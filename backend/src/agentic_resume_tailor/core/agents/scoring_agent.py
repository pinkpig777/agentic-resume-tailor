from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
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
from agentic_resume_tailor.core.prompts.scoring import SCORING_PROMPT_VERSION, build_scoring_prompt
from agentic_resume_tailor.core.retrieval import _compute_quant_bonus
from agentic_resume_tailor.core.scorer import clamp01, compute_coverage_norm, compute_retrieval_norm

_WORD_RE = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)


class ScoringAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    must_missing_bullets_only: List[str] = Field(default_factory=list)
    nice_missing_bullets_only: List[str] = Field(default_factory=list)
    must_missing_all: List[str] = Field(default_factory=list)
    nice_missing_all: List[str] = Field(default_factory=list)
    candidate_boost_terms: List[str] = Field(default_factory=list)
    summary: str = ""
    notes: List[str] = Field(default_factory=list)


@dataclass
class ScoreResult:
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
    semantic_summary: str = ""
    semantic_notes: List[str] = field(default_factory=list)
    candidate_boost_terms: List[str] = field(default_factory=list)
    agent_used: bool = False
    agent_fallback: bool = False
    agent_model: str | None = None
    prompt_version: str = SCORING_PROMPT_VERSION


class ScoreResultV3(ScoreResult):
    """Deprecated. Use ScoreResult."""


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
        normalized = (term or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _filter_terms(terms: List[str], allowed: List[str]) -> List[str]:
    allowed_map = {term.lower(): term for term in allowed if term}
    out: List[str] = []
    seen = set()
    for term in terms or []:
        normalized = (term or "").strip().lower()
        if not normalized or normalized not in allowed_map or normalized in seen:
            continue
        seen.add(normalized)
        out.append(allowed_map[normalized])
    return out


def _deterministic_components(
    target_profile: Any | None,
    selected_candidates: List[Any],
    all_candidates: List[Any],
    selected_bullets_original: List[Dict[str, Any]],
    rewritten_bullets: Dict[str, str],
    skills_text: str,
    settings: Any,
) -> Dict[str, Any]:
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

    components = {
        "retrieval_score": retrieval_score,
        "length_by_bullet": length_by_bullet,
        "rewritten_texts": rewritten_texts,
        "length_score": length_score,
        "redundancy_pairs": redundancy_pairs,
        "redundancy_penalty": redundancy_penalty,
        "quality_score": quality_score,
        "coverage_bullets_only": retrieval_score,
        "coverage_all": retrieval_score,
        "must_missing_bullets_only": [],
        "nice_missing_bullets_only": [],
        "must_missing_all": [],
        "nice_missing_all": [],
    }

    if target_profile is None:
        return components

    profile_keywords = extract_profile_keywords(target_profile)
    selected_bullets_for_match = [
        {"bullet_id": bid, "text_latex": txt, "meta": {"section": "selected"}}
        for bid, txt in rewritten_texts.items()
    ]
    must_evs = match_keywords_against_bullets(profile_keywords["must_have"], selected_bullets_for_match)
    nice_evs = match_keywords_against_bullets(profile_keywords["nice_to_have"], selected_bullets_for_match)
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

    components.update(
        {
            "coverage_bullets_only": coverage_bullets_only,
            "coverage_all": coverage_all,
            "must_missing_bullets_only": must_missing_bullets,
            "nice_missing_bullets_only": nice_missing_bullets,
            "must_missing_all": must_missing_all,
            "nice_missing_all": nice_missing_all,
        }
    )
    return components


def _deterministic_score(
    target_profile: Any | None,
    selected_candidates: List[Any],
    all_candidates: List[Any],
    selected_bullets_original: List[Dict[str, Any]],
    rewritten_bullets: Dict[str, str],
    skills_text: str,
    settings: Any,
    *,
    semantic_summary: str = "",
    semantic_notes: List[str] | None = None,
    candidate_boost_terms: List[str] | None = None,
    agent_used: bool = False,
    agent_model: str | None = None,
    agent_fallback: bool | None = None,
) -> ScoreResult:
    fallback_flag = agent_fallback if agent_fallback is not None else bool(agent_used)
    components = _deterministic_components(
        target_profile,
        selected_candidates,
        all_candidates,
        selected_bullets_original,
        rewritten_bullets,
        skills_text,
        settings,
    )
    base = clamp01(
        settings.alpha * components["retrieval_score"]
        + (1.0 - settings.alpha) * components["coverage_bullets_only"]
    )
    final = clamp01(
        base
        + settings.length_weight * components["length_score"]
        + settings.quality_weight * components["quality_score"]
        - settings.redundancy_weight * components["redundancy_penalty"]
    )

    must_missing = _normalize_terms(components["must_missing_bullets_only"])
    candidate_boost_terms = candidate_boost_terms or must_missing[: settings.boost_top_n_missing]
    boost_terms = _filter_terms(candidate_boost_terms, must_missing)[: settings.boost_top_n_missing]

    return ScoreResult(
        final_score=int(round(100 * final)),
        retrieval_score=components["retrieval_score"],
        coverage_bullets_only=components["coverage_bullets_only"],
        coverage_all=components["coverage_all"],
        length_score=components["length_score"],
        redundancy_penalty=components["redundancy_penalty"],
        quality_score=components["quality_score"],
        must_missing_bullets_only=components["must_missing_bullets_only"],
        nice_missing_bullets_only=components["nice_missing_bullets_only"],
        must_missing_all=components["must_missing_all"],
        nice_missing_all=components["nice_missing_all"],
        length_by_bullet=components["length_by_bullet"],
        redundancy_pairs=components["redundancy_pairs"],
        boost_terms=boost_terms,
        semantic_summary=semantic_summary,
        semantic_notes=semantic_notes or [],
        candidate_boost_terms=boost_terms,
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
) -> ScoreResult:
    """Score a resume draft using deterministic metrics plus optional semantic feedback."""
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

    components = _deterministic_components(
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
        "retrieval_score": components["retrieval_score"],
        "coverage_bullets_only": components["coverage_bullets_only"],
        "coverage_all": components["coverage_all"],
        "length_score": components["length_score"],
        "redundancy_penalty": components["redundancy_penalty"],
        "quality_score": components["quality_score"],
        "must_missing_bullets_only": components["must_missing_bullets_only"],
        "nice_missing_bullets_only": components["nice_missing_bullets_only"],
        "must_missing_all": components["must_missing_all"],
        "nice_missing_all": components["nice_missing_all"],
    }

    try:
        system_prompt, prompt = build_scoring_prompt(
            jd_text=jd_text,
            target_profile_json=json.dumps(
                target_profile.model_dump() if hasattr(target_profile, "model_dump") else target_profile,
                ensure_ascii=False,
            ),
            skills_text=skills_text or "",
            selected_bullets_json=json.dumps(selected_bullets_original, ensure_ascii=False),
            rewritten_bullets_json=json.dumps(components["rewritten_texts"], ensure_ascii=False),
            signals_json=json.dumps(signals, ensure_ascii=False),
            min_chars=settings.rewrite_min_chars,
            max_chars=settings.rewrite_max_chars,
        )
        output = call_llm_json(
            prompt,
            ScoringAgentOutput,
            system_prompt=system_prompt,
            settings=settings,
            model=model,
        )

        profile_keywords = extract_profile_keywords(target_profile)
        must_allowed = _normalize_terms(
            [item.get("canonical") or item.get("raw") for item in profile_keywords.get("must_have", [])]
        )
        _normalize_terms(
            [item.get("canonical") or item.get("raw") for item in profile_keywords.get("nice_to_have", [])]
        )

        candidate_boost_terms = _filter_terms(output.candidate_boost_terms, must_allowed)
        return _deterministic_score(
            target_profile,
            selected_candidates,
            all_candidates,
            selected_bullets_original,
            rewritten_bullets,
            skills_text,
            settings,
            semantic_summary=output.summary,
            semantic_notes=list(output.notes or []),
            candidate_boost_terms=candidate_boost_terms,
            agent_used=True,
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
            agent_fallback=True,
        )
