from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

from agentic_resume_tailor.core.agents.query_agent import QueryPlanItem, build_query_plan
from agentic_resume_tailor.core.agents.rewrite_agent import (
    RewriteConstraints,
    RewriteResult,
    build_rewrite_allowlist_by_bullet,
    build_rewrite_context,
    rewrite_bullets,
)
from agentic_resume_tailor.core.agents.scoring_agent import ScoreResult, score_resume
from agentic_resume_tailor.core.artifacts import (
    process_and_render_artifacts,
    render_pdf,
    trim_to_single_page,
)
from agentic_resume_tailor.core.jd_utils import build_jd_excerpt
from agentic_resume_tailor.core.prompts.query import QUERY_PROMPT_VERSION
from agentic_resume_tailor.core.prompts.rewrite import REWRITE_PROMPT_VERSION
from agentic_resume_tailor.core.prompts.scoring import SCORING_PROMPT_VERSION
from agentic_resume_tailor.core.retrieval import multi_query_retrieve
from agentic_resume_tailor.core.selection import select_topk


@dataclass
class RunArtifacts:
    run_id: str
    selected_ids: List[str]
    rewritten_bullets: Dict[str, str]
    best_score: ScoreResult
    iteration_trace: List[Dict[str, Any]]
    pdf_path: str
    tex_path: str
    report_path: str
    best_iteration_index: int
    profile_used: bool


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        value = (item or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _query_items_with_boosts(
    base_items: List[QueryPlanItem],
    boost_terms: List[str],
    boost_weight: float,
) -> List[QueryPlanItem]:
    boost_terms = _dedupe_keep_order(boost_terms)
    items: List[QueryPlanItem] = []
    for item in base_items:
        boosts = _dedupe_keep_order(list(item.boost_keywords) + boost_terms)
        items.append(
            QueryPlanItem(
                text=item.text,
                purpose=item.purpose,
                weight=item.weight if item.weight > 0 else 1.0,
                boost_keywords=boosts,
            )
        )
    if boost_terms:
        items.append(
            QueryPlanItem(
                text=" ".join(boost_terms),
                purpose="general",
                weight=float(boost_weight),
                boost_keywords=[],
            )
        )
    return items


def _query_payload(items: List[QueryPlanItem]) -> Dict[str, Any]:
    return {
        "retrieval_plan": {
            "experience_queries": [
                {
                    "query": item.text,
                    "purpose": item.purpose,
                    "weight": item.weight,
                    "boost_keywords": list(item.boost_keywords),
                }
                for item in items
            ]
        }
    }


def _queries_used(items: List[QueryPlanItem]) -> List[str]:
    queries: List[str] = []
    for item in items:
        parts = [item.text]
        if item.boost_keywords:
            parts.extend(item.boost_keywords)
        queries.append(" ".join(parts).strip())
    return queries


def _skills_text(static_export: Dict[str, Any]) -> str:
    skills = static_export.get("skills", {}) or {}
    parts = []
    for key in ["languages_frameworks", "ai_ml", "db_tools"]:
        value = skills.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts).strip()


def _collect_selected_bullets(candidates: List[Any], selected_ids: List[str]) -> List[Dict[str, Any]]:
    selected_set = set(selected_ids)
    bullets: List[Dict[str, Any]] = []
    for candidate in candidates:
        bullet_id = getattr(candidate, "bullet_id", "")
        if bullet_id not in selected_set:
            continue
        bullets.append(
            {
                "bullet_id": bullet_id,
                "text_latex": getattr(candidate, "text_latex", ""),
                "meta": getattr(candidate, "meta", {}) or {},
            }
        )
    return bullets


def _length_violations(
    length_by_bullet: Dict[str, int], min_chars: int, max_chars: int
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bullet_id, count in length_by_bullet.items():
        if count < min_chars:
            out[bullet_id] = "too_short"
        elif count > max_chars:
            out[bullet_id] = "too_long"
    return out


def _rewrite_report_entries(
    rewrite_info: Dict[str, Any], selected_ids: List[str]
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for bullet_id in selected_ids:
        info = rewrite_info.get(bullet_id)
        if not info:
            continue
        entries.append(
            {
                "bullet_id": bullet_id,
                "original_text": info.original_text,
                "rewritten_text": info.rewritten_text,
                "changed": info.changed,
                "fallback_used": info.fallback_used,
                "violations": info.validation.violations,
                "new_numbers": info.validation.new_numbers,
                "new_tools": info.validation.new_tools,
            }
        )
    return entries


def _rewrite_conditioning_report(context: Any | None, settings: Any) -> Dict[str, Any]:
    if context is None:
        return {
            "has_target_profile": False,
            "has_jd_excerpt": False,
            "must_have_keywords": [],
            "query_plan_top": [],
        }
    profile_summary = context.target_profile_summary or {}
    must_have = list(profile_summary.get("must_have") or [])
    top_n = int(getattr(settings, "rewrite_report_query_plan_top_n", 5) or 0)
    query_plan = list(context.query_plan_summary or [])
    top_items = []
    if top_n > 0:
        for item in query_plan[:top_n]:
            top_items.append(
                {
                    "query": item.get("query"),
                    "purpose": item.get("purpose"),
                    "weight": item.get("weight"),
                }
            )
    return {
        "has_target_profile": context.target_profile_summary is not None,
        "has_jd_excerpt": bool(context.jd_excerpt),
        "must_have_keywords": must_have,
        "query_plan_top": top_items,
    }


def _run_id(settings: Any) -> str:
    override = getattr(settings, "run_id", None)
    if override:
        return override
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]


def generate_run_id(settings: Any) -> str:
    """Return a run id using the standard server scheme."""
    return _run_id(settings)


def _render_pdf(settings: Any, context: Dict[str, Any], run_id: str):
    return render_pdf(settings, context, run_id)


def _trim_to_single_page(
    settings: Any,
    run_id: str,
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any],
    rewritten_bullets: Dict[str, str],
    pdf_path: str,
):
    return trim_to_single_page(
        settings,
        run_id,
        static_data,
        selected_ids,
        selected_candidates,
        pdf_path,
        rewritten_bullets=rewritten_bullets,
    )


def run_loop(
    jd_text: str,
    *,
    collection: Any,
    embedding_fn: Any,
    static_export: Dict[str, Any],
    settings: Any,
    run_id: str | None = None,
    progress_cb: Callable[[Dict[str, Any]], None] | None = None,
) -> RunArtifacts:
    run_id = run_id or _run_id(settings)
    total_iters = int(getattr(settings, "max_iters", 0) or 0)

    def _notify(stage: str, iteration: int | None = None) -> None:
        if not progress_cb:
            return
        payload: Dict[str, Any] = {"stage": stage, "max_iters": total_iters}
        if iteration is not None:
            payload["iteration"] = iteration
        progress_cb(payload)

    _notify("query")
    logger.info("[%s] QUERYING  — building target profile and retrieval plan (use_jd_parser=%s)",
                run_id, getattr(settings, "use_jd_parser", True))
    query_plan = build_query_plan(jd_text, settings)
    base_profile = query_plan.profile
    jd_excerpt = build_jd_excerpt(jd_text, max_chars=settings.jd_excerpt_max_chars)
    rewrite_context = build_rewrite_context(base_profile, query_plan.items, jd_excerpt)
    logger.info("[%s] QUERYING  — done: profile_used=%s, queries=%d",
                run_id, query_plan.profile_used, len(query_plan.items))

    iterations: List[Dict[str, Any]] = []
    best_score: ScoreResult | None = None
    best_selected_ids: List[str] = []
    best_rewrites: Dict[str, str] = {}
    best_candidates: List[Any] = []
    best_rewrite_info: Dict[str, Any] = {}
    best_idx = 0

    boost_terms: List[str] = []
    for iteration in range(settings.max_iters):
        _notify("retrieve", iteration=iteration)
        logger.info("[%s] RETRIEVING — iter=%d/%d  queries=%d  boost_terms=%s",
                    run_id, iteration + 1, total_iters,
                    len(query_plan.items), boost_terms or [])
        items = _query_items_with_boosts(query_plan.items, boost_terms, settings.boost_weight)
        payload = _query_payload(items)
        queries_used = _queries_used(items)

        candidates = multi_query_retrieve(
            collection=collection,
            embedding_fn=embedding_fn,
            jd_parser_result=payload,
            per_query_k=settings.per_query_k,
            final_k=settings.final_k,
        )
        logger.info("[%s] RETRIEVING — done: candidates=%d", run_id, len(candidates))

        _notify("select", iteration=iteration)
        selected_ids, _ = select_topk(candidates, max_bullets=settings.max_bullets)
        selected_candidates = [candidate for candidate in candidates if candidate.bullet_id in set(selected_ids)]
        selected_bullets = _collect_selected_bullets(candidates, selected_ids)
        logger.info("[%s] SELECTING  — iter=%d/%d  selected=%d bullets",
                    run_id, iteration + 1, total_iters, len(selected_ids))

        _notify("rewrite", iteration=iteration)
        _rewrite_style = getattr(settings, "rewrite_style", "conservative")
        logger.info("[%s] REWRITING  — iter=%d/%d  enabled=%s  style=%s  bullets=%d",
                    run_id, iteration + 1, total_iters,
                    settings.enable_bullet_rewrite, _rewrite_style, len(selected_bullets))
        allowlist = build_rewrite_allowlist_by_bullet(selected_bullets, settings=settings)
        constraints = RewriteConstraints(
            enabled=settings.enable_bullet_rewrite,
            min_chars=settings.rewrite_min_chars,
            max_chars=settings.rewrite_max_chars,
            style=_rewrite_style,
        )
        rewrite_result: RewriteResult = rewrite_bullets(
            rewrite_context=rewrite_context,
            bullets_original=selected_bullets,
            allowlist=allowlist,
            constraints=constraints,
            settings=settings,
        )
        changed = sum(1 for info in rewrite_result.bullet_info.values() if info.changed)
        fallbacks = sum(1 for info in rewrite_result.bullet_info.values() if info.fallback_used)
        logger.info("[%s] REWRITING  — done: changed=%d  fallbacks=%d",
                    run_id, changed, fallbacks)

        _notify("score", iteration=iteration)
        logger.info("[%s] SCORING    — iter=%d/%d  (threshold=%d)",
                    run_id, iteration + 1, total_iters, settings.threshold)
        score = score_resume(
            jd_text,
            target_profile=base_profile,
            selected_candidates=selected_candidates,
            all_candidates=candidates,
            selected_bullets_original=selected_bullets,
            rewritten_bullets=rewrite_result.rewritten_bullets,
            skills_text=_skills_text(static_export),
            settings=settings,
        )
        logger.info(
            "[%s] SCORING    — done: score=%d  retrieval=%.2f  coverage=%.2f  "
            "length=%.2f  redundancy=%.2f  must_missing=%d",
            run_id, score.final_score, score.retrieval_score,
            score.coverage_bullets_only, score.length_score,
            score.redundancy_penalty, len(score.must_missing_bullets_only),
        )

        length_violations = _length_violations(
            score.length_by_bullet, settings.rewrite_min_chars, settings.rewrite_max_chars
        )
        iteration_entry = {
            "iteration": iteration,
            "queries_used": queries_used,
            "candidate_count": len(candidates),
            "selected_ids": selected_ids,
            "score": {
                "final": score.final_score,
                "retrieval": score.retrieval_score,
                "coverage_bullets_only": score.coverage_bullets_only,
                "coverage_all": score.coverage_all,
                "length_score": score.length_score,
                "redundancy_penalty": score.redundancy_penalty,
                "quality_score": score.quality_score,
            },
            "missing": {
                "must_bullets_only": score.must_missing_bullets_only,
                "nice_bullets_only": score.nice_missing_bullets_only,
                "must_all": score.must_missing_all,
                "nice_all": score.nice_missing_all,
            },
            "rewrites": _rewrite_report_entries(rewrite_result.bullet_info, selected_ids),
            "rewrite_conditioning": _rewrite_conditioning_report(rewrite_context, settings),
            "boost_terms": score.boost_terms,
            "length": {
                "by_bullet": score.length_by_bullet,
                "violations": length_violations,
            },
            "redundancy": {
                "penalty": score.redundancy_penalty,
                "pairs": [{"a": a, "b": b, "score": s} for a, b, s in score.redundancy_pairs],
            },
            "prompt_versions": {
                "query": query_plan.prompt_version,
                "rewrite": rewrite_result.prompt_version,
                "scoring": score.prompt_version,
            },
            "rewrite_style": rewrite_result.rewrite_style,
            "scoring_semantic_feedback": {
                "summary": score.semantic_summary,
                "notes": score.semantic_notes,
                "candidate_boost_terms": score.candidate_boost_terms,
            },
            "agents": {
                "rewrite": {
                    "model": rewrite_result.agent_model,
                    "used": rewrite_result.agent_used,
                    "fallback": rewrite_result.agent_fallback,
                },
                "scoring": {
                    "model": score.agent_model,
                    "used": score.agent_used,
                    "fallback": score.agent_fallback,
                },
            },
        }
        iterations.append(iteration_entry)

        if best_score is None or score.final_score > best_score.final_score:
            best_score = score
            best_selected_ids = list(selected_ids)
            best_rewrites = dict(rewrite_result.rewritten_bullets)
            best_candidates = list(selected_candidates)
            best_rewrite_info = dict(rewrite_result.bullet_info)
            best_idx = iteration

        if base_profile is None:
            break
        if score.final_score >= settings.threshold:
            break
        boost_terms = _dedupe_keep_order(score.boost_terms)[: settings.boost_top_n_missing]

    if best_score is None:
        raise ValueError("No candidates selected; cannot build artifacts.")

    _notify("render")
    logger.info("[%s] RENDERING  — best_iter=%d  final_score=%d  bullets=%d",
                run_id, best_idx, best_score.final_score, len(best_selected_ids))
    
    base_report = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_used": query_plan.profile_used,
        "target_profile_summary": query_plan.profile_summary,
        "prompt_versions": {
            "query": QUERY_PROMPT_VERSION,
            "rewrite": REWRITE_PROMPT_VERSION,
            "scoring": SCORING_PROMPT_VERSION,
        },
        "rewrite_style": getattr(settings, "rewrite_style", "conservative"),
        "agents": {
            "query": {
                "model": query_plan.agent_model,
                "used": query_plan.agent_used,
                "fallback": query_plan.agent_fallback,
            }
        },
        "best_iteration_index": best_idx,
        "best_score": {
            "final_score": best_score.final_score,
            "retrieval_score": best_score.retrieval_score,
            "coverage_bullets_only": best_score.coverage_bullets_only,
            "coverage_all": best_score.coverage_all,
            "length_score": best_score.length_score,
            "redundancy_penalty": best_score.redundancy_penalty,
            "quality_score": best_score.quality_score,
            "must_missing_bullets_only": best_score.must_missing_bullets_only,
            "nice_missing_bullets_only": best_score.nice_missing_bullets_only,
            "must_missing_all": best_score.must_missing_all,
            "nice_missing_all": best_score.nice_missing_all,
            "boost_terms": best_score.boost_terms,
        },
        "scoring_semantic_feedback": {
            "summary": best_score.semantic_summary,
            "notes": best_score.semantic_notes,
            "candidate_boost_terms": best_score.candidate_boost_terms,
        },
        "iterations": iterations,
        "rewritten_bullets": [
            {
                "bullet_id": bullet_id,
                "original_text": info.original_text,
                "rewritten_text": info.rewritten_text,
                "changed": info.changed,
                "fallback_used": info.fallback_used,
                "violations": info.validation.violations,
                "new_numbers": info.validation.new_numbers,
                "new_tools": info.validation.new_tools,
            }
            for bullet_id, info in best_rewrite_info.items()
            if bullet_id in set(best_selected_ids)
        ]
    }

    pdf_path, tex_path, report_path, best_selected_ids, best_candidates = process_and_render_artifacts(
        settings,
        run_id,
        static_export,
        best_selected_ids,
        best_candidates,
        rewritten_bullets=best_rewrites,
        base_report=base_report,
    )

    logger.info("[%s] DONE       — pdf=%s  report=%s",
                run_id, pdf_path, report_path)
    _notify("done")
    return RunArtifacts(
        run_id=run_id,
        selected_ids=best_selected_ids,
        rewritten_bullets=best_rewrites,
        best_score=best_score,
        iteration_trace=iterations,
        pdf_path=pdf_path,
        tex_path=tex_path,
        report_path=report_path,
        best_iteration_index=best_idx,
        profile_used=query_plan.profile_used,
    )
