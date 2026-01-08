from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jinja2
from pypdf import PdfReader

from agentic_resume_tailor.core.agents.query_agent import QueryPlanItem, build_query_plan
from agentic_resume_tailor.core.agents.rewrite_agent import (
    RewriteConstraints,
    RewriteResult,
    build_rewrite_allowlist,
    rewrite_bullets,
)
from agentic_resume_tailor.core.agents.scoring_agent import ScoreResultV3, score_resume
from agentic_resume_tailor.core.retrieval import multi_query_retrieve
from agentic_resume_tailor.core.selection import select_topk


@dataclass
class RunArtifactsV3:
    run_id: str
    selected_ids: List[str]
    rewritten_bullets: Dict[str, str]
    best_score: ScoreResultV3
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
        val = (item or "").strip().lower()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
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
        val = skills.get(key)
        if val:
            parts.append(str(val))
    return " | ".join(parts).strip()


def _collect_selected_bullets(
    candidates: List[Any], selected_ids: List[str]
) -> List[Dict[str, Any]]:
    selected_set = set(selected_ids)
    bullets: List[Dict[str, Any]] = []
    for c in candidates:
        bid = getattr(c, "bullet_id", "")
        if bid in selected_set:
            bullets.append(
                {
                    "bullet_id": bid,
                    "text_latex": getattr(c, "text_latex", ""),
                    "meta": getattr(c, "meta", {}) or {},
                }
            )
    return bullets


def _length_violations(
    length_by_bullet: Dict[str, int], min_chars: int, max_chars: int
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bid, count in length_by_bullet.items():
        if count < min_chars:
            out[bid] = "too_short"
        elif count > max_chars:
            out[bid] = "too_long"
    return out


def _rewrite_report_entries(
    rewrite_info: Dict[str, Any], selected_ids: List[str]
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for bid in selected_ids:
        info = rewrite_info.get(bid)
        if not info:
            continue
        entries.append(
            {
                "bullet_id": bid,
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


def _run_id(settings: Any) -> str:
    override = getattr(settings, "run_id", None)
    if override:
        return override
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]


def _output_pdf_alias_path(settings: Any) -> str | None:
    name = getattr(settings, "output_pdf_name", None)
    if not name:
        return None
    filename = os.path.basename(str(name).strip())
    if not filename:
        return None
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return os.path.join(settings.output_dir, filename)


def _write_output_pdf_alias(settings: Any, pdf_path: str) -> None:
    alias_path = _output_pdf_alias_path(settings)
    if not alias_path:
        return
    if os.path.abspath(alias_path) == os.path.abspath(pdf_path):
        return
    try:
        shutil.copyfile(pdf_path, alias_path)
    except Exception:
        return


def _render_pdf(settings: Any, context: Dict[str, Any], run_id: str) -> Tuple[str, str]:
    os.makedirs(settings.output_dir, exist_ok=True)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(settings.template_dir),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
        autoescape=False,
    )

    local_template = os.path.join(settings.template_dir, "resume.local.tex")
    template_name = "resume.local.tex" if os.path.exists(local_template) else "resume.tex"
    template = env.get_template(template_name)
    tex_content = template.render(context)

    tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    if settings.skip_pdf:
        pdf_path = os.path.join(settings.output_dir, f"{run_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"")
        _write_output_pdf_alias(settings, pdf_path)
        return pdf_path, tex_path

    subprocess.run(
        ["tectonic", tex_path, "--outdir", settings.output_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    pdf_path = os.path.join(settings.output_dir, f"{run_id}.pdf")
    _write_output_pdf_alias(settings, pdf_path)
    return pdf_path, tex_path


def _pdf_page_count(path: str) -> int | None:
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception:
        return None


def _trim_to_single_page(
    settings: Any,
    run_id: str,
    static_export: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any],
    rewritten_bullets: Dict[str, str],
    pdf_path: str,
) -> Tuple[str, str, List[str], Dict[str, str]]:
    if settings.skip_pdf:
        tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
        return pdf_path, tex_path, selected_ids, rewritten_bullets

    score_map: Dict[str, float] = {}
    for c in selected_candidates:
        score = getattr(c, "selection_score", None)
        if score is None:
            score = getattr(getattr(c, "best_hit", None), "weighted", 0.0)
        score_map[getattr(c, "bullet_id", "")] = float(score or 0.0)

    page_count = _pdf_page_count(pdf_path)
    while page_count is not None and page_count > 1 and len(selected_ids) > 1:
        ranked = [(score_map.get(bid, 0.0), bid) for bid in selected_ids]
        ranked.sort(key=lambda item: (item[0], item[1]))
        drop_id = ranked[0][1] if ranked else ""
        if not drop_id:
            break
        selected_ids = [bid for bid in selected_ids if bid != drop_id]
        rewritten_bullets.pop(drop_id, None)
        selected_candidates = [
            c for c in selected_candidates if getattr(c, "bullet_id", "") != drop_id
        ]
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        tailored = _build_tailored_snapshot(static_export, selected_ids, rewritten_bullets)
        pdf_path, tex_path = _render_pdf(settings, tailored, run_id)
        page_count = _pdf_page_count(pdf_path)

    tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
    return pdf_path, tex_path, selected_ids, rewritten_bullets


def _build_tailored_snapshot(
    static_export: Dict[str, Any],
    selected_ids: List[str],
    rewritten_bullets: Dict[str, str],
) -> Dict[str, Any]:
    selected_set = set(selected_ids)
    order_map = {bid: idx for idx, bid in enumerate(selected_ids)}
    tailored = copy.deepcopy(static_export)

    new_exps = []
    for exp in tailored.get("experiences", []) or []:
        job_id = exp.get("job_id")
        kept: List[Tuple[int, str, str]] = []
        for idx, b in enumerate(exp.get("bullets", []) or []):
            local_id = b.get("id")
            if not job_id or not local_id:
                continue
            bid = f"exp:{job_id}:{local_id}"
            if bid in selected_set:
                order = order_map.get(bid, len(order_map))
                text = rewritten_bullets.get(bid, b.get("text_latex", ""))
                tie = local_id or f"idx:{idx:04d}"
                kept.append((order, tie, text))
        if kept:
            kept.sort(key=lambda item: (item[0], item[1]))
            exp["bullets"] = [text for _, _, text in kept]
            new_exps.append(exp)

    new_projs = []
    for proj in tailored.get("projects", []) or []:
        project_id = proj.get("project_id")
        kept = []
        for idx, b in enumerate(proj.get("bullets", []) or []):
            local_id = b.get("id")
            if not project_id or not local_id:
                continue
            bid = f"proj:{project_id}:{local_id}"
            if bid in selected_set:
                order = order_map.get(bid, len(order_map))
                text = rewritten_bullets.get(bid, b.get("text_latex", ""))
                tie = local_id or f"idx:{idx:04d}"
                kept.append((order, tie, text))
        if kept:
            kept.sort(key=lambda item: (item[0], item[1]))
            proj["bullets"] = [text for _, _, text in kept]
            new_projs.append(proj)

    tailored["experiences"] = new_exps
    tailored["projects"] = new_projs
    return tailored


def run_loop_v3(
    jd_text: str,
    *,
    collection: Any,
    embedding_fn: Any,
    static_export: Dict[str, Any],
    settings: Any,
) -> RunArtifactsV3:
    run_id = _run_id(settings)
    query_plan = build_query_plan(jd_text, settings)
    base_profile = query_plan.profile

    iterations: List[Dict[str, Any]] = []
    best_score: ScoreResultV3 | None = None
    best_selected_ids: List[str] = []
    best_rewrites: Dict[str, str] = {}
    best_candidates: List[Any] = []
    best_rewrite_info: Dict[str, Any] = {}
    best_idx = 0

    boost_terms: List[str] = []
    for it in range(settings.max_iters):
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

        selected_ids, _ = select_topk(candidates, max_bullets=settings.max_bullets)
        selected_candidates = [c for c in candidates if c.bullet_id in set(selected_ids)]
        selected_bullets = _collect_selected_bullets(candidates, selected_ids)

        allowlist = build_rewrite_allowlist(
            [b["text_latex"] for b in selected_bullets], settings=settings
        )
        constraints = RewriteConstraints(
            enabled=settings.enable_bullet_rewrite,
            min_chars=settings.rewrite_min_chars,
            max_chars=settings.rewrite_max_chars,
        )
        rewrite_result: RewriteResult = rewrite_bullets(
            target_profile=base_profile,
            bullets_original=selected_bullets,
            allowlist=allowlist,
            constraints=constraints,
        )

        score = score_resume(
            target_profile=base_profile,
            selected_candidates=selected_candidates,
            all_candidates=candidates,
            selected_bullets_original=selected_bullets,
            rewritten_bullets=rewrite_result.rewritten_bullets,
            skills_text=_skills_text(static_export),
            settings=settings,
        )

        length_violations = _length_violations(
            score.length_by_bullet, settings.rewrite_min_chars, settings.rewrite_max_chars
        )

        iteration_entry = {
            "iteration": it,
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
            "length": {
                "by_bullet": score.length_by_bullet,
                "violations": length_violations,
            },
            "redundancy": {
                "penalty": score.redundancy_penalty,
                "pairs": [
                    {"a": a, "b": b, "score": s} for a, b, s in score.redundancy_pairs
                ],
            },
        }
        iterations.append(iteration_entry)

        if best_score is None or score.final_score > best_score.final_score:
            best_score = score
            best_selected_ids = list(selected_ids)
            best_rewrites = dict(rewrite_result.rewritten_bullets)
            best_candidates = list(selected_candidates)
            best_rewrite_info = dict(rewrite_result.bullet_info)
            best_idx = it

        if base_profile is None:
            break

        if score.final_score >= settings.threshold:
            break

        boost_terms = _dedupe_keep_order(score.must_missing_bullets_only)[
            : settings.boost_top_n_missing
        ]

    if best_score is None:
        raise ValueError("No candidates selected; cannot build v3 artifacts.")

    tailored = _build_tailored_snapshot(static_export, best_selected_ids, best_rewrites)
    pdf_path, tex_path = _render_pdf(settings, tailored, run_id)
    pdf_path, tex_path, best_selected_ids, best_rewrites = _trim_to_single_page(
        settings,
        run_id,
        static_export,
        best_selected_ids,
        best_candidates,
        best_rewrites,
        pdf_path,
    )

    report = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_used": query_plan.profile_used,
        "best_iteration_index": best_idx,
        "selected_ids": best_selected_ids,
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
        },
        "iterations": iterations,
        "rewritten_bullets": [
            {
                "bullet_id": bid,
                "original_text": info.original_text,
                "rewritten_text": info.rewritten_text,
                "changed": info.changed,
                "fallback_used": info.fallback_used,
                "violations": info.validation.violations,
                "new_numbers": info.validation.new_numbers,
                "new_tools": info.validation.new_tools,
            }
            for bid, info in best_rewrite_info.items()
            if bid in set(best_selected_ids)
        ],
        "artifacts": {
            "pdf": os.path.basename(pdf_path),
            "tex": os.path.basename(tex_path),
        },
    }

    report_path = os.path.join(settings.output_dir, f"{run_id}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return RunArtifactsV3(
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
