from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agentic_resume_tailor.core.keyword_matcher import (
    extract_profile_keywords,
    match_keywords_against_bullets,
)
from agentic_resume_tailor.core.retrieval import multi_query_retrieve
from agentic_resume_tailor.core.scorer import score as hybrid_score
from agentic_resume_tailor.core.selection import select_topk


@dataclass
class LoopConfig:
    max_iters: int = 3
    threshold: int = 80

    per_query_k: int = 10
    final_k: int = 30
    max_bullets: int = 16

    alpha: float = 0.7
    must_weight: float = 0.8

    # boosting behavior
    boost_weight: float = 1.6
    boost_top_n_missing: int = 6


@dataclass
class LoopResult:
    best_iteration_index: int
    best_selected_ids: List[str]
    best_candidates: List[Any]  # list[Candidate]
    best_selected_candidates: List[Any]  # list[Candidate]
    best_hybrid: Optional[Any]  # ScoreResult
    # must/nice evidences (bullets_only/all_plus_skills)
    best_evidence: Dict[str, Any]
    iterations: List[Dict[str, Any]]  # explain trace


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    """Deduplicate strings while preserving order.

    Args:
        xs: The xs value.

    Returns:
        List of results.
    """
    seen = set()
    out = []
    for x in xs:
        k = (x or "").strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def build_skills_pseudo_bullet(static_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a pseudo-bullet for skills to include in scoring.

    Args:
        static_data: Exported resume data snapshot.

    Returns:
        Dictionary result.
    """
    skills = static_data.get("skills", {}) or {}
    parts = []
    for k in ["languages_frameworks", "ai_ml", "db_tools"]:
        v = skills.get(k)
        if v:
            parts.append(str(v))
    txt = " | ".join(parts).strip()
    if not txt:
        return None
    return {"bullet_id": "__skills__", "text_latex": txt, "meta": {"section": "skills"}}


def _profile_to_query_payload(profile: Any) -> Dict[str, Any]:
    """Convert a parsed JD profile into a retrieval payload.

    Args:
        profile: The profile value.

    Returns:
        Dictionary result.
    """
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    if isinstance(profile, dict):
        return copy.deepcopy(profile)
    # last resort: try object -> dict
    return dict(profile)


def _boost_query_payload(
    base_profile_or_queries: Any,
    boost_terms: List[str],
    boost_weight: float,
) -> Tuple[Any, List[str]]:
    """Apply boost terms and return payload plus queries used.

    Args:
        base_profile_or_queries: The base profile or queries value.
        boost_terms: The boost terms value.
        boost_weight: Boost query weight for missing terms.

    Returns:
        Tuple of results.
    """
    boost_terms = _dedupe_keep_order([t.lower().strip() for t in boost_terms])

    # Case A: TargetProfile-like
    if not (
        isinstance(base_profile_or_queries, list)
        and all(isinstance(x, str) for x in base_profile_or_queries)
    ):
        payload = _profile_to_query_payload(base_profile_or_queries)
        rp = payload.get("retrieval_plan", {}) or {}
        eq = rp.get("experience_queries", []) or []

        new_eq = []
        queries_used = []
        for it in eq:
            if not isinstance(it, dict) or not it.get("query"):
                continue
            cur_boost = list(it.get("boost_keywords") or [])
            cur_boost = _dedupe_keep_order([b.lower().strip() for b in cur_boost] + boost_terms)

            new_it = dict(it)
            new_it["boost_keywords"] = cur_boost

            # optionally increase weights a bit when boosting is active
            if boost_terms:
                try:
                    new_it["weight"] = float(max(float(new_it.get("weight", 1.0)), 1.0))
                except Exception:
                    new_it["weight"] = 1.0

            new_eq.append(new_it)
            # what will actually be embedded by retrieval.py:
            qtxt = str(new_it["query"]).strip()
            if cur_boost:
                qtxt = qtxt + " " + " ".join(cur_boost)
            queries_used.append(qtxt)

        # Add one dedicated boost query to pull missing terms hard
        if boost_terms:
            boost_query = {
                "query": " ".join(boost_terms),
                "purpose": "general",
                "boost_keywords": [],
                "weight": float(boost_weight),
            }
            new_eq.append(boost_query)
            queries_used.append(boost_query["query"])

        payload["retrieval_plan"] = dict(rp)
        payload["retrieval_plan"]["experience_queries"] = new_eq
        return payload, queries_used

    # Case B: list[str] fallback queries
    base_queries: List[str] = [q.strip() for q in base_profile_or_queries if q.strip()]
    queries_used = list(base_queries)

    if boost_terms:
        # Add a boost-only query + append boosts to the first few queries
        queries_used = []
        for i, q in enumerate(base_queries):
            if i < 3:
                queries_used.append(q + " " + " ".join(boost_terms))
            else:
                queries_used.append(q)
        queries_used.append(" ".join(boost_terms))  # boost-only

    # retrieval.py can take list[str] directly
    return queries_used, queries_used


def run_loop(
    *,
    jd_text: str,
    static_data: Dict[str, Any],
    collection: Any,
    embedding_fn: Any,
    base_profile_or_queries: Any,  # TargetProfile OR list[str]
    cfg: LoopConfig,
) -> LoopResult:
    """Run the retrieval/selection/scoring loop with optional boosts.

    The loop does not call OpenAI; it only re-runs retrieval with boosted terms
    based on missing must-have keywords (bullets-only).

    Args:
        jd_text: Job description text.
        static_data: Exported resume data snapshot.
        collection: Chroma collection instance.
        embedding_fn: Embedding function.
        base_profile_or_queries: The base profile or queries value.
        cfg: The config value.

    Returns:
        Result value.
    """

    iterations: List[Dict[str, Any]] = []

    best_score = -1
    best_idx = 0
    best_selected_ids: List[str] = []
    best_candidates: List[Any] = []
    best_selected_candidates: List[Any] = []
    best_hybrid = None
    best_evidence: Dict[str, Any] = {}

    # We can only do scoring if we have a real profile with must/nice lists.
    has_profile = not (
        isinstance(base_profile_or_queries, list)
        and all(isinstance(x, str) for x in base_profile_or_queries)
    )

    pk = None
    if has_profile:
        pk = extract_profile_keywords(base_profile_or_queries)

    boost_terms: List[str] = []

    for it in range(cfg.max_iters):
        jd_payload, queries_used = _boost_query_payload(
            base_profile_or_queries=base_profile_or_queries,
            boost_terms=boost_terms,
            boost_weight=cfg.boost_weight,
        )

        # Node 2: retrieval
        cands = multi_query_retrieve(
            collection=collection,
            embedding_fn=embedding_fn,
            jd_parser_result=jd_payload,
            per_query_k=cfg.per_query_k,
            final_k=cfg.final_k,
        )

        # Node 3: selection
        selected_ids, _ = select_topk(cands, max_bullets=cfg.max_bullets)
        selected_set = set(selected_ids)
        selected_candidates = [c for c in cands if c.bullet_id in selected_set]

        # Default iteration record (even if no scoring)
        iter_entry: Dict[str, Any] = {
            "iteration": it,
            "queries_used": queries_used,
            "selected_ids": selected_ids,
            "candidate_count": len(cands),
            "scored": False,
        }

        # If no profile, we can't compute keyword coverage.
        if not has_profile or pk is None:
            iterations.append(iter_entry)
            # still track "best" by retrieval-only heuristic (mean total_weighted)
            retrieval_only = 0.0
            if selected_candidates:
                retrieval_only = sum(
                    float(getattr(c, "effective_total_weighted", c.total_weighted))
                    for c in selected_candidates
                ) / len(selected_candidates)
            if retrieval_only > best_score:
                best_score = retrieval_only
                best_idx = it
                best_selected_ids = selected_ids
                best_candidates = cands
                best_selected_candidates = selected_candidates
                best_hybrid = None
                best_evidence = {}
            continue

        # Build bullet payloads for matcher
        selected_bullets = [
            {"bullet_id": c.bullet_id, "text_latex": c.text_latex, "meta": c.meta}
            for c in selected_candidates
        ]
        all_bullets = [
            {"bullet_id": c.bullet_id, "text_latex": c.text_latex, "meta": c.meta} for c in cands
        ]
        skills_b = build_skills_pseudo_bullet(static_data)
        all_plus_skills = all_bullets + ([skills_b] if skills_b else [])

        must_evs_bullets_only = match_keywords_against_bullets(pk["must_have"], selected_bullets)
        nice_evs_bullets_only = match_keywords_against_bullets(pk["nice_to_have"], selected_bullets)

        must_evs_all = match_keywords_against_bullets(pk["must_have"], all_plus_skills)
        nice_evs_all = match_keywords_against_bullets(pk["nice_to_have"], all_plus_skills)

        hybrid = hybrid_score(
            selected_candidates=selected_candidates,
            all_candidates=cands,
            profile_keywords=pk,
            must_evs_all=must_evs_all,
            nice_evs_all=nice_evs_all,
            must_evs_bullets_only=must_evs_bullets_only,
            nice_evs_bullets_only=nice_evs_bullets_only,
            alpha=cfg.alpha,
            must_weight=cfg.must_weight,
        )

        iter_entry["scored"] = True
        iter_entry["scores"] = {
            "final": hybrid.final_score,
            "retrieval": hybrid.retrieval_score,
            "coverage_bullets_only": hybrid.coverage_bullets_only,
            "coverage_all": hybrid.coverage_all,
        }
        iter_entry["missing"] = {
            "must_bullets_only": list(hybrid.must_missing_bullets_only),
            "nice_bullets_only": list(hybrid.nice_missing_bullets_only),
            "must_all": list(hybrid.must_missing_all),
            "nice_all": list(hybrid.nice_missing_all),
        }

        iterations.append(iter_entry)

        # Track best
        if hybrid.final_score > best_score:
            best_score = hybrid.final_score
            best_idx = it
            best_selected_ids = selected_ids
            best_candidates = cands
            best_selected_candidates = selected_candidates
            best_hybrid = hybrid
            best_evidence = {
                "must_evs_bullets_only": must_evs_bullets_only,
                "nice_evs_bullets_only": nice_evs_bullets_only,
                "must_evs_all": must_evs_all,
                "nice_evs_all": nice_evs_all,
            }

        # Stop if good enough
        if hybrid.final_score >= cfg.threshold:
            break

        # Prepare next iteration boosts:
        # We boost what is missing ON PAGE (bullets-only), because that's what we want to fix.
        missing = list(hybrid.must_missing_bullets_only)
        boost_terms = _dedupe_keep_order(missing)[: cfg.boost_top_n_missing]

    return LoopResult(
        best_iteration_index=best_idx,
        best_selected_ids=best_selected_ids,
        best_candidates=best_candidates,
        best_selected_candidates=best_selected_candidates,
        best_hybrid=best_hybrid,
        best_evidence=best_evidence,
        iterations=iterations,
    )
