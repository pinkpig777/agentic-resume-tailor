# src/loop_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from retrieval import multi_query_retrieve
from selection import select_topk
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
from scorer import score as hybrid_score


@dataclass
class IterationResult:
    iter_idx: int
    selected_ids: List[str]
    selected_candidates: List[Any]
    score: Optional[Any]  # ScoreResult from scorer.py
    must_missing: List[str]
    nice_missing: List[str]
    boosted_terms: List[str]
    used_queries: List[str]


def _profile_to_dict(profile: Any) -> Dict[str, Any]:
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    if isinstance(profile, dict):
        return profile
    raise TypeError("profile must be a TargetProfile-like object or dict")


def _get_queries_from_profile(profile_dict: Dict[str, Any]) -> List[str]:
    rp = (profile_dict.get("retrieval_plan") or {})
    eq = (rp.get("experience_queries") or [])
    out = []
    for it in eq:
        if isinstance(it, dict) and it.get("query"):
            out.append(str(it["query"]))
    return out


def apply_boosts_to_profile(
    profile: Any,
    boosted_terms: List[str],
    top_n: int = 10,
    per_query_append: int = 4,
    add_boost_queries: bool = True,
) -> Dict[str, Any]:
    """
    Generic boosting (no hardcoding):
    - take top N boosted terms
    - append a few (per_query_append) to each existing query's boost_keywords
    - optionally add 1-2 extra "boost-only" queries with higher weight
    """
    profile_dict = _profile_to_dict(profile)
    boosted_terms = [t.strip().lower()
                     for t in (boosted_terms or []) if t and t.strip()]
    boosted_terms = boosted_terms[:top_n]

    rp = profile_dict.setdefault("retrieval_plan", {})
    eq = rp.setdefault("experience_queries", [])

    # append boosts to existing queries
    if isinstance(eq, list):
        for it in eq:
            if not isinstance(it, dict):
                continue
            b = it.get("boost_keywords") or []
            if not isinstance(b, list):
                b = []
            # add a few boosts to each query to avoid bloating embeddings
            for term in boosted_terms[:per_query_append]:
                if term not in b:
                    b.append(term)
            it["boost_keywords"] = b

    # add extra boost-only queries
    if add_boost_queries and boosted_terms:
        # One “must coverage” query + one “systems” query; both are generic.
        boost_q1 = " ".join(["must have skills"] +
                            boosted_terms[: min(8, len(boosted_terms))])
        boost_q2 = " ".join(["production experience"] +
                            boosted_terms[: min(8, len(boosted_terms))])

        eq.append(
            {
                "query": boost_q1,
                "purpose": "general",
                "boost_keywords": [],
                "weight": 1.8,
            }
        )
        eq.append(
            {
                "query": boost_q2,
                "purpose": "general",
                "boost_keywords": [],
                "weight": 1.6,
            }
        )

    # safety: cap total queries to avoid query explosion
    rp["experience_queries"] = eq[:9]
    return profile_dict


def build_selected_payloads(
    candidates: List[Any], selected_ids: List[str]
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Returns:
      selected_candidates: Candidate objects (for scorer normalization)
      selected_bullets: list of dicts for keyword_matcher
    """
    selected_set = set(selected_ids)
    selected_candidates: List[Any] = []
    selected_bullets: List[Dict[str, Any]] = []

    for c in candidates:
        if c.bullet_id in selected_set:
            selected_candidates.append(c)
            selected_bullets.append(
                {
                    "bullet_id": c.bullet_id,
                    "text_latex": c.text_latex,
                    "meta": c.meta,
                }
            )
    return selected_candidates, selected_bullets


def run_loop(
    *,
    jd_text: str,
    collection: Any,
    embedding_fn: Any,
    profile: Optional[Any],
    per_query_k: int = 10,
    final_k: int = 30,
    max_bullets: int = 16,
    threshold: int = 80,
    max_iters: int = 3,
    alpha: float = 0.7,
) -> Tuple[IterationResult, List[IterationResult]]:
    """
    Agentic loop:
      - retrieve -> select -> keyword match -> hybrid score
      - if score < threshold: boost missing must-have and retry
      - always returns best iteration
    """
    history: List[IterationResult] = []
    best: Optional[IterationResult] = None

    # If no profile: single-pass retrieval only, no loop / no scoring
    if profile is None:
        candidates = multi_query_retrieve(
            collection=collection,
            embedding_fn=embedding_fn,
            jd_parser_result=[jd_text],  # just use the JD as a single query
            per_query_k=per_query_k,
            final_k=final_k,
        )
        selected_ids, _ = select_topk(candidates, max_bullets=max_bullets)
        sel_cands, sel_bullets = build_selected_payloads(
            candidates, selected_ids)
        it = IterationResult(
            iter_idx=1,
            selected_ids=selected_ids,
            selected_candidates=sel_cands,
            score=None,
            must_missing=[],
            nice_missing=[],
            boosted_terms=[],
            used_queries=[jd_text],
        )
        history.append(it)
        return it, history

    # Start with the original profile, then boosted variants
    current_profile: Any = profile
    boosted_terms: List[str] = []

    for it_idx in range(1, max_iters + 1):
        candidates = multi_query_retrieve(
            collection=collection,
            embedding_fn=embedding_fn,
            jd_parser_result=current_profile,
            per_query_k=per_query_k,
            final_k=final_k,
        )
        selected_ids, _ = select_topk(candidates, max_bullets=max_bullets)
        sel_cands, sel_bullets = build_selected_payloads(
            candidates, selected_ids)

        pk = extract_profile_keywords(current_profile)
        must_evs = match_keywords_against_bullets(pk["must_have"], sel_bullets)
        nice_evs = match_keywords_against_bullets(
            pk["nice_to_have"], sel_bullets)

        s = hybrid_score(
            selected_candidates=sel_cands,
            all_candidates=candidates,
            profile_keywords=pk,
            must_evs=must_evs,
            nice_evs=nice_evs,
            alpha=alpha,
        )

        used_queries = _get_queries_from_profile(
            _profile_to_dict(current_profile))

        result = IterationResult(
            iter_idx=it_idx,
            selected_ids=selected_ids,
            selected_candidates=sel_cands,
            score=s,
            must_missing=list(s.must_missing),
            nice_missing=list(s.nice_missing),
            boosted_terms=list(boosted_terms),
            used_queries=used_queries,
        )
        history.append(result)

        if best is None or (result.score and best.score and result.score.final_score > best.score.final_score):
            best = result
        elif best is None:
            best = result

        # stop if good enough
        if s.final_score >= threshold:
            break

        # boost missing MUST terms for next iteration
        boosted_terms = s.must_missing[:12]
        current_profile = apply_boosts_to_profile(profile, boosted_terms)

    return best or history[-1], history
