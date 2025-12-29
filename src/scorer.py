from dataclasses import dataclass
from typing import Any, Dict, List
import math


@dataclass
class ScoreResult:
    final_score: int
    retrieval_score: float     # 0..1
    coverage_score: float      # 0..1
    must_missing: List[str]
    nice_missing: List[str]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_retrieval_norm(selected_candidates: List[Any], all_candidates: List[Any]) -> float:
    """
    selected_candidates: list of Candidate objects (from retrieval.py) for selected ids
    all_candidates: full candidate list for normalization reference
    """
    if not selected_candidates or not all_candidates:
        return 0.0

    # Use total_weighted (multi-hit reward). If you prefer best_hit, swap it.
    best = max(c.total_weighted for c in all_candidates) or 1e-9
    avg_sel = sum(c.total_weighted for c in selected_candidates) / \
        len(selected_candidates)
    return clamp01(avg_sel / best)


def compute_coverage_norm(profile_keywords: Dict[str, List[Dict[str, str]]], must_evs, nice_evs) -> (float, List[str], List[str]):
    must = [(k.get("canonical") or k.get("raw") or "").strip().lower()
            for k in profile_keywords.get("must_have", [])]
    nice = [(k.get("canonical") or k.get("raw") or "").strip().lower()
            for k in profile_keywords.get("nice_to_have", [])]
    must = [k for k in must if k]
    nice = [k for k in nice if k]

    must_cov = {e.keyword for e in must_evs if e.tier != "none"}
    nice_cov = {e.keyword for e in nice_evs if e.tier != "none"}

    must_covered = sum(1 for k in must if k in must_cov)
    nice_covered = sum(1 for k in nice if k in nice_cov)

    must_frac = (must_covered / len(must)) if must else 1.0
    nice_frac = (nice_covered / len(nice)) if nice else 1.0

    coverage = clamp01(0.8 * must_frac + 0.2 * nice_frac)

    must_missing = [k for k in must if k not in must_cov]
    nice_missing = [k for k in nice if k not in nice_cov]
    return coverage, must_missing, nice_missing


def score(
    selected_candidates: List[Any],
    all_candidates: List[Any],
    profile_keywords: Dict[str, List[Dict[str, str]]],
    must_evs,
    nice_evs,
    alpha: float = 0.3,
) -> ScoreResult:
    r = compute_retrieval_norm(selected_candidates, all_candidates)
    c, must_missing, nice_missing = compute_coverage_norm(
        profile_keywords, must_evs, nice_evs)

    final = int(round(100 * clamp01(alpha * r + (1 - alpha) * c)))

    return ScoreResult(
        final_score=final,
        retrieval_score=r,
        coverage_score=c,
        must_missing=must_missing,
        nice_missing=nice_missing,
    )
