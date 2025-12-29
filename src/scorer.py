from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# Higher tiers count more toward coverage.
# (keyword_matcher currently emits: exact | family | substring | none)
# Keep alias for future compatibility.
TIER_WEIGHTS: Dict[str, float] = {
    "exact": 1.00,
    "alias": 0.85,
    "family": 0.80,
    "substring": 0.50,
    "none": 0.00,
}


@dataclass
class ScoreResult:
    """
    Two coverage views:
    - bullets_only: only the selected bullets (proof for what appears on the page)
    - all: all retrieved bullets + (optional) a pseudo "skills" bullet (to avoid false "missing" when skill exists)
    """
    final_score: int                 # 0..100
    retrieval_score: float           # 0..1
    coverage_bullets_only: float     # 0..1
    coverage_all: float              # 0..1

    must_missing_bullets_only: List[str]
    nice_missing_bullets_only: List[str]
    must_missing_all: List[str]
    nice_missing_all: List[str]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def compute_retrieval_norm(selected_candidates: List[Any], all_candidates: List[Any]) -> float:
    """
    Normalize retrieval strength by comparing:
      mean(selected.total_weighted)
    against the ceiling:
      mean(top-N all_candidates.total_weighted), where N = len(selected_candidates)

    This makes the score stable across different JDs and different final_k.
    """
    if not selected_candidates or not all_candidates:
        return 0.0

    n = min(len(selected_candidates), len(all_candidates))
    if n <= 0:
        return 0.0

    selected_mean = _mean([float(c.total_weighted)
                          for c in selected_candidates])

    all_vals = sorted((float(c.total_weighted)
                      for c in all_candidates), reverse=True)
    best_possible_mean = _mean(all_vals[:n])

    if best_possible_mean <= 1e-9:
        return 0.0

    return clamp01(selected_mean / best_possible_mean)


def _canonical_list(profile_keywords: Dict[str, List[Dict[str, str]]], key: str) -> List[str]:
    """
    profile_keywords should come from keyword_matcher.extract_profile_keywords(profile),
    which returns lists of dicts like: {raw, canonical, ...}.

    We always lowercase and de-dupe while preserving order.
    """
    items = profile_keywords.get(key, []) or []
    out: List[str] = []
    for k in items:
        v = (k.get("canonical") or k.get("raw") or "").strip().lower()
        if v:
            out.append(v)

    seen = set()
    deduped: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _best_tier_per_keyword(keywords: List[str], evidences) -> Tuple[float, List[str]]:
    """
    For each keyword, take the best tier score among evidences.
    Returns (avg_score, missing_keywords)
    """
    if not keywords:
        return 1.0, []

    best: Dict[str, float] = {k: 0.0 for k in keywords}

    for e in evidences:
        kw = getattr(e, "keyword", None)
        if not kw:
            continue
        kw = str(kw).strip().lower()
        if kw not in best:
            continue

        tier = str(getattr(e, "tier", "none") or "none").strip().lower()
        score = float(TIER_WEIGHTS.get(tier, 0.0))
        if score > best[kw]:
            best[kw] = score

    avg = _mean(list(best.values()))
    missing = [k for k, v in best.items() if v <= 1e-9]
    return clamp01(avg), missing


def compute_coverage_norm(
    profile_keywords: Dict[str, List[Dict[str, str]]],
    must_evs,
    nice_evs,
    must_weight: float = 0.8,
) -> Tuple[float, List[str], List[str]]:
    """
    Coverage is tier-weighted (exact > alias/family > substring).
    must_weight controls how much must-have dominates coverage.
    """
    must_weight = clamp01(float(must_weight))
    nice_weight = 1.0 - must_weight

    must = _canonical_list(profile_keywords, "must_have")
    nice = _canonical_list(profile_keywords, "nice_to_have")

    must_score, must_missing = _best_tier_per_keyword(must, must_evs)
    nice_score, nice_missing = _best_tier_per_keyword(nice, nice_evs)

    coverage = clamp01(must_weight * must_score + nice_weight * nice_score)
    return coverage, must_missing, nice_missing


def score(
    selected_candidates: List[Any],
    all_candidates: List[Any],
    profile_keywords: Dict[str, List[Dict[str, str]]],
    must_evs_all,
    nice_evs_all,
    must_evs_bullets_only,
    nice_evs_bullets_only,
    alpha: float = 0.7,
    must_weight: float = 0.8,
) -> ScoreResult:
    """
    alpha blends retrieval vs coverage (bullets-only coverage).
    final_score is 0..100 (int).
    """
    alpha = clamp01(float(alpha))

    r = compute_retrieval_norm(selected_candidates, all_candidates)

    cov_bullets, must_missing_b, nice_missing_b = compute_coverage_norm(
        profile_keywords=profile_keywords,
        must_evs=must_evs_bullets_only,
        nice_evs=nice_evs_bullets_only,
        must_weight=must_weight,
    )

    cov_all, must_missing_all, nice_missing_all = compute_coverage_norm(
        profile_keywords=profile_keywords,
        must_evs=must_evs_all,
        nice_evs=nice_evs_all,
        must_weight=must_weight,
    )

    final = int(round(100 * clamp01(alpha * r + (1.0 - alpha) * cov_bullets)))

    return ScoreResult(
        final_score=final,
        retrieval_score=r,
        coverage_bullets_only=cov_bullets,
        coverage_all=cov_all,
        must_missing_bullets_only=must_missing_b,
        nice_missing_bullets_only=nice_missing_b,
        must_missing_all=must_missing_all,
        nice_missing_all=nice_missing_all,
    )
