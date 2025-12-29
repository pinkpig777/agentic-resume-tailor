from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


TIER_WEIGHTS = {
    "exact": 1.0,
    "family": 0.80,
    "substring": 0.50,
    "none": 0.0,
}


@dataclass
class ScoreResult:
    final_score: int
    retrieval_score: float     # 0..1
    coverage_score: float      # 0..1

    # Missing in the FINAL resume text (skills + selected bullets). This is the one users care about.
    must_missing: List[str]
    nice_missing: List[str]

    # Missing in selected bullets only (proof signal). Optional but useful.
    must_missing_bullets_only: List[str]
    nice_missing_bullets_only: List[str]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_retrieval_norm(selected_candidates: List[Any], all_candidates: List[Any]) -> float:
    """
    Normalization you requested:
    - numerator: mean(total_weighted) over selected candidates (after selection)
    - denominator: mean(top-N total_weighted) among all candidates (ceiling)
    """
    if not selected_candidates or not all_candidates:
        return 0.0

    n = min(len(selected_candidates), len(all_candidates))
    if n <= 0:
        return 0.0

    selected_mean = sum(float(c.total_weighted)
                        for c in selected_candidates) / len(selected_candidates)

    all_vals = sorted((float(c.total_weighted)
                      for c in all_candidates), reverse=True)
    best_possible_mean = sum(all_vals[:n]) / n

    if best_possible_mean <= 1e-9:
        return 0.0

    return clamp01(selected_mean / best_possible_mean)


def _canonical_list(profile_keywords: Dict[str, List[Dict[str, str]]], key: str) -> List[str]:
    items = profile_keywords.get(key, []) or []
    out = []
    for k in items:
        v = (k.get("canonical") or k.get("raw") or "").strip().lower()
        if v:
            out.append(v)

    seen = set()
    deduped = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _best_tier_per_keyword(keywords: List[str], evidences) -> Tuple[float, List[str]]:
    if not keywords:
        return 1.0, []

    best = {k: 0.0 for k in keywords}

    for e in evidences:
        kw = getattr(e, "keyword", None)
        if not kw or kw not in best:
            continue
        tier = getattr(e, "tier", "none")
        score = float(TIER_WEIGHTS.get(tier, 0.0))
        if score > best[kw]:
            best[kw] = score

    avg = _mean(list(best.values()))
    missing = [k for k, v in best.items() if v <= 1e-9]
    return clamp01(avg), missing


def compute_coverage_norm(
    profile_keywords: Dict[str, List[Dict[str, str]]],
    must_evs_all,
    nice_evs_all,
    must_weight: float = 0.8,
) -> Tuple[float, List[str], List[str]]:
    """
    Coverage computed against the FINAL resume text (skills + selected bullets).
    """
    must_weight = clamp01(float(must_weight))
    nice_weight = 1.0 - must_weight

    must = _canonical_list(profile_keywords, "must_have")
    nice = _canonical_list(profile_keywords, "nice_to_have")

    must_score, must_missing = _best_tier_per_keyword(must, must_evs_all)
    nice_score, nice_missing = _best_tier_per_keyword(nice, nice_evs_all)

    coverage = clamp01(must_weight * must_score + nice_weight * nice_score)
    return coverage, must_missing, nice_missing


def score(
    selected_candidates: List[Any],
    all_candidates: List[Any],
    profile_keywords: Dict[str, List[Dict[str, str]]],
    must_evs,
    nice_evs,
    alpha: float = 0.7,
    must_weight: float = 0.8,
) -> ScoreResult:
    """
    alpha blends retrieval vs coverage.
    must_weight blends must-have vs nice-to-have inside coverage.
    final_score is 0..100 (int).
    """
    alpha = clamp01(float(alpha))
    must_weight = clamp01(float(must_weight))

    r = compute_retrieval_norm(selected_candidates, all_candidates)
    c, must_missing, nice_missing = compute_coverage_norm(
        profile_keywords, must_evs, nice_evs, must_weight=must_weight
    )

    final = int(round(100 * clamp01(alpha * r + (1 - alpha) * c)))

    return ScoreResult(
        final_score=final,
        retrieval_score=r,
        coverage_score=c,
        must_missing=must_missing,
        nice_missing=nice_missing,
    )
