from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Higher tiers count more toward coverage.
TIER_WEIGHTS = {
    "exact": 1.0,
    "alias": 0.85,  # kept for compatibility if you add alias tier later
    "family": 0.80,
    "substring": 0.50,
    "none": 0.0,
}


@dataclass
class ScoreResult:
    final_score: int
    retrieval_score: float  # 0..1
    coverage_bullets_only: float  # 0..1
    coverage_all: float  # 0..1 (includes skills pseudo-bullet)
    must_missing_bullets_only: List[str]
    nice_missing_bullets_only: List[str]
    must_missing_all: List[str]
    nice_missing_all: List[str]


def clamp01(x: float) -> float:
    """Clamp a float value to the [0, 1] range.

    Args:
        x: Input vector.

    Returns:
        Float result.
    """
    return max(0.0, min(1.0, x))


def _mean(xs: List[float]) -> float:
    """Compute the mean of a list of floats.

    Args:
        xs: The xs value.

    Returns:
        Float result.
    """
    return sum(xs) / len(xs) if xs else 0.0


def _candidate_weight(c: Any) -> float:
    """Return the selection weight for a candidate.

    Args:
        c: The c value.

    Returns:
        Float result.
    """
    return float(getattr(c, "effective_total_weighted", getattr(c, "total_weighted", 0.0)) or 0.0)


def compute_retrieval_norm(selected_candidates: List[Any], all_candidates: List[Any]) -> float:
    """Compute a normalized retrieval score.

    Args:
        selected_candidates: Selected candidate bullets.
        all_candidates: All candidate bullets.

    Returns:
        Float result.
    """
    if not selected_candidates or not all_candidates:
        return 0.0

    n = min(len(selected_candidates), len(all_candidates))
    if n <= 0:
        return 0.0

    selected_mean = sum(_candidate_weight(c) for c in selected_candidates) / len(
        selected_candidates
    )

    all_vals = sorted((_candidate_weight(c) for c in all_candidates), reverse=True)
    best_possible_mean = sum(all_vals[:n]) / n

    if best_possible_mean <= 1e-9:
        return 0.0

    return clamp01(selected_mean / best_possible_mean)


def _canonical_list(profile_keywords: Dict[str, List[Dict[str, str]]], key: str) -> List[str]:
    """Extract canonical keywords for a category.

    Args:
        profile_keywords: The profile keywords value.
        key: The key value.

    Returns:
        List of results.
    """
    items = profile_keywords.get(key, []) or []
    out: List[str] = []
    for k in items:
        v = (k.get("canonical") or k.get("raw") or "").strip().lower()
        if v:
            out.append(v)

    # de-dupe, preserve order
    seen = set()
    deduped: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _best_tier_per_keyword(keywords: List[str], evidences) -> Tuple[float, List[str]]:
    """Compute coverage tier and missing keywords.

    Args:
        keywords: The keywords value.
        evidences: The evidences value.

    Returns:
        Tuple of results.
    """
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
    must_evs,
    nice_evs,
    must_weight: float = 0.8,
) -> Tuple[float, List[str], List[str]]:
    """Compute a normalized coverage score.

    Args:
        profile_keywords: The profile keywords value.
        must_evs: The must evs value.
        nice_evs: The nice evs value.
        must_weight: Must-have weight in coverage scoring (optional).

    Returns:
        Tuple of results.
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
    *,
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
    """Compute the hybrid score for a selection.

    Args:
        selected_candidates: Selected candidate bullets.
        all_candidates: All candidate bullets.
        profile_keywords: The profile keywords value.
        must_evs_all: The must evs all value.
        nice_evs_all: The nice evs all value.
        must_evs_bullets_only: The must evs bullets only value.
        nice_evs_bullets_only: The nice evs bullets only value.
        alpha: Retrieval weight in hybrid scoring (optional).
        must_weight: Must-have weight in coverage scoring (optional).

    Returns:
        Result value.
    """
    alpha = clamp01(float(alpha))
    must_weight = clamp01(float(must_weight))

    r = compute_retrieval_norm(selected_candidates, all_candidates)

    cov_bullets, must_missing_bullets, nice_missing_bullets = compute_coverage_norm(
        profile_keywords, must_evs_bullets_only, nice_evs_bullets_only, must_weight=must_weight
    )
    cov_all, must_missing_all, nice_missing_all = compute_coverage_norm(
        profile_keywords, must_evs_all, nice_evs_all, must_weight=must_weight
    )

    # Final score emphasizes bullet proof (not skills section).
    final = int(round(100 * clamp01(alpha * r + (1.0 - alpha) * cov_bullets)))

    return ScoreResult(
        final_score=final,
        retrieval_score=r,
        coverage_bullets_only=cov_bullets,
        coverage_all=cov_all,
        must_missing_bullets_only=must_missing_bullets,
        nice_missing_bullets_only=nice_missing_bullets,
        must_missing_all=must_missing_all,
        nice_missing_all=nice_missing_all,
    )
