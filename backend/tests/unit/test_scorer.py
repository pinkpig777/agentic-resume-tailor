from types import SimpleNamespace

import pytest

from agentic_resume_tailor.core.scorer import compute_retrieval_norm, score


def test_compute_retrieval_norm_caps_at_one() -> None:
    """Test compute retrieval norm caps at one."""
    selected = [SimpleNamespace(total_weighted=2.0), SimpleNamespace(total_weighted=1.0)]
    all_candidates = [
        SimpleNamespace(total_weighted=2.0),
        SimpleNamespace(total_weighted=1.0),
        SimpleNamespace(total_weighted=0.5),
    ]

    assert compute_retrieval_norm(selected, all_candidates) == 1.0


def test_compute_retrieval_norm_uses_effective_weight() -> None:
    """Test compute retrieval norm uses effective weight."""
    selected = [
        SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
        SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
    ]
    all_candidates = [
        SimpleNamespace(total_weighted=1.0, effective_total_weighted=1.0),
        SimpleNamespace(total_weighted=0.5, effective_total_weighted=0.7),
    ]

    assert compute_retrieval_norm(selected, all_candidates) == pytest.approx(0.8235294)


def test_score_returns_full_match() -> None:
    """Test score returns full match."""
    selected = [SimpleNamespace(total_weighted=2.0), SimpleNamespace(total_weighted=1.0)]
    all_candidates = [
        SimpleNamespace(total_weighted=2.0),
        SimpleNamespace(total_weighted=1.0),
        SimpleNamespace(total_weighted=0.5),
    ]
    profile_keywords = {
        "must_have": [{"raw": "Python", "canonical": "python"}],
        "nice_to_have": [],
    }
    evidence = [SimpleNamespace(keyword="python", tier="exact")]

    result = score(
        selected_candidates=selected,
        all_candidates=all_candidates,
        profile_keywords=profile_keywords,
        must_evs_all=evidence,
        nice_evs_all=[],
        must_evs_bullets_only=evidence,
        nice_evs_bullets_only=[],
        alpha=0.5,
        must_weight=1.0,
    )

    assert result.final_score == 100
    assert result.must_missing_bullets_only == []


def test_compute_retrieval_norm_empty() -> None:
    """Test compute retrieval norm with empty inputs."""
    assert compute_retrieval_norm([], []) == 0.0
