from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _safe_candidate_dict(c: Any) -> Dict[str, Any]:
    """Serialize a candidate into a compact JSON-friendly dict."""
    d = {
        "bullet_id": getattr(c, "bullet_id", ""),
        "source": getattr(c, "source", ""),
        "text_latex": getattr(c, "text_latex", ""),
        "meta": getattr(c, "meta", {}) or {},
        "best_hit": None,
        "total_weighted": float(getattr(c, "total_weighted", 0.0) or 0.0),
        "hits": [],
    }

    bh = getattr(c, "best_hit", None)
    if bh is not None:
        d["best_hit"] = {
            "query": getattr(bh, "query", ""),
            "purpose": getattr(bh, "purpose", ""),
            "weight": float(getattr(bh, "weight", 0.0) or 0.0),
            "cosine": float(getattr(bh, "cosine", 0.0) or 0.0),
            "weighted": float(getattr(bh, "weighted", 0.0) or 0.0),
        }

    hits = getattr(c, "hits", []) or []
    out_hits = []
    for h in hits:
        out_hits.append(
            {
                "query": getattr(h, "query", ""),
                "purpose": getattr(h, "purpose", ""),
                "weight": float(getattr(h, "weight", 0.0) or 0.0),
                "cosine": float(getattr(h, "cosine", 0.0) or 0.0),
                "weighted": float(getattr(h, "weighted", 0.0) or 0.0),
            }
        )
    d["hits"] = out_hits[:8]  # cap for size
    return d


def _evidence_list(evidences: List[Any]) -> List[Dict[str, Any]]:
    """Serialize evidence items into JSON-friendly dicts."""
    out: List[Dict[str, Any]] = []
    for e in evidences or []:
        out.append(
            {
                "keyword": getattr(e, "keyword", ""),
                "tier": getattr(e, "tier", "none"),
                "satisfied_by": getattr(e, "satisfied_by", None),
                "bullet_ids": list(getattr(e, "bullet_ids", []) or [])[:10],
                "notes": getattr(e, "notes", "") or "",
            }
        )
    return out


def build_report(
    *,
    jd_text: str,
    profile: Optional[Any],
    config: Dict[str, Any],
    final_iteration_index: int,
    selected_ids: List[str],
    selected_candidates: List[Any],
    all_candidates: List[Any],
    # scoring result
    hybrid: Optional[Any],
    # evidences
    must_evs_bullets_only: Optional[List[Any]] = None,
    nice_evs_bullets_only: Optional[List[Any]] = None,
    must_evs_all: Optional[List[Any]] = None,
    nice_evs_all: Optional[List[Any]] = None,
    # loop history
    iterations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable report for a generation run."""
    profile_dump = None
    if profile is not None:
        try:
            profile_dump = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
        except Exception:
            profile_dump = None

    report: Dict[str, Any] = {
        "schema_version": "resume_report_v1",
        "created_at_utc": utc_now_iso(),
        "job": {
            "jd_text_preview": jd_text[:8000],  # keep some proof, cap size
            "profile_used": profile is not None,
            "profile": profile_dump,
        },
        "config": config,
        "final": {
            "final_iteration_index": final_iteration_index,
            "selected_ids": selected_ids,
            "candidate_count": len(all_candidates),
        },
        "selected_bullets": [_safe_candidate_dict(c) for c in selected_candidates],
        "coverage": {
            "bullets_only": {
                "must_evidence": _evidence_list(must_evs_bullets_only or []),
                "nice_evidence": _evidence_list(nice_evs_bullets_only or []),
            },
            "all_plus_skills": {
                "must_evidence": _evidence_list(must_evs_all or []),
                "nice_evidence": _evidence_list(nice_evs_all or []),
            },
        },
        "scores": None,
        "iterations": iterations or [],
    }

    if hybrid is not None:
        report["scores"] = {
            "final_score": int(getattr(hybrid, "final_score", 0) or 0),
            "retrieval_score": float(getattr(hybrid, "retrieval_score", 0.0) or 0.0),
            "coverage_bullets_only": float(getattr(hybrid, "coverage_bullets_only", 0.0) or 0.0),
            "coverage_all": float(getattr(hybrid, "coverage_all", 0.0) or 0.0),
            "must_missing_bullets_only": list(
                getattr(hybrid, "must_missing_bullets_only", []) or []
            ),
            "nice_missing_bullets_only": list(
                getattr(hybrid, "nice_missing_bullets_only", []) or []
            ),
            "must_missing_all": list(getattr(hybrid, "must_missing_all", []) or []),
            "nice_missing_all": list(getattr(hybrid, "nice_missing_all", []) or []),
        }

    return report


def write_report_json(
    report: Dict[str, Any], output_dir: str, filename: str = "resume_report.json"
) -> str:
    """Write the report to disk and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
