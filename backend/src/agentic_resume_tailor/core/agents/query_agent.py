from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentic_resume_tailor.core.jd_utils import fallback_queries_from_jd, try_parse_jd
from agentic_resume_tailor.core.retrieval import normalize_query_text


@dataclass(frozen=True)
class QueryPlanItem:
    text: str
    purpose: str = "general"
    weight: float = 1.0
    boost_keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    items: List[QueryPlanItem]
    profile: Optional[Any] = None
    profile_used: bool = False


def _items_from_profile(profile: Any) -> List[QueryPlanItem]:
    profile_dict = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
    retrieval_plan = profile_dict.get("retrieval_plan", {}) or {}
    experience_queries = retrieval_plan.get("experience_queries", []) or []

    items: List[QueryPlanItem] = []
    for entry in experience_queries:
        if not isinstance(entry, dict):
            continue
        query = (entry.get("query") or "").strip()
        if not query:
            continue
        items.append(
            QueryPlanItem(
                text=normalize_query_text(query),
                purpose=str(entry.get("purpose") or "general"),
                weight=float(entry.get("weight", 1.0) or 1.0),
                boost_keywords=[str(k) for k in (entry.get("boost_keywords") or []) if str(k)],
            )
        )
    return items


def build_query_plan(jd_text: str, settings: Any) -> QueryPlan:
    """Build a query plan from a JD using optional parsing."""
    profile = None
    if getattr(settings, "use_jd_parser", False):
        profile = try_parse_jd(jd_text)

    if profile is not None:
        items = _items_from_profile(profile)
        if items:
            return QueryPlan(items=items, profile=profile, profile_used=True)

    fallback = [q for q in fallback_queries_from_jd(jd_text) if q.strip()]
    items = [QueryPlanItem(text=normalize_query_text(q)) for q in fallback]
    return QueryPlan(items=items, profile=None, profile_used=False)
