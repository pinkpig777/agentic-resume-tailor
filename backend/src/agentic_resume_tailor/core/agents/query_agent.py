from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentic_resume_tailor.core.agents.llm_client import call_llm_json
from agentic_resume_tailor.core.jd_utils import fallback_queries_from_jd
from agentic_resume_tailor.core.retrieval import normalize_query_text
from agentic_resume_tailor.jd_parser import canonicalize

logger = logging.getLogger(__name__)


KeywordType = Literal["hard_skill", "soft_skill", "tool", "framework", "domain", "responsibility"]
QueryPurpose = Literal[
    "core_stack",
    "domain_fit",
    "deployment",
    "scale_reliability",
    "leadership",
    "general",
]


class QueryKeywordItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    type: KeywordType
    priority: int = Field(ge=1, le=5)


class QueryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=8)
    purpose: QueryPurpose
    boost_keywords: List[str] = Field(default_factory=list)
    weight: float = Field(ge=0.1, le=3.0)


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experience_queries: List[QueryItem] = Field(default_factory=list)


class TargetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role_title: str = ""
    role_summary: str = ""
    must_have: List[QueryKeywordItem] = Field(default_factory=list)
    nice_to_have: List[QueryKeywordItem] = Field(default_factory=list)
    responsibilities: List[QueryKeywordItem] = Field(default_factory=list)
    domain_terms: List[QueryKeywordItem] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)


class QueryAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_profile: TargetProfile
    retrieval_plan: RetrievalPlan


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
    profile_summary: Optional[Dict[str, Any]] = None
    agent_used: bool = False
    agent_fallback: bool = False
    agent_model: Optional[str] = None


SYSTEM_PROMPT = """You are the Query Agent for Agentic Resume Tailor.

Goal: parse the job description into a target profile and a retrieval plan.

Rules:
- Use ONLY information present in the JD; do not guess.
- Canonical terms must be lowercase, concise, and tool/skill specific.
- Include must-have vs nice-to-have skill lists and responsibilities.
- retrieval_plan.experience_queries: 3-7 queries, 5-12 words each, no boolean operators.
- Queries should be specific and technical (avoid single generic words).
- Return STRICT JSON only; no extra keys or commentary.
"""


USER_TEMPLATE = """Job description:
{jd_text}

Return JSON with this shape:
{{
  "target_profile": {{
    "role_title": "...",
    "role_summary": "...",
    "must_have": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 1}}],
    "nice_to_have": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 2}}],
    "responsibilities": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 2}}],
    "domain_terms": [{{"raw": "...", "canonical": "...", "type": "...", "priority": 3}}],
    "retrieval_plan": {{"experience_queries": [{{"query": "...", "purpose": "...", "boost_keywords": [], "weight": 1.0}}]}}
  }},
  "retrieval_plan": {{"experience_queries": [{{"query": "...", "purpose": "...", "boost_keywords": [], "weight": 1.0}}]}}
}}
"""


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


def _normalize_profile(profile: TargetProfile) -> TargetProfile:
    profile_dict = profile.model_dump()
    for bucket in ("must_have", "nice_to_have", "responsibilities", "domain_terms"):
        items = profile_dict.get(bucket, []) or []
        for item in items:
            raw = str(item.get("raw") or "").strip()
            canonical_val = str(item.get("canonical") or "").strip()
            canonical_val = canonical_val or raw
            item["raw"] = raw or canonical_val
            item["canonical"] = canonicalize(item["raw"] or canonical_val)
    return TargetProfile(**profile_dict)


def _summarize_profile(profile: Any) -> Dict[str, Any]:
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()
    summary = {
        "role_title": profile.get("role_title", ""),
        "role_summary": profile.get("role_summary", ""),
        "must_have": [k.get("canonical") or k.get("raw") for k in profile.get("must_have", [])],
        "nice_to_have": [
            k.get("canonical") or k.get("raw") for k in profile.get("nice_to_have", [])
        ],
        "responsibilities": [
            k.get("canonical") or k.get("raw") for k in profile.get("responsibilities", [])
        ],
        "domain_terms": [
            k.get("canonical") or k.get("raw") for k in profile.get("domain_terms", [])
        ],
    }
    for key, vals in summary.items():
        if isinstance(vals, list):
            summary[key] = [v for v in vals if v]
    return summary


def build_query_plan(jd_text: str, settings: Any) -> QueryPlan:
    """Build a query plan from a JD using optional parsing."""
    profile = None
    profile_summary = None
    agent_used = False
    agent_fallback = False
    model = getattr(settings, "agent_model", None) or getattr(settings, "jd_model", None)

    if getattr(settings, "use_jd_parser", False):
        agent_used = True
        try:
            output = call_llm_json(
                USER_TEMPLATE.format(jd_text=jd_text),
                QueryAgentOutput,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                model=model,
            )
            profile = _normalize_profile(output.target_profile)
            if not profile.retrieval_plan.experience_queries and output.retrieval_plan:
                profile = profile.model_copy(update={"retrieval_plan": output.retrieval_plan})
            items = _items_from_profile(profile)
            profile_summary = _summarize_profile(profile)
            if items:
                return QueryPlan(
                    items=items,
                    profile=profile,
                    profile_used=True,
                    profile_summary=profile_summary,
                    agent_used=agent_used,
                    agent_fallback=False,
                    agent_model=model,
                )
            agent_fallback = True
        except Exception:
            logger.exception("Query agent failed; falling back to heuristic queries.")
            agent_fallback = True

    fallback = [q for q in fallback_queries_from_jd(jd_text) if q.strip()]
    items = [QueryPlanItem(text=normalize_query_text(q)) for q in fallback]
    return QueryPlan(
        items=items,
        profile=profile,
        profile_used=profile is not None,
        profile_summary=profile_summary,
        agent_used=agent_used,
        agent_fallback=agent_fallback,
        agent_model=model if agent_used else None,
    )
