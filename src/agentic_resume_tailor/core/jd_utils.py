from __future__ import annotations

import logging
from typing import Any, List

from agentic_resume_tailor import jd_parser
from agentic_resume_tailor.settings import get_settings

logger = logging.getLogger(__name__)


def try_parse_jd(jd_text: str) -> Any | None:
    """Parse a JD with the optional JD parser, falling back on failure."""
    settings = get_settings()
    if not settings.use_jd_parser:
        return None
    if not hasattr(jd_parser, "parse_job_description"):
        logger.error("jd_parser.parse_job_description not found")
        return None
    try:
        return jd_parser.parse_job_description(jd_text, model=settings.jd_model)
    except TypeError:
        return jd_parser.parse_job_description(jd_text)
    except Exception:
        logger.exception("JD parser failed; falling back to manual queries.")
        return None


def fallback_queries_from_jd(jd_text: str, max_q: int = 6) -> List[str]:
    """Build heuristic fallback queries from JD text."""
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    bulletish = [
        ln.lstrip("-•* ").strip() for ln in lines if ln.strip().startswith(("-", "•", "*"))
    ]

    out: List[str] = []
    for b in bulletish:
        if len(b) >= 12:
            out.append(b)

    condensed = " ".join(lines[:20])
    condensed = " ".join(condensed.split())
    if condensed and condensed not in out:
        out.insert(0, condensed)

    seen = set()
    deduped: List[str] = []
    for q in out:
        qn = q.lower()
        if qn not in seen:
            seen.add(qn)
            deduped.append(q)
        if len(deduped) >= max_q:
            break

    return deduped[:max_q] if deduped else [jd_text.strip()]
