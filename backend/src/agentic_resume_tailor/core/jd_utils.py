from __future__ import annotations

import logging
import re
from typing import Any, List

from agentic_resume_tailor import jd_parser
from agentic_resume_tailor.settings import get_settings

logger = logging.getLogger(__name__)

_JD_HEADING_KEYWORDS = (
    "must-have",
    "must have",
    "requirements",
    "responsibilities",
)


def _is_heading(line: str) -> bool:
    cleaned = re.sub(r"[:-]+$", "", line or "").strip().lower()
    if not cleaned:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", "", cleaned).strip()
    for heading in _JD_HEADING_KEYWORDS:
        if normalized == heading:
            return True
        if normalized.startswith(f"{heading} "):
            return True
    return False


def try_parse_jd(jd_text: str) -> Any | None:
    """Parse a JD with the optional JD parser, falling back on failure.

    Args:
        jd_text: Job description text.

    Returns:
        Result value.
    """
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
    """Build heuristic fallback queries from JD text.

    Args:
        jd_text: Job description text.
        max_q: Maximum q (optional).

    Returns:
        List of results.
    """
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


def build_jd_excerpt(jd_text: str, max_chars: int | None = None) -> str:
    """Build a short JD excerpt for tone/phrasing.

    Args:
        jd_text: Job description text.
        max_chars: Maximum characters (optional).

    Returns:
        Short excerpt.
    """
    settings = get_settings()
    if max_chars is None:
        max_chars = int(getattr(settings, "jd_excerpt_max_chars", 1200))
    if max_chars <= 0:
        return ""
    if not jd_text or not jd_text.strip():
        return ""

    raw_lines = [ln.rstrip() for ln in jd_text.splitlines()]
    chosen_lines: List[str] = []
    capture = False
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            if capture:
                capture = False
            continue
        if _is_heading(stripped):
            capture = True
            continue
        if capture:
            chosen_lines.append(stripped)

    if not chosen_lines:
        chosen_lines = [ln.strip() for ln in raw_lines if ln.strip()]

    excerpt_lines: List[str] = []
    total = 0
    for line in chosen_lines:
        if not line:
            continue
        sep = 1 if excerpt_lines else 0
        remaining = max_chars - total - sep
        if remaining <= 0:
            break
        if len(line) > remaining:
            excerpt_lines.append(line[:remaining].rstrip())
            total = max_chars
            break
        excerpt_lines.append(line)
        total += len(line) + sep

    return "\n".join(excerpt_lines).strip()
