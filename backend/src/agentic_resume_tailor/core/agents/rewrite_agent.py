from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from agentic_resume_tailor.core.agents.rewrite_validation import (
    ValidationResult,
    validate_rewrite,
)
from agentic_resume_tailor.settings import get_settings


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./#-]*")


@dataclass(frozen=True)
class RewriteConstraints:
    enabled: bool
    min_chars: int
    max_chars: int


@dataclass
class RewriteBulletInfo:
    bullet_id: str
    original_text: str
    rewritten_text: str
    changed: bool
    fallback_used: bool
    validation: ValidationResult


@dataclass
class RewriteResult:
    rewritten_bullets: Dict[str, str]
    bullet_info: Dict[str, RewriteBulletInfo]


def _tokenize_terms(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _load_json(path: str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_rewrite_allowlist(original_texts: Iterable[str], settings: Any | None = None) -> Set[str]:
    """Build an allowlist from canonicalization/families config + original tokens."""
    settings = settings or get_settings()
    allow: Set[str] = set()

    for text in original_texts:
        allow.update(_tokenize_terms(text))

    canon_cfg = _load_json(settings.canon_config)
    for group in canon_cfg.get("canon_groups", []) or []:
        canonical = group.get("canonical")
        if isinstance(canonical, str):
            allow.update(_tokenize_terms(canonical))
        for variant in group.get("variants") or []:
            if isinstance(variant, str):
                allow.update(_tokenize_terms(variant))

    families_cfg = _load_json(settings.family_config)
    for fam in families_cfg.get("families", []) or []:
        generic = fam.get("generic")
        if isinstance(generic, str):
            allow.update(_tokenize_terms(generic))
        for term in fam.get("satisfied_by") or []:
            if isinstance(term, str):
                allow.update(_tokenize_terms(term))

    return {t for t in allow if t}


def _rewrite_local(text: str, constraints: RewriteConstraints) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    cleaned = cleaned.lstrip("-•* ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    cleaned = re.sub(
        r"^(responsible for|worked on|assisted with|helped|tasked with|in charge of)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    if len(cleaned) > constraints.max_chars:
        shortened = re.sub(r"\s+\([^)]*\)", "", cleaned).strip()
        if len(shortened) > constraints.max_chars:
            shortened = re.split(r"[;:]", shortened, maxsplit=1)[0].strip()
        shortened = re.sub(r"\s+", " ", shortened).strip()
        if shortened:
            cleaned = shortened

    return cleaned


def rewrite_bullets(
    target_profile: Any | None,
    bullets_original: List[Dict[str, Any]],
    allowlist: Iterable[str],
    constraints: RewriteConstraints,
) -> RewriteResult:
    """Rewrite bullets safely using local-only rules."""
    rewritten: Dict[str, str] = {}
    info: Dict[str, RewriteBulletInfo] = {}

    allow = {t.lower() for t in allowlist if t}

    for bullet in bullets_original:
        bullet_id = str(bullet.get("bullet_id") or "")
        original = str(bullet.get("text_latex") or "")
        if not bullet_id:
            continue

        candidate = original
        if constraints.enabled:
            candidate = _rewrite_local(original, constraints)
            if not candidate:
                candidate = original

        validation = validate_rewrite(original, candidate, allow)
        fallback_used = False
        if not validation.ok:
            candidate = original
            fallback_used = True
            validation = validate_rewrite(original, candidate, allow)

        rewritten[bullet_id] = candidate
        info[bullet_id] = RewriteBulletInfo(
            bullet_id=bullet_id,
            original_text=original,
            rewritten_text=candidate,
            changed=candidate != original,
            fallback_used=fallback_used,
            validation=validation,
        )

    return RewriteResult(rewritten_bullets=rewritten, bullet_info=info)
