from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict

from agentic_resume_tailor.settings import get_settings

DEFAULT_CANON_CONFIG: Dict[str, Any] = {
    "schema_version": "canon_config_v1",
    "options": {
        "keep_chars": "+#./-",
        "collapse_whitespace": True,
        "slash_to_space": True,
        "dash_to_space": True,
        "separator_exceptions": ["ci/cd"],
    },
    "canon_groups": [],
}

DEFAULT_FAMILY_CONFIG: Dict[str, Any] = {
    "schema_version": "families_v1",
    "families": [],
}

logger = logging.getLogger(__name__)


def _load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return fallback
        return data
    except FileNotFoundError:
        logger.warning("%s not found; using empty config", path)
        return fallback
    except Exception as exc:
        logger.warning("Failed to load %s: %s; using empty config", path, exc)
        return fallback


@lru_cache(maxsize=32)
def load_matching_configs(canon_path: str, family_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load canonicalization + family configs keyed by path."""
    return _load_json(canon_path, DEFAULT_CANON_CONFIG), _load_json(
        family_path, DEFAULT_FAMILY_CONFIG
    )


def current_matching_configs(settings: Any | None = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load configs for the active settings snapshot."""
    settings = settings or get_settings()
    return load_matching_configs(settings.canon_config, settings.family_config)


def _base_normalize(text: str, keep_chars: str, collapse_ws: bool = True) -> str:
    s = (text or "").lower().strip()
    if collapse_ws:
        s = re.sub(r"\s+", " ", s)

    keep = re.escape(keep_chars)
    s = re.sub(rf"[^a-z0-9{keep}\s]+", " ", s)
    if collapse_ws:
        s = re.sub(r"\s+", " ", s)
    return s.strip()


def _build_variant_to_canon_map(canon_cfg: Dict[str, Any]) -> Dict[str, str]:
    opts = canon_cfg.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))

    variant_to_canon: Dict[str, str] = {}
    for group in canon_cfg.get("canon_groups") or []:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical")
        variants = group.get("variants") or []
        if not isinstance(canonical, str) or not canonical.strip():
            continue

        canon_norm = _base_normalize(canonical, keep_chars, collapse_ws)
        variant_to_canon[canon_norm] = canon_norm

        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, str) and variant.strip():
                    var_norm = _base_normalize(variant, keep_chars, collapse_ws)
                    variant_to_canon[var_norm] = canon_norm
    return variant_to_canon


def canonicalize_term(term: str, *, canon_cfg: Dict[str, Any] | None = None) -> str:
    canon_cfg = canon_cfg or current_matching_configs()[0]
    opts = canon_cfg.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))
    slash_to_space = bool(opts.get("slash_to_space", True))
    dash_to_space = bool(opts.get("dash_to_space", True))
    exceptions = set(opts.get("separator_exceptions") or [])
    variant_to_canon = _build_variant_to_canon_map(canon_cfg)

    normalized = _base_normalize(term, keep_chars, collapse_ws)
    if normalized not in exceptions:
        if slash_to_space:
            normalized = normalized.replace("/", " ")
        if dash_to_space:
            normalized = normalized.replace("-", " ")
        if collapse_ws:
            normalized = re.sub(r"\s+", " ", normalized).strip()
    return variant_to_canon.get(normalized, normalized)


def canonicalize_text(text: str, *, canon_cfg: Dict[str, Any] | None = None) -> str:
    canon_cfg = canon_cfg or current_matching_configs()[0]
    opts = canon_cfg.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))
    variant_to_canon = _build_variant_to_canon_map(canon_cfg)

    normalized = _base_normalize(text, keep_chars, collapse_ws)
    for variant, canon in variant_to_canon.items():
        if not variant or variant == canon:
            continue
        if " " in variant:
            normalized = normalized.replace(variant, canon)
        else:
            normalized = re.sub(
                rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", canon, normalized
            )

    if collapse_ws:
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_family_map(
    *, family_cfg: Dict[str, Any] | None = None, canon_cfg: Dict[str, Any] | None = None
) -> Dict[str, list[str]]:
    if canon_cfg is None or family_cfg is None:
        canon_cfg_current, family_cfg_current = current_matching_configs()
        canon_cfg = canon_cfg or canon_cfg_current
        family_cfg = family_cfg or family_cfg_current
    if family_cfg.get("schema_version") != "families_v1":
        return {}

    families: Dict[str, list[str]] = {}
    for family in family_cfg.get("families") or []:
        if not isinstance(family, dict):
            continue
        generic = family.get("generic")
        satisfied_by = family.get("satisfied_by") or []
        if not isinstance(generic, str) or not generic.strip():
            continue
        generic_canon = canonicalize_term(generic, canon_cfg=canon_cfg)
        specific_terms = [
            canonicalize_term(item, canon_cfg=canon_cfg)
            for item in satisfied_by
            if isinstance(item, str) and item.strip()
        ]
        families[generic_canon] = list(dict.fromkeys(specific_terms))
    return families
