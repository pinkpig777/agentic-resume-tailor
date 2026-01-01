import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agentic_resume_tailor.settings import get_settings

# ----------------------------
# Config loaders
# ----------------------------

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
    """Load a JSON file with fallback defaults.

    Args:
        path: Filesystem path.
        fallback: The fallback value.

    Returns:
        Dictionary result.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return fallback
        return data
    except FileNotFoundError:
        logger.warning("%s not found; using empty config", path)
        return fallback
    except Exception as e:
        logger.warning("Failed to load %s: %s; using empty config", path, e)
        return fallback


_settings = get_settings()
CANON_PATH = _settings.canon_config
FAMILY_PATH = _settings.family_config

_CANON_CONFIG = _load_json(CANON_PATH, DEFAULT_CANON_CONFIG)
_FAMILY_CONFIG = _load_json(FAMILY_PATH, DEFAULT_FAMILY_CONFIG)


# ----------------------------
# Canonicalization helpers
# ----------------------------


def _base_normalize(text: str, keep_chars: str, collapse_ws: bool = True) -> str:
    """Normalize text for canonicalization.

    Args:
        text: The text value.
        keep_chars: The keep chars value.
        collapse_ws: The collapse ws value (optional).

    Returns:
        String result.
    """
    s = (text or "").lower().strip()
    if collapse_ws:
        s = re.sub(r"\s+", " ", s)

    keep = re.escape(keep_chars)
    s = re.sub(rf"[^a-z0-9{keep}\s]+", " ", s)
    if collapse_ws:
        s = re.sub(r"\s+", " ", s)
    return s.strip()


def _build_variant_to_canon_map(canon_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Build a variant-to-canonical mapping.

    Args:
        canon_cfg: The canon config value.

    Returns:
        Dictionary result.
    """
    opts = canon_cfg.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))

    m: Dict[str, str] = {}
    for g in canon_cfg.get("canon_groups") or []:
        if not isinstance(g, dict):
            continue
        canonical = g.get("canonical")
        variants = g.get("variants") or []
        if not isinstance(canonical, str) or not canonical.strip():
            continue

        canon_norm = _base_normalize(canonical, keep_chars, collapse_ws)
        m[canon_norm] = canon_norm

        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, str) and v.strip():
                    var_norm = _base_normalize(v, keep_chars, collapse_ws)
                    m[var_norm] = canon_norm
    return m


_VARIANT_TO_CANON = _build_variant_to_canon_map(_CANON_CONFIG)


def canonicalize_term(term: str) -> str:
    """Canonicalize a single term for matching.

    Args:
        term: The term value.

    Returns:
        String result.
    """
    opts = _CANON_CONFIG.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))
    slash_to_space = bool(opts.get("slash_to_space", True))
    dash_to_space = bool(opts.get("dash_to_space", True))
    exceptions = set(opts.get("separator_exceptions") or [])

    s = _base_normalize(term, keep_chars, collapse_ws)

    if s not in exceptions:
        if slash_to_space:
            s = s.replace("/", " ")
        if dash_to_space:
            s = s.replace("-", " ")
        if collapse_ws:
            s = re.sub(r"\s+", " ", s).strip()

    return _VARIANT_TO_CANON.get(s, s)


def canonicalize_text(text: str) -> str:
    """Canonicalize free text for matching.

    Args:
        text: The text value.

    Returns:
        String result.
    """
    opts = _CANON_CONFIG.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    collapse_ws = bool(opts.get("collapse_whitespace", True))

    s = _base_normalize(text, keep_chars, collapse_ws)

    # Replace variants by scanning (simple, fine for small configs)
    for var, canon in _VARIANT_TO_CANON.items():
        if not var or var == canon:
            continue
        if " " in var:
            s = s.replace(var, canon)
        else:
            s = re.sub(rf"(?<![a-z0-9]){re.escape(var)}(?![a-z0-9])", canon, s)

    if collapse_ws:
        s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Bullet text normalization (LaTeX-aware)
# ----------------------------

_LATEX_TWO_ARGS = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}\{([^{}]*)\}")
_LATEX_ONE_ARG = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_LATEX_CMD_ONLY = re.compile(r"\\[a-zA-Z]+\*?")
_BRACES = re.compile(r"[{}]")


def latex_to_plain_for_matching(latex: str) -> str:
    """Strip LaTeX markup for matching.

    Args:
        latex: The latex value.

    Returns:
        String result.
    """
    s = latex or ""

    s = (
        s.replace(r"\\%", "%")
        .replace(r"\\&", "&")
        .replace(r"\\$", "$")
        .replace(r"\\_", "_")
        .replace(r"\\#", "#")
    )

    s = re.sub(r"\$(.*?)\$", r" \1 ", s)

    for _ in range(6):
        prev = s
        s = _LATEX_TWO_ARGS.sub(r" \2 ", s)
        s = _LATEX_ONE_ARG.sub(r" \1 ", s)
        if s == prev:
            break

    s = _LATEX_CMD_ONLY.sub(" ", s)
    s = _BRACES.sub(" ", s)
    s = s.replace("\\", " ")
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ----------------------------
# Families (generic -> specifics)
# ----------------------------


def load_families() -> Dict[str, List[str]]:
    """Load family mappings for keyword matching.

    Returns:
        Dictionary result.
    """
    fam_cfg = _FAMILY_CONFIG
    if fam_cfg.get("schema_version") != "families_v1":
        return {}

    out: Dict[str, List[str]] = {}
    for f in fam_cfg.get("families") or []:
        if not isinstance(f, dict):
            continue
        generic = f.get("generic")
        sats = f.get("satisfied_by") or []
        if not isinstance(generic, str) or not generic.strip():
            continue
        generic_c = canonicalize_term(generic)
        sats_c = [canonicalize_term(x) for x in sats if isinstance(x, str) and x.strip()]
        out[generic_c] = list(dict.fromkeys(sats_c))
    return out


_FAMILIES = load_families()


# ----------------------------
# Matching outputs
# ----------------------------

MatchTier = str  # "exact" | "family" | "substring" | "none"


@dataclass
class MatchEvidence:
    keyword: str  # canonical
    tier: MatchTier
    satisfied_by: Optional[str]
    bullet_ids: List[str]
    notes: str = ""


def _safe_word_boundary_regex(phrase: str) -> re.Pattern:
    """Build a word-boundary regex for a phrase.

    Args:
        phrase: The phrase value.

    Returns:
        Result value.
    """
    parts = [re.escape(p) for p in phrase.split()]
    pat = r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])"
    return re.compile(pat)


def _is_safe_substring_token(t: str) -> bool:
    """Check if a token is safe for substring matching.

    Args:
        t: The t value.

    Returns:
        True if the condition is met, otherwise False.
    """
    if len(t) < 6:
        return False
    return bool(re.fullmatch(r"[a-z0-9]+", t))


def match_keywords_against_bullets(
    keywords: List[Dict[str, Any]],
    bullets: List[Dict[str, Any]],
) -> List[MatchEvidence]:
    """Match profile keywords against bullet text.

    Tiers: exact phrase, family mapping, safe substring, or none.

    Args:
        keywords: The keywords value.
        bullets: The bullets value.

    Returns:
        List of results.
    """
    bullet_text: Dict[str, str] = {}
    for b in bullets:
        bid = b["bullet_id"]
        plain = latex_to_plain_for_matching(b.get("text_latex", ""))
        canon_txt = canonicalize_text(plain)
        bullet_text[bid] = canon_txt

    evidences: List[MatchEvidence] = []

    for kw in keywords:
        raw = kw.get("raw", "") or ""
        canonical = kw.get("canonical", "") or raw
        k = canonicalize_term(canonical)

        # Tier 1: exact phrase match
        rx = _safe_word_boundary_regex(k) if k else None
        exact_hits = []
        if rx:
            for bid, txt in bullet_text.items():
                if rx.search(txt):
                    exact_hits.append(bid)

        if exact_hits:
            evidences.append(
                MatchEvidence(keyword=k, tier="exact", satisfied_by=k, bullet_ids=exact_hits)
            )
            continue

        # Tier 2: family match (generic keyword satisfied by specific)
        fam = _FAMILIES.get(k)
        if fam:
            fam_hits = []
            sat_term = None
            for spec in fam:
                rx2 = _safe_word_boundary_regex(spec)
                hit_bids = [bid for bid, txt in bullet_text.items() if rx2.search(txt)]
                if hit_bids:
                    fam_hits = hit_bids
                    sat_term = spec
                    break
            if fam_hits:
                evidences.append(
                    MatchEvidence(
                        keyword=k, tier="family", satisfied_by=sat_term, bullet_ids=fam_hits
                    )
                )
                continue

        # Tier 3: controlled substring (safe long tokens only)
        sub_hits = []
        sat_term = None
        if _is_safe_substring_token(k):
            for bid, txt in bullet_text.items():
                if k in txt:
                    sub_hits.append(bid)
            if sub_hits:
                sat_term = k

        if sub_hits:
            evidences.append(
                MatchEvidence(
                    keyword=k, tier="substring", satisfied_by=sat_term, bullet_ids=sub_hits
                )
            )
        else:
            evidences.append(
                MatchEvidence(keyword=k, tier="none", satisfied_by=None, bullet_ids=[])
            )

    return evidences


def extract_profile_keywords(profile: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Extract must-have and nice-to-have lists.

    Args:
        profile: The profile value.

    Returns:
        Dictionary result.
    """
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()

    must_have = profile.get("must_have", []) or []
    nice_to_have = profile.get("nice_to_have", []) or []
    return {
        "must_have": must_have,
        "nice_to_have": nice_to_have,
    }


# ----------------------------
# NEW: build match corpus (skills + selected bullets)
# ----------------------------


def build_match_corpus(
    resume_data: Dict[str, Any],
    selected_bullets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a matching corpus from skills and bullets.

    Args:
        resume_data: Resume data snapshot.
        selected_bullets: The selected bullets value.

    Returns:
        List of results.
    """
    corpus: List[Dict[str, Any]] = []

    skills = (resume_data or {}).get("skills", {}) or {}
    if isinstance(skills, dict):
        for field, val in skills.items():
            if isinstance(val, str) and val.strip():
                corpus.append(
                    {
                        "bullet_id": f"skills:{field}",
                        "text_latex": val,
                        "meta": {"section": "skills", "field": field},
                    }
                )

    # Add selected bullets as-is
    corpus.extend(selected_bullets or [])
    return corpus
