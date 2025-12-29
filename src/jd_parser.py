import os
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Literal, Dict, Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================
# Target Profile v1 (STRICT schema for Structured Outputs)
# =============================

KeywordType = Literal["hard_skill", "soft_skill",
                      "tool", "framework", "domain", "responsibility"]
QueryPurpose = Literal["core_stack", "domain_fit",
                       "deployment", "scale_reliability", "leadership", "general"]


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # NOTE: We DO NOT trust model-provided offsets. We repair them server-side.
    start: int = Field(
        ge=0, description="Character start offset into original jd_text (repaired server-side).")
    end: int = Field(
        ge=0, description="Character end offset into original jd_text (repaired server-side).")
    snippet: str = Field(
        min_length=1, description="Exact substring copied from jd_text.")


class KeywordItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str = Field(
        min_length=1, description="Original phrase as it appears in the JD.")
    canonical: str = Field(
        min_length=1, description="Canonicalized (lowercase, normalized).")
    type: KeywordType
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    priority: int = Field(ge=1, le=5, description="1 is highest priority.")


class QueryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, description="Embedding-friendly phrase, no boolean operators.")
    purpose: QueryPurpose
    boost_keywords: List[str] = Field(
        default_factory=list, description="Canonical keywords to boost.")
    weight: float = Field(ge=0.1, le=3.0, description="Relative query weight.")


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_queries: List[QueryItem] = Field(default_factory=list)


class MetaInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_model: str
    jd_hash: str
    created_at_utc: str


class TargetProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["target_profile_v1"] = "target_profile_v1"
    role_title: str = Field(
        default="", description="Job title if identifiable.")
    role_summary: str = Field(default="", description="1-2 sentence summary.")
    must_have: List[KeywordItem] = Field(default_factory=list)
    nice_to_have: List[KeywordItem] = Field(default_factory=list)
    responsibilities: List[KeywordItem] = Field(default_factory=list)
    domain_terms: List[KeywordItem] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    meta: MetaInfo


# =============================
# Canonicalization config (file-driven, canonical-first schema)
# =============================

DEFAULT_CANON_CONFIG: Dict[str, Any] = {
    "schema_version": "canon_config_v1",
    "options": {
        "keep_chars": "+#./-",
        "slash_to_space": True,
        "dash_to_space": True,
        "separator_exceptions": ["ci/cd"],
    },
    "canon_groups": [],
}


def load_canon_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("canonicalization config must be a JSON object")
        if data.get("schema_version") != "canon_config_v1":
            raise ValueError(
                "canonicalization config schema_version must be canon_config_v1")
        return data
    except FileNotFoundError:
        print(f"⚠️ {path} not found; using empty canonical config")
        return DEFAULT_CANON_CONFIG
    except Exception as e:
        print(
            f"⚠️ Failed to load canonicalization config ({e}); using empty canonical config")
        return DEFAULT_CANON_CONFIG


CANON_CONFIG_PATH = os.environ.get(
    "ART_CANON_CONFIG", "config/canonicalization.json")
_CANON_CONFIG = load_canon_config(CANON_CONFIG_PATH)


def _base_normalize(text: str, keep_chars: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    keep = re.escape(keep_chars)
    s = re.sub(rf"[^a-z0-9{keep}\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_variant_to_canonical_map(config: Dict[str, Any]) -> Dict[str, str]:
    opts = config.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))

    out: Dict[str, str] = {}
    for group in (config.get("canon_groups", []) or []):
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical")
        variants = group.get("variants", [])

        if not isinstance(canonical, str) or not canonical.strip():
            continue

        canon_norm = _base_normalize(canonical, keep_chars)
        out[canon_norm] = canon_norm

        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, str) and v.strip():
                    var_norm = _base_normalize(v, keep_chars)
                    out[var_norm] = canon_norm

    return out


_VARIANT_TO_CANON = build_variant_to_canonical_map(_CANON_CONFIG)


def canonicalize(text: str) -> str:
    opts = _CANON_CONFIG.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))
    exceptions = set(opts.get("separator_exceptions") or [])

    slash_to_space = bool(opts.get("slash_to_space", True))
    dash_to_space = bool(opts.get("dash_to_space", True))

    s = _base_normalize(text, keep_chars)

    if s not in exceptions:
        if slash_to_space:
            s = s.replace("/", " ")
        if dash_to_space:
            s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()

    return _VARIANT_TO_CANON.get(s, s)


# =============================
# Evidence span repair (the key fix)
# =============================

def find_all_spans(haystack: str, needle: str) -> List[tuple]:
    spans = []
    if not needle:
        return spans
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def repair_evidence_items(jd_text: str, items: List[KeywordItem]) -> None:
    """
    We do NOT trust model offsets. We trust snippet.
    For each evidence span, recompute start/end by searching jd_text.
    - exact match first
    - whitespace-normalized fallback second
    """
    for item in items:
        repaired: List[EvidenceSpan] = []
        for ev in item.evidence:
            snip = (ev.snippet or "").strip("\n")

            # 1) exact match
            spans = find_all_spans(jd_text, snip)

            # 2) whitespace-normalized fallback
            if not spans:
                jd_norm = re.sub(r"\s+", " ", jd_text)
                snip_norm = re.sub(r"\s+", " ", snip).strip()
                if snip_norm:
                    spans_norm = find_all_spans(jd_norm, snip_norm)
                    if spans_norm:
                        # try to locate snip_norm in original too
                        spans = find_all_spans(jd_text, snip_norm)

            if spans:
                s, e = spans[0]
                ev.start = s
                ev.end = e
                ev.snippet = jd_text[s:e]
                repaired.append(ev)
            else:
                # keep it; downstream validation decides if it's acceptable
                repaired.append(ev)

        item.evidence = repaired


def validate_evidence_spans(jd_text: str, item: KeywordItem) -> List[str]:
    errs = []
    n = len(jd_text)

    for ev in item.evidence:
        if ev.start < 0 or ev.end <= ev.start:
            errs.append(
                f"Invalid offsets for '{item.raw}': start={ev.start}, end={ev.end}")
            continue
        if ev.end > n:
            errs.append(
                f"Out of range for '{item.raw}': end={ev.end} > len={n}")
            continue
        expected = jd_text[ev.start:ev.end]
        if ev.snippet != expected:
            errs.append(
                f"Snippet mismatch for '{item.raw}' at [{ev.start}:{ev.end}]")

    return errs


# =============================
# Postprocess + constraints
# =============================

def jd_hash(jd_text: str) -> str:
    return hashlib.sha256((jd_text or "").encode("utf-8")).hexdigest()


def dedupe_by_canonical(items: List[KeywordItem]) -> List[KeywordItem]:
    seen = set()
    out = []
    for it in items:
        if it.canonical in seen:
            continue
        seen.add(it.canonical)
        out.append(it)
    return out


def sanitize_query_for_embeddings(q: str) -> str:
    q = (q or "")
    q = re.sub(r"\bAND\b|\bOR\b|\bNOT\b", " ", q, flags=re.IGNORECASE)
    q = q.replace("(", " ").replace(")", " ").replace(
        '"', " ").replace("'", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def postprocess(profile: TargetProfileV1, jd_text: str, model_name: str) -> TargetProfileV1:
    # canonicalize & dedupe
    for group_name in ["must_have", "nice_to_have", "responsibilities", "domain_terms"]:
        group = getattr(profile, group_name)
        for it in group:
            it.canonical = canonicalize(it.canonical or it.raw)
        setattr(profile, group_name, dedupe_by_canonical(group))

    # enforce must_have evidence presence
    for it in profile.must_have:
        if not it.evidence:
            raise ValueError(f"Must-have missing evidence: '{it.raw}'")

    # retrieval_plan constraints
    eq = profile.retrieval_plan.experience_queries
    if len(eq) < 3:
        raise ValueError(
            "retrieval_plan.experience_queries must have at least 3 queries")
    if len(eq) > 7:
        profile.retrieval_plan.experience_queries = eq[:7]

    # sanitize queries and canonicalize boosts
    for qi in profile.retrieval_plan.experience_queries:
        qi.query = sanitize_query_for_embeddings(qi.query)
        qi.boost_keywords = [canonicalize(b)
                             for b in (qi.boost_keywords or []) if b]
        qi.weight = min(3.0, max(0.1, float(qi.weight)))

    # fill meta
    profile.meta = MetaInfo(
        parser_model=model_name,
        jd_hash=jd_hash(jd_text),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    return profile


# =============================
# Prompts
# =============================

SYSTEM_PROMPT = """
You are an expert technical recruiter helping tailor a resume using a semantic vector database.

Return a Target Profile (v1) used for retrieval + explainability.

Hard requirements:
- Output MUST strictly match the TargetProfile v1 schema.
- For every item in must_have:
  - include >= 1 evidence entry
  - evidence.snippet MUST be copied EXACTLY from the provided jd_text (verbatim, including casing and spacing)
  - Set evidence.start=0 and evidence.end=0 (the server will compute offsets). Do NOT try to count characters.
- canonical fields must be lowercase, concise, and dedup-friendly.
- retrieval_plan.experience_queries:
  - 3 to 7 queries
  - embedding-friendly phrases only
  - NO boolean operators (AND/OR/NOT), no parentheses, no quotes
  - each query targets a distinct angle (core stack, domain, deployment, scale/reliability, leadership if present)
  - include boost_keywords (canonical) and weight (0.8–1.8 typically)
"""

USER_TEMPLATE = """Create a TargetProfile v1 from this job description.

jd_text:
{jd_text}
"""


# =============================
# Main parse function (Node 1)
# =============================

def parse_job_description(
    jd_text: str,
    model: str = "gpt-4.1-nano-2025-04-14",
    max_attempts: int = 2,
) -> TargetProfileV1:
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text is empty")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        max_retries=0,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": USER_TEMPLATE.format(
            jd_text=jd_text).strip()},
    ]

    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        print(
            f"🤖 Analyzing Job Description (attempt {attempt}/{max_attempts})...")

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=TargetProfileV1,
        )
        profile = completion.choices[0].message.parsed

        # Repair evidence spans based on snippet matches
        repair_evidence_items(jd_text, profile.must_have)

        # Validate evidence spans after repair
        errors: List[str] = []
        for it in profile.must_have:
            errors.extend(validate_evidence_spans(jd_text, it))

        if errors:
            last_error = "Evidence span validation failed after repair:\n" + \
                "\n".join(errors[:10])
            messages.append({
                "role": "user",
                "content": (
                    "Your previous output had evidence snippets that could not be matched exactly in jd_text.\n"
                    f"{last_error}\n\n"
                    "Regenerate the full TargetProfile v1. "
                    "Copy evidence.snippet EXACTLY from jd_text (verbatim). "
                    "Remember: start/end must be 0."
                ),
            })
            continue

        # Postprocess constraints
        try:
            profile = postprocess(profile, jd_text, model)
        except Exception as e:
            last_error = str(e)
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous output failed contract checks: {last_error}\n"
                    "Regenerate the full TargetProfile v1 to satisfy all constraints."
                ),
            })
            continue

        return profile

    raise ValueError(last_error or "Failed to parse JD into TargetProfile v1")


if __name__ == "__main__":
    sample_jd = """
We are looking for a Python Engineer with experience in Computer Vision.
Must know PyTorch, OpenCV, and have experience deploying models to edge devices.
Bonus if you have worked with Docker and CI/CD pipelines.
""".lstrip("\n")

    profile = parse_job_description(sample_jd)
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))
