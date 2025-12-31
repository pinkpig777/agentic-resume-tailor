import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from agentic_resume_tailor.settings import get_settings

logger = logging.getLogger(__name__)

# =============================
# Target Profile v1 (STRICT schema for Structured Outputs)
# =============================

KeywordType = Literal["hard_skill", "soft_skill", "tool", "framework", "domain", "responsibility"]
QueryPurpose = Literal[
    "core_stack", "domain_fit", "deployment", "scale_reliability", "leadership", "general"
]


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # We do NOT trust model-provided offsets. We'll repair them server-side.
    start: int = Field(
        ge=0, description="Character start offset into original jd_text (repaired server-side)."
    )
    end: int = Field(
        ge=0, description="Character end offset into original jd_text (repaired server-side)."
    )
    snippet: str = Field(min_length=1, description="Exact substring copied from jd_text.")


class KeywordItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw: str = Field(min_length=1, description="Original phrase as it appears in the JD.")
    canonical: str = Field(min_length=1, description="Canonicalized (lowercase, normalized).")
    type: KeywordType
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    priority: int = Field(ge=1, le=5, description="1 is highest priority.")


class QueryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        min_length=10,
        description=(
            "A specific, descriptive phrase (5-12 words). "
            "BAD: 'deployment'. GOOD: 'deploying computer vision models on edge devices'."
        ),
    )
    purpose: QueryPurpose
    boost_keywords: List[str] = Field(
        default_factory=list,
        description="Specific technical nouns to append (e.g., 'tensorrt', 'onnx', 'cuda'). Avoid generic words.",
    )
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
    role_title: str = Field(default="", description="Job title if identifiable.")
    role_summary: str = Field(default="", description="1-2 sentence summary.")
    must_have: List[KeywordItem] = Field(default_factory=list)
    nice_to_have: List[KeywordItem] = Field(default_factory=list)
    responsibilities: List[KeywordItem] = Field(default_factory=list)
    domain_terms: List[KeywordItem] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    # Make meta optional so the model doesn't have to output it; we fill it postprocess.
    meta: Optional[MetaInfo] = None


# =============================
# Canonicalization config (file-driven, canonical-first schema)
# =============================

DEFAULT_CANON_CONFIG: Dict[str, Any] = {
    "schema_version": "canon_config_v1",
    "options": {
        "keep_chars": "+#./-",
        "collapse_whitespace": True,
        "slash_to_space": True,
        "dash_to_space": True,
        "separator_exceptions": ["ci/cd", "node.js"],
    },
    "canon_groups": [],
}


def load_canon_config(path: str) -> Dict[str, Any]:
    """Load canonicalization config from JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("canonicalization config must be a JSON object")
        if data.get("schema_version") != "canon_config_v1":
            raise ValueError("canonicalization config schema_version must be canon_config_v1")
        return data
    except FileNotFoundError:
        logger.warning("%s not found; using empty canonical config", path)
        return DEFAULT_CANON_CONFIG
    except Exception as e:
        logger.warning(
            "Failed to load canonicalization config (%s); using empty canonical config",
            e,
        )
        return DEFAULT_CANON_CONFIG


settings = get_settings()
CANON_CONFIG_PATH = settings.canon_config
_CANON_CONFIG = load_canon_config(CANON_CONFIG_PATH)


def _base_normalize(text: str, keep_chars: str) -> str:
    """Normalize text for canonicalization."""
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    keep = re.escape(keep_chars)
    s = re.sub(rf"[^a-z0-9{keep}\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_variant_to_canonical_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Build a variant-to-canonical lookup map."""
    opts = config.get("options", {})
    keep_chars = str(opts.get("keep_chars", "+#./-"))

    out: Dict[str, str] = {}
    for group in config.get("canon_groups", []) or []:
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
    """Canonicalize a term using the loaded config."""
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
# Evidence span repair + salvage (best-effort; never hard-fail)
# =============================


def find_all_spans(haystack: str, needle: str) -> List[Tuple[int, int]]:
    """Find all occurrences of a substring in text."""
    spans: List[Tuple[int, int]] = []
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
    """Repair evidence spans based on JD snippets."""
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
                        spans = find_all_spans(jd_text, snip_norm)

            if spans:
                s, e = spans[0]
                ev.start = s
                ev.end = e
                ev.snippet = jd_text[s:e]
                repaired.append(ev)
            else:
                # Keep as-is for now; salvage step may fix it.
                repaired.append(ev)

        item.evidence = repaired


def validate_evidence_spans(jd_text: str, item: KeywordItem) -> List[str]:
    """Validate evidence spans against the JD text."""
    errs: List[str] = []
    n = len(jd_text)

    for ev in item.evidence:
        if ev.start < 0 or ev.end <= ev.start:
            errs.append(f"Invalid offsets for '{item.raw}': start={ev.start}, end={ev.end}")
            continue
        if ev.end > n:
            errs.append(f"Out of range for '{item.raw}': end={ev.end} > len={n}")
            continue
        expected = jd_text[ev.start : ev.end]
        if ev.snippet != expected:
            errs.append(f"Snippet mismatch for '{item.raw}' at [{ev.start}:{ev.end}]")
    return errs


def _first_case_insensitive_span(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    """Find a case-insensitive span for a substring."""
    if not haystack or not needle:
        return None
    n = needle.strip()
    if not n:
        return None
    idx = haystack.lower().find(n.lower())
    if idx == -1:
        return None
    return (idx, idx + len(n))


def salvage_evidence_for_item(jd_text: str, item: KeywordItem) -> None:
    """Best-effort salvage of evidence spans for an item."""
    candidates: List[str] = []
    if item.raw:
        candidates.append(item.raw)
    if item.canonical:
        candidates.append(item.canonical)
    if item.raw:
        candidates.append(canonicalize(item.raw))

    for cand in candidates:
        span = _first_case_insensitive_span(jd_text, cand)
        if span:
            s, e = span
            item.evidence = [EvidenceSpan(start=s, end=e, snippet=jd_text[s:e])]
            return

    item.evidence = []


# =============================
# Postprocess + constraints
# =============================


def jd_hash(jd_text: str) -> str:
    """Compute a stable hash for the JD text."""
    return hashlib.sha256((jd_text or "").encode("utf-8")).hexdigest()


def dedupe_by_canonical(items: List[KeywordItem]) -> List[KeywordItem]:
    """Deduplicate keyword items by canonical text."""
    seen = set()
    out: List[KeywordItem] = []
    for it in items:
        key = (it.canonical or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def sanitize_query_for_embeddings(q: str) -> str:
    """Clean a query string for embeddings."""
    q = q or ""
    q = re.sub(r"\bAND\b|\bOR\b|\bNOT\b", " ", q, flags=re.IGNORECASE)
    q = q.replace("(", " ").replace(")", " ").replace('"', " ").replace("'", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def postprocess(profile: TargetProfileV1, jd_text: str, model_name: str) -> TargetProfileV1:
    """Apply constraints and metadata to a parsed profile."""
    # canonicalize & dedupe keywords
    for group_name in ["must_have", "nice_to_have", "responsibilities", "domain_terms"]:
        group: List[KeywordItem] = getattr(profile, group_name)
        for it in group:
            it.canonical = canonicalize(it.canonical or it.raw)
        setattr(profile, group_name, dedupe_by_canonical(group))

    # Retrieval plan constraints
    eq = profile.retrieval_plan.experience_queries
    if len(eq) < 3:
        raise ValueError("retrieval_plan.experience_queries must have at least 3 queries")
    if len(eq) > 7:
        profile.retrieval_plan.experience_queries = eq[:7]

    # sanitize queries and canonicalize boosts
    for qi in profile.retrieval_plan.experience_queries:
        qi.query = sanitize_query_for_embeddings(qi.query)
        qi.boost_keywords = [canonicalize(b) for b in (qi.boost_keywords or []) if b]
        qi.weight = min(3.0, max(0.1, float(qi.weight)))

    # Evidence is best-effort: never fail pipeline if missing
    missing = [it.raw for it in profile.must_have if not it.evidence]
    if missing:
        logger.warning(
            "Must-have items missing evidence (best-effort mode): %s",
            missing[:12],
        )

    # Fill meta
    profile.meta = MetaInfo(
        parser_model=model_name,
        jd_hash=jd_hash(jd_text),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    return profile


# =============================
# Prompts (atomic skills enforcement)
# =============================

SYSTEM_PROMPT = """
You are an expert technical recruiter helping tailor a resume using a semantic vector database.

Return a Target Profile (v1) used for retrieval + explainability.

### CRITICAL: ATOMIC SKILLS ONLY
When you output keywords in must_have / nice_to_have / domain_terms:
- Each KeywordItem MUST be a SINGLE atomic term.
  Examples (GOOD): "python", "fastapi", "ruby on rails", "postgresql", "docker", "websockets", "ci/cd"
  Examples (BAD): "python (fastapi)", "docker and kubernetes", "sql database experience", "aws/gcp/azure"
- Split combined phrases into separate items.
- Avoid parentheses, slashes, and multi-skill phrases in KeywordItem.raw/canonical.

### CRITICAL RETRIEVAL INSTRUCTIONS
The database contains mixed generic IT and specialized AI/ML experience.
To retrieve the correct experience, your queries must be SEMANTICALLY DENSE.

1) Avoid single keywords like "deployment" or "leadership" alone.
2) Bind context to action:
   - BAD: query="deployment"
   - GOOD: query="deploying computer vision models to edge devices using docker"
3) Use technical specificity: include libraries/tools directly in the query string.

### Evidence (best-effort)
- If a must-have skill appears in jd_text, include evidence with snippet copied EXACTLY.
- If you are unsure, OMIT evidence rather than guessing.
- If you provide evidence, set start=0 and end=0 (server will repair).

### Schema rules
- Output MUST strictly match TargetProfile v1 schema.
- retrieval_plan.experience_queries:
  - 3 to 7 queries max.
  - NO boolean operators (AND/OR/NOT).
  - weight: 1.0 (standard) to 1.8 (critical skills).
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
    """Parse a JD into TargetProfile v1 using OpenAI."""
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text is empty")

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=60.0,
        max_retries=0,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": USER_TEMPLATE.format(jd_text=jd_text).strip()},
    ]

    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Analyzing job description (attempt %s/%s)...", attempt, max_attempts)

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=TargetProfileV1,
        )
        profile: TargetProfileV1 = completion.choices[0].message.parsed

        # 1) Canonicalize the canonical fields early (helps salvage matching)
        for grp in [
            profile.must_have,
            profile.nice_to_have,
            profile.responsibilities,
            profile.domain_terms,
        ]:
            for it in grp:
                it.canonical = canonicalize(it.canonical or it.raw)

        # 2) Repair evidence spans based on snippet matches (exact/whitespace)
        repair_evidence_items(jd_text, profile.must_have)

        # 3) Validate evidence; if mismatched, SALVAGE LOCALLY (do not fail)
        all_errors: List[str] = []
        for it in profile.must_have:
            errs = validate_evidence_spans(jd_text, it)
            if errs:
                salvage_evidence_for_item(jd_text, it)
                # If still invalid after salvage, drop evidence.
                if validate_evidence_spans(jd_text, it):
                    it.evidence = []
                all_errors.extend(errs)

        if all_errors:
            logger.warning("Evidence issues detected; continuing in best-effort mode.")
            last_error = "Evidence had mismatches; best-effort salvage applied."

        # 4) Postprocess contract checks (queries, dedupe, meta)
        try:
            profile = postprocess(profile, jd_text, model)
        except Exception as e:
            last_error = str(e)
            # Ask for a retry if contract checks fail (not evidence-related)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed contract checks: {last_error}\n"
                        "Regenerate the full TargetProfile v1 to satisfy all constraints.\n"
                        "Remember: atomic skills only, 3-7 retrieval queries, no boolean operators."
                    ),
                }
            )
            continue

        return profile

    raise ValueError(last_error or "Failed to parse JD into TargetProfile v1")


if __name__ == "__main__":
    sample_jd = """
Software Engineer (Full Stack), AI Product

We are building a user-facing AI product and are hiring a full stack engineer to deliver features end-to-end. You will build web UIs, APIs, background jobs, and integrate ML components.

Must-have:
- Backend development with Python (FastAPI) or Ruby on Rails
- SQL database experience
- Experience building REST APIs and real-time features (WebSockets)
- Docker and production deployment
- Strong collaboration and ownership

Nice-to-have:
- Redis / background jobs (Sidekiq, Celery)
- React or modern frontend frameworks
- CI/CD pipelines
- Experience integrating LLMs into products
- Cloud deployment (AWS/GCP/Azure)

Responsibilities:
- Ship user-facing features quickly with good engineering quality
- Improve system reliability and observability
- Collaborate with PM/design and lead small initiatives
""".lstrip("\n")

    profile = parse_job_description(sample_jd)
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))
