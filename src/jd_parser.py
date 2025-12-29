import os
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Literal, Dict, Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict


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
        min_length=10,
        description="A specific, descriptive phrase (6-14 words). Avoid single keywords."
    )
    purpose: QueryPurpose
    boost_keywords: List[str] = Field(
        default_factory=list,
        description="Specific technical nouns to append. Avoid generic words."
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
# Canonicalization config (file-driven)
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
# Evidence span repair (robust)
# =============================

def _clean_snippet(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.strip(" \t\r\n")
    s = s.strip(".,;:()[]{}")
    return s


def find_first_span(haystack: str, needle: str, case_insensitive: bool = False):
    if not haystack or not needle:
        return None
    if case_insensitive:
        h = haystack.lower()
        n = needle.lower()
        idx = h.find(n)
        if idx == -1:
            return None
        return (idx, idx + len(needle))
    else:
        idx = haystack.find(needle)
        if idx == -1:
            return None
        return (idx, idx + len(needle))


def repair_evidence_items(jd_text: str, items: List[KeywordItem]) -> None:
    """
    Repair evidence spans robustly:
    1) exact snippet match
    2) case-insensitive snippet match
    3) whitespace-normalized match
    4) fallback to keyword raw
    5) fallback to keyword canonical
    If found, overwrite start/end/snippet using exact substring from jd_text.
    """
    jd_norm = re.sub(r"\s+", " ", jd_text)

    for item in items:
        repaired: List[EvidenceSpan] = []
        raw_kw = _clean_snippet(item.raw)
        canon_kw = _clean_snippet(item.canonical or "")

        for ev in item.evidence:
            snip0 = _clean_snippet(ev.snippet or "")

            candidates = [snip0, raw_kw, canon_kw]
            candidates = [c for c in candidates if c]

            found = None

            for cand in candidates:
                # exact
                found = find_first_span(jd_text, cand, case_insensitive=False)
                if found:
                    break

                # case-insensitive
                found = find_first_span(jd_text, cand, case_insensitive=True)
                if found:
                    break

                # whitespace normalized (try in norm text then map back best-effort)
                cand_norm = re.sub(r"\s+", " ", cand).strip()
                if cand_norm:
                    if jd_norm.find(cand_norm) != -1:
                        found2 = find_first_span(
                            jd_text, cand_norm, case_insensitive=False)
                        if not found2:
                            found2 = find_first_span(
                                jd_text, cand_norm, case_insensitive=True)
                        if found2:
                            found = found2
                            break

            if found:
                s, e = found
                ev.start = s
                ev.end = e
                ev.snippet = jd_text[s:e]  # force exact substring
                repaired.append(ev)
            else:
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


def hard_fallback_evidence(jd_text: str, items: List[KeywordItem]) -> None:
    """
    If model evidence is unusable, create one valid evidence span by searching the keyword itself.
    This prevents the whole pipeline from collapsing on trivial casing/punctuation issues.
    """
    for it in items:
        # if evidence missing or invalid, try to generate one from raw/canonical
        if not it.evidence or validate_evidence_spans(jd_text, it):
            for cand in [_clean_snippet(it.raw), _clean_snippet(it.canonical or "")]:
                if not cand:
                    continue
                span = find_first_span(jd_text, cand, case_insensitive=True)
                if span:
                    s, e = span
                    it.evidence = [EvidenceSpan(
                        start=s, end=e, snippet=jd_text[s:e])]
                    break


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


_BANNED_WORDS = {
    "engineer", "developer", "experience", "experienced", "years", "must",
    "required", "looking", "seeking", "ability", "proficient", "familiarity",
    "knowledge", "strong", "role", "position"
}


def looks_atomic(s: str) -> bool:
    if not s:
        return False
    toks = s.split()
    if len(toks) > 3:
        return False
    if any(t in _BANNED_WORDS for t in toks):
        return False
    return True


def postprocess(profile: TargetProfileV1, jd_text: str, model_name: str) -> TargetProfileV1:
    # canonicalize & dedupe
    for group_name in ["must_have", "nice_to_have", "responsibilities", "domain_terms"]:
        group = getattr(profile, group_name)
        for it in group:
            it.canonical = canonicalize(it.canonical or it.raw)
        setattr(profile, group_name, dedupe_by_canonical(group))

    # must_have should be atomic; demote non-atomic to responsibilities (soft enforcement)
    demoted = []
    kept = []
    for it in profile.must_have:
        if looks_atomic(it.canonical):
            kept.append(it)
        else:
            it.type = "responsibility"
            demoted.append(it)
    profile.must_have = kept
    if demoted:
        profile.responsibilities = dedupe_by_canonical(
            profile.responsibilities + demoted)

    # require must_have evidence presence (after demotion)
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
# Prompts (UPDATED: force atomic skills)
# =============================

SYSTEM_PROMPT = r"""
You are an expert technical recruiter helping tailor a resume using a semantic vector database.

Return a Target Profile (v1) used for retrieval + explainability.

==============================
A) ATOMIC KEYWORD RULES (STRICT)
==============================
All keywords in must_have, nice_to_have, responsibilities, domain_terms MUST be ATOMIC.

Definition of ATOMIC:
- A single skill/tool/library/platform/domain noun.
- 1 to 3 tokens max (examples: "python", "computer vision", "edge devices", "ci/cd", "pytorch", "opencv", "docker").
- NO role phrases and NO requirement-language.

BANNED WORDS anywhere inside keyword raw/canonical:
engineer, developer, role, position, experience, experienced, years, must, required, looking for, seeking, ability, proficient, familiarity, knowledge, strong

Examples:
- If JD says "Python Engineer", output must_have includes "python" (NOT "python engineer").
- If JD says "experience in Computer Vision", output "computer vision" (NOT "computer vision experience").

Split phrases into atomic items:
- "deploying models to edge devices" -> "deployment", "edge devices"
- "Docker and CI/CD pipelines" -> "docker", "ci/cd"

==============================
B) EVIDENCE RULES (STRICT)
==============================
- For EVERY item in must_have, include >= 1 evidence entry.
- evidence.snippet MUST be copied EXACTLY from jd_text (verbatim substring).
- Keep evidence.snippet short (<= 60 chars if possible).
- You may set start/end to 0. Offsets are repaired server-side.

==============================
C) RETRIEVAL PLAN RULES (STRICT)
==============================
The database contains mixed generic IT and specialized AI/ML experience.
To retrieve the correct experience, your queries must be SEMANTICALLY DENSE.

1) Avoid single keywords: never use broad terms like "deployment", "leadership", or "python" alone.
2) Bind context to action:
   - BAD: "deployment"
   - GOOD: "deploying deep learning models to edge devices using docker"
3) Use technical specificity when relevant: include "cuda", "onnx", "tensorrt", "rtsp", etc.

retrieval_plan.experience_queries:
- 3 to 7 queries max.
- NO boolean operators (AND/OR/NOT).
- Each query must be 6 to 14 words.
- weight: 1.0 standard, up to 1.8 for critical.

==============================
D) OUTPUT SCHEMA RULES
==============================
- Output MUST strictly match the TargetProfile v1 schema.
- No extra fields. No commentary. JSON only.
"""

USER_TEMPLATE = """Create a TargetProfile v1 from this job description.

IMPORTANT:
- Extract ATOMIC skills only (1-3 tokens).
- Do NOT include role phrases like "python engineer" or "computer vision experience".
- Split combined requirements into atomic items.

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

        # Repair evidence spans based on snippet/keyword matches
        repair_evidence_items(jd_text, profile.must_have)

        # Hard fallback repair pass (prevents brittle failures)
        hard_fallback_evidence(jd_text, profile.must_have)

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
                    "Regenerate the full TargetProfile v1.\n"
                    "- Copy evidence.snippet EXACTLY from jd_text (verbatim).\n"
                    "- Keep snippets short.\n"
                    "- Set start=end=0 if unsure.\n"
                    "- Remember: must_have keywords must be atomic (1-3 tokens), no 'engineer'/'experience'."
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
                    "Regenerate the full TargetProfile v1 to satisfy all constraints.\n"
                    "Remember: must_have keywords must be atomic (1-3 tokens)."
                ),
            })
            continue

        return profile

    raise ValueError(last_error or "Failed to parse JD into TargetProfile v1")


if __name__ == "__main__":
    # Stress-test JD (more complex)
    sample_jd = """
Software Engineer, Machine Learning (Edge Vision)

We are looking for a software engineer to build and deploy computer vision models to edge devices for real-time video analytics. You will work on end-to-end systems including data pipelines, model training, optimization, and deployment.

Must-have:
- Python proficiency and strong software engineering fundamentals
- Computer vision experience (object detection preferred)
- Experience deploying ML models to edge devices or resource-constrained environments
- Familiarity with CUDA or GPU acceleration
- Experience with Docker and CI/CD workflows

Nice-to-have:
- PyTorch and OpenCV
- TensorRT / ONNX optimization
- RTSP video streaming or multi-camera systems
- Monitoring dashboards or UI tools for non-technical users
- Cloud experience (AWS/GCP/Azure)

Responsibilities:
- Build low-latency video ingestion and processing pipelines
- Optimize inference performance (FPS/latency/memory)
- Collaborate cross-functionally and lead small projects when needed
""".lstrip("\n")

    profile = parse_job_description(sample_jd)
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))
