from scorer import score as hybrid_score  # retrieval+coverage hybrid
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
import os
import json
import traceback
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from retrieval import multi_query_retrieve
from selection import select_topk
from keyword_matcher import (
    extract_profile_keywords,
    match_keywords_against_bullets,
    build_match_corpus,
    canonicalize_term,
)
from scorer import score as hybrid_score

DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "resume_experience"
EMBED_MODEL = os.environ.get("ART_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

USE_JD_PARSER = True
JD_MODEL = os.environ.get("ART_JD_MODEL", "gpt-4.1-nano-2025-04-14")


def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef)
    print(
        f"✅ Loaded collection '{COLLECTION_NAME}' with {collection.count()} records")
    return collection, ef


def try_parse_jd(jd_text: str):
    if not USE_JD_PARSER:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    try:
        import jd_parser  # src/jd_parser.py
        if not hasattr(jd_parser, "parse_job_description"):
            raise RuntimeError("jd_parser.parse_job_description not found")
        return jd_parser.parse_job_description(jd_text, model=JD_MODEL)
    except Exception:
        print("⚠️ JD parser failed; falling back to local heuristic keywords + manual queries.")
        traceback.print_exc()
        return None


def local_extract_keywords(jd_text: str) -> Dict[str, List[Dict[str, str]]]:
    must: List[str] = []
    nice: List[str] = []

    section = None
    for raw_line in (jd_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("must-have"):
            section = "must"
            continue
        if low.startswith("nice-to-have") or low.startswith("nice to have"):
            section = "nice"
            continue
        if low.startswith("responsibilities"):
            section = None
            continue

        if line.startswith("-") or line.startswith("•"):
            item = line.lstrip("-•").strip()
            if not item:
                continue
            if section == "must":
                must.append(item)
            elif section == "nice":
                nice.append(item)

    if not must and not nice:
        must = ["python", "rest api", "sql", "docker"]

    def to_items(xs: List[str]) -> List[Dict[str, str]]:
        out = []
        seen = set()
        for x in xs:
            c = canonicalize_term(x)
            if not c or c in seen:
                continue
            seen.add(c)
            out.append({"raw": x, "canonical": c})
        return out

    return {"must_have": to_items(must), "nice_to_have": to_items(nice)}


def fallback_queries_for_jd(jd_text: str) -> List[str]:
    return [
        "fastapi backend python restful api websocket realtime features",
        "ruby on rails postgresql redis sidekiq background jobs",
        "docker deployment production ci/cd github actions",
        "react typescript javascript web ui product features",
        "llm integration gpt rag vector database product",
        jd_text[:400],
    ]


def print_results(cands, top=12, show_hits=True):
    print("\n🔎 RETRIEVAL RESULTS")
    print("-" * 60)
    for i, c in enumerate(cands[:top], start=1):
        print(f"[{i}] {c.source}")
        print(f"ID: {c.bullet_id}")
        print(
            f"Best: weighted={c.best_hit.weighted:.4f} cos={c.best_hit.cosine:.4f}")
        print(f"Purpose: {c.best_hit.purpose}")
        print(f"Best query: {c.best_hit.query}")
        print(f"Total weighted: {c.total_weighted:.4f}")
        print(f"LATEX: {c.text_latex}")
        if show_hits:
            print("  hits:")
            for h in c.hits[:3]:
                print(
                    f"   - w={h.weight:.2f} cos={h.cosine:.4f} weighted={h.weighted:.4f} | {h.purpose} | {h.query}"
                )
        print()


def load_resume_json() -> Dict[str, Any]:
    with open("data/my_experience.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    collection, ef = load_collection()
    resume_data = load_resume_json()

    jd_text = """
Software Engineer (Full Stack), AI Product

We are building a user-facing AI product and are hiring a full stack engineer to deliver features end-to-end.
You will build web UIs, APIs, background jobs, and integrate ML components.

Must-have:
- Backend development with Python (FastAPI) or Ruby on Rails
- SQL database experience (PostgreSQL preferred)
- Experience building REST APIs and real-time features (WebSockets)
- Docker and production deployment
- Strong collaboration, ownership, and ability to ship

Nice-to-have:
- Redis / background jobs (Sidekiq, Celery)
- React or modern frontend frameworks
- CI/CD pipelines (GitHub Actions)
- Experience integrating LLMs into products (RAG, vector databases)
- Cloud deployment (AWS/GCP/Azure)
""".strip()

    profile = try_parse_jd(jd_text)

    if profile is not None:
        retrieval_input = profile
        pk = extract_profile_keywords(profile)
    else:
        retrieval_input = fallback_queries_for_jd(jd_text)
        pk = local_extract_keywords(jd_text)

    # Retrieval
    cands = multi_query_retrieve(
        collection=collection,
        embedding_fn=ef,
        jd_parser_result=retrieval_input,
        per_query_k=10,
        final_k=30,
    )

    print_results(cands, top=12, show_hits=True)

    # Selection
    selected_ids, _ = select_topk(cands, max_bullets=16)
    print("\n✅ SELECTED (Top-K)")
    print("-" * 60)
    for i, bid in enumerate(selected_ids, start=1):
        print(f"[{i}] {bid}")

    selected_set = set(selected_ids)
    selected_bullets = []
    selected_candidates = []
    for c in cands:
        if c.bullet_id in selected_set:
            selected_candidates.append(c)
            selected_bullets.append(
                {"bullet_id": c.bullet_id, "text_latex": c.text_latex, "meta": c.meta})

    # Coverage evidence:
    # bullets-only = proof in selected bullets
    must_evs_bullets_only = match_keywords_against_bullets(
        pk["must_have"], selected_bullets)
    nice_evs_bullets_only = match_keywords_against_bullets(
        pk["nice_to_have"], selected_bullets)

    # skills + bullets = "do I have it on resume at all"
    corpus = build_match_corpus(resume_data, selected_bullets)
    must_evs_all = match_keywords_against_bullets(pk["must_have"], corpus)
    nice_evs_all = match_keywords_against_bullets(pk["nice_to_have"], corpus)

    # Hybrid score
    import os


DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "resume_experience"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

USE_JD_PARSER = True


def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef)
    print(
        f"✅ Loaded collection '{COLLECTION_NAME}' with {collection.count()} records")
    return collection, ef


def try_parse_jd(jd_text: str):
    if not USE_JD_PARSER:
        return None

    try:
        import jd_parser  # src/jd_parser.py
        if not hasattr(jd_parser, "parse_job_description"):
            raise RuntimeError("jd_parser.parse_job_description not found")

        model = os.environ.get("ART_JD_MODEL", "gpt-4.1-nano-2025-04-14")
        try:
            return jd_parser.parse_job_description(jd_text, model=model)
        except TypeError:
            return jd_parser.parse_job_description(jd_text)

    except Exception:
        import traceback
        print("⚠️ JD parser failed; falling back to manual queries.")
        traceback.print_exc()
        return None


def print_results(cands, top=12, show_hits=True):
    print("\n🔎 RETRIEVAL RESULTS")
    print("-" * 60)
    for i, c in enumerate(cands[:top], start=1):
        print(f"[{i}] {c.source}")
        print(f"ID: {c.bullet_id}")
        print(
            f"Best: weighted={c.best_hit.weighted:.4f} cos={c.best_hit.cosine:.4f}")
        print(f"Purpose: {c.best_hit.purpose}")
        print(f"Best query: {c.best_hit.query}")
        print(f"LATEX: {c.text_latex}")
        if show_hits:
            print("  hits:")
            for h in c.hits[:3]:
                print(
                    f"   - w={h.weight:.2f} cos={h.cosine:.4f} weighted={h.weighted:.4f} | {h.purpose} | {h.query}"
                )
        print()


def build_skills_pseudo_bullet() -> dict:
    """
    Optional: allow the matcher to detect must-have keywords that exist in Skills section.
    This does NOT change bullets; it's just for diagnostics (and optionally coverage_all).
    """
    try:
        import json
        data_path = os.environ.get(
            "ART_EXPERIENCE_JSON", "data/my_experience.json")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        skills = data.get("skills", {}) or {}
        parts = []
        for k in ["languages_frameworks", "ai_ml", "db_tools"]:
            v = skills.get(k)
            if v:
                parts.append(str(v))
        skill_text = " | ".join(parts).strip()
        if not skill_text:
            return {}
        return {"bullet_id": "__skills__", "text_latex": skill_text, "meta": {"section": "skills"}}
    except Exception:
        return {}


if __name__ == "__main__":
    collection, ef = load_collection()

    jd_text = """
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
""".strip()

    profile = try_parse_jd(jd_text)

    # Fallback: manual multi-query aligned to THIS JD (full-stack + AI product)
    if profile is None:
        print("ℹ️ No TargetProfile produced. Using manual multi-query retrieval only.")
        profile_or_queries = [
            "fastapi backend python restful api websocket realtime features",
            "ruby on rails postgresql sql database migrations",
            "docker deployment production ci cd pipelines",
            "redis sidekiq background jobs async queues",
            "react typescript modern frontend web ui",
            "llm integration gpt product features",
            "cloud deployment aws gcp azure",
        ]
    else:
        profile_or_queries = profile

    # Retrieval (Node 2)
    cands = multi_query_retrieve(
        collection=collection,
        embedding_fn=ef,
        jd_parser_result=profile_or_queries,
        per_query_k=int(os.environ.get("ART_PER_QUERY_K", "10")),
        final_k=int(os.environ.get("ART_FINAL_K", "30")),
    )

    print_results(cands, top=12, show_hits=True)

    # Selection (Node 3)
    selected_ids, _ = select_topk(cands, max_bullets=int(
        os.environ.get("ART_MAX_BULLETS", "16")))
    print("\n✅ SELECTED (Top-K)")
    print("-" * 60)
    for i, bid in enumerate(selected_ids, start=1):
        print(f"[{i}] {bid}")

    # Build selected bullets payload + selected candidates (for hybrid scoring)
    selected_set = set(selected_ids)
    selected_bullets = []
    selected_candidates = []
    for c in cands:
        if c.bullet_id in selected_set:
            selected_candidates.append(c)
            selected_bullets.append(
                {"bullet_id": c.bullet_id, "text_latex": c.text_latex, "meta": c.meta})

    # Hybrid scoring only if we have a TargetProfile
    if profile is None:
        print("\nℹ️ Skipping hybrid scoring because no TargetProfile was produced.")
        raise SystemExit(0)

    pk = extract_profile_keywords(profile)

    # Evidence on selected bullets only
    must_evs_bullets_only = match_keywords_against_bullets(
        pk["must_have"], selected_bullets)
    nice_evs_bullets_only = match_keywords_against_bullets(
        pk["nice_to_have"], selected_bullets)

    # Evidence on "all" bullets (what you define as all = retrieved candidates)
    all_bullets = [{"bullet_id": c.bullet_id,
                    "text_latex": c.text_latex, "meta": c.meta} for c in cands]

    # Optional: add skills pseudo-bullet for diagnostics
    skills_b = build_skills_pseudo_bullet()
    if skills_b:
        all_bullets_plus_skills = all_bullets + [skills_b]
    else:
        all_bullets_plus_skills = all_bullets

    must_evs_all = match_keywords_against_bullets(
        pk["must_have"], all_bullets_plus_skills)
    nice_evs_all = match_keywords_against_bullets(
        pk["nice_to_have"], all_bullets_plus_skills)

    hybrid = hybrid_score(
        selected_candidates=selected_candidates,
        all_candidates=cands,
        profile_keywords=pk,
        must_evs_all=must_evs_all,
        nice_evs_all=nice_evs_all,
        must_evs_bullets_only=must_evs_bullets_only,
        nice_evs_bullets_only=nice_evs_bullets_only,
        alpha=float(os.environ.get("ART_SCORE_ALPHA", "0.7")),
        must_weight=float(os.environ.get("ART_MUST_WEIGHT", "0.8")),
    )

    print("\n🧪 HYBRID SCORE")
    print(
        f"final={hybrid.final_score} | retrieval={hybrid.retrieval_score:.3f} "
        f"| coverage(bullets_only)={hybrid.coverage_bullets_only:.3f}"
    )
    if hybrid.coverage_all is not None:
        print(f"coverage(all+skills)={hybrid.coverage_all:.3f}")

    print("\nMissing (bullets only, proof):")
    print("  must:", hybrid.must_missing_bullets_only)
    print("  nice:", hybrid.nice_missing_bullets_only)

    if hybrid.must_missing_all is not None or hybrid.nice_missing_all is not None:
        print("\nMissing (all retrieved candidates + skills pseudo-bullet):")
        print("  must:", hybrid.must_missing_all or [])
        print("  nice:", hybrid.nice_missing_all or [])

    print("\n✅ MATCH EVIDENCE (must-have, bullets-only)")
    for ev in must_evs_bullets_only:
        if ev.tier != "none":
            print(
                f"- {ev.keyword} [{ev.tier}] satisfied_by={ev.satisfied_by} bullets={ev.bullet_ids[:3]}")

    print("\n🧪 HYBRID SCORE")
    print(
        f"final={hybrid.final_score} | retrieval={hybrid.retrieval_score:.3f} | coverage={hybrid.coverage_score:.3f}"
    )

    print("\nMissing (skills + bullets):")
    print("  must:", hybrid.must_missing)
    print("  nice:", hybrid.nice_missing)

    print("\nMissing (bullets only, proof):")
    print("  must:", hybrid.must_missing_bullets_only)
    print("  nice:", hybrid.nice_missing_bullets_only)

    print("\n✅ MATCH EVIDENCE (must-have, bullets only)")
    for ev in must_evs_bullets_only:
        if ev.tier != "none":
            print(
                f"- {ev.keyword} [{ev.tier}] satisfied_by={ev.satisfied_by} bullets={ev.bullet_ids[:3]}")

    print("\n✅ MATCH EVIDENCE (must-have, skills + bullets)")
    for ev in must_evs_all:
        if ev.tier != "none":
            print(
                f"- {ev.keyword} [{ev.tier}] satisfied_by={ev.satisfied_by} hits={ev.bullet_ids[:3]}")
