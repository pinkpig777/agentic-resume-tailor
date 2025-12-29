import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

from retrieval import multi_query_retrieve
from selection import select_topk
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
from scorer import score as hybrid_score  # retrieval+coverage hybrid


DB_PATH = os.environ.get("ART_DB_PATH", "data/processed/chroma_db")
COLLECTION_NAME = os.environ.get("ART_COLLECTION", "resume_experience")
EMBED_MODEL = os.environ.get("ART_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

USE_JD_PARSER = os.environ.get("ART_USE_JD_PARSER", "1") == "1"


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
        print("⚠️ JD parser failed; falling back to manual queries.")
        traceback.print_exc()
        return None


def fallback_queries_from_jd(jd_text: str, max_q: int = 6) -> List[str]:
    """
    Minimal heuristic fallback.
    Produces embedding-friendly queries from bullet lines + a condensed full query.
    """
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    bulletish = [ln.lstrip("-•* ").strip()
                 for ln in lines if ln.strip().startswith(("-", "•", "*"))]

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


def build_skills_pseudo_bullet(static_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Skills are NOT embedded in Chroma, but users still want credit if a must-have exists in Skills.
    We add a pseudo bullet for coverage_all computations only.
    """
    skills = static_data.get("skills", {}) or {}
    parts = []
    for k in ["languages_frameworks", "ai_ml", "db_tools"]:
        v = skills.get(k)
        if v:
            parts.append(str(v))
    txt = " | ".join(parts).strip()
    if not txt:
        return None
    return {"bullet_id": "__skills__", "text_latex": txt, "meta": {"section": "skills"}}


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
                    f"   - w={h.weight:.2f} cos={h.cosine:.4f} weighted={h.weighted:.4f} | {h.purpose} | {h.query}")
        print()


if __name__ == "__main__":
    collection, ef = load_collection()

    # Robust JD for stress testing
    jd_text = """
Software Engineer (Full Stack), AI Product

We are building a user-facing AI product and are hiring a full stack engineer to deliver features end-to-end.
You will build web UIs, APIs, background jobs, and integrate ML components.

Must-have:
- Backend development with Python (FastAPI) or Ruby on Rails
- SQL database experience
- Experience building REST APIs and real-time features (WebSockets)
- Docker and production deployment
- Strong collaboration and ownership

Nice-to-have:
- Redis / background jobs (Sidekiq, Celery)
- React or modern frontend frameworks
- CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- Experience integrating LLMs into products (RAG, prompt engineering)
- Cloud deployment (AWS/GCP/Azure)

Responsibilities:
- Ship user-facing features quickly with good engineering quality
- Improve system reliability and observability
- Collaborate with PM/design and lead small initiatives
""".strip()

    profile = try_parse_jd(jd_text)

    if profile is None:
        print("ℹ️ No TargetProfile produced. Using manual multi-query retrieval only.")
        profile_or_queries = fallback_queries_from_jd(jd_text)
    else:
        profile_or_queries = profile

    # Retrieval
    cands = multi_query_retrieve(
        collection=collection,
        embedding_fn=ef,
        jd_parser_result=profile_or_queries,
        per_query_k=int(os.environ.get("ART_PER_QUERY_K", "10")),
        final_k=int(os.environ.get("ART_FINAL_K", "30")),
    )

    print_results(cands, top=12, show_hits=True)

    # Selection (Top-K)
    selected_ids, _ = select_topk(cands, max_bullets=int(
        os.environ.get("ART_MAX_BULLETS", "16")))
    print("\n✅ SELECTED (Top-K)")
    print("-" * 60)
    for i, bid in enumerate(selected_ids, start=1):
        print(f"[{i}] {bid}")

    # Build selected bullets payload + selected candidates (for hybrid scoring)
    selected_set = set(selected_ids)
    selected_candidates = [c for c in cands if c.bullet_id in selected_set]

    selected_bullets = [{"bullet_id": c.bullet_id, "text_latex": c.text_latex,
                         "meta": c.meta} for c in selected_candidates]
    all_bullets = [{"bullet_id": c.bullet_id,
                    "text_latex": c.text_latex, "meta": c.meta} for c in cands]

    # Hybrid scoring only if we have a TargetProfile
    if profile is None:
        print("\nℹ️ Skipping hybrid scoring because no TargetProfile was produced.")
    else:
        pk = extract_profile_keywords(profile)

        # bullets-only: what will actually appear on the rendered page
        must_evs_bullets_only = match_keywords_against_bullets(
            pk["must_have"], selected_bullets)
        nice_evs_bullets_only = match_keywords_against_bullets(
            pk["nice_to_have"], selected_bullets)

        # all: allow matching against all retrieved bullets + Skills (so user isn't misled)
        # NOTE: this does NOT change selection. It's scoring/explainability only.
        try:
            import json
            with open(os.environ.get("ART_DATA_FILE", "data/my_experience.json"), "r", encoding="utf-8") as f:
                static_data = json.load(f)
        except Exception:
            static_data = {}

        skills_b = build_skills_pseudo_bullet(static_data)
        all_bullets_plus_skills = all_bullets + \
            ([skills_b] if skills_b else [])

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
        print(f"final={hybrid.final_score} | retrieval={hybrid.retrieval_score:.3f} | cov(bullets)={hybrid.coverage_bullets_only:.3f} | cov(all)={hybrid.coverage_all:.3f}")

        print("\nMissing (bullets only, proof):")
        print("  must:", hybrid.must_missing_bullets_only)
        print("  nice:", hybrid.nice_missing_bullets_only)

        print("\nMissing (all + skills):")
        print("  must:", hybrid.must_missing_all)
        print("  nice:", hybrid.nice_missing_all)
