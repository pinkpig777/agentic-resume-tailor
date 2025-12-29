import os
import chromadb
from chromadb.utils import embedding_functions

from retrieval import multi_query_retrieve
from selection import select_topk
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
from scorer import score as hybrid_score  # retrieval+coverage hybrid

DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "resume_experience"
EMBED_MODEL = "all-MiniLM-L6-v2"

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
            "fastapi backend restful api websocket realtime features",
            "ruby on rails postgresql redis sidekiq background jobs",
            "docker deployment production ci cd pipelines",
            "react frontend modern javascript typescript web ui",
            "llm integration gpt personalized content product features",
        ]
    else:
        profile_or_queries = profile

    # Retrieval
    cands = multi_query_retrieve(
        collection=collection,
        embedding_fn=ef,
        jd_parser_result=profile_or_queries,
        per_query_k=10,
        final_k=30,
    )

    print_results(cands, top=12, show_hits=True)

    # Selection (Top-K)
    selected_ids, _ = select_topk(cands, max_bullets=16)
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
                {
                    "bullet_id": c.bullet_id,
                    "text_latex": c.text_latex,
                    "meta": c.meta,
                }
            )

    # Hybrid scoring only if we have a TargetProfile
    if profile is None:
        print("\nℹ️ Skipping hybrid scoring because no TargetProfile was produced.")
    else:
        pk = extract_profile_keywords(profile)

        must_evs = match_keywords_against_bullets(
            pk["must_have"], selected_bullets)
        nice_evs = match_keywords_against_bullets(
            pk["nice_to_have"], selected_bullets)

        hybrid = hybrid_score(
            selected_candidates=selected_candidates,
            all_candidates=cands,
            profile_keywords=pk,
            must_evs=must_evs,
            nice_evs=nice_evs,
            alpha=float(os.environ.get("ART_SCORE_ALPHA", "0.5")),
        )

        print("\n🧪 HYBRID SCORE")
        print(
            f"Hybrid: {hybrid.final_score} | retrieval={hybrid.retrieval_score:.3f} coverage={hybrid.coverage_score:.3f}"
        )
        print("missing must:", hybrid.must_missing)
        print("missing nice:", hybrid.nice_missing)

        print("\n✅ MATCH EVIDENCE (must-have)")
        for ev in must_evs:
            if ev.tier != "none":
                print(
                    f"- {ev.keyword} [{ev.tier}] satisfied_by={ev.satisfied_by} bullets={ev.bullet_ids[:3]}")
