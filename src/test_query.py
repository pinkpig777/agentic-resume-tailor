import os
import json
import traceback
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from loop_controller import run_loop
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
from scorer import score as hybrid_score


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

    # de-dupe keep order
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


def _load_static_data() -> Dict[str, Any]:
    path = os.environ.get("ART_DATA_FILE", "data/my_experience.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return {}


def print_top_candidates(cands: List[Any], top: int = 12, show_hits: bool = True):
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

    static_data = _load_static_data()
    skills_b = build_skills_pseudo_bullet(static_data)

    profile = try_parse_jd(jd_text)
    initial_queries = fallback_queries_from_jd(jd_text)

    best_it, history, best_candidates = run_loop(
        jd_text=jd_text,
        collection=collection,
        embedding_fn=ef,
        profile=profile,  # if None -> retrieval-only mode
        initial_queries=initial_queries,
        skills_pseudo_bullet=skills_b,
        per_query_k=int(os.environ.get("ART_PER_QUERY_K", "10")),
        final_k=int(os.environ.get("ART_FINAL_K", "30")),
        max_bullets=int(os.environ.get("ART_MAX_BULLETS", "16")),
        threshold=int(os.environ.get("ART_SCORE_THRESHOLD", "80")),
        max_iters=int(os.environ.get("ART_MAX_ITERS", "3")),
        alpha=float(os.environ.get("ART_SCORE_ALPHA", "0.7")),
        must_weight=float(os.environ.get("ART_MUST_WEIGHT", "0.8")),
    )

    print("\n🧠 LOOP SUMMARY")
    print("-" * 60)
    print(f"profile_used: {profile is not None}")
    print(f"iters_ran: {len(history)}")
    print(f"best_iter: {best_it.iter_idx}")

    # show top candidates of the best iteration
    print_top_candidates(best_candidates, top=12, show_hits=True)

    print("\n✅ SELECTED (Top-K) [best iteration]")
    print("-" * 60)
    for i, bid in enumerate(best_it.selected_ids, start=1):
        print(f"[{i}] {bid}")

    # If profile is None: no scoring
    if profile is None or best_it.score is None:
        print("\nℹ️ Skipping hybrid scoring because no TargetProfile was produced.")
        raise SystemExit(0)

    # Best iteration score
    s = best_it.score
    print("\n🧪 HYBRID SCORE (best iteration)")
    print(
        f"final={s.final_score} | retrieval={s.retrieval_score:.3f} | "
        f"cov(bullets)={s.coverage_bullets_only:.3f} | cov(all)={s.coverage_all:.3f}"
    )

    print("\nMissing (bullets only, proof):")
    print("  must:", s.must_missing_bullets_only)
    print("  nice:", s.nice_missing_bullets_only)

    print("\nMissing (all + skills):")
    print("  must:", s.must_missing_all)
    print("  nice:", s.nice_missing_all)

    print("\n📜 ITERATION HISTORY")
    print("-" * 60)
    for it in history:
        if it.score is None:
            print(
                f"[iter {it.iter_idx}] score=None boosted_terms={it.boosted_terms}")
        else:
            print(
                f"[iter {it.iter_idx}] final={it.score.final_score} "
                f"retrieval={it.score.retrieval_score:.3f} "
                f"cov(bullets)={it.score.coverage_bullets_only:.3f} "
                f"boosted_terms={it.boosted_terms}"
            )
