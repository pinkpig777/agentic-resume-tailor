from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from agentic_resume_tailor.core.jd_utils import fallback_queries_from_jd, try_parse_jd
from agentic_resume_tailor.core.loop_controller import LoopConfig, run_loop
from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

DB_PATH = settings.db_path
COLLECTION_NAME = settings.collection_name
EMBED_MODEL = settings.embed_model


def load_collection() -> tuple[Any, Any]:
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    logger.info("Loaded collection '%s' with %s records", COLLECTION_NAME, collection.count())
    return collection, ef


def _load_static_data() -> Dict[str, Any]:
    try:
        with open(settings.data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", settings.data_file, e)
        return {}


def print_top_candidates(cands: List[Any], top: int = 12, show_hits: bool = True) -> None:
    print("\nRETRIEVAL RESULTS")
    print("-" * 60)
    for i, c in enumerate(cands[:top], start=1):
        print(f"[{i}] {c.source}")
        print(f"ID: {c.bullet_id}")
        print(f"Best: weighted={c.best_hit.weighted:.4f} cos={c.best_hit.cosine:.4f}")
        print(f"Purpose: {c.best_hit.purpose}")
        print(f"Best query: {c.best_hit.query}")
        print(f"LATEX: {c.text_latex}")
        if show_hits:
            print("  hits:")
            for h in c.hits[:3]:
                print(
                    f"   - w={h.weight:.2f} cos={h.cosine:.4f} weighted={h.weighted:.4f} "
                    f"| {h.purpose} | {h.query}"
                )
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

    profile = try_parse_jd(jd_text)
    base_profile_or_queries = profile or fallback_queries_from_jd(jd_text)

    cfg = LoopConfig(
        max_iters=settings.max_iters,
        threshold=settings.threshold,
        per_query_k=settings.per_query_k,
        final_k=settings.final_k,
        max_bullets=settings.max_bullets,
        alpha=settings.alpha,
        must_weight=settings.must_weight,
        boost_weight=settings.boost_weight,
        boost_top_n_missing=settings.boost_top_n_missing,
    )

    loop = run_loop(
        jd_text=jd_text,
        static_data=static_data,
        collection=collection,
        embedding_fn=ef,
        base_profile_or_queries=base_profile_or_queries,
        cfg=cfg,
    )

    print("\nLOOP SUMMARY")
    print("-" * 60)
    print(f"profile_used: {profile is not None}")
    print(f"iters_ran: {len(loop.iterations)}")
    print(f"best_iter: {loop.best_iteration_index}")

    print_top_candidates(loop.best_candidates, top=12, show_hits=True)

    print("\nSELECTED (Top-K) [best iteration]")
    print("-" * 60)
    for i, bid in enumerate(loop.best_selected_ids, start=1):
        print(f"[{i}] {bid}")

    if loop.best_hybrid is None:
        print("\nSkipping hybrid scoring because no TargetProfile was produced.")
        raise SystemExit(0)

    s = loop.best_hybrid
    print("\nHYBRID SCORE (best iteration)")
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

    print("\nITERATION HISTORY")
    print("-" * 60)
    for it in loop.iterations:
        scores = it.get("scores") or {}
        missing = it.get("missing") or {}
        if not scores:
            print(f"[iter {it.get('iteration')}] score=None")
        else:
            print(
                f"[iter {it.get('iteration')}] final={scores.get('final')} "
                f"retrieval={scores.get('retrieval'):.3f} "
                f"cov_bullets={scores.get('coverage_bullets_only'):.3f} "
                f"cov_all={scores.get('coverage_all'):.3f} "
                f"missing_must={len(missing.get('must_bullets_only') or [])}"
            )
