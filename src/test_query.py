import os
import json
import chromadb
from chromadb.utils import embedding_functions

from retrieval import multi_query_retrieve


DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "resume_experience"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Optional: if your jd_parser.py exists and has parse_job_description()
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
        # Your jd_parser.py should expose parse_job_description(jd_text, ...)
        import jd_parser  # src/jd_parser.py
        if hasattr(jd_parser, "parse_job_description"):
            model = os.environ.get("ART_JD_MODEL", "gpt-4.1-nano-2025-04-14")
            # If your parse_job_description signature supports model, pass it; otherwise call without.
            try:
                return jd_parser.parse_job_description(jd_text, model=model)
            except TypeError:
                return jd_parser.parse_job_description(jd_text)
    except Exception as e:
        print(
            f"⚠️ JD parser unavailable or failed, falling back to manual queries. Reason: {e}")

    return None


def print_results(cands, show_hits=False, top=10):
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

    # Example JD (replace with real JD paste)
    jd_text = """
We are looking for a Python Engineer with experience in Computer Vision.
Must know PyTorch, OpenCV, and have experience deploying models to edge devices.
Bonus if you have worked with Docker and CI/CD pipelines.
""".strip()

    parsed = try_parse_jd(jd_text)

    if parsed is None:
        # Fallback: manual multi-query (still no hardcoding in retrieval logic)
        parsed = [
            "python computer vision pytorch opencv edge deployment",
            "docker ci/cd model deployment",
            "real-time inference performance optimization",
        ]

    cands = multi_query_retrieve(
        collection=collection,
        embedding_fn=ef,
        jd_parser_result=parsed,
        per_query_k=10,
        final_k=30,
    )

    print("\n🔎 RETRIEVAL RESULTS (multi-query + cosine rerank)")
    print("-" * 60)
    print_results(cands, show_hits=True, top=12)
