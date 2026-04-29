from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

import chromadb
from tqdm import tqdm

from agentic_resume_tailor.db.session import SessionLocal, init_db
from agentic_resume_tailor.db.sync import export_resume_data, write_resume_json
from agentic_resume_tailor.settings import load_settings
from agentic_resume_tailor.utils.embeddings import build_sentence_transformer_ef
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def strip_latex(s: str) -> str:
    """Strip LaTeX markup for embedding-friendly text."""
    if not s:
        return ""
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def create_chroma_client(settings: Any | None = None) -> Any:
    settings = settings or load_settings()
    return chromadb.PersistentClient(path=settings.db_path)


def create_embedding_function(settings: Any | None = None) -> Any:
    settings = settings or load_settings()
    return build_sentence_transformer_ef(settings.embed_model, disable_progress=True)


def ingest(
    data: dict | None = None,
    json_path: str | None = None,
    *,
    settings: Any | None = None,
    client_factory: Callable[[Any | None], Any] | None = None,
    embedding_factory: Callable[[Any | None], Any] | None = None,
) -> int:
    """Ingest resume bullets into Chroma from JSON or DB."""
    settings = settings or load_settings()
    client_factory = client_factory or create_chroma_client
    embedding_factory = embedding_factory or create_embedding_function
    logger.info("Initializing ChromaDB client")
    client = client_factory(settings)

    logger.info("Loading embedding model")
    embedding_fn = embedding_factory(settings)
    logger.info("Model loaded. Starting JSON processing.")
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if data is None:
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with SessionLocal() as db:
                data = export_resume_data(db)

    documents = []
    metadatas = []
    ids = []

    total_items = len(data.get("experiences", [])) + len(data.get("projects", []))
    pbar = tqdm(total=total_items, desc="Processing Experience & Projects", disable=True)

    for exp in data.get("experiences", []):
        job_id = exp["job_id"]
        company = exp.get("company", "")
        role = exp.get("role", "")
        dates = exp.get("dates", "")
        location = exp.get("location", "")

        for bullet in exp.get("bullets", []):
            local_id = bullet["id"]
            text_latex = bullet["text_latex"]
            bullet_id = f"exp:{job_id}:{local_id}"

            documents.append(strip_latex(text_latex))
            metadatas.append(
                {
                    "section": "experience",
                    "job_id": job_id,
                    "company": company,
                    "role": role,
                    "dates": dates,
                    "location": location,
                    "local_bullet_id": local_id,
                    "text_latex": text_latex,
                }
            )
            ids.append(bullet_id)

        pbar.update(1)

    for proj in data.get("projects", []):
        project_id = proj["project_id"]
        name = proj.get("name", "")
        technologies = proj.get("technologies", "")

        for bullet in proj.get("bullets", []):
            local_id = bullet["id"]
            text_latex = bullet["text_latex"]
            bullet_id = f"proj:{project_id}:{local_id}"

            documents.append(strip_latex(text_latex))
            metadatas.append(
                {
                    "section": "project",
                    "project_id": project_id,
                    "name": name,
                    "technologies": technologies,
                    "local_bullet_id": local_id,
                    "text_latex": text_latex,
                }
            )
            ids.append(bullet_id)

        pbar.update(1)

    pbar.close()

    if documents:
        logger.info("Generating embeddings and storing %s bullets", len(documents))
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Successfully stored in ChromaDB.")
        return len(documents)

    logger.warning("No bullets found to ingest.")
    return 0


def main() -> None:
    """Export current DB to JSON and ingest into Chroma."""
    settings = load_settings()
    init_db()
    with SessionLocal() as db:
        write_resume_json(db, settings.export_file)
    ingest(json_path=settings.export_file, settings=settings)


if __name__ == "__main__":
    main()
