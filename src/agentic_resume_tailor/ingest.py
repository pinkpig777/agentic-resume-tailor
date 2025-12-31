import json
import logging
import re

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

DB_PATH = settings.db_path
COLLECTION_NAME = settings.collection_name
JSON_PATH = settings.data_file


def strip_latex(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ingest() -> int:
    logger.info("Initializing ChromaDB client")
    client = chromadb.PersistentClient(path=DB_PATH)

    logger.info("Loading embedding model")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)
    logger.info("Model loaded. Starting JSON processing.")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    # Calculate total items for the main progress bar
    total_items = len(data.get("experiences", [])) + len(data.get("projects", []))

    # 2. Initialize the main progress bar
    pbar = tqdm(total=total_items, desc="Processing Experience & Projects")

    # Experiences
    for exp in data.get("experiences", []):
        job_id = exp["job_id"]
        company = exp.get("company", "")
        role = exp.get("role", "")
        dates = exp.get("dates", "")
        location = exp.get("location", "")

        for b in exp.get("bullets", []):
            local_id = b["id"]
            text_latex = b["text_latex"]
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

        pbar.update(1)  # Update for each job processed

    # Projects
    for proj in data.get("projects", []):
        project_id = proj["project_id"]
        name = proj.get("name", "")
        technologies = proj.get("technologies", "")

        for b in proj.get("bullets", []):
            local_id = b["id"]
            text_latex = b["text_latex"]
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

        pbar.update(1)  # Update for each project processed

    pbar.close()

    if documents:
        logger.info("Generating embeddings and storing %s bullets", len(documents))
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Successfully stored in ChromaDB.")
        return len(documents)

    logger.warning("No bullets found to ingest.")
    return 0


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()
