import json
import re
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm  #

DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "resume_experience"
JSON_PATH = "data/my_experience.json"


def strip_latex(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ingest():
    print("⏳ Initializing ChromaDB Client...")
    client = chromadb.PersistentClient(path=DB_PATH)

    print("⏳ Loading Embedding Model (this usually takes a few seconds)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print("🚀 Model Loaded. Starting JSON processing...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    data = json.load(open(JSON_PATH, "r", encoding="utf-8"))

    documents = []
    metadatas = []
    ids = []

    # Calculate total items for the main progress bar
    total_items = len(data.get("experiences", [])) + \
        len(data.get("projects", []))

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
            metadatas.append({
                "section": "experience",
                "job_id": job_id,
                "company": company,
                "role": role,
                "dates": dates,
                "location": location,
                "local_bullet_id": local_id,
                "text_latex": text_latex,
            })
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
            metadatas.append({
                "section": "project",
                "project_id": project_id,
                "name": name,
                "technologies": technologies,
                "local_bullet_id": local_id,
                "text_latex": text_latex,
            })
            ids.append(bullet_id)

        pbar.update(1)  # Update for each project processed

    pbar.close()

    if documents:
        # 3. Add a specialized message for the embedding process
        print(
            f"✨ Generating embeddings and storing {len(documents)} bullets...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ Successfully stored in ChromaDB.")
    else:
        print("No bullets found to ingest.")


if __name__ == "__main__":
    ingest()
