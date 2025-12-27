import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

# 1. Setup ChromaDB
# This creates a local folder 'data/processed/chroma_db' to store the vectors
client = chromadb.PersistentClient(path="data/processed/chroma_db")

# Use a free, local embedding model (no API key needed yet)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# Create (or get) a collection for your resume content
collection = client.get_or_create_collection(
    name="resume_experience", embedding_function=ef)


def load_data():
    with open("data/my_experience.json", "r") as f:
        return json.load(f)


def ingest():
    data = load_data()

    documents = []
    metadatas = []
    ids = []

    # 1. Ingest Work Experience
    print("Ingesting Work Experience...")
    for job in data.get("experiences", []):
        company = job["company"]
        role = job["role"]

        for bullet in job["bullets"]:
            # The "text" the AI searches:
            documents.append(bullet)

            # The "metadata" helps us filter later (e.g., only show "SaturnAI" bullets):
            metadatas.append({
                "type": "experience",
                "company": company,
                "role": role,
                "category": "work"  # generic tag
            })
            ids.append(str(uuid.uuid4()))

    # 2. Ingest Projects
    print("Ingesting Projects...")
    for project in data.get("projects", []):
        name = project["name"]
        tech = project["technologies"]

        for bullet in project["bullets"]:
            documents.append(bullet)
            metadatas.append({
                "type": "project",
                "name": name,
                "technologies": tech,
                "category": "project"
            })
            ids.append(str(uuid.uuid4()))

    # 3. Save to DB
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(
            f"Successfully stored {len(documents)} resume bullet points in ChromaDB!")
    else:
        print("No bullets found to ingest.")


if __name__ == "__main__":
    ingest()
