import chromadb
from chromadb.utils import embedding_functions

# 1. Connect to the Database
# We use the exact same path where ingest.py stored the data
client = chromadb.PersistentClient(path="data/processed/chroma_db")

# 2. Load the Embedding Function
# This converts text into numbers so we can compare "meaning"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# 3. Get Your Collection
collection = client.get_collection(
    name="resume_experience", embedding_function=ef)


def test_query(query_text):
    print(f"\n🔎 QUESTION: '{query_text}'")
    print("-" * 50)

    # Ask the database for the top 3 most relevant bullets
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # Print results
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        source = meta.get('company') or meta.get('name')
        print(f"[{source}]")
        print(f"👉 {doc}\n")


if __name__ == "__main__":
    # Test 1: Ask about specific tech (Python/Video)
    test_query(
        "Experience with high performance video processing and Python threading")

    # Test 2: Ask about a soft skill or leadership
    test_query("Leading teams and optimizing workflows")
