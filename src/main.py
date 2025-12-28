import os
import chromadb
from chromadb.utils import embedding_functions
from jd_parser import parse_job_description


def get_tailored_resume(jd_text):
    # 1. Parse the Job Description
    print("1.  Analyzing JD...")
    requirements = parse_job_description(jd_text)
    print(
        f"   -> Extracted {len(requirements.experience_queries)} search queries.")

    # 2. Setup Database Connection
    client = chromadb.PersistentClient(path="data/processed/chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(
        name="resume_experience", embedding_function=ef)

    # 3. Search for matching bullets
    print("2.  Searching Resume Database...")
    selected_bullets = set()

    for query in requirements.experience_queries:
        print(f"   🔎 Query: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=2,  # Get top 2 matches per query
            include=["documents", "metadatas"]
        )

        # Add unique found bullets to our list
        for idx, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][idx]
            # Format: [Company] Bullet text
            formatted_bullet = f"[{meta.get('company', 'Project')}] {doc}"
            selected_bullets.add(formatted_bullet)

    return list(selected_bullets)


if __name__ == "__main__":
    # Real test: Paste a real JD snippet here!
    test_jd = """
    Software Engineer - AI Platform
    We need a backend engineer who knows Python and FastAPl.
    Experience with Vector Databases (Chroma/Pinecone) and LLM integration is a huge plus.
    Must have experience building REST APIs and working with Docker.
    """

    print("\n🚀 Starting Resume Tailoring Engine...\n")
    tailored_content = get_tailored_resume(test_jd)

    print("\n✅ SELECTED CONTENT FOR THIS JOB:")
    print("-" * 50)
    for bullet in tailored_content:
        print(f"• {bullet}")
    print("-" * 50)
