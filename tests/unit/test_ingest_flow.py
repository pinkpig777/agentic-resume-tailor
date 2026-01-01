import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.ingest import ingest


class DummyCollection:
    def __init__(self) -> None:
        self.add_calls = []

    def add(self, documents, metadatas, ids) -> None:
        self.add_calls.append({"documents": documents, "metadatas": metadatas, "ids": ids})


class DummyClient:
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        DummyClient.last_instance = self
        self.collection = DummyCollection()

    def delete_collection(self, name) -> None:
        return None

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        return self.collection


class DummyEmbedding:
    def __init__(self, *args, **kwargs) -> None:
        pass


class TestIngestFlow(unittest.TestCase):
    @patch("agentic_resume_tailor.ingest.embedding_functions.SentenceTransformerEmbeddingFunction", DummyEmbedding)
    @patch("agentic_resume_tailor.ingest.chromadb.PersistentClient", DummyClient)
    def test_ingest_counts_bullets(self) -> None:
        """Test ingest counts bullets from data input."""
        data = {
            "experiences": [
                {
                    "job_id": "job_a",
                    "company": "Acme",
                    "role": "Eng",
                    "dates": "2020",
                    "location": "Remote",
                    "bullets": [{"id": "b01", "text_latex": "Did X"}],
                }
            ],
            "projects": [
                {
                    "project_id": "proj_a",
                    "name": "Proj",
                    "technologies": "Python",
                    "bullets": [{"id": "b01", "text_latex": "Built Y"}],
                }
            ],
        }

        count = ingest(data=data)
        self.assertEqual(count, 2)

        client = DummyClient.last_instance
        self.assertIsNotNone(client)
        add_calls = client.collection.add_calls
        self.assertEqual(len(add_calls), 1)
        ids = add_calls[0]["ids"]
        self.assertEqual(set(ids), {"exp:job_a:b01", "proj:proj_a:b01"})

    @patch("agentic_resume_tailor.ingest.embedding_functions.SentenceTransformerEmbeddingFunction", DummyEmbedding)
    @patch("agentic_resume_tailor.ingest.chromadb.PersistentClient", DummyClient)
    def test_ingest_handles_empty(self) -> None:
        """Test ingest handles empty data."""
        data = {"experiences": [], "projects": []}
        count = ingest(data=data)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
