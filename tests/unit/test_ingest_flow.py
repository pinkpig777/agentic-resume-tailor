from agentic_resume_tailor import ingest as ingest_module


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


def test_ingest_counts_bullets(monkeypatch) -> None:
    """Test ingest counts bullets from data input."""
    monkeypatch.setattr(ingest_module.chromadb, "PersistentClient", DummyClient)
    monkeypatch.setattr(
        ingest_module.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        DummyEmbedding,
    )
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

    count = ingest_module.ingest(data=data)
    assert count == 2

    client = DummyClient.last_instance
    assert client is not None
    add_calls = client.collection.add_calls
    assert len(add_calls) == 1
    ids = add_calls[0]["ids"]
    assert set(ids) == {"exp:job_a:b01", "proj:proj_a:b01"}


def test_ingest_handles_empty(monkeypatch) -> None:
    """Test ingest handles empty data."""
    monkeypatch.setattr(ingest_module.chromadb, "PersistentClient", DummyClient)
    monkeypatch.setattr(
        ingest_module.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        DummyEmbedding,
    )
    data = {"experiences": [], "projects": []}
    count = ingest_module.ingest(data=data)
    assert count == 0
