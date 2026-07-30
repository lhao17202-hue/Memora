import json
import sqlite3

from memora.config import MemoryConfig
from memora.embeddings import EmbeddingVector, SparseVector
from memora.vector_store import SQLiteVectorStore, cosine_similarity


def test_cosine_similarity_orders_matching_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) > cosine_similarity([1.0, 0.0], [0.0, 1.0])


def test_sqlite_vector_store_upsert_search_delete_and_verify(tmp_path):
    store = SQLiteVectorStore(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    store.init_storage()

    store.upsert(
        "mem_1",
        EmbeddingVector(dense=[1.0, 0.0], sparse=SparseVector(indices=[1], values=[0.5])),
        {
            "memory_id": "mem_1",
            "user_id": "alice",
            "project_id": "project-a",
            "workspace_id": None,
            "type": "preference",
            "status": "active",
            "tags": ["preference"],
            "provider": "hash",
            "model": "memora-hash-v1",
            "dimension": 2,
            "text_hash": "hash-1",
        },
    )
    store.upsert(
        "mem_2",
        [0.0, 1.0],
        {
            "memory_id": "mem_2",
            "user_id": "bob",
            "project_id": "project-b",
            "workspace_id": None,
            "type": "project",
            "status": "active",
            "tags": ["tool"],
            "provider": "hash",
            "model": "memora-hash-v1",
            "dimension": 2,
            "text_hash": "hash-2",
        },
    )

    hits = store.search(EmbeddingVector(dense=[1.0, 0.0]), top_k=2)
    alice_hits = store.search(EmbeddingVector(dense=[1.0, 0.0]), top_k=2, filters={"user_id": "alice", "tags": ["preference"]})

    assert [hit.memory_id for hit in hits] == ["mem_1", "mem_2"]
    assert hits[0].dense_score == hits[0].score
    assert hits[0].source == "dense"
    assert [hit.memory_id for hit in alice_hits] == ["mem_1"]
    assert store.verify({"mem_1"})["vector_orphans"] == ["mem_2"]

    store.upsert(
        "mem_1",
        EmbeddingVector(dense=[0.0, 1.0]),
        {
            "memory_id": "mem_1",
            "user_id": "alice",
            "project_id": "project-a",
            "workspace_id": None,
            "type": "preference",
            "status": "active",
            "tags": ["preference"],
            "provider": "hash",
            "model": "memora-hash-v1",
            "dimension": 2,
            "text_hash": "hash-1-updated",
        },
    )
    updated_hits = store.search(EmbeddingVector(dense=[0.0, 1.0]), top_k=1, filters={"user_id": "alice"})

    assert updated_hits[0].memory_id == "mem_1"
    assert store.get_metadata("mem_1")["text_hash"] == "hash-1-updated"

    store.delete("mem_1")

    assert store.search(EmbeddingVector(dense=[0.0, 1.0]), top_k=2, filters={"user_id": "alice"}) == []
    assert store.verify({"mem_1", "mem_2"})["vector_missing"] == ["mem_1"]


def test_sqlite_vector_store_stores_only_dense_vector(tmp_path):
    store = SQLiteVectorStore(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))

    store.upsert(
        "mem_1",
        EmbeddingVector(dense=[1.0, 2.0], sparse=SparseVector(indices=[10], values=[0.75])),
        {"provider": "hash", "model": "memora-hash-v1", "dimension": 2, "text_hash": "hash"},
    )

    with sqlite3.connect(store.db_path) as connection:
        vector_json = connection.execute("SELECT vector_json FROM memory_vectors WHERE memory_id = ?", ("mem_1",)).fetchone()[0]

    assert json.loads(vector_json) == [1.0, 2.0]


def test_sqlite_vector_store_uses_sqlite_path_when_configured(tmp_path):
    db_path = tmp_path / "custom.sqlite3"
    store = SQLiteVectorStore(MemoryConfig(root_dir=tmp_path / ".memora", sqlite_path=db_path, rag_enabled=True))

    store.init_storage()

    assert db_path.exists()
